"""HTTP transports that put Habits under an SDK client.

The OpenAI and Anthropic SDKs, LangChain's chat models and most other
clients send requests through an httpx client. ThinkLess replaces that
client's transport with one that looks at every chat completion request:
answered locally when ThinkLess may, sent on unchanged otherwise. Every other
request passes straight through.

``httpx`` 0.x and ``httpx2`` (used by ``openai`` 3 and ``anthropic`` 1) share
the transport API, so the transport classes are built for whichever library
the client uses.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator, Callable, Iterator
from functools import cache
from types import ModuleType
from typing import Any

from ..logs import get_logger
from .apis import APIS, Api, api_for_path
from .runtime import Habits, Ticket, default_habits

logger = get_logger("habits.transport")

__all__ = ["async_http_client", "http_client", "sdk_http_library", "transport_classes", "wrap"]

SITE_HEADER = "x-thinkless-site"
DROPPED = ("content-encoding", "content-length", "transfer-encoding")


def _api(request: Any) -> Api | None:
    """The LLM API of a request, or ``None`` for requests ThinkLess passes straight on."""
    if request.method != "POST":
        return None
    return api_for_path(request.url.path)


def _body(request: Any) -> Any:
    try:
        return json.loads(request.content or b"null")
    except (ValueError, UnicodeDecodeError):
        return None


def _headers(response: Any) -> list[tuple[str, str]]:
    return [(k, v) for k, v in response.headers.items() if k.lower() not in DROPPED]


def _begin(habits: Habits, request: Any, api: Api) -> Ticket | None:
    """``habits.begin``, failing open: any error sends the call to the LLM unchanged."""
    try:
        return habits.begin(_body(request), request.headers, request.url.host, api)
    except Exception:
        logger.exception("ThinkLess failed on a request; it goes to the LLM unchanged")
        return None


def _answer(habits: Habits, ticket: Ticket) -> tuple[bytes, dict[str, str]] | None:
    """The local answer's body and headers, or ``None`` to send the call to the LLM instead."""
    try:
        parts = _local_parts(ticket)
    except Exception:
        logger.exception("ThinkLess could not build a local answer; the call goes to the LLM")
        return None
    try:
        habits.served(ticket)
    except Exception:
        logger.exception("ThinkLess could not record a local answer")
    return parts


def _finish(habits: Habits, ticket: Ticket, body: Any, status_code: int) -> None:
    try:
        habits.finish(ticket, body, status_code=status_code)
    except Exception:
        logger.exception("ThinkLess could not record an LLM answer")


def _local_parts(ticket: Ticket) -> tuple[bytes, dict[str, str]]:
    assert ticket.response is not None
    confidence = ticket.extra.get("confidence", 0.0)
    headers = {
        "x-thinkless-habit": "local",
        "x-thinkless-site": ticket.site or "",
        "x-thinkless-confidence": f"{confidence:.4f}",
    }
    if ticket.call.stream:
        headers["content-type"] = "text/event-stream"
        return APIS[ticket.call.api].build_stream(ticket.call, ticket.response), headers
    headers["content-type"] = "application/json"
    return json.dumps(ticket.response, ensure_ascii=False).encode("utf-8"), headers


@cache
def transport_classes(module_name: str = "httpx") -> tuple[type, type]:
    """The sync and async Habits transports for ``httpx`` or ``httpx2``."""
    lib: ModuleType = importlib.import_module(module_name)

    def upstream(request: Any, content: bytes) -> Any:
        headers = [(k, v) for k, v in request.headers.items() if k.lower() != SITE_HEADER]
        headers = [(k, v) for k, v in headers if k.lower() != "accept-encoding"]
        # Uncompressed bodies let ThinkLess read the answer without decoding twice.
        headers.append(("accept-encoding", "identity"))
        return lib.Request(
            request.method,
            request.url,
            headers=headers,
            content=content,
            extensions=request.extensions,
        )

    class TeeStream(lib.SyncByteStream):  # type: ignore[name-defined]
        def __init__(self, stream: Any, done: Callable[[list[bytes]], None]) -> None:
            self._stream = stream
            self._done = done
            self._chunks: list[bytes] = []
            self._finished = False

        def __iter__(self) -> Iterator[bytes]:
            for chunk in self._stream:
                self._chunks.append(chunk)
                yield chunk

        def close(self) -> None:
            self._stream.close()
            if not self._finished:
                self._finished = True
                self._done(self._chunks)

    class AsyncTeeStream(lib.AsyncByteStream):  # type: ignore[name-defined]
        def __init__(self, stream: Any, done: Callable[[list[bytes]], None]) -> None:
            self._stream = stream
            self._done = done
            self._chunks: list[bytes] = []
            self._finished = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            async for chunk in self._stream:
                self._chunks.append(chunk)
                yield chunk

        async def aclose(self) -> None:
            await self._stream.aclose()
            if not self._finished:
                self._finished = True
                self._done(self._chunks)

    def finisher(habits: Habits, ticket: Ticket) -> Callable[[list[bytes]], None]:
        def done(chunks: list[bytes]) -> None:
            try:
                body = APIS[ticket.call.api].collect_stream(chunks)
            except Exception:
                logger.exception("ThinkLess could not read a streamed answer")
                return
            _finish(habits, ticket, body, 200)

        return done

    class HabitsTransport(lib.BaseTransport):  # type: ignore[name-defined]
        """Sends chat completions through Habits, everything else straight on."""

        def __init__(self, habits: Habits | None = None, inner: Any = None) -> None:
            self.habits = habits or default_habits()
            self.inner = inner if inner is not None else lib.HTTPTransport()

        def handle_request(self, request: Any) -> Any:
            api = _api(request)
            if api is None:
                return self.inner.handle_request(request)
            content = request.read()
            ticket = _begin(self.habits, request, api)
            if ticket is None:
                return self.inner.handle_request(request)
            if ticket.answered_locally:
                parts = _answer(self.habits, ticket)
                if parts is not None:
                    return lib.Response(200, headers=parts[1], content=parts[0], request=request)
                ticket.response = None
            response = self.inner.handle_request(upstream(request, content))
            if ticket.call.stream and response.status_code == 200:
                return lib.Response(
                    response.status_code,
                    headers=response.headers,
                    stream=TeeStream(response.stream, finisher(self.habits, ticket)),
                    request=request,
                    extensions=response.extensions,
                )
            data = response.read()
            parsed = None
            if response.status_code == 200:
                try:
                    parsed = json.loads(data)
                except ValueError:
                    parsed = None
            _finish(self.habits, ticket, parsed, response.status_code)
            return lib.Response(
                response.status_code,
                headers=_headers(response),
                content=data,
                request=request,
                extensions={k: v for k, v in response.extensions.items() if k != "network_stream"},
            )

        def close(self) -> None:
            self.inner.close()

    class AsyncHabitsTransport(lib.AsyncBaseTransport):  # type: ignore[name-defined]
        """The async version of :class:`HabitsTransport`."""

        def __init__(self, habits: Habits | None = None, inner: Any = None) -> None:
            self.habits = habits or default_habits()
            self.inner = inner if inner is not None else lib.AsyncHTTPTransport()

        async def handle_async_request(self, request: Any) -> Any:
            api = _api(request)
            if api is None:
                return await self.inner.handle_async_request(request)
            content = await request.aread()
            ticket = _begin(self.habits, request, api)
            if ticket is None:
                return await self.inner.handle_async_request(request)
            if ticket.answered_locally:
                parts = _answer(self.habits, ticket)
                if parts is not None:
                    return lib.Response(200, headers=parts[1], content=parts[0], request=request)
                ticket.response = None
            response = await self.inner.handle_async_request(upstream(request, content))
            if ticket.call.stream and response.status_code == 200:
                return lib.Response(
                    response.status_code,
                    headers=response.headers,
                    stream=AsyncTeeStream(response.stream, finisher(self.habits, ticket)),
                    request=request,
                    extensions=response.extensions,
                )
            data = await response.aread()
            parsed = None
            if response.status_code == 200:
                try:
                    parsed = json.loads(data)
                except ValueError:
                    parsed = None
            _finish(self.habits, ticket, parsed, response.status_code)
            return lib.Response(
                response.status_code,
                headers=_headers(response),
                content=data,
                request=request,
                extensions={k: v for k, v in response.extensions.items() if k != "network_stream"},
            )

        async def aclose(self) -> None:
            await self.inner.aclose()

    return HabitsTransport, AsyncHabitsTransport


def sdk_http_library() -> str:
    """``httpx2`` when the installed OpenAI SDK uses it, ``httpx`` otherwise."""
    try:
        import openai
    except ImportError:
        return "httpx"
    base = getattr(openai, "DefaultHttpxClient", None)
    for cls in getattr(base, "__mro__", ()):
        root = cls.__module__.split(".")[0]
        if root in ("httpx", "httpx2"):
            return root
    return "httpx"


def http_client(
    habits: Habits | None = None,
    *,
    library: str | None = None,
    transport: Any = None,
    **kwargs: Any,
) -> Any:
    """An httpx client that sends chat completions through Habits.

    Pass it where a library takes an HTTP client, for example
    ``ChatOpenAI(http_client=http_client())`` or
    ``OpenAI(http_client=http_client())``.

    Args:
        habits: The Habits to use. Defaults to the process-wide one.
        library: ``httpx`` or ``httpx2``. Defaults to what the installed
            OpenAI SDK expects.
        transport: The transport that reaches the LLM, of the same library.
            Defaults to a plain HTTP transport.
        **kwargs: Passed to the client, for example ``timeout``.
    """
    name = library or sdk_http_library()
    lib = importlib.import_module(name)
    sync, _ = transport_classes(name)
    return lib.Client(transport=sync(habits, inner=transport), **kwargs)


def async_http_client(
    habits: Habits | None = None,
    *,
    library: str | None = None,
    transport: Any = None,
    **kwargs: Any,
) -> Any:
    """The async version of :func:`http_client`."""
    name = library or sdk_http_library()
    lib = importlib.import_module(name)
    _, asynchronous = transport_classes(name)
    return lib.AsyncClient(transport=asynchronous(habits, inner=transport), **kwargs)


def _wrap_http(http: Any, habits: Habits | None) -> bool:
    module = type(http).__module__.split(".")[0]
    if module not in ("httpx", "httpx2") or not hasattr(http, "_transport"):
        return False
    lib = importlib.import_module(module)
    sync, asynchronous = transport_classes(module)
    cls = asynchronous if isinstance(http, lib.AsyncClient) else sync
    if not isinstance(http._transport, cls):
        http._transport = cls(habits, inner=http._transport)
    mounts = getattr(http, "_mounts", None)
    if isinstance(mounts, dict):
        for pattern, mounted in list(mounts.items()):
            if mounted is not None and not isinstance(mounted, cls):
                mounts[pattern] = cls(habits, inner=mounted)
    return True


def wrap(client: Any, habits: Habits | None = None) -> Any:
    """Put Habits under an existing client, in place, and return it.

    Works with ``openai.OpenAI`` and ``openai.AsyncOpenAI`` (and anything
    built on them, such as OpenRouter, Azure, Groq, vLLM or Ollama clients),
    LangChain's ``ChatOpenAI``, and a bare ``httpx`` or ``httpx2`` client.

    Args:
        client: The client to wrap.
        habits: The Habits to use. Defaults to the process-wide one.

    Raises:
        TypeError: If the client does not expose an httpx transport.
    """
    wrapped = False
    for attr in ("root_client", "root_async_client"):
        inner = getattr(client, attr, None)
        if inner is not None:
            wrap(inner, habits)
            wrapped = True
    if wrapped:
        return client
    http = getattr(client, "_client", None)
    if http is not None and _wrap_http(http, habits):
        return client
    if _wrap_http(client, habits):
        return client
    raise TypeError(
        f"cannot put Habits under {type(client).__name__}: it does not expose an httpx "
        "transport. Pass http_client=thinkless.habits.http_client() when creating it, "
        "or run `thinkless habits proxy` and point its base URL there."
    )
