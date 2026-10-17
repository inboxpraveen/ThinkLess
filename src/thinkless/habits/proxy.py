"""An OpenAI-compatible proxy with Habits inside, for clients in any language.

Point any OpenAI-compatible client at the proxy instead of the provider::

    thinkless habits proxy --upstream https://api.openai.com/v1
    export OPENAI_BASE_URL=http://localhost:8787/v1

Every request goes to the upstream unchanged, with the caller's own API key,
except chat completions for sites ThinkLess has proven, which it answers
itself. ``GET /habits/status`` returns the current plan.

One proxy in front of many workers also gives them one shared call log and
one set of students.

Needs ``pip install "thinkless[server]"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .._version import __version__
from .runtime import Habits
from .transport import transport_classes

__all__ = ["create_proxy"]

HOP_BY_HOP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "proxy-authorization",
    "te",
    "trailer",
}
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


async def _body(response: httpx.Response) -> AsyncIterator[bytes]:
    """The response body as it arrived, or as ThinkLess built it.

    Responses ThinkLess answered or read in full already hold their content,
    decoded; everything else streams through untouched.
    """
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    except httpx.StreamConsumed:
        yield response.content


def create_proxy(
    upstream: str, habits: Habits, *, timeout: float = 600.0, transport: Any = None
) -> FastAPI:
    """The proxy app.

    Args:
        upstream: The provider's API base URL, for example
            ``https://api.openai.com/v1`` or ``https://openrouter.ai/api/v1``.
        habits: The Habits that answers and records.
        timeout: Seconds to wait for the upstream.
        transport: The httpx transport that reaches the upstream. Defaults to
            a plain HTTP transport; tests pass a mock.
    """
    base = upstream.rstrip("/")
    _, habits_transport = transport_classes("httpx")
    client = httpx.AsyncClient(transport=habits_transport(habits, inner=transport), timeout=timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()
        habits.close()

    app = FastAPI(title="ThinkLess Habits proxy", version=__version__, lifespan=lifespan)

    @app.get("/habits/status")
    async def status() -> dict[str, Any]:
        plan = habits.plan or {}
        return {
            "version": __version__,
            "mode": habits.mode,
            "upstream": base,
            "counts": dict(habits.counts),
            "sites": {
                site_id: {
                    key: entry.get(key)
                    for key in ("name", "status", "threshold", "evidence", "reason", "calls")
                }
                for site_id, entry in plan.get("sites", {}).items()
            },
        }

    @app.api_route("/{path:path}", methods=METHODS)
    async def forward(path: str, request: Request) -> Any:
        # Clients use http://proxy/v1 as their base URL; the upstream URL
        # already carries its own version prefix.
        if path.startswith("v1/"):
            path = path[3:]
        url = f"{base}/{path}"
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP]
        body = await request.body()
        outgoing = client.build_request(
            request.method, url, params=request.query_params, headers=headers, content=body
        )
        try:
            response = await client.send(outgoing, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": {"message": f"upstream request failed: {exc}", "type": "proxy_error"}},
                status_code=502,
            )
        reply_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP}
        return StreamingResponse(
            _body(response),
            status_code=response.status_code,
            headers=reply_headers,
            background=BackgroundTask(response.aclose),
        )

    return app
