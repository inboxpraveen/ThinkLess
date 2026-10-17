"""The LLM APIs ThinkLess understands.

* ``chat``: OpenAI Chat Completions (``POST .../chat/completions``), also
  spoken by OpenRouter, Azure OpenAI, Groq, Together, Fireworks, DeepSeek,
  Mistral, Ollama, vLLM, LM Studio and llama.cpp.
* ``responses``: the OpenAI Responses API (``POST .../responses``), the
  default of the OpenAI Agents SDK.
* ``messages``: the Anthropic Messages API (``POST .../v1/messages``).

Each API turns request and response bodies into the same :class:`Call` and
:class:`Output`, so sites, students and the gate work the same way for all
of them. Local answers to streaming requests are built for Chat Completions;
streaming Responses and Messages calls are recorded and learned from, and
always answered by the model.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from .wire import (
    Call,
    Message,
    Output,
    answer_payload,
    build_response,
    build_stream,
    collect_stream,
    fingerprint,
    make_output,
    parse_request,
    parse_response,
)

__all__ = ["APIS", "Api", "api_for_path", "api_for_record"]


class Api(Protocol):
    name: str
    streams_locally: bool

    def matches(self, path: str) -> bool: ...

    def parse_request(self, body: Any, headers: Mapping[str, str] | None = None) -> Call | None: ...

    def parse_response(self, call: Call, body: Any) -> Output | None: ...

    def build_response(
        self, call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def build_stream(self, call: Call, response: Mapping[str, Any]) -> bytes: ...

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None: ...


def _events(chunks: Iterable[bytes]) -> list[dict[str, Any]]:
    """The JSON payloads of a server-sent event stream."""
    text = b"".join(chunks).decode("utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _site_name(headers: Mapping[str, str] | None) -> str | None:
    return (headers or {}).get("x-thinkless-site") or None


def _reason(has_media: bool, contract: str, extra: str = "") -> str:
    if has_media:
        return "the input holds images, audio or files"
    if extra:
        return extra
    if contract == "tools":
        return "the model may call tools"
    return ""


class ChatApi:
    """OpenAI Chat Completions."""

    name = "chat"
    streams_locally = True

    def matches(self, path: str) -> bool:
        return path.rstrip("/").endswith("/chat/completions")

    def parse_request(self, body: Any, headers: Mapping[str, str] | None = None) -> Call | None:
        return parse_request(body, headers)

    def parse_response(self, call: Call, body: Any) -> Output | None:
        return parse_response(call, body)

    def build_response(
        self, call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
    ) -> dict[str, Any]:
        return build_response(call, fields, template)

    def build_stream(self, call: Call, response: Mapping[str, Any]) -> bytes:
        return build_stream(call, response)

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None:
        return collect_stream(chunks)


class ResponsesApi:
    """The OpenAI Responses API."""

    name = "responses"
    streams_locally = False

    def matches(self, path: str) -> bool:
        return path.rstrip("/").endswith("/responses")

    def parse_request(self, body: Any, headers: Mapping[str, str] | None = None) -> Call | None:
        if not isinstance(body, Mapping) or "input" not in body or "model" not in body:
            return None
        messages: list[Message] = []
        has_media = False
        if isinstance(body.get("instructions"), str):
            messages.append(Message("system", body["instructions"]))
        items = body["input"]
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        if not isinstance(items, Sequence):
            return None
        for item in items:
            if not isinstance(item, Mapping):
                return None
            kind = item.get("type", "message")
            if kind == "message" or "role" in item:
                text, other = _parts(item.get("content"))
                has_media = has_media or other
                messages.append(Message(str(item.get("role") or "user"), text))
            elif kind == "function_call":
                messages.append(
                    Message("assistant", f"{item.get('name')}({item.get('arguments') or ''})")
                )
            elif kind == "function_call_output":
                messages.append(Message("tool", str(item.get("output") or "")))
            else:
                has_media = True
        contract, definition, schema, name = self._contract(body)
        extra = ""
        if body.get("previous_response_id"):
            extra = "the conversation continues a stored response ThinkLess cannot see"
        reason = _reason(has_media, contract, extra) or ("no input" if not messages else "")
        return Call(
            model=str(body.get("model") or ""),
            messages=tuple(messages),
            contract=contract,
            contract_id=fingerprint([self.name, contract, definition]),
            schema=schema,
            schema_name=str(name) if name else None,
            stream=bool(body.get("stream")),
            answerable=not reason,
            reason=reason,
            site_name=_site_name(headers),
            body=body,
            api=self.name,
        )

    @staticmethod
    def _contract(body: Mapping[str, Any]) -> tuple[str, Any, dict[str, Any] | None, str | None]:
        tools = [t for t in body.get("tools") or [] if isinstance(t, Mapping)]
        choice = body.get("tool_choice")
        if isinstance(choice, Mapping) and choice.get("type") == "function" and choice.get("name"):
            forced = str(choice["name"])
            for tool in tools:
                if tool.get("name") == forced:
                    schema = tool.get("parameters")
                    return "tool", tool, schema if isinstance(schema, dict) else None, forced
            return "tool", {"name": forced}, None, forced
        if tools and choice != "none":
            return "tools", {"tools": tools, "choice": choice}, None, None
        text = body.get("text")
        fmt = text.get("format") if isinstance(text, Mapping) else None
        if isinstance(fmt, Mapping):
            if fmt.get("type") == "json_schema":
                schema = fmt.get("schema")
                return (
                    "json_schema",
                    fmt,
                    schema if isinstance(schema, dict) else None,
                    fmt.get("name"),
                )
            if fmt.get("type") == "json_object":
                return "json_object", "json_object", None, None
        return "text", "text", None, None

    def parse_response(self, call: Call, body: Any) -> Output | None:
        if not isinstance(body, Mapping) or body.get("status") not in (None, "completed"):
            return None
        texts: list[str] = []
        tool_calls: list[tuple[str, Any]] = []
        for item in body.get("output") or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, Mapping) and part.get("type") == "refusal":
                        return None
                    if isinstance(part, Mapping) and part.get("type") == "output_text":
                        texts.append(str(part.get("text") or ""))
            elif item.get("type") == "function_call":
                tool_calls.append((str(item.get("name")), item.get("arguments") or "{}"))
        usage = _mapping(body.get("usage"))
        return make_output(
            call,
            text="".join(texts),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            model=body.get("model"),
            usage=(int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), None),
            raw=body,
        )

    def build_response(
        self, call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
    ) -> dict[str, Any]:
        text, arguments = answer_payload(call, fields, template)
        if call.contract == "tool":
            item: dict[str, Any] = {
                "type": "function_call",
                "id": f"fc_thinkless_{uuid.uuid4().hex[:20]}",
                "call_id": f"call_{uuid.uuid4().hex[:24]}",
                "name": call.schema_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
                "status": "completed",
            }
        else:
            item = {
                "type": "message",
                "id": f"msg_thinkless_{uuid.uuid4().hex[:20]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text or "", "annotations": []}],
            }
        body = call.body
        return {
            "id": f"resp_thinkless_{uuid.uuid4().hex[:20]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": call.model,
            "output": [item],
            "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
            "tool_choice": body.get("tool_choice", "auto"),
            "tools": list(body.get("tools") or []),
            "text": body.get("text") or {"format": {"type": "text"}},
            "error": None,
            "incomplete_details": None,
            "instructions": body.get("instructions"),
            "metadata": body.get("metadata") or {},
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 0,
            },
        }

    def build_stream(self, call: Call, response: Mapping[str, Any]) -> bytes:
        raise NotImplementedError("streaming Responses calls are always sent to the model")

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None:
        for event in reversed(_events(chunks)):
            if event.get("type") == "response.completed" and isinstance(
                event.get("response"), dict
            ):
                return dict(event["response"])
        return None


def _parts(content: Any) -> tuple[str, bool]:
    """Text of Responses or Anthropic content, and whether it held anything else."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    texts: list[str] = []
    other = False
    for part in content if isinstance(content, Sequence) else [content]:
        if not isinstance(part, Mapping):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            texts.append(str(part.get("text") or ""))
        elif kind == "tool_use":
            texts.append(f"{part.get('name')}({json.dumps(part.get('input'), sort_keys=True)})")
        elif kind == "tool_result":
            inner, nested = _parts(part.get("content"))
            texts.append(inner)
            other = other or nested
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            other = True
    return "\n".join(texts), other


class MessagesApi:
    """The Anthropic Messages API."""

    name = "messages"
    streams_locally = False

    def matches(self, path: str) -> bool:
        return path.rstrip("/").endswith("/messages")

    def parse_request(self, body: Any, headers: Mapping[str, str] | None = None) -> Call | None:
        if not isinstance(body, Mapping) or "max_tokens" not in body:
            return None
        raw = body.get("messages")
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            return None
        messages: list[Message] = []
        has_media = False
        system = body.get("system")
        if system:
            text, other = _parts(system)
            has_media = has_media or other
            messages.append(Message("system", text))
        for item in raw:
            if not isinstance(item, Mapping):
                return None
            content = item.get("content")
            blocks = content if isinstance(content, list) else []
            role = str(item.get("role") or "user")
            if role == "user" and any(
                isinstance(b, Mapping) and b.get("type") == "tool_result" for b in blocks
            ):
                role = "tool"
            text, other = _parts(content)
            has_media = has_media or other
            messages.append(Message(role, text))
        contract, definition, schema, name = self._contract(body)
        extra = ""
        thinking = body.get("thinking")
        if isinstance(thinking, Mapping) and thinking.get("type") not in (None, "disabled"):
            extra = "extended thinking was requested"
        reason = _reason(has_media, contract, extra) or ("no messages" if not messages else "")
        return Call(
            model=str(body.get("model") or ""),
            messages=tuple(messages),
            contract=contract,
            contract_id=fingerprint([self.name, contract, definition]),
            schema=schema,
            schema_name=str(name) if name else None,
            stream=bool(body.get("stream")),
            answerable=not reason,
            reason=reason,
            site_name=_site_name(headers),
            body=body,
            api=self.name,
        )

    @staticmethod
    def _contract(body: Mapping[str, Any]) -> tuple[str, Any, dict[str, Any] | None, str | None]:
        tools = [t for t in body.get("tools") or [] if isinstance(t, Mapping)]
        choice = _mapping(body.get("tool_choice"))
        forced = None
        if choice.get("type") == "tool" and choice.get("name"):
            forced = str(choice["name"])
        elif choice.get("type") == "any" and len(tools) == 1:
            forced = str(tools[0].get("name"))
        if forced is not None:
            for tool in tools:
                if tool.get("name") == forced:
                    schema = tool.get("input_schema")
                    return "tool", tool, schema if isinstance(schema, dict) else None, forced
            return "tool", {"name": forced}, None, forced
        if tools and choice.get("type") != "none":
            return "tools", {"tools": tools, "choice": choice}, None, None
        fmt = body.get("output_format")
        if isinstance(fmt, Mapping) and fmt.get("type") == "json_schema":
            schema = fmt.get("schema")
            return "json_schema", fmt, schema if isinstance(schema, dict) else None, None
        return "text", "text", None, None

    def parse_response(self, call: Call, body: Any) -> Output | None:
        if not isinstance(body, Mapping) or body.get("type") not in (None, "message"):
            return None
        if body.get("stop_reason") == "refusal":
            return None
        texts: list[str] = []
        tool_calls: list[tuple[str, Any]] = []
        for block in body.get("content") or []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                tool_calls.append((str(block.get("name")), block.get("input") or {}))
        usage = _mapping(body.get("usage"))
        return make_output(
            call,
            text="".join(texts),
            tool_calls=tool_calls,
            finish_reason=body.get("stop_reason"),
            model=body.get("model"),
            usage=(int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), None),
            raw=body,
        )

    def build_response(
        self, call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
    ) -> dict[str, Any]:
        text, arguments = answer_payload(call, fields, template)
        if call.contract == "tool":
            content: list[dict[str, Any]] = [
                {
                    "type": "tool_use",
                    "id": f"toolu_thinkless_{uuid.uuid4().hex[:18]}",
                    "name": call.schema_name,
                    "input": arguments,
                }
            ]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": text or ""}]
            stop = "end_turn"
        return {
            "id": f"msg_thinkless_{uuid.uuid4().hex[:20]}",
            "type": "message",
            "role": "assistant",
            "model": call.model,
            "content": content,
            "stop_reason": template.get("finish_reason") or stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def build_stream(self, call: Call, response: Mapping[str, Any]) -> bytes:
        raise NotImplementedError("streaming Messages calls are always sent to the model")

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None:
        message: dict[str, Any] | None = None
        blocks: dict[int, dict[str, Any]] = {}
        partial: dict[int, list[str]] = {}
        for event in _events(chunks):
            kind = event.get("type")
            if kind == "message_start" and isinstance(event.get("message"), dict):
                message = dict(event["message"])
                message["content"] = []
            elif kind == "content_block_start":
                index = int(event.get("index", 0))
                blocks[index] = dict(event.get("content_block") or {})
                partial[index] = []
            elif kind == "content_block_delta":
                index = int(event.get("index", 0))
                delta = event.get("delta") or {}
                block = blocks.setdefault(index, {"type": "text", "text": ""})
                if delta.get("type") == "text_delta":
                    block["text"] = str(block.get("text") or "") + str(delta.get("text") or "")
                elif delta.get("type") == "input_json_delta":
                    partial.setdefault(index, []).append(str(delta.get("partial_json") or ""))
            elif kind == "message_delta" and message is not None:
                delta = event.get("delta") or {}
                if delta.get("stop_reason"):
                    message["stop_reason"] = delta["stop_reason"]
                usage = event.get("usage")
                if isinstance(usage, Mapping):
                    message.setdefault("usage", {}).update(usage)
        if message is None:
            return None
        for index in sorted(blocks):
            block = blocks[index]
            if block.get("type") == "tool_use" and partial.get(index):
                try:
                    block["input"] = json.loads("".join(partial[index]))
                except json.JSONDecodeError:
                    return None
            message["content"].append(block)
        return message


CHAT = ChatApi()
RESPONSES = ResponsesApi()
MESSAGES = MessagesApi()
APIS: dict[str, Api] = {"chat": CHAT, "responses": RESPONSES, "messages": MESSAGES}


def api_for_path(path: str) -> Api | None:
    """The API a request path belongs to, or ``None`` for everything else."""
    for api in (CHAT, RESPONSES, MESSAGES):
        if api.matches(path):
            return api
    return None


def api_for_record(request: Any, response: Any = None) -> Api:
    """Guess the API of a logged request and response pair from their shapes."""
    if isinstance(request, Mapping):
        if "input" in request and "messages" not in request:
            return RESPONSES
        if "max_tokens" in request and (
            "system" in request
            or (isinstance(response, Mapping) and response.get("type") == "message")
        ):
            return MESSAGES
    return CHAT
