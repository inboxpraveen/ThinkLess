"""The LLM APIs ThinkLess understands.

* ``chat``: OpenAI Chat Completions (``POST .../chat/completions``), also
  spoken by OpenRouter, Azure OpenAI, Groq, Together, Fireworks, DeepSeek,
  Mistral, Ollama, vLLM, LM Studio and llama.cpp.
* ``responses``: the OpenAI Responses API (``POST .../responses``), the
  default of the OpenAI Agents SDK.
* ``messages``: the Anthropic Messages API (``POST .../v1/messages``).
* ``gemini``: the Gemini API (``POST .../models/<model>:generateContent``
  and ``:streamGenerateContent``), on Google AI and Vertex AI.
* ``converse``: the Amazon Bedrock Converse API, through boto3's events
  rather than HTTP (see :mod:`thinkless.habits.bedrock`).

Each API turns request and response bodies into the same :class:`Call` and
:class:`Output`, so sites, students and the gate work the same way for all
of them, and a local answer to a streaming request arrives as the event
stream the API defines.
"""

from __future__ import annotations

import json
import re
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
    streams_locally = True

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
        elif body.get("conversation"):
            extra = "the call belongs to a stored conversation ThinkLess cannot see"
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
        """The events a streaming Responses client expects, for a finished response."""
        events: list[dict[str, Any]] = []

        def emit(kind: str, **fields: Any) -> None:
            events.append({"type": kind, "sequence_number": len(events), **fields})

        started = {**response, "status": "in_progress", "output": [], "usage": None}
        emit("response.created", response=started)
        emit("response.in_progress", response=started)
        for index, item in enumerate(response.get("output") or []):
            if item.get("type") == "function_call":
                emit(
                    "response.output_item.added",
                    output_index=index,
                    item={**item, "arguments": "", "status": "in_progress"},
                )
                emit(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=index,
                    delta=item["arguments"],
                )
                emit(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=index,
                    name=item.get("name"),
                    arguments=item["arguments"],
                )
            else:
                emit(
                    "response.output_item.added",
                    output_index=index,
                    item={**item, "content": [], "status": "in_progress"},
                )
                for part_index, part in enumerate(item.get("content") or []):
                    where = {
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": part_index,
                    }
                    emit("response.content_part.added", **where, part={**part, "text": ""})
                    emit("response.output_text.delta", **where, delta=part["text"], logprobs=[])
                    emit("response.output_text.done", **where, text=part["text"], logprobs=[])
                    emit("response.content_part.done", **where, part=part)
            emit("response.output_item.done", output_index=index, item=item)
        emit("response.completed", response=dict(response))
        return _sse(events)

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None:
        for event in reversed(_events(chunks)):
            if event.get("type") == "response.completed" and isinstance(
                event.get("response"), dict
            ):
                return dict(event["response"])
        return None


def _sse(events: Sequence[Mapping[str, Any]]) -> bytes:
    """Server-sent events, each named by its type, as both APIs send them."""
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    ).encode("utf-8")


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
    streams_locally = True

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
        """The events a streaming Messages client expects, for a finished message."""
        events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    **response,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                },
            }
        ]
        for index, block in enumerate(response.get("content") or []):
            if block.get("type") == "tool_use":
                start = {**block, "input": {}}
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                }
            else:
                start = {**block, "text": ""}
                delta = {"type": "text_delta", "text": block.get("text") or ""}
            events.append({"type": "content_block_start", "index": index, "content_block": start})
            events.append({"type": "content_block_delta", "index": index, "delta": delta})
            events.append({"type": "content_block_stop", "index": index})
        events.append(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": response.get("stop_reason"),
                    "stop_sequence": response.get("stop_sequence"),
                },
                "usage": {"output_tokens": (response.get("usage") or {}).get("output_tokens", 0)},
            }
        )
        events.append({"type": "message_stop"})
        return _sse(events)

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None:
        message: dict[str, Any] | None = None
        blocks: dict[int, dict[str, Any]] = {}
        partial: dict[int, list[str]] = {}
        stopped = False
        for event in _events(chunks):
            kind = event.get("type")
            if kind == "message_stop":
                stopped = True
            elif kind == "message_start" and isinstance(event.get("message"), dict):
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
        # A stream the caller stopped reading, or that broke off, is not an answer.
        if message is None or not stopped:
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


_GEMINI_PATH = re.compile(r"/models/([^/:]+):(stream)?generateContent$", re.IGNORECASE)


def _either(body: Mapping[str, Any], camel: str, snake: str) -> Any:
    return body[camel] if camel in body else body.get(snake)


class GeminiApi:
    """The Gemini API (``POST .../models/<model>:generateContent``), also on Vertex AI.

    The model and whether the call streams are in the URL, not the body, so
    the transport copies them into the body ThinkLess parses and records,
    under ``model`` and ``stream``; the body sent to Google is unchanged.
    """

    name = "gemini"
    streams_locally = True

    def matches(self, path: str) -> bool:
        return _GEMINI_PATH.search(path.rstrip("/")) is not None

    def annotate(self, body: Any, path: str, query: str = "") -> Any:
        """``body`` with the model and stream flag of its URL, for parsing and the log.

        A stream is sent as server-sent events only with ``alt=sse``; without
        it Gemini streams one JSON array, which ThinkLess records but does
        not build, so such calls are always answered by the model.
        """
        found = _GEMINI_PATH.search(path.rstrip("/"))
        if found is None or not isinstance(body, Mapping):
            return body
        annotated = {**body, "model": found.group(1), "stream": bool(found.group(2))}
        if found.group(2) and "alt=sse" not in query:
            annotated["stream_format"] = "json"
        return annotated

    def parse_request(self, body: Any, headers: Mapping[str, str] | None = None) -> Call | None:
        if not isinstance(body, Mapping):
            return None
        raw = body.get("contents")
        if isinstance(raw, Mapping):
            raw = [raw]
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            return None
        messages: list[Message] = []
        has_media = False
        system = _either(body, "systemInstruction", "system_instruction")
        if system:
            text, other = _gemini_parts(
                system.get("parts") if isinstance(system, Mapping) else system
            )
            has_media = has_media or other
            messages.append(Message("system", text))
        for item in raw:
            if not isinstance(item, Mapping):
                return None
            parts = item.get("parts") or []
            role = "assistant" if item.get("role") == "model" else str(item.get("role") or "user")
            if any(
                isinstance(p, Mapping) and _either(p, "functionResponse", "function_response")
                for p in parts
            ):
                role = "tool"
            text, other = _gemini_parts(parts)
            has_media = has_media or other
            messages.append(Message(role, text))
        contract, definition, schema, name = self._contract(body)
        config = _mapping(_either(body, "generationConfig", "generation_config"))
        extra = ""
        if int(_either(config, "candidateCount", "candidate_count") or 1) > 1:
            extra = "several candidates were requested"
        elif _either(config, "responseLogprobs", "response_logprobs"):
            extra = "log probabilities were requested"
        elif _either(body, "cachedContent", "cached_content"):
            extra = "the call continues cached content that is not in the request"
        elif body.get("stream_format") == "json":
            extra = "the answer is streamed as a JSON array, which ThinkLess does not build"
        reason = _reason(has_media, contract, extra) or ("no contents" if not messages else "")
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
        declarations: list[Mapping[str, Any]] = []
        for tool in body.get("tools") or []:
            if isinstance(tool, Mapping):
                found = _either(tool, "functionDeclarations", "function_declarations") or []
                declarations += [d for d in found if isinstance(d, Mapping)]
        tool_config = _mapping(_either(body, "toolConfig", "tool_config"))
        calling = _mapping(_either(tool_config, "functionCallingConfig", "function_calling_config"))
        mode = str(calling.get("mode") or "AUTO").upper()
        allowed = list(_either(calling, "allowedFunctionNames", "allowed_function_names") or [])
        if declarations and mode == "ANY" and (len(allowed) == 1 or len(declarations) == 1):
            forced = str(allowed[0]) if len(allowed) == 1 else str(declarations[0].get("name"))
            for declaration in declarations:
                if declaration.get("name") == forced:
                    schema = _either(declaration, "parametersJsonSchema", "parameters_json_schema")
                    if schema is None:
                        schema = declaration.get("parameters")
                    return "tool", declaration, schema if isinstance(schema, dict) else None, forced
            return "tool", {"name": forced}, None, forced
        if declarations and mode != "NONE":
            return "tools", {"tools": declarations, "config": calling}, None, None
        config = _mapping(_either(body, "generationConfig", "generation_config"))
        mime = _either(config, "responseMimeType", "response_mime_type")
        if mime == "application/json":
            schema = _either(config, "responseJsonSchema", "response_json_schema")
            if schema is None:
                schema = _either(config, "responseSchema", "response_schema")
            definition = {"mime": mime, "schema": schema}
            return "json_schema", definition, schema if isinstance(schema, dict) else None, None
        if mime == "text/x.enum":
            schema = _either(config, "responseSchema", "response_schema")
            return "text", {"mime": mime, "schema": schema}, None, None
        return "text", "text", None, None

    def parse_response(self, call: Call, body: Any) -> Output | None:
        if not isinstance(body, Mapping):
            return None
        candidates = body.get("candidates") or []
        if not candidates or not isinstance(candidates[0], Mapping):
            return None
        candidate = candidates[0]
        reason = candidate.get("finishReason") or candidate.get("finish_reason")
        if reason not in (None, "STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"):
            return None  # blocked by a safety filter, a recitation check or similar
        texts: list[str] = []
        tool_calls: list[tuple[str, Any]] = []
        for part in _mapping(candidate.get("content")).get("parts") or []:
            if not isinstance(part, Mapping) or part.get("thought"):
                continue
            call_part = _either(part, "functionCall", "function_call")
            if isinstance(call_part, Mapping):
                tool_calls.append((str(call_part.get("name")), call_part.get("args") or {}))
            elif isinstance(part.get("text"), str):
                texts.append(part["text"])
        usage = _mapping(_either(body, "usageMetadata", "usage_metadata"))
        return make_output(
            call,
            text="".join(texts),
            tool_calls=tool_calls,
            finish_reason=reason,
            model=_either(body, "modelVersion", "model_version") or call.model,
            usage=(
                int(_either(usage, "promptTokenCount", "prompt_token_count") or 0),
                int(_either(usage, "candidatesTokenCount", "candidates_token_count") or 0),
                None,
            ),
            raw=body,
        )

    def build_response(
        self, call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
    ) -> dict[str, Any]:
        text, arguments = answer_payload(call, fields, template)
        if call.contract == "tool":
            part: dict[str, Any] = {"functionCall": {"name": call.schema_name, "args": arguments}}
        else:
            part = {"text": text or ""}
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [part]},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0,
            },
            "modelVersion": call.model,
            "responseId": f"thinkless_{uuid.uuid4().hex[:20]}",
        }

    def build_stream(self, call: Call, response: Mapping[str, Any]) -> bytes:
        """A streamed Gemini answer is a series of whole responses; a local one is a single one."""
        return f"data: {json.dumps(response, ensure_ascii=False)}\r\n\r\n".encode()

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None:
        chunks = list(chunks)
        events = _events(chunks)
        if not events:
            # Without alt=sse, the stream is one JSON array of responses.
            try:
                whole = json.loads(b"".join(chunks) or b"null")
            except ValueError:
                return None
            events = [e for e in whole if isinstance(e, dict)] if isinstance(whole, list) else []
            if not events:
                return None
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        finish = None
        for event in events:
            for candidate in event.get("candidates") or []:
                if not isinstance(candidate, Mapping) or candidate.get("index", 0) != 0:
                    continue
                for part in _mapping(candidate.get("content")).get("parts") or []:
                    if not isinstance(part, Mapping) or part.get("thought"):
                        continue
                    if _either(part, "functionCall", "function_call"):
                        calls.append(dict(part))
                    elif isinstance(part.get("text"), str):
                        texts.append(part["text"])
                finish = candidate.get("finishReason") or candidate.get("finish_reason") or finish
        if finish is None:
            return None  # the stream broke off or the caller stopped reading
        parts: list[dict[str, Any]] = [{"text": "".join(texts)}] if texts else []
        last = events[-1]
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts + calls},
                    "finishReason": finish,
                    "index": 0,
                }
            ],
            "usageMetadata": _either(last, "usageMetadata", "usage_metadata") or {},
            "modelVersion": _either(last, "modelVersion", "model_version"),
            "responseId": _either(last, "responseId", "response_id"),
        }


def _gemini_parts(parts: Any) -> tuple[str, bool]:
    """Text of Gemini parts, and whether they held anything else (images, files, code)."""
    if isinstance(parts, str):
        return parts, False
    if isinstance(parts, Mapping):
        parts = [parts]
    texts: list[str] = []
    other = False
    for part in parts or []:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, Mapping) or part.get("thought"):
            continue
        call = _either(part, "functionCall", "function_call")
        result = _either(part, "functionResponse", "function_response")
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif isinstance(call, Mapping):
            texts.append(f"{call.get('name')}({json.dumps(call.get('args'), sort_keys=True)})")
        elif isinstance(result, Mapping):
            texts.append(json.dumps(result.get("response"), sort_keys=True, ensure_ascii=False))
        else:
            other = True
    return "\n".join(texts), other


class ConverseApi:
    """The Amazon Bedrock Converse API, through boto3.

    boto3 signs and sends requests itself, not through httpx, so Habits
    joins it through botocore's events instead of a transport; see
    :func:`thinkless.habits.bedrock.wrap_bedrock`. Request and response
    bodies are the dictionaries boto3 sends and returns, with the model id
    added to the request ThinkLess parses and records.
    """

    name = "converse"
    streams_locally = False

    def matches(self, path: str) -> bool:
        return False

    def parse_request(self, body: Any, headers: Mapping[str, str] | None = None) -> Call | None:
        if not isinstance(body, Mapping):
            return None
        raw = body.get("messages")
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            return None
        messages: list[Message] = []
        has_media = False
        system = body.get("system")
        if system:
            text, other = _converse_blocks(system)
            has_media = has_media or other
            messages.append(Message("system", text))
        for item in raw:
            if not isinstance(item, Mapping):
                return None
            blocks = item.get("content") or []
            role = str(item.get("role") or "user")
            if role == "user" and any(isinstance(b, Mapping) and "toolResult" in b for b in blocks):
                role = "tool"
            text, other = _converse_blocks(blocks)
            has_media = has_media or other
            messages.append(Message(role, text))
        contract, definition, schema, name = self._contract(body)
        extra = ""
        fields = _mapping(body.get("additionalModelRequestFields"))
        if fields.get("thinking") or fields.get("reasoning_config"):
            extra = "reasoning was requested"
        elif body.get("promptVariables"):
            extra = "the prompt comes from a prompt template that is not in the request"
        reason = _reason(has_media, contract, extra) or ("no messages" if not messages else "")
        return Call(
            model=str(body.get("modelId") or ""),
            messages=tuple(messages),
            contract=contract,
            contract_id=fingerprint([self.name, contract, definition]),
            schema=schema,
            schema_name=str(name) if name else None,
            stream=False,
            answerable=not reason,
            reason=reason,
            site_name=_site_name(headers),
            body=body,
            api=self.name,
        )

    @staticmethod
    def _contract(body: Mapping[str, Any]) -> tuple[str, Any, dict[str, Any] | None, str | None]:
        config = _mapping(body.get("toolConfig"))
        specs = [
            _mapping(t.get("toolSpec"))
            for t in config.get("tools") or []
            if isinstance(t, Mapping) and "toolSpec" in t
        ]
        choice = _mapping(config.get("toolChoice"))
        forced = None
        if isinstance(choice.get("tool"), Mapping) and choice["tool"].get("name"):
            forced = str(choice["tool"]["name"])
        elif "any" in choice and len(specs) == 1:
            forced = str(specs[0].get("name"))
        if forced is not None:
            for spec in specs:
                if spec.get("name") == forced:
                    schema = _mapping(spec.get("inputSchema")).get("json")
                    return "tool", spec, schema if isinstance(schema, dict) else None, forced
            return "tool", {"name": forced}, None, forced
        if specs:
            return "tools", {"tools": specs, "choice": choice}, None, None
        return "text", "text", None, None

    def parse_response(self, call: Call, body: Any) -> Output | None:
        if not isinstance(body, Mapping):
            return None
        stop = body.get("stopReason")
        if stop in ("guardrail_intervened", "content_filtered"):
            return None
        texts: list[str] = []
        tool_calls: list[tuple[str, Any]] = []
        message = _mapping(_mapping(body.get("output")).get("message"))
        for block in message.get("content") or []:
            if not isinstance(block, Mapping):
                continue
            if isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif isinstance(block.get("toolUse"), Mapping):
                use = block["toolUse"]
                tool_calls.append((str(use.get("name")), use.get("input") or {}))
        usage = _mapping(body.get("usage"))
        return make_output(
            call,
            text="".join(texts),
            tool_calls=tool_calls,
            finish_reason=stop,
            model=call.model,
            usage=(int(usage.get("inputTokens") or 0), int(usage.get("outputTokens") or 0), None),
            raw={k: v for k, v in body.items() if k != "ResponseMetadata"},
        )

    def build_response(
        self, call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
    ) -> dict[str, Any]:
        text, arguments = answer_payload(call, fields, template)
        if call.contract == "tool":
            block: dict[str, Any] = {
                "toolUse": {
                    "toolUseId": f"tooluse_thinkless_{uuid.uuid4().hex[:16]}",
                    "name": call.schema_name,
                    "input": arguments,
                }
            }
            stop = "tool_use"
        else:
            block = {"text": text or ""}
            stop = "end_turn"
        return {
            "output": {"message": {"role": "assistant", "content": [block]}},
            "stopReason": stop,
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 0},
        }

    def build_stream(self, call: Call, response: Mapping[str, Any]) -> bytes:
        raise NotImplementedError("streamed Converse calls are always sent to the model")

    def collect_stream(self, chunks: Iterable[bytes]) -> dict[str, Any] | None:
        return None


def _converse_blocks(blocks: Any) -> tuple[str, bool]:
    """Text of Converse content blocks, and whether they held anything else."""
    if isinstance(blocks, Mapping):
        blocks = [blocks]
    texts: list[str] = []
    other = False
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        if isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif isinstance(block.get("toolUse"), Mapping):
            use = block["toolUse"]
            texts.append(f"{use.get('name')}({json.dumps(use.get('input'), sort_keys=True)})")
        elif isinstance(block.get("toolResult"), Mapping):
            inner, nested = _converse_blocks(block["toolResult"].get("content"))
            texts.append(inner)
            other = other or nested
        elif isinstance(block.get("json"), (Mapping, list)):
            texts.append(json.dumps(block["json"], sort_keys=True, ensure_ascii=False))
        elif any(k in block for k in ("reasoningContent", "cachePoint", "guardContent")):
            continue
        else:
            other = True  # images, documents, video
    return "\n".join(texts), other


CHAT = ChatApi()
RESPONSES = ResponsesApi()
MESSAGES = MessagesApi()
GEMINI = GeminiApi()
CONVERSE = ConverseApi()
APIS: dict[str, Api] = {
    "chat": CHAT,
    "responses": RESPONSES,
    "messages": MESSAGES,
    "gemini": GEMINI,
    "converse": CONVERSE,
}


def api_for_path(path: str) -> Api | None:
    """The API a request path belongs to, or ``None`` for everything else."""
    for api in (CHAT, RESPONSES, MESSAGES, GEMINI):
        if api.matches(path):
            return api
    return None


def api_for_record(request: Any, response: Any = None) -> Api:
    """Guess the API of a logged request and response pair from their shapes."""
    if isinstance(request, Mapping):
        if "contents" in request and "messages" not in request:
            return GEMINI
        if "input" in request and "messages" not in request:
            return RESPONSES
        if "max_tokens" in request and (
            "system" in request
            or (isinstance(response, Mapping) and response.get("type") == "message")
        ):
            return MESSAGES
    return CHAT
