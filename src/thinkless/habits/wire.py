"""The OpenAI Chat Completions wire format, as ThinkLess sees it.

ThinkLess works at the HTTP layer, under whatever SDK or framework makes the
call, so it reads requests and responses as JSON. This module turns a request
body into a :class:`Call` (what was asked, and under which output contract),
a response body into an :class:`Output` (what the model said), and builds a
response body, or a stream of server-sent events, from an answer Habits
made locally.

Output contracts:

* ``json_schema``: ``response_format`` with a JSON schema (what
  ``with_structured_output`` and ``client.chat.completions.parse`` send).
* ``json_object``: JSON mode without a schema.
* ``tool``: ``tool_choice`` forces one function, whose arguments are the answer.
* ``tools``: tools offered with the model free to pick one or none.
* ``text``: a plain completion.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MISSING",
    "Call",
    "Output",
    "answer_payload",
    "build_response",
    "build_stream",
    "collect_stream",
    "fingerprint",
    "label_of",
    "make_output",
    "normalize_text",
    "parse_request",
    "parse_response",
    "value_of",
]

MISSING = "~missing"
"""Label for a JSON key the model left out. It cannot collide with a JSON value."""

TEXT_FIELD = "$text"
TOOLS_FIELD = "$tools"
ROOT_FIELD = "$"

_WS = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s.!?,;:]+$")


def fingerprint(value: Any, length: int = 12) -> str:
    """A short stable hash of JSON-like data."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(data.encode("utf-8"), digest_size=16).hexdigest()[:length]


def normalize_text(text: str) -> str:
    """Case, spacing and trailing punctuation removed, so "Yes." and "yes" match."""
    text = text.strip().strip("\"'`*").strip()
    return _TRAILING.sub("", _WS.sub(" ", text)).lower()


def label_of(value: Any) -> str:
    """The label a JSON value is counted under. Strings are kept JSON quoted."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def value_of(label: str) -> Any:
    """The JSON value behind a label from :func:`label_of`."""
    return json.loads(label)


def _content_text(content: Any) -> tuple[str, bool]:
    """Text of a message's content, and whether it held anything besides text."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, Sequence):
        texts: list[str] = []
        other = False
        for part in content:
            if isinstance(part, Mapping) and part.get("type") in ("text", "input_text"):
                texts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                texts.append(part)
            else:
                other = True
        return "\n".join(texts), other
    return str(content), True


@dataclass(frozen=True)
class Message:
    """One message of the conversation, reduced to text."""

    role: str
    text: str


@dataclass(frozen=True)
class Call:
    """One chat completion request.

    Attributes:
        model: The model the request names.
        messages: The conversation as text. Tool calls an assistant made
            earlier are rendered as ``name(arguments)``.
        contract: The output contract (see the module docstring).
        contract_id: Hash of the schema or tools that define the contract.
        schema: The JSON schema of the answer, for ``json_schema`` and ``tool``.
        schema_name: The schema's or forced tool's name.
        stream: Whether the caller asked for a stream of events.
        include_usage: Whether a streamed response should end with usage.
        answerable: Whether ThinkLess may answer this call itself.
        reason: Why not, when ``answerable`` is false.
        site_name: A site name the caller set with the ``x-thinkless-site`` header.
        api: ``chat`` (Chat Completions), ``responses`` (the OpenAI Responses
            API) or ``messages`` (the Anthropic Messages API).
    """

    model: str
    messages: tuple[Message, ...]
    contract: str
    contract_id: str
    schema: dict[str, Any] | None = None
    schema_name: str | None = None
    stream: bool = False
    include_usage: bool = False
    answerable: bool = True
    reason: str = ""
    site_name: str | None = None
    body: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    api: str = "chat"

    @property
    def roles(self) -> str:
        """A coarse shape of the conversation: whether tools ran, and who spoke last."""
        has_tool = any(m.role == "tool" for m in self.messages)
        last = self.messages[-1].role if self.messages else "none"
        return f"{'tool+' if has_tool else ''}{last}"


def _render_tool_calls(calls: Any) -> str:
    parts = []
    for call in calls or []:
        function = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(function, Mapping):
            parts.append(f"{function.get('name')}({function.get('arguments') or ''})")
    return "\n".join(parts)


def _contract(body: Mapping[str, Any]) -> tuple[str, Any, dict[str, Any] | None, str | None]:
    tools = body.get("tools") or []
    choice = body.get("tool_choice")
    forced: str | None = None
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if isinstance(function, Mapping):
            forced = function.get("name")
    if forced is None and isinstance(body.get("function_call"), Mapping):
        forced = body["function_call"].get("name")
        tools = [{"type": "function", "function": f} for f in body.get("functions") or []]
    if forced is not None:
        for tool in tools:
            function = tool.get("function") if isinstance(tool, Mapping) else None
            if isinstance(function, Mapping) and function.get("name") == forced:
                schema = function.get("parameters")
                return "tool", function, schema if isinstance(schema, dict) else None, forced
        return "tool", {"name": forced}, None, forced
    usable_tools = tools and choice != "none"
    if usable_tools:
        definition = [
            (t.get("function") or {}).get("name") if isinstance(t, Mapping) else None for t in tools
        ]
        return "tools", {"tools": tools, "choice": choice, "names": definition}, None, None
    fmt = body.get("response_format")
    if isinstance(fmt, Mapping):
        if fmt.get("type") == "json_schema":
            spec = fmt.get("json_schema") or {}
            schema = spec.get("schema") if isinstance(spec, Mapping) else None
            name = spec.get("name") if isinstance(spec, Mapping) else None
            return "json_schema", spec, schema if isinstance(schema, dict) else None, name
        if fmt.get("type") == "json_object":
            return "json_object", "json_object", None, None
    return "text", "text", None, None


def parse_request(body: Any, headers: Mapping[str, str] | None = None) -> Call | None:
    """Read a Chat Completions request body. ``None`` if it is not one."""
    if not isinstance(body, Mapping):
        return None
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, str):
        return None
    messages: list[Message] = []
    has_media = False
    for item in raw_messages:
        if not isinstance(item, Mapping):
            return None
        text, other = _content_text(item.get("content"))
        has_media = has_media or other
        calls = _render_tool_calls(item.get("tool_calls"))
        if calls:
            text = f"{text}\n{calls}" if text else calls
        messages.append(Message(role=str(item.get("role") or "user"), text=text))
    contract, definition, schema, name = _contract(body)
    stream = bool(body.get("stream"))
    options = body.get("stream_options")
    include_usage = bool(isinstance(options, Mapping) and options.get("include_usage"))
    reason = ""
    if has_media:
        reason = "the input holds images, audio or files"
    elif (body.get("n") or 1) != 1:
        reason = "more than one choice was requested"
    elif body.get("logprobs"):
        reason = "log probabilities were requested"
    elif contract == "tools":
        reason = "the model may call tools"
    elif not messages:
        reason = "no messages"
    site_name = None
    if headers is not None:
        site_name = headers.get("x-thinkless-site") or None
    return Call(
        model=str(body.get("model") or ""),
        messages=tuple(messages),
        contract=contract,
        contract_id=fingerprint([contract, definition]),
        schema=schema,
        schema_name=str(name) if name else None,
        stream=stream,
        include_usage=include_usage,
        answerable=not reason,
        reason=reason,
        site_name=site_name,
        body=body,
    )


@dataclass(frozen=True)
class Output:
    """What the model answered.

    Attributes:
        fields: The answer as labeled fields. A JSON object answer has one
            field per key; a plain text answer has the single field ``$text``
            holding its normalized text; a tool call answer holds the
            arguments' keys.
        text: The raw text of the answer, for text answers.
        finish_reason: As reported.
        model: The model that answered, as reported.
        input_tokens: Prompt tokens reported.
        output_tokens: Completion tokens reported.
        cost_usd: Cost reported in ``usage.cost`` (OpenRouter), if any.
        key_order: The order the answer's JSON keys came in.
    """

    fields: dict[str, str]
    text: str | None = None
    finish_reason: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    key_order: tuple[str, ...] = ()
    message: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


def _json_fields(
    value: Any, schema: Mapping[str, Any] | None
) -> tuple[dict[str, str], tuple[str, ...]]:
    if isinstance(value, Mapping):
        fields = {str(k): label_of(v) for k, v in value.items()}
        properties = (schema or {}).get("properties")
        if isinstance(properties, Mapping):
            for key in properties:
                fields.setdefault(str(key), MISSING)
        return fields, tuple(str(k) for k in value)
    return {ROOT_FIELD: label_of(value)}, ()


def _usage(body: Mapping[str, Any]) -> tuple[int, int, float | None]:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return 0, 0, None
    cost = usage.get("cost")
    try:
        cost_usd = None if cost is None else float(cost)
    except (TypeError, ValueError):
        cost_usd = None
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), cost_usd


def make_output(
    call: Call,
    *,
    text: str,
    tool_calls: Sequence[tuple[str, Any]] = (),
    finish_reason: str | None = None,
    model: str | None = None,
    usage: tuple[int, int, float | None] = (0, 0, None),
    raw: Mapping[str, Any] | None = None,
) -> Output | None:
    """An answer in any API's format, as labeled fields.

    Args:
        text: The answer's text content.
        tool_calls: ``(name, arguments)`` for each tool call, with arguments
            as a JSON string or an already decoded object.

    Returns:
        ``None`` when the answer breaks the call's contract (malformed JSON,
        or no call to the forced tool).
    """
    common: dict[str, Any] = {
        "finish_reason": finish_reason,
        "model": model,
        "input_tokens": usage[0],
        "output_tokens": usage[1],
        "cost_usd": usage[2],
        "message": raw or {},
    }

    def decoded(arguments: Any) -> Any:
        if isinstance(arguments, str):
            return json.loads(arguments or "{}")
        return arguments

    if call.contract == "tool":
        for name, arguments in tool_calls:
            if name == call.schema_name:
                try:
                    value = decoded(arguments)
                except json.JSONDecodeError:
                    return None
                fields, order = _json_fields(value, call.schema)
                return Output(fields=fields, key_order=order, **common)
        return None
    if call.contract == "tools":
        rendered = sorted(
            f"{name}:{json.dumps(decoded(arguments), sort_keys=True)}"
            for name, arguments in tool_calls
        )
        if rendered:
            return Output(fields={TOOLS_FIELD: label_of(rendered)}, **common)
        return Output(fields={TEXT_FIELD: label_of(normalize_text(text))}, text=text, **common)
    if call.contract in ("json_schema", "json_object"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        fields, order = _json_fields(value, call.schema)
        return Output(fields=fields, key_order=order, text=text, **common)
    return Output(fields={TEXT_FIELD: label_of(normalize_text(text))}, text=text, **common)


def output_from_message(
    call: Call,
    message: Mapping[str, Any],
    *,
    finish_reason: str | None = None,
    model: str | None = None,
    usage: tuple[int, int, float | None] = (0, 0, None),
) -> Output | None:
    """Read a Chat Completions assistant message. ``None`` for refusals and broken contracts."""
    if message.get("refusal"):
        return None
    text, _ = _content_text(message.get("content"))
    tool_calls = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, Mapping) else None
        if isinstance(function, Mapping):
            tool_calls.append((str(function.get("name")), function.get("arguments") or "{}"))
    return make_output(
        call,
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        model=model,
        usage=usage,
        raw=message,
    )


def parse_response(call: Call, body: Any) -> Output | None:
    """Read a Chat Completions response body. ``None`` if it holds no usable answer."""
    if not isinstance(body, Mapping):
        return None
    choices = body.get("choices")
    if not isinstance(choices, Sequence) or len(choices) != 1:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
        return None
    return output_from_message(
        call,
        choice["message"],
        finish_reason=choice.get("finish_reason"),
        model=body.get("model"),
        usage=_usage(body),
    )


def answer_payload(
    call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
) -> tuple[str | None, Any]:
    """What a local answer says: ``(text, None)``, or ``(None, arguments)`` for a forced tool.

    ``template`` carries what ThinkLess learned about how the model formats
    its answers on the site: JSON key order and the raw text behind each
    text label.
    """
    if TEXT_FIELD in fields:
        raw = template.get("texts", {}).get(fields[TEXT_FIELD])
        return (raw if isinstance(raw, str) else value_of(fields[TEXT_FIELD])), None
    order = list(template.get("key_order") or [])
    order += [k for k in fields if k not in order]
    if ROOT_FIELD in fields:
        value: Any = value_of(fields[ROOT_FIELD])
    else:
        value = {k: value_of(fields[k]) for k in order if k in fields and fields[k] != MISSING}
    if call.contract == "tool":
        return None, value
    return json.dumps(value, ensure_ascii=False), None


def _message(call: Call, fields: Mapping[str, str], template: Mapping[str, Any]) -> dict[str, Any]:
    """The Chat Completions assistant message for an answer, shaped like the model's own."""
    text, arguments = answer_payload(call, fields, template)
    if call.contract == "tool":
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": call.schema_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        return {"role": "assistant", "content": None, "tool_calls": [tool_call], "refusal": None}
    return {"role": "assistant", "content": text, "refusal": None}


def build_response(
    call: Call, fields: Mapping[str, str], template: Mapping[str, Any]
) -> dict[str, Any]:
    """A complete Chat Completions response body for an answer made locally.

    ``template`` carries what ThinkLess learned about how the model formats
    its answers on this site: JSON key order, the raw text behind each text
    label, and the finish reason it reports.
    """
    message = _message(call, fields, template)
    default_finish = "tool_calls" if call.contract == "tool" else "stop"
    return {
        "id": f"chatcmpl-thinkless-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": call.model,
        "system_fingerprint": "thinkless-habits",
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": template.get("finish_reason") or default_finish,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_stream(call: Call, response: Mapping[str, Any]) -> bytes:
    """The server-sent events a streaming client expects, for a finished response."""
    choice = response["choices"][0]
    message = choice["message"]
    base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
        "system_fingerprint": response.get("system_fingerprint"),
    }
    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        delta["content"] = None
        delta["tool_calls"] = [
            {"index": i, **tool_call} for i, tool_call in enumerate(message["tool_calls"])
        ]
    events = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        },
    ]
    if call.include_usage:
        events.append({**base, "choices": [], "usage": response["usage"]})
    lines = [f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def collect_stream(chunks: Iterable[bytes]) -> dict[str, Any] | None:
    """Rebuild a response body from the server-sent events of a streamed completion."""
    buffer = b"".join(chunks).decode("utf-8", errors="replace")
    content: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    model = None
    usage: dict[str, Any] | None = None
    seen = False
    for line in buffer.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        seen = True
        model = event.get("model") or model
        if isinstance(event.get("usage"), Mapping):
            usage = dict(event["usage"])
        for choice in event.get("choices") or []:
            if choice.get("index", 0) != 0:
                return None
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for part in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    int(part.get("index", 0)),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if part.get("id"):
                    slot["id"] = part["id"]
                function = part.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] += function["name"]
                if function.get("arguments"):
                    slot["function"]["arguments"] += function["arguments"]
            finish_reason = choice.get("finish_reason") or finish_reason
    if not seen:
        return None
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    body: dict[str, Any] = {
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    return body
