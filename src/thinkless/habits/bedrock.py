"""Habits under boto3's Bedrock Runtime client, through botocore's events.

boto3 does not send requests through httpx, and its requests are signed
(SigV4), so a transport or a forwarding proxy cannot sit under it. botocore
emits an event before and after every API call instead: ``before-call`` may
return a response, which botocore then hands back without sending anything,
and ``after-call`` sees what the service answered. Habits uses the first to
answer a confident call locally and the second to record the rest.

Only ``converse`` is covered. ``converse_stream``, ``invoke_model`` and every
other operation pass through untouched.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote

from ..logs import get_logger
from .apis import CONVERSE
from .runtime import Habits, Ticket, default_habits

__all__ = ["is_bedrock_client", "wrap_bedrock"]

logger = get_logger("habits")

_TICKET = "thinkless_ticket"
_MARK = "_thinkless_habits"


def is_bedrock_client(client: Any) -> bool:
    """Whether ``client`` is a boto3 ``bedrock-runtime`` client."""
    meta = getattr(client, "meta", None)
    service = getattr(getattr(meta, "service_model", None), "service_name", None)
    return service == "bedrock-runtime" and hasattr(meta, "events")


def _model_id(params: Any) -> str:
    path = str(params.get("url_path") or "") if isinstance(params, dict) else ""
    # /model/{modelId}/converse, with the id percent-encoded
    parts = path.split("/")
    if len(parts) >= 3 and parts[1] == "model":
        return unquote(parts[2])
    return ""


def _request_body(params: Any) -> Any:
    body = params.get("body") if isinstance(params, dict) else None
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    try:
        return json.loads(body) if isinstance(body, str) and body else None
    except ValueError:
        return None


def wrap_bedrock(client: Any, habits: Habits | None = None) -> Any:
    """Put Habits under a boto3 ``bedrock-runtime`` client's ``converse`` calls."""
    if getattr(client, _MARK, None) is not None:
        return client
    runtime = habits or default_habits()
    events = client.meta.events

    def before(params: Any, context: dict[str, Any], **_: Any) -> Any:
        try:
            body = _request_body(params)
            if not isinstance(body, dict):
                return None
            body = {**body, "modelId": _model_id(params)}
            headers = {k.lower(): str(v) for k, v in (params.get("headers") or {}).items()}
            ticket = runtime.begin(
                body, headers, getattr(client.meta, "region_name", None), CONVERSE
            )
        except Exception:
            logger.exception("ThinkLess failed on a request; it goes to the model unchanged")
            return None
        if ticket is None:
            return None
        context[_TICKET] = ticket
        if not ticket.answered_locally or ticket.response is None:
            return None
        try:
            runtime.served(ticket)
        except Exception:
            logger.exception("ThinkLess could not record a local answer")
        return _local(ticket)

    def after(http_response: Any, parsed: Any, context: dict[str, Any], **_: Any) -> None:
        ticket = context.pop(_TICKET, None)
        if not isinstance(ticket, Ticket) or ticket.answered_locally:
            return
        try:
            status = int(getattr(http_response, "status_code", 0) or 0)
            body = {k: v for k, v in parsed.items() if k != "ResponseMetadata"}
            runtime.finish(ticket, body, status_code=status)
        except Exception:
            logger.exception("ThinkLess could not record a model answer")

    events.register("before-call.bedrock-runtime.Converse", before, unique_id="thinkless-before")
    events.register("after-call.bedrock-runtime.Converse", after, unique_id="thinkless-after")
    setattr(client, _MARK, runtime)
    return client


def _local(ticket: Ticket) -> tuple[Any, dict[str, Any]]:
    """What botocore returns in place of a call it never sent."""
    from botocore.awsrequest import AWSResponse

    confidence = ticket.extra.get("confidence", 0.0)
    headers = {
        "x-thinkless-habit": "local",
        "x-thinkless-site": ticket.site or "",
        "x-thinkless-confidence": f"{confidence:.4f}",
    }
    http = AWSResponse(url="", status_code=200, headers=headers, raw=None)
    assert ticket.response is not None
    parsed = {
        **ticket.response,
        "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": headers, "RetryAttempts": 0},
    }
    return http, parsed
