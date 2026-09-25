"""Hosted or self-hosted System One servers over HTTP."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx

from ..decision import Plane, Usage
from ..logs import get_logger
from ..questions import Kind, Question
from .base import DecisionProvider, ProviderResult, State
from .wire import from_wire, to_wire

__all__ = ["SystemOne", "SystemOneError"]

logger = get_logger("providers.systemone")

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class SystemOneError(RuntimeError):
    """A System One server returned an error that retries did not fix."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"System One request failed with HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class SystemOne(DecisionProvider):
    """Any server that implements ``POST /v1/systemone``.

    Works with TypeSafe's hosted Jev and with self-hosted servers that copy
    its wire format, such as Kev and OpenJev. All questions of one call are
    answered in a single request.

    Args:
        name: Provider name in traces.
        base_url: Server root, without the ``/v1/systemone`` path.
        model: Model id or alias, for example ``jev-latest``.
        api_key: Bearer token. Defaults to the ``api_key_env`` variable.
        api_key_env: Environment variable read when ``api_key`` is omitted.
        price_key: Provider id for price lookup. ``typesafe`` for Jev,
            ``local`` for self-hosted servers.
        timeout: Per-request timeout in seconds.
        max_retries: Retries on 429, 529, 5xx and connection errors, with
            exponential backoff that honors ``Retry-After``.
        client: A preconfigured ``httpx.Client`` (tests, proxies).

    Use :meth:`jev` and :meth:`self_hosted` for the common setups.
    """

    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset({Kind.CHOICE, Kind.SCORE, Kind.YES_NO})

    def __init__(
        self,
        *,
        name: str = "jev",
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        api_key: str | None = None,
        api_key_env: str | None = "TYPESAFE_API_KEY",
        price_key: str = "typesafe",
        timeout: float = 30.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.price_key = price_key
        self._url = base_url.rstrip("/") + "/v1/systemone"
        self._api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self._max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def jev(
        cls, *, api_key: str | None = None, model: str = "jev-latest", **kwargs: Any
    ) -> SystemOne:
        """TypeSafe's hosted Jev. Reads ``TYPESAFE_API_KEY`` by default."""
        return cls(name="jev", model=model, api_key=api_key, **kwargs)

    @classmethod
    def self_hosted(
        cls, base_url: str, *, name: str = "systemone", model: str = "latest", **kwargs: Any
    ) -> SystemOne:
        """A self-hosted server such as Kev or OpenJev. Cost is recorded as 0."""
        kwargs.setdefault("api_key_env", None)
        return cls(name=name, base_url=base_url, model=model, price_key="local", **kwargs)

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        payload = {"state": state, "model": self.model, "questions": to_wire(questions)}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = self._post(payload, headers)
        usage = data.get("usage") or {}
        return ProviderResult(
            answers=from_wire(data.get("answers") or {}, questions),
            model=data.get("model") or self.model,
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            ),
            meta={"endpoint": self._url},
        )

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                response = self._client.post(self._url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise
                delay = self._backoff(attempt, None)
                logger.warning("System One connection error (%s), retrying in %.1f s", exc, delay)
            else:
                if response.status_code < 400:
                    result: dict[str, Any] = response.json()
                    return result
                if response.status_code not in RETRYABLE or attempt >= self._max_retries:
                    raise SystemOneError(response.status_code, response.text)
                delay = self._backoff(attempt, response.headers.get("retry-after"))
                logger.warning(
                    "System One HTTP %s, retrying in %.1f s", response.status_code, delay
                )
            time.sleep(delay)
            attempt += 1

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        return min(0.5 * (2**attempt), 8.0) * (0.75 + random.random() * 0.5)

    def close(self) -> None:
        self._client.close()
