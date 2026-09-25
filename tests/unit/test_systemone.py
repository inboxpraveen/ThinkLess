from __future__ import annotations

import json

import httpx
import pytest

from thinkless import Choice, Score, Status, YesNo
from thinkless.providers import SystemOne, SystemOneError
from thinkless.providers.wire import from_wire, to_wire

QUESTIONS = {
    "team": Choice("Which team?", options={"billing": "payments", "technical": None}),
    "urgency": Score("How urgent?", levels=["low", "medium", "high"]),
    "urgent": YesNo("Does this need attention now?", yes_means="outage", no_means="question"),
}

RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.94, "technical": 0.06},
            "confidence": 0.88,
        },
        "urgency": {
            "type": "score",
            "score": 1.7,
            "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75},
            "legend": {"0": "low"},
        },
        "urgent": {"type": "noul", "noul": 0.97, "confidence": 0.94},
    },
    "usage": {"input_tokens": 296, "output_tokens": 20},
}


def test_wire_encoding() -> None:
    wire = to_wire(QUESTIONS)
    assert wire["team"] == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "payments", "technical": None},
    }
    assert wire["urgency"]["criteria"] == ["low", "medium", "high"]
    assert wire["urgent"] == {
        "type": "noul",
        "instructions": "Does this need attention now?",
        "criteria": {"true": "outage", "false": "question"},
    }


def test_wire_decoding() -> None:
    answers = from_wire(RESPONSE["answers"], QUESTIONS)
    assert answers["team"].value == "billing"
    assert answers["urgency"].probabilities == {"low": 0.05, "medium": 0.2, "high": 0.75}
    assert answers["urgent"].probabilities["yes"] == pytest.approx(0.97)
    assert from_wire({"team": {"choice": "sales"}}, QUESTIONS)["team"] is None


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_request_shape_and_cost(make_engine, memory) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPONSE)

    provider = SystemOne.jev(api_key="test-key", client=_client(handler))
    engine = make_engine([provider], threshold=0.8)
    decisions = engine.decide_many("The deploy failed twice", QUESTIONS)

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "jev-latest"
    assert set(seen["body"]["questions"]) == set(QUESTIONS)
    assert decisions["team"].status is Status.ACCEPTED
    assert decisions["urgent"].value is True
    assert decisions["urgency"].level == "high"
    attempt = next(s for s in memory.spans if s.kind == "attempt")
    assert attempt.attributes["cost_usd"] == pytest.approx(296 * 0.042 / 1e6)
    assert attempt.attributes["model"] == "jev-1.13.0"


def test_retries_on_429(monkeypatch) -> None:
    monkeypatch.setattr("thinkless.providers.systemone.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        return httpx.Response(200, json=RESPONSE)

    provider = SystemOne.jev(api_key="k", client=_client(handler))
    result = provider.answer("x", QUESTIONS)
    assert calls["n"] == 3
    assert result.answers["team"].value == "billing"


def test_non_retryable_error_raises(monkeypatch) -> None:
    provider = SystemOne.jev(
        api_key="bad", client=_client(lambda r: httpx.Response(401, text="invalid key"))
    )
    with pytest.raises(SystemOneError) as info:
        provider.answer("x", QUESTIONS)
    assert info.value.status == 401


def test_self_hosted_is_free(make_engine, memory) -> None:
    provider = SystemOne.self_hosted(
        "http://localhost:8009",
        name="kev",
        model="kev-latest",
        client=_client(lambda r: httpx.Response(200, json=RESPONSE)),
    )
    make_engine([provider]).decide_many("x", QUESTIONS)
    attempt = next(s for s in memory.spans if s.kind == "attempt")
    assert attempt.attributes["cost_usd"] == 0.0
    assert attempt.attributes["endpoint"] == "http://localhost:8009/v1/systemone"
