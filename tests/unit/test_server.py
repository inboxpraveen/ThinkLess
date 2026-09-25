from __future__ import annotations

import asyncio
import json

import pytest

from tests.conftest import FakeProvider, choice_answer, yes_answer
from thinkless import Answer, Choice, Engine, Score, Status, Tracer, YesNo
from thinkless.providers import Rules, SystemOne

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from thinkless.server.app import create_app

INTENT = Choice(
    "What does the customer want?",
    options={"refund": "wants money back", "order_status": "asks about an order", "other": None},
    name="intent",
)
URGENT = YesNo("Is it urgent?", name="urgent")
URGENCY = Score("How urgent is it?", levels=["low", "medium", "high"], name="urgency")


def _engine() -> Engine:
    rules = Rules()
    rules.match("intent", r"\brefund\b", "refund")
    fake = FakeProvider(
        "fake",
        {
            "intent": choice_answer(
                "order_status", {"refund": 0.02, "order_status": 0.96, "other": 0.02}
            ),
            "urgent": yes_answer(0.97),
            "urgency": Answer(value=1.8, probabilities={"low": 0.05, "medium": 0.1, "high": 0.85}),
        },
    )
    return Engine([rules, fake], tracer=Tracer())


def _client(**kwargs) -> TestClient:
    return TestClient(create_app(_engine(), [INTENT, URGENT], warmup=False, **kwargs))


def test_health_and_registry() -> None:
    client = _client()
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert [p["name"] for p in health["providers"]] == ["rules", "fake"]
    assert set(client.get("/v1/questions").json()) == {"intent", "urgent"}


def test_decide_registered_and_ad_hoc_questions() -> None:
    client = _client()
    response = client.post(
        "/v1/decide", json={"state": "please refund me", "questions": ["intent", "urgent"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decisions"]["intent"]["value"] == "refund"
    assert body["decisions"]["intent"]["provider"] == "rules"
    assert body["decisions"]["urgent"]["status"] == "accepted"
    assert body["trace_id"]

    ad_hoc = client.post(
        "/v1/decide", json={"state": "where is it", "questions": {"urgency": URGENCY.spec()}}
    )
    assert ad_hoc.status_code == 200
    assert ad_hoc.json()["decisions"]["urgency"]["level"] == "high"


def test_decide_rejects_bad_requests() -> None:
    client = _client(allow_ad_hoc=False, max_questions=1)
    assert client.post("/v1/decide", json={"state": "x", "questions": ["nope"]}).status_code == 404
    assert client.post("/v1/decide", json={"state": "x", "questions": []}).status_code == 422
    too_many = client.post("/v1/decide", json={"state": "x", "questions": ["intent", "urgent"]})
    assert too_many.status_code == 422
    ad_hoc = client.post("/v1/decide", json={"state": "x", "questions": {"u": URGENCY.spec()}})
    assert ad_hoc.status_code == 422
    open_client = _client()
    bad_spec = open_client.post(
        "/v1/decide", json={"state": "x", "questions": {"u": {"kind": "poem"}}}
    )
    assert bad_spec.status_code == 422


def test_api_key() -> None:
    client = _client(api_key="s3cret")
    assert client.get("/healthz").status_code == 200
    body = {"state": "x", "questions": ["intent"]}
    assert client.post("/v1/decide", json=body).status_code == 401
    wrong = client.post("/v1/decide", json=body, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    ok = client.post("/v1/decide", json=body, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_systemone_endpoint_speaks_the_sdk_format() -> None:
    sdk = pytest.importorskip("typesafe_sdk")
    from thinkless.providers.wire import to_wire

    client = _client()
    questions = {"intent": INTENT, "urgent": URGENT, "urgency": URGENCY}
    response = client.post(
        "/v1/systemone", json={"state": "where is my order", "questions": to_wire(questions)}
    )
    assert response.status_code == 200
    parsed = sdk.SystemOneResponse.model_validate_json(response.text)
    answers = parsed.model_dump(mode="json")["answers"]
    assert answers["intent"]["choice"] == "order_status"
    assert answers["urgent"]["noul"] == pytest.approx(0.97)
    assert answers["urgency"]["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert response.json()["thinkless"]["decisions"]["intent"]["provider"] == "fake"


def test_a_thinkless_server_is_a_system_one_provider() -> None:
    server = _client()
    remote = SystemOne.self_hosted("http://testserver", client=server, max_retries=0)
    downstream = Engine([remote], tracer=Tracer())
    decisions = downstream.decide_many("where is my order", [INTENT, URGENT])
    assert decisions["intent"].value == "order_status"
    assert decisions["intent"].status is Status.ACCEPTED
    assert decisions["intent"].provider == "systemone"
    assert decisions["urgent"].value is True


def test_mcp_server_exposes_one_tool_per_question() -> None:
    pytest.importorskip("mcp")
    from thinkless.server.mcp import create_mcp_server

    server = create_mcp_server(_engine(), [INTENT, URGENT])
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(tools) == {"decide_intent", "decide_urgent"}
    assert "refund (wants money back)" in tools["decide_intent"].description
    result = asyncio.run(server.call_tool("decide_intent", {"text": "I want a refund"}))
    content = result.content if hasattr(result, "content") else result[0]
    payload = json.loads(content[0].text)
    assert payload["value"] == "refund"
    assert payload["status"] == "accepted"


def test_cli_loads_an_engine_and_its_questions(tmp_path, monkeypatch) -> None:
    import typer

    from thinkless.cli.serve import load_engine_and_questions

    (tmp_path / "my_decisions.py").write_text(
        "from thinkless import Engine, YesNo\n"
        "from thinkless.providers import Rules\n"
        "QUESTIONS = [YesNo('Is it urgent?', name='urgent')]\n"
        "def build():\n"
        "    return Engine([Rules()])\n"
        "engine = build()\n"
        "not_an_engine = 3\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    engine, questions = load_engine_and_questions("my_decisions:engine", None)
    assert isinstance(engine, Engine)
    assert [q.key for q in questions] == ["urgent"]
    built, _ = load_engine_and_questions("my_decisions:build", "my_decisions:QUESTIONS")
    assert isinstance(built, Engine)
    with pytest.raises(typer.BadParameter):
        load_engine_and_questions("my_decisions:not_an_engine", None)
    with pytest.raises(typer.BadParameter):
        load_engine_and_questions("my_decisions", None)
