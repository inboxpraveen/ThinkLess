from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import httpx
import pytest

from thinkless.habits import Habits, discover, read_calls, select_threshold, wrap
from thinkless.habits.calls import local_answers, observation_from
from thinkless.habits.features import featurize
from thinkless.habits.gate import binomial_lower_bound, binomial_upper_bound, min_evidence
from thinkless.habits.learn import LearnOptions, learn, load_plan
from thinkless.habits.report import build_report
from thinkless.habits.sites import profile_field
from thinkless.habits.student import LinearModel, ShapeExtractor, SiteModel
from thinkless.habits.transport import transport_classes
from thinkless.habits.wire import (
    MISSING,
    build_response,
    build_stream,
    collect_stream,
    label_of,
    normalize_text,
    parse_request,
    parse_response,
)

INTENTS = {
    "refund": ["refund", "money back", "charged twice", "reimburse me"],
    "order_status": ["where is my order", "tracking number", "has not arrived", "delivery date"],
    "cancel": ["cancel", "stop my order", "do not want it anymore"],
    "other": ["hello there", "change my password", "update my address", "opening hours"],
}
SYSTEM = "You route customer support messages.\nPick the intent and say whether they want a person."
SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "wants_human": {"type": "boolean"},
    },
    "required": ["intent", "wants_human"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "Route", "schema": SCHEMA, "strict": True},
}


def ticket(rng: random.Random) -> tuple[str, str]:
    intent = rng.choice(list(INTENTS))
    phrase = rng.choice(INTENTS[intent])
    opener = rng.choice(["Hi,", "Hello team.", "", "Quick one:", "Ugh."])
    order = rng.choice(
        ["", f" order #{rng.randint(1000, 99999)}", f" for order {rng.randint(1000, 99999)}"]
    )
    human = rng.random() < 0.2
    return (
        f"{opener} {phrase}{order}{' let me talk to a person' if human else ''} please".strip(),
        intent,
    )


def teacher_intent(text: str) -> str:
    return next((k for k, phrases in INTENTS.items() if any(p in text for p in phrases)), "other")


def completion(
    content: str | None, *, tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 12,
            "total_tokens": 132,
            "cost": 0.0002,
        },
    }


class FakeLLM:
    """A stand-in LLM: routes by keyword, and can be told to change its mind."""

    def __init__(self) -> None:
        self.calls = 0
        self.flipped = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        assert "x-thinkless-site" not in request.headers
        text = body["messages"][-1]["content"]
        intent = "other" if self.flipped else teacher_intent(text)
        if body.get("stream"):
            chunks = [
                {
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": '{"intent": '}}
                    ]
                },
                {
                    "choices": [
                        {"index": 0, "delta": {"content": json.dumps(intent) + ', "wants_human": '}}
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": json.dumps("person" in text) + "}"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ]
            data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(
                200, content=data.encode(), headers={"content-type": "text/event-stream"}
            )
        content = json.dumps({"intent": intent, "wants_human": "person" in text})
        return httpx.Response(200, json=completion(content))


def request_body(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "gpt-test",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "response_format": RESPONSE_FORMAT,
        **extra,
    }


def client_for(habits: Habits, llm: FakeLLM) -> httpx.Client:
    sync, _ = transport_classes("httpx")
    return httpx.Client(
        base_url="https://api.example.com/v1",
        transport=sync(habits, inner=httpx.MockTransport(llm)),
    )


# Wire format


def test_parse_request_contracts() -> None:
    call = parse_request(request_body("hi"))
    assert call is not None
    assert call.contract == "json_schema"
    assert call.schema_name == "Route"
    assert call.answerable
    assert call.roles == "user"

    forced = parse_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "route", "parameters": SCHEMA}}],
            "tool_choice": {"type": "function", "function": {"name": "route"}},
        }
    )
    assert forced is not None
    assert forced.contract == "tool"
    assert forced.schema == SCHEMA

    auto = parse_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        }
    )
    assert auto is not None
    assert auto.contract == "tools"
    assert not auto.answerable

    image = parse_request(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "data:"}},
                    ],
                }
            ],
        }
    )
    assert image is not None
    assert not image.answerable
    assert "images" in image.reason
    assert parse_request({"model": "m"}) is None
    assert parse_request({"model": "m", "messages": [], "n": 2}).reason  # type: ignore[union-attr]


def test_parse_response_fields_and_rebuild() -> None:
    call = parse_request(request_body("I want a refund"))
    assert call is not None
    output = parse_response(call, completion('{"intent": "refund", "wants_human": false}'))
    assert output is not None
    assert output.fields == {"intent": '"refund"', "wants_human": "false"}
    assert output.cost_usd == 0.0002
    rebuilt = build_response(call, output.fields, {"key_order": ["intent", "wants_human"]})
    content = rebuilt["choices"][0]["message"]["content"]
    assert json.loads(content) == {"intent": "refund", "wants_human": False}
    assert rebuilt["system_fingerprint"] == "thinkless-habits"
    assert parse_response(call, completion("not json")) is None
    refusal = completion(None)
    refusal["choices"][0]["message"]["refusal"] = "no"
    assert parse_response(call, refusal) is None


def test_text_labels_normalize_and_keep_raw_form() -> None:
    call = parse_request({"model": "m", "messages": [{"role": "user", "content": "ok?"}]})
    assert call is not None
    output = parse_response(call, completion(" Yes. "))
    assert output is not None
    assert output.fields == {"$text": label_of("yes")}
    assert normalize_text('"No!"') == "no"
    rebuilt = build_response(call, output.fields, {"texts": {label_of("yes"): "Yes."}})
    assert rebuilt["choices"][0]["message"]["content"] == "Yes."


def test_forced_tool_answers_rebuild_as_tool_calls() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "route", "parameters": SCHEMA}}],
        "tool_choice": {"type": "function", "function": {"name": "route"}},
    }
    call = parse_request(body)
    assert call is not None
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "route", "arguments": '{"intent": "cancel", "wants_human": true}'},
    }
    output = parse_response(call, completion(None, tool_calls=[tool_call]))
    assert output is not None
    assert output.fields["intent"] == '"cancel"'
    rebuilt = build_response(call, output.fields, {})
    message = rebuilt["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "route"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["wants_human"] is True


def test_stream_round_trip() -> None:
    call = parse_request(request_body("x", stream=True, stream_options={"include_usage": True}))
    assert call is not None
    response = build_response(call, {"intent": '"refund"', "wants_human": "false"}, {})
    events = build_stream(call, response)
    assert events.endswith(b"data: [DONE]\n\n")
    rebuilt = collect_stream([events[:40], events[40:]])
    assert rebuilt is not None
    assert json.loads(rebuilt["choices"][0]["message"]["content"])["intent"] == "refund"
    assert rebuilt["usage"]["total_tokens"] == 0


# Call logs


def test_readers_accept_finetune_and_pair_formats() -> None:
    finetune = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "refund please"},
            {"role": "assistant", "content": '{"intent": "refund", "wants_human": false}'},
        ],
        "response_format": RESPONSE_FORMAT,
    }
    observation = observation_from(finetune)
    assert observation is not None
    assert observation.output is not None
    assert observation.output.fields["intent"] == '"refund"'
    pair = {
        "request": {"body": request_body("where is my order")},
        "response": {"body": completion('{"intent": "order_status", "wants_human": false}')},
    }
    observation = observation_from(pair)
    assert observation is not None
    assert observation.cost_usd == 0.0002
    assert observation_from({"unrelated": 1}) is None


# Sites


def _observations(texts_and_answers: list[tuple[str, str, dict[str, Any]]]) -> list[Any]:
    found = []
    for system, text, answer in texts_and_answers:
        body = {
            "model": "gpt-test",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "response_format": {"type": "json_object"},
        }
        found.append(
            observation_from({"request": body, "response": completion(json.dumps(answer))})
        )
    return found


def test_discover_separates_sites_by_template() -> None:
    rng = random.Random(0)
    rows = []
    for _ in range(40):
        text, intent = ticket(rng)
        rows.append(("Route the message.\nReply in JSON.", text, {"intent": intent}))
        rows.append(("Is the customer angry?\nReply in JSON.", text, {"angry": rng.random() < 0.3}))
    sites = discover(_observations(rows))
    assert len(sites) == 2
    by_field = {next(iter(s.fields)): s for s in sites}
    assert by_field["intent"].fields["intent"].kind == "choice"
    assert by_field["angry"].fields["angry"].kind == "yes_no"
    assert by_field["intent"].verdict == "decision"
    assert "Route the message." in by_field["intent"].template_lines
    assert all("Reply in JSON." not in text for text in by_field["intent"].inputs())


def test_profile_field_kinds() -> None:
    inputs = [f"order {1000 + i} is late" for i in range(60)]
    extract = profile_field("order_id", [label_of(str(1000 + i)) for i in range(60)], inputs)
    assert extract.kind == "extract"
    free = profile_field(
        "reply", [label_of(f"Sorry about order {i}, we will help") for i in range(60)], inputs
    )
    assert free.kind == "open"
    score = profile_field("stars", [label_of(i % 5 + 1) for i in range(60)], inputs)
    assert score.kind == "score"
    constant = profile_field("ok", ["true"] * 60, inputs)
    assert constant.kind == "constant"
    schema = {"properties": {"tier": {"type": "string", "enum": ["a", "b", "c"]}}}
    enum = profile_field("tier", [label_of("a")] * 5, inputs[:5], schema)
    assert enum.kind == "choice"
    assert set(enum.options) == {'"a"', '"b"', '"c"'}


def test_mixed_sites_say_how_to_split() -> None:
    rng = random.Random(1)
    rows = []
    for i in range(60):
        text, intent = ticket(rng)
        rows.append(
            (
                "Triage.\nJSON only.",
                text,
                {"intent": intent, "reply": f"Thanks for message {i}, {text}"},
            )
        )
    (site,) = discover(_observations(rows))
    assert site.verdict == "mixed"
    plan = learn_sites_plan(site)
    assert plan["status"] == "llm"
    assert "split the call" in plan["reason"]


def learn_sites_plan(site: Any) -> dict[str, Any]:
    from thinkless.habits.learn import learn_sites

    return learn_sites([site], Path("unused"))["sites"][site.id]


# Gate


def test_binomial_bounds() -> None:
    assert binomial_lower_bound(100, 100, 0.95) == pytest.approx(0.05 ** (1 / 100))
    low = binomial_lower_bound(95, 100, 0.95)
    assert 0.88 < low < 0.91
    # The bound is where seeing 95 or more has probability 5 percent.
    tail = sum(math.comb(100, i) * low**i * (1 - low) ** (100 - i) for i in range(95, 101))
    assert tail == pytest.approx(0.05, abs=1e-3)
    assert binomial_upper_bound(0, 50, 0.95) == pytest.approx(1 - 0.05 ** (1 / 50))
    assert binomial_lower_bound(0, 10) == 0.0
    assert min_evidence(0.97, 0.95) == 99


def test_select_threshold_fixed_sequence() -> None:
    rng = random.Random(0)
    scores = [rng.random() for _ in range(2000)]
    # Agreement rises with confidence: perfect above 0.5, a coin flip below.
    agreed = [s > 0.5 or rng.random() < 0.5 for s in scores]
    gate = select_threshold(scores, agreed, target=0.97, confidence=0.95)
    assert gate is not None
    assert 0.35 < gate.threshold < 0.6
    assert gate.lower_bound >= 0.97
    assert gate.agreement >= 0.97
    assert select_threshold(scores[:50], agreed[:50]) is None
    noisy = [rng.random() < 0.9 for _ in scores]
    assert select_threshold(scores, noisy) is None


# Students


def test_linear_student_learns_and_round_trips(tmp_path: Path) -> None:
    rng = random.Random(0)
    rows = [ticket(rng) for _ in range(300)]
    texts = [t for t, _ in rows]
    labels = {"intent": [label_of(i) for _, i in rows]}
    model = SiteModel.fit(texts, labels, {"intent": "choice"})
    assert model is not None
    held = [ticket(rng) for _ in range(100)]
    hits = sum(model.predict(t)[0]["intent"] == label_of(i) for t, i in held)
    assert hits >= 95
    model.save(tmp_path / "site")
    loaded = SiteModel.load(tmp_path / "site")
    for text, _ in held[:10]:
        assert loaded.predict(text) == pytest.approx(model.predict(text))


def test_shape_extractor_finds_order_numbers() -> None:
    texts = [f"my order #{1000 + i * 37} has not arrived" for i in range(50)]
    texts += [f"hello, a question about my account {i}" for i in range(10)]
    labels = [label_of(str(1000 + i * 37)) for i in range(50)] + ["null"] * 10
    extractor = ShapeExtractor.fit(texts, labels)
    assert extractor is not None
    label, confidence = extractor.predict(featurize("x"), "where is order 5512?")
    assert label == label_of("5512")
    assert confidence > 0.9
    assert extractor.predict({}, "no number here")[0] == "null"


def test_linear_model_handles_unknown_features() -> None:
    model = LinearModel.fit([featurize("a b"), featurize("c d")] * 5, ["x", "y"] * 5)
    probs = model.proba({123456789: 1.0})
    assert probs.sum() == pytest.approx(1.0)


# Runtime, end to end


def _run(
    client: httpx.Client, rng: random.Random, count: int, **extra: Any
) -> list[httpx.Response]:
    responses = []
    for _ in range(count):
        text, _ = ticket(rng)
        responses.append(client.post("/chat/completions", json=request_body(text, **extra)))
    return responses


def test_habits_learns_then_answers_locally(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = client_for(habits, llm)
    rng = random.Random(3)
    first = _run(client, rng, 250)
    assert all(r.headers.get("x-thinkless-habit") is None for r in first)
    assert llm.calls == 250
    plan = habits.learn()
    (site,) = plan["sites"].values()
    assert site["status"] == "active"
    assert site["evidence"]["lower_bound"] >= 0.97
    second = _run(client, rng, 100)
    local = [r for r in second if r.headers.get("x-thinkless-habit") == "local"]
    assert len(local) >= 80
    assert llm.calls == 250 + 100 - len(local)
    body = local[0].json()
    answer = json.loads(body["choices"][0]["message"]["content"])
    assert set(answer) == {"intent", "wants_human"}
    assert body["usage"]["total_tokens"] == 0
    report = build_report(
        read_calls([tmp_path / "calls"]),
        plan=load_plan(tmp_path),
        local=local_answers([tmp_path / "calls"]),
    )
    assert report.local == len(local)
    assert report.sites[0].saved_usd == pytest.approx(0.0002 * len(local))
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert '"activated"' in events
    habits.close()


def test_audits_suspend_a_site_that_stops_agreeing(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=1.0, seed=0)
    client = client_for(habits, llm)
    rng = random.Random(4)
    _run(client, rng, 250)
    habits.learn()
    # The LLM starts answering differently: every audit now disagrees.
    llm.flipped = True
    _run(client, rng, 120)
    plan = load_plan(tmp_path)
    assert plan is not None
    (site,) = plan["sites"].values()
    assert site["status"] == "suspended"
    assert "suspended" in (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    # A suspended site goes back to the LLM for everything.
    after = _run(client, rng, 20)
    assert all(r.headers.get("x-thinkless-habit") is None for r in after)
    habits.close()


def test_modes(tmp_path: Path) -> None:
    llm = FakeLLM()
    trainer = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    _run(client_for(trainer, llm), random.Random(5), 250)
    trainer.learn()
    trainer.close()
    for mode in ("shadow", "record", "off"):
        habits = Habits(tmp_path, mode=mode, learn_every=None, audit=0.0)
        responses = _run(client_for(habits, llm), random.Random(6), 20)
        assert all(r.headers.get("x-thinkless-habit") is None for r in responses)
        habits.close()
    shadowed = [o for o in read_calls([tmp_path / "calls"]) if o.local]
    assert len(shadowed) >= 20
    with pytest.raises(ValueError):
        Habits(tmp_path, mode="fly")


def test_streaming_calls_are_recorded_and_answered(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = client_for(habits, llm)
    rng = random.Random(7)
    for _ in range(200):
        text, _ = ticket(rng)
        with client.stream(
            "POST", "/chat/completions", json=request_body(text, stream=True)
        ) as response:
            assert b"[DONE]" in response.read()
    teacher = [o for o in read_calls([tmp_path / "calls"]) if o.teacher]
    assert len(teacher) == 200
    habits.learn()
    text, _ = ticket(rng)
    response = client.post("/chat/completions", json=request_body(text, stream=True))
    assert response.headers["x-thinkless-habit"] == "local"
    assert response.headers["content-type"] == "text/event-stream"
    rebuilt = collect_stream([response.content])
    assert rebuilt is not None
    assert "intent" in rebuilt["choices"][0]["message"]["content"]
    habits.close()


def test_other_requests_pass_through(tmp_path: Path) -> None:
    seen = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    habits = Habits(tmp_path, confirm_calls=0, learn_every=None)
    sync, _ = transport_classes("httpx")
    client = httpx.Client(
        base_url="https://api.example.com/v1",
        transport=sync(habits, inner=httpx.MockTransport(upstream)),
    )
    assert client.get("/models").json() == {"data": []}
    assert client.post("/embeddings", json={"input": "x"}).status_code == 200
    assert seen == ["/v1/models", "/v1/embeddings"]
    assert not (tmp_path / "calls").exists()


def test_background_learning(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=120, audit=0.0, seed=0, min_calls=100)
    client = client_for(habits, llm)
    _run(client, random.Random(8), 240)
    habits.wait(60)
    plan = load_plan(tmp_path)
    assert plan is not None
    assert next(iter(plan["sites"].values()))["status"] == "active"
    habits.close()


def test_async_transport(tmp_path: Path) -> None:
    import asyncio

    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    _, asynchronous = transport_classes("httpx")

    async def main() -> list[httpx.Response]:
        async with httpx.AsyncClient(
            base_url="https://api.example.com/v1",
            transport=asynchronous(habits, inner=httpx.MockTransport(llm)),
        ) as client:
            rng = random.Random(9)
            for _ in range(220):
                text, _ = ticket(rng)
                await client.post("/chat/completions", json=request_body(text))
            habits.learn()
            return [
                await client.post("/chat/completions", json=request_body(ticket(rng)[0]))
                for _ in range(20)
            ]

    responses = asyncio.run(main())
    assert any(r.headers.get("x-thinkless-habit") == "local" for r in responses)
    habits.close()


def test_site_header_names_a_site_and_is_not_forwarded(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None)
    client = client_for(habits, llm)
    for text in ("refund please", "cancel it"):
        client.post(
            "/chat/completions", json=request_body(text), headers={"x-thinkless-site": "router"}
        )
    (site,) = discover(read_calls([tmp_path / "calls"]))
    assert site.name == "router"
    habits.close()


# SDK integration


def test_wrap_openai_sdk(tmp_path: Path) -> None:
    openai = pytest.importorskip("openai")
    pydantic = pytest.importorskip("pydantic")
    from typing import Literal

    from thinkless.habits.transport import sdk_http_library

    library = sdk_http_library()
    lib = pytest.importorskip(library)
    llm = FakeLLM()

    def handler(request: Any) -> Any:
        response = llm(
            httpx.Request(
                request.method,
                str(request.url),
                headers=dict(request.headers),
                content=request.content,
            )
        )
        return lib.Response(
            response.status_code, headers=dict(response.headers), content=response.content
        )

    class Route(pydantic.BaseModel):
        intent: Literal["refund", "order_status", "cancel", "other"]
        wants_human: bool

    client = openai.OpenAI(
        api_key="test", http_client=lib.Client(transport=lib.MockTransport(handler))
    )
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    assert wrap(client, habits) is client
    wrap(client, habits)  # wrapping twice is harmless
    rng = random.Random(10)

    def ask() -> Any:
        text, _ = ticket(rng)
        return client.chat.completions.parse(
            model="gpt-test",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            response_format=Route,
        )

    for _ in range(220):
        ask()
    habits.learn()
    parsed = [ask().choices[0].message.parsed for _ in range(30)]
    assert all(isinstance(p, Route) for p in parsed)
    assert llm.calls < 220 + 30
    habits.close()


def test_wrap_rejects_unknown_clients() -> None:
    with pytest.raises(TypeError, match="proxy"):
        wrap(object())


# Learning options and CLI


def test_learning_waits_for_enough_calls(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None)
    _run(client_for(habits, llm), random.Random(11), 40)
    plan = learn(tmp_path, options=LearnOptions())
    (site,) = plan["sites"].values()
    assert site["status"] == "learning"
    assert "40 of 100" in site["reason"]
    habits.close()


def test_cli_scan_learn_status(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from thinkless.cli.main import app

    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None)
    _run(client_for(habits, llm), random.Random(12), 150)
    habits.close()
    runner = CliRunner()
    scanned = runner.invoke(app, ["habits", "scan", "--dir", str(tmp_path)])
    assert scanned.exit_code == 0, scanned.output
    assert "decision" in scanned.output
    as_json = runner.invoke(app, ["habits", "scan", "--dir", str(tmp_path), "--json"])
    assert json.loads(as_json.output)["decision_calls"] == 150
    learned = runner.invoke(app, ["habits", "learn", "--dir", str(tmp_path)])
    assert learned.exit_code == 0, learned.output
    plan = load_plan(tmp_path)
    assert plan is not None
    assert next(iter(plan["sites"].values()))["status"] == "probation"
    status = runner.invoke(app, ["habits", "status", "--dir", str(tmp_path)])
    assert status.exit_code == 0, status.output
    reset = runner.invoke(app, ["habits", "reset", "--dir", str(tmp_path), "--yes"])
    assert reset.exit_code == 0
    assert not (tmp_path / "plan.json").exists()
    missing = runner.invoke(app, ["habits", "scan", str(tmp_path / "nope")])
    assert missing.exit_code == 1


def test_proxy_forwards_and_answers(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from thinkless.habits.proxy import create_proxy

    llm = FakeLLM()
    seen_auth = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-test"}]})
        assert str(request.url).startswith("https://api.example.com/v1/chat/completions")
        return llm(request)

    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    app = create_proxy(
        "https://api.example.com/v1", habits, transport=httpx.MockTransport(upstream)
    )
    with TestClient(app) as client:
        assert client.get("/v1/models", headers={"authorization": "Bearer k"}).json()["data"]
        rng = random.Random(13)
        for _ in range(220):
            text, _ = ticket(rng)
            assert client.post("/v1/chat/completions", json=request_body(text)).status_code == 200
        habits.learn()
        replies = [
            client.post("/v1/chat/completions", json=request_body(ticket(rng)[0]))
            for _ in range(20)
        ]
        assert any(r.headers.get("x-thinkless-habit") == "local" for r in replies)
        status = client.get("/habits/status").json()
        assert next(iter(status["sites"].values()))["status"] == "active"
    assert seen_auth[0] == "Bearer k"
    assert MISSING not in json.dumps(replies[0].json())


# The OpenAI Responses API and the Anthropic Messages API


def _library_of(sdk: Any) -> Any:
    import importlib

    base = getattr(sdk, "DefaultHttpxClient", None)
    for cls in getattr(base, "__mro__", ()):
        root = cls.__module__.split(".")[0]
        if root in ("httpx", "httpx2"):
            return importlib.import_module(root)
    return httpx


def test_openai_responses_api(tmp_path: Path) -> None:
    openai = pytest.importorskip("openai")
    pydantic = pytest.importorskip("pydantic")
    from typing import Literal

    lib = _library_of(openai)
    calls = {"n": 0}

    def handler(request: Any) -> Any:
        calls["n"] += 1
        body = json.loads(request.content)
        assert request.url.path.endswith("/responses")
        assert body["text"]["format"]["type"] == "json_schema"
        text = body["input"][-1]["content"]
        content = json.dumps({"intent": teacher_intent(text), "wants_human": "person" in text})
        return lib.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-test",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 90, "output_tokens": 11, "total_tokens": 101},
            },
        )

    class Route(pydantic.BaseModel):
        intent: Literal["refund", "order_status", "cancel", "other"]
        wants_human: bool

    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = wrap(
        openai.OpenAI(api_key="test", http_client=lib.Client(transport=lib.MockTransport(handler))),
        habits,
    )
    rng = random.Random(14)

    def ask() -> Any:
        text, _ = ticket(rng)
        return client.responses.parse(
            model="gpt-test",
            instructions=SYSTEM,
            input=[{"role": "user", "content": text}],
            text_format=Route,
        )

    for _ in range(220):
        ask()
    plan = habits.learn()
    assert next(iter(plan["sites"].values()))["status"] == "active"
    before = calls["n"]
    answers = [ask() for _ in range(20)]
    assert calls["n"] - before < 20
    assert all(isinstance(a.output_parsed, Route) for a in answers)
    habits.close()


def test_anthropic_messages_api(tmp_path: Path) -> None:
    anthropic = pytest.importorskip("anthropic")
    lib = _library_of(anthropic)
    calls = {"n": 0}
    tool = {"name": "route", "description": "Route the message.", "input_schema": SCHEMA}

    def handler(request: Any) -> Any:
        calls["n"] += 1
        body = json.loads(request.content)
        assert request.url.path.endswith("/v1/messages")
        text = body["messages"][-1]["content"]
        return lib.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "route",
                        "input": {"intent": teacher_intent(text), "wants_human": "person" in text},
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 300, "output_tokens": 40},
            },
        )

    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = wrap(
        anthropic.Anthropic(
            api_key="test", http_client=lib.Client(transport=lib.MockTransport(handler))
        ),
        habits,
    )
    rng = random.Random(15)

    def ask() -> Any:
        text, _ = ticket(rng)
        return client.messages.create(
            model="claude-test",
            max_tokens=200,
            system=SYSTEM,
            messages=[{"role": "user", "content": text}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "route"},
        )

    for _ in range(220):
        ask()
    observations = list(read_calls([tmp_path / "calls"]))
    assert observations[0].call.api == "messages"
    assert observations[0].call.contract == "tool"
    plan = habits.learn()
    assert next(iter(plan["sites"].values()))["status"] == "active"
    before = calls["n"]
    replies = [ask() for _ in range(20)]
    assert calls["n"] - before < 20
    for reply in replies:
        block = reply.content[0]
        assert block.type == "tool_use"
        assert block.name == "route"
        assert set(block.input) == {"intent", "wants_human"}
    habits.close()


def test_streamed_messages_and_responses_are_recorded() -> None:
    from thinkless.habits.apis import MESSAGES, RESPONSES

    anthropic_events = [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "c",
                "content": [],
                "usage": {"input_tokens": 5},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "route", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"intent": "ref'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'und"}'},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]
    stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in anthropic_events)
    message = MESSAGES.collect_stream([stream.encode()])
    assert message is not None
    assert message["content"][0]["input"] == {"intent": "refund"}
    assert message["stop_reason"] == "tool_use"

    completed = {
        "type": "response.completed",
        "response": {"id": "r", "status": "completed", "output": []},
    }
    events = (
        f"data: {json.dumps({'type': 'response.created'})}\n\ndata: {json.dumps(completed)}\n\n"
    )
    assert RESPONSES.collect_stream([events.encode()]) == completed["response"]


def test_api_specific_requests_are_not_answered_when_unsafe() -> None:
    from thinkless.habits.apis import MESSAGES, RESPONSES, api_for_path

    stored = RESPONSES.parse_request(
        {"model": "m", "input": "hi", "previous_response_id": "resp_1"}
    )
    assert stored is not None
    assert not stored.answerable
    thinking = MESSAGES.parse_request(
        {
            "model": "c",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }
    )
    assert thinking is not None
    assert not thinking.answerable
    assert MESSAGES.parse_request({"role": "user", "content": "assistants api body"}) is None
    assert api_for_path("/v1/chat/completions").name == "chat"  # type: ignore[union-attr]
    assert api_for_path("/v1/responses").name == "responses"  # type: ignore[union-attr]
    assert api_for_path("/v1/messages").name == "messages"  # type: ignore[union-attr]
    assert api_for_path("/v1/embeddings") is None


# Frameworks on top of the SDKs


def _chat_handler(lib: Any, counter: dict[str, int]) -> Any:
    def handler(request: Any) -> Any:
        counter["n"] += 1
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        answer = json.dumps({"intent": teacher_intent(text), "wants_human": "person" in text})
        message: dict[str, Any] = {"role": "assistant", "content": answer}
        if body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": name, "arguments": answer},
                    }
                ],
            }
        return lib.Response(
            200,
            json={
                **completion(None),
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    return handler


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
def test_langchain_structured_output(tmp_path: Path, method: str) -> None:
    langchain_openai = pytest.importorskip("langchain_openai")
    openai = pytest.importorskip("openai")
    pydantic = pytest.importorskip("pydantic")
    from typing import Literal

    lib = _library_of(openai)
    counter = {"n": 0}
    handler = _chat_handler(lib, counter)

    class Route(pydantic.BaseModel):
        intent: Literal["refund", "order_status", "cancel", "other"]
        wants_human: bool

    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    chat = langchain_openai.ChatOpenAI(
        model="gpt-test",
        api_key="test",
        http_client=lib.Client(transport=lib.MockTransport(handler)),
        http_async_client=lib.AsyncClient(transport=lib.MockTransport(handler)),
    )
    router = habits_wrap(chat, habits).with_structured_output(Route, method=method)
    rng = random.Random(16)
    for _ in range(220):
        router.invoke([("system", SYSTEM), ("user", ticket(rng)[0])])
    habits.learn()
    before = counter["n"]
    outputs = [router.invoke([("system", SYSTEM), ("user", ticket(rng)[0])]) for _ in range(20)]
    assert counter["n"] - before < 20
    assert all(isinstance(o, Route) for o in outputs)
    habits.close()


def habits_wrap(client: Any, habits: Habits) -> Any:
    return wrap(client, habits)


def test_openai_agents_sdk(tmp_path: Path) -> None:
    agents = pytest.importorskip("agents")
    openai = pytest.importorskip("openai")
    pydantic = pytest.importorskip("pydantic")
    import asyncio
    from typing import Literal

    lib = _library_of(openai)
    counter = {"n": 0}

    def handler(request: Any) -> Any:
        counter["n"] += 1
        body = json.loads(request.content)
        text = body["input"][-1]["content"]
        answer = json.dumps({"intent": teacher_intent(text)})
        return lib.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-test",
                "output": [
                    {
                        "type": "message",
                        "id": "m1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": answer, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
            },
        )

    class Route(pydantic.BaseModel):
        intent: Literal["refund", "order_status", "cancel", "other"]

    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = wrap(
        openai.AsyncOpenAI(
            api_key="test", http_client=lib.AsyncClient(transport=lib.MockTransport(handler))
        ),
        habits,
    )
    model = agents.OpenAIResponsesModel(model="gpt-test", openai_client=client)
    agent = agents.Agent(name="router", instructions=SYSTEM, model=model, output_type=Route)
    rng = random.Random(17)

    async def main() -> list[Any]:
        for _ in range(220):
            await agents.Runner.run(
                agent, ticket(rng)[0], run_config=agents.RunConfig(tracing_disabled=True)
            )
        habits.learn()
        return [
            (
                await agents.Runner.run(
                    agent, ticket(rng)[0], run_config=agents.RunConfig(tracing_disabled=True)
                )
            ).final_output
            for _ in range(20)
        ]

    before_total = 220
    outputs = asyncio.run(main())
    assert counter["n"] - before_total < 20
    assert all(isinstance(o, Route) for o in outputs)
    habits.close()


# Production behavior


def test_local_answers_are_logged_compactly(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = client_for(habits, llm)
    rng = random.Random(18)
    _run(client, rng, 220)
    habits.learn()
    _run(client, rng, 30)
    habits.close()
    lines = [
        json.loads(line)
        for path in (tmp_path / "calls").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    local = [r for r in lines if r["answered_by"] == "habit"]
    assert local
    assert all(r["request"] is None and r["response"] is None for r in local)
    assert all(r["local"]["fields"] for r in local)
    assert sum(local_answers([tmp_path / "calls"]).values()) == len(local)
    # Reading the log back for learning skips them.
    assert all(o.answered_by != "habit" for o in read_calls([tmp_path / "calls"]))


def test_one_learning_run_at_a_time(tmp_path: Path) -> None:
    from thinkless.habits.learn import LearningInProgressError

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / ".learning.lock").write_text("123", encoding="utf-8")
    with pytest.raises(LearningInProgressError):
        learn(tmp_path)
    # A lock left by a crashed process goes stale.
    import os

    old = 1_000_000_000
    os.utime(tmp_path / ".learning.lock", (old, old))
    plan = learn(tmp_path)
    assert plan["sites"] == {}
    assert not (tmp_path / ".learning.lock").exists()


def test_learning_reads_the_most_recent_calls(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None)
    _run(client_for(habits, llm), random.Random(19), 150)
    habits.close()
    plan = learn(tmp_path, options=LearnOptions(max_observations=60))
    (site,) = plan["sites"].values()
    assert site["calls"] == 60


def test_a_serving_student_stays_until_a_newer_one_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    learn_module = importlib.import_module("thinkless.habits.learn")

    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = client_for(habits, llm)
    rng = random.Random(20)
    _run(client, rng, 250)
    (first,) = habits.learn()["sites"].values()
    assert first["status"] == "active"
    # The next gate fails, as it can once confident calls stop reaching the LLM.
    monkeypatch.setattr(learn_module, "select_threshold", lambda *a, **k: None)
    _run(client, rng, 50)
    (second,) = habits.learn()["sites"].values()
    assert second["status"] == "active"
    assert second["student"] == first["student"]
    assert second["threshold"] == first["threshold"]
    assert "not reached" in second["candidate"]
    assert (tmp_path / "models" / f"{first['student']}.npz").exists()
    local = _run(client, rng, 20)
    assert any(r.headers.get("x-thinkless-habit") == "local" for r in local)
    # A newer student that passes replaces it, and the old files are removed.
    monkeypatch.undo()
    (third,) = habits.learn()["sites"].values()
    assert third["student"] != first["student"]
    assert not (tmp_path / "models" / f"{first['student']}.npz").exists()
    habits.close()


def test_habits_fails_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, confirm_calls=0, learn_every=None, audit=0.0, seed=0)
    client = client_for(habits, llm)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("corrupt plan")

    monkeypatch.setattr(habits, "begin", broken)
    response = client.post("/chat/completions", json=request_body("refund please"))
    assert response.status_code == 200
    assert llm.calls == 1
    monkeypatch.undo()
    monkeypatch.setattr(habits, "finish", broken)
    response = client.post("/chat/completions", json=request_body("cancel it"))
    assert response.status_code == 200
    assert json.loads(response.json()["choices"][0]["message"]["content"])["intent"] == "cancel"
    habits.close()


def test_recurring_input_lines_do_not_split_a_site() -> None:
    rng = random.Random(21)
    rows = []
    for i in range(60):
        text, intent = ticket(rng)
        if i % 5 < 2:
            # Boilerplate that recurs in 40% of inputs, like a retrieved passage.
            text = f"{text}\nSent from my phone"
        rows.append(("Route the message.\nReply in JSON.", text, {"intent": intent}))
    sites = discover(_observations(rows))
    assert len(sites) == 1


# Probation


def test_probation_confirms_on_new_calls_before_answering(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, learn_every=None, audit=0.0, seed=0, confirm_calls=100)
    client = client_for(habits, llm)
    rng = random.Random(22)
    _run(client, rng, 250)
    (site,) = habits.learn()["sites"].values()
    assert site["status"] == "probation"
    # On probation nothing is answered locally; the student is scored in shadow.
    during = _run(client, rng, 60)
    assert all(r.headers.get("x-thinkless-habit") is None for r in during)
    (site,) = habits.learn()["sites"].values()
    assert site["status"] == "probation"
    assert "of 100 so far" in site["reason"]
    _run(client, rng, 60)
    (site,) = habits.learn()["sites"].values()
    assert site["status"] == "active"
    assert site["evidence"]["confirmation"]["calls"] == 100
    assert site["evidence"]["confirmation"]["agreed"] == 100
    after = _run(client, rng, 30)
    assert any(r.headers.get("x-thinkless-habit") == "local" for r in after)
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert '"probation"' in events
    assert '"activated"' in events
    habits.close()


def test_a_student_that_fails_probation_is_replaced(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, learn_every=None, audit=0.0, seed=0, confirm_calls=100)
    client = client_for(habits, llm)
    rng = random.Random(23)
    _run(client, rng, 250)
    (site,) = habits.learn()["sites"].values()
    assert site["status"] == "probation"
    first = site["student"]
    # The LLM changes its answers while the student is on probation.
    llm.flipped = True
    _run(client, rng, 120)
    (site,) = habits.learn()["sites"].values()
    assert site["status"] != "active"
    assert site["last_probation"]["student"] == first
    assert site["last_probation"]["agreed"] < 100
    habits.close()


def test_confirm_calls_zero_skips_probation(tmp_path: Path) -> None:
    llm = FakeLLM()
    habits = Habits(tmp_path, learn_every=None, audit=0.0, seed=0, confirm_calls=0)
    _run(client_for(habits, llm), random.Random(24), 250)
    (site,) = habits.learn()["sites"].values()
    assert site["status"] == "active"
    habits.close()
