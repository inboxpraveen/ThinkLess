from __future__ import annotations

import asyncio

import pytest

from tests.conftest import FakeProvider, choice_answer, yes_answer
from thinkless import (
    Answer,
    Choice,
    ConfigurationError,
    Extract,
    Kind,
    Plane,
    Score,
    Status,
    YesNo,
    summarize,
)
from thinkless.llm import ScriptedLLM

INTENT = Choice(
    "What does the customer want?", options=["refund", "status", "other"], name="intent"
)
URGENT = YesNo("Is it urgent?", name="urgent")


def test_accepts_first_confident_answer(make_engine) -> None:
    fast = FakeProvider(
        "fast", {"intent": choice_answer("refund", {"refund": 0.96, "status": 0.02, "other": 0.02})}
    )
    slow = FakeProvider(
        "slow", {"intent": choice_answer("status", {"refund": 0, "status": 1, "other": 0})}
    )
    engine = make_engine([fast, slow], threshold=0.8)

    decision = engine.decide("charged twice", INTENT)

    assert decision.status is Status.ACCEPTED
    assert decision.value == "refund"
    assert decision.provider == "fast"
    assert decision.confidence == pytest.approx((3 * 0.96 - 1) / 2)
    assert decision.probability == pytest.approx(0.96)
    assert slow.calls == []
    assert decision.is_("refund")
    assert not decision.escalated


def test_escalates_below_threshold(make_engine) -> None:
    fast = FakeProvider(
        "fast", {"intent": choice_answer("refund", {"refund": 0.5, "status": 0.4, "other": 0.1})}
    )
    slow = FakeProvider(
        "slow", {"intent": choice_answer("status", {"refund": 0.02, "status": 0.97, "other": 0.01})}
    )
    engine = make_engine([fast, slow], threshold=0.8)

    decision = engine.decide("where is it", INTENT)

    assert decision.provider == "slow"
    assert decision.value == "status"
    assert [a.provider for a in decision.attempts] == ["fast", "slow"]
    assert [a.reason for a in decision.attempts] == ["below_threshold", "met_threshold"]
    assert decision.escalated


def test_uncertain_returns_best_answer(make_engine) -> None:
    a = FakeProvider(
        "a", {"intent": choice_answer("refund", {"refund": 0.6, "status": 0.3, "other": 0.1})}
    )
    b = FakeProvider(
        "b", {"intent": choice_answer("other", {"refund": 0.2, "status": 0.3, "other": 0.5})}
    )
    engine = make_engine([a, b], threshold=0.9)

    decision = engine.decide("hmm", INTENT)

    assert decision.status is Status.UNCERTAIN
    assert decision.value == "refund"
    assert decision.provider == "a"
    assert not decision.is_("refund")
    assert len(decision.attempts) == 2


def test_abstained_when_nobody_answers(make_engine) -> None:
    engine = make_engine([FakeProvider("a", {"intent": None})])
    decision = engine.decide("?", INTENT)
    assert decision.status is Status.ABSTAINED
    assert decision.value is None
    assert decision.attempts[0].reason == "abstained"


def test_batches_only_pending_questions(make_engine) -> None:
    fast = FakeProvider(
        "fast",
        {
            "intent": choice_answer("refund", {"refund": 0.99, "status": 0.005, "other": 0.005}),
            "urgent": yes_answer(0.6),
        },
    )
    slow = FakeProvider("slow", {"urgent": yes_answer(0.98)})
    engine = make_engine([fast, slow], threshold=0.8)

    decisions = engine.decide_many("text", [INTENT, URGENT])

    assert fast.calls == [["intent", "urgent"]]
    assert slow.calls == [["urgent"]]
    assert decisions["intent"].provider == "fast"
    assert decisions["urgent"].provider == "slow"
    assert decisions["urgent"].value is True
    assert list(decisions) == ["intent", "urgent"]


def test_skips_providers_that_do_not_support_the_kind(make_engine) -> None:
    choice_only = FakeProvider("choice_only", {}, kinds=frozenset({Kind.CHOICE}))
    anything = FakeProvider("anything", {"urgent": yes_answer(0.99)})
    engine = make_engine([choice_only, anything])
    decision = engine.decide("x", URGENT)
    assert choice_only.calls == []
    assert decision.provider == "anything"


def test_question_allowlist_restricts_providers(make_engine) -> None:
    weak = FakeProvider("weak", {"approve": yes_answer(0.99)})
    strong = FakeProvider("strong", {"approve": yes_answer(0.97)})
    engine = make_engine([weak, strong])
    decision = engine.decide("x", YesNo("Approve?", name="approve", providers=["strong"]))
    assert weak.calls == []
    assert decision.provider == "strong"


def test_threshold_precedence(make_engine) -> None:
    answer = choice_answer(
        "refund", {"refund": 0.9, "status": 0.05, "other": 0.05}
    )  # confidence 0.85
    provider = FakeProvider("p", {"intent": answer, "strict": answer, "loose": answer})
    engine = make_engine([provider], threshold=0.8, thresholds={"strict": 0.95})

    assert engine.decide("x", INTENT).status is Status.ACCEPTED
    assert engine.decide("x", INTENT, name="strict").status is Status.UNCERTAIN
    loose = Choice("q", options=["refund", "status", "other"], name="loose", threshold=0.5)
    assert engine.decide("x", loose).threshold == 0.5


def test_uncalibrated_provider_is_trusted_by_default(make_engine) -> None:
    llm = FakeProvider("llm", {"intent": Answer(value="status")}, plane=Plane.LLM, calibrated=False)
    decision = make_engine([llm]).decide("x", INTENT)
    assert decision.status is Status.ACCEPTED
    assert decision.confidence is None
    assert decision.attempts[0].reason == "trusted_uncalibrated"

    strict = make_engine([llm], trust_uncalibrated=False).decide("x", INTENT)
    assert strict.status is Status.UNCERTAIN


def test_provider_errors_fall_through(make_engine, memory) -> None:
    broken = FakeProvider("broken", error=RuntimeError("model crashed"))
    backup = FakeProvider(
        "backup", {"intent": choice_answer("other", {"refund": 0, "status": 0, "other": 1})}
    )
    decision = make_engine([broken, backup]).decide("x", INTENT)
    assert decision.provider == "backup"
    assert decision.attempts[0].reason == "error: RuntimeError"
    errored = [s for s in memory.spans if s.status == "error"]
    assert errored
    assert errored[0].name == "broken"


def test_on_error_raise(make_engine) -> None:
    broken = FakeProvider("broken", error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        make_engine([broken], on_error="raise").decide("x", INTENT)


def test_score_decision_level_and_value(make_engine) -> None:
    score = Score("How urgent?", levels=["low", "medium", "high"], name="urgency")
    answer = Answer(value=1.8, probabilities={"low": 0.05, "medium": 0.1, "high": 0.85})
    decision = make_engine([FakeProvider("p", {"urgency": answer})], threshold=0.7).decide(
        "x", score
    )
    assert decision.level == "high"
    assert decision.value == pytest.approx(1.8)
    assert decision.probability == pytest.approx(0.85)


def test_extract_confidence_rules(make_engine) -> None:
    fields = Extract(
        fields={"order_id": "order number", "email": "email"}, required=["order_id"], name="order"
    )
    found = Answer(
        value={"order_id": "4471", "email": None}, fields={"order_id": 0.93, "email": None}
    )
    missing = Answer(
        value={"order_id": None, "email": "a@b.co"}, fields={"order_id": None, "email": 0.99}
    )
    backup = Answer(
        value={"order_id": "4471", "email": None}, fields={"order_id": 1.0, "email": None}
    )

    ok = make_engine([FakeProvider("p", {"order": found})], threshold=0.9).decide("x", fields)
    assert ok.status is Status.ACCEPTED
    assert ok.confidence == pytest.approx(0.93)

    escalated = make_engine(
        [FakeProvider("p", {"order": missing}), FakeProvider("q", {"order": backup})]
    ).decide("x", fields)
    assert escalated.provider == "q"
    assert escalated.attempts[0].confidence == 0.0


def test_extract_shorthand(make_engine) -> None:
    answer = Answer(value={"order_id": "123"}, fields={"order_id": 0.99})
    engine = make_engine([FakeProvider("p", {"order": answer})])
    decision = engine.extract("order 123", {"order_id": "order number"}, name="order")
    assert decision.value == {"order_id": "123"}


def test_generate_records_llm_span(make_engine, memory) -> None:
    llm = ScriptedLLM(["Sorry about that. The refund is on its way."])
    engine = make_engine([], llm=llm)
    with engine.run("reply") as run:
        completion = engine.generate("Write an apology", system="Be brief", name="reply")
    assert completion.text.startswith("Sorry")
    llm_spans = [s for s in memory.spans if s.kind == "llm"]
    assert llm_spans[0].attributes["completion"] == completion.text
    summary = run.summary()
    assert summary.llm_calls == 1
    assert summary.generation_calls == 1
    assert summary.llm_output_tokens > 0


def test_generate_without_llm_raises(make_engine) -> None:
    with pytest.raises(ConfigurationError):
        make_engine([]).generate("hi")


def test_rule_is_counted_as_a_decision(make_engine) -> None:
    engine = make_engine(
        [
            FakeProvider(
                "p", {"intent": choice_answer("refund", {"refund": 1, "status": 0, "other": 0})}
            )
        ]
    )
    with engine.run("ticket", mode="hybrid") as run:
        assert engine.rule("authenticated", True) is True
        with engine.step("triage"):
            engine.decide("x", INTENT)
    summary = run.summary()
    assert summary.decisions == 2
    assert summary.decisions_by_plane == {"rule": 1, "model": 1}
    assert summary.attributes["mode"] == "hybrid"
    kinds = {s["kind"] for s in run.spans}
    assert kinds == {"run", "rule", "step", "decide", "attempt"}


def test_span_tree_is_linked(make_engine, memory) -> None:
    engine = make_engine(
        [
            FakeProvider(
                "p", {"intent": choice_answer("refund", {"refund": 1, "status": 0, "other": 0})}
            )
        ]
    )
    with engine.run("ticket"), engine.step("triage"):
        decision = engine.decide("x", INTENT)
    by_id = {s.span_id: s for s in memory.spans}
    decide = by_id[decision.span_id]
    step = by_id[decide.parent_id]
    run = by_id[step.parent_id]
    assert (decide.kind, step.kind, run.kind) == ("decide", "step", "run")
    assert run.parent_id is None
    assert len({s.trace_id for s in memory.spans}) == 1


def test_capture_content_off_redacts(make_engine, memory) -> None:
    from thinkless import Tracer

    # Markers use letters outside 0-9a-f so they can never appear by chance
    # inside random hex trace ids or numeric timestamps.
    engine = make_engine(
        [
            FakeProvider(
                "p", {"order": Answer(value={"order_id": "zqx-order"}, fields={"order_id": 1.0})}
            )
        ],
        tracer=Tracer([memory], capture_content=False),
    )
    with engine.run("ticket", input={"message": "wyz-private-note"}):
        engine.extract("my order zqx-order", {"order_id": "order"}, name="order")
    text = str([s.to_dict() for s in memory.spans])
    assert "zqx-order" not in text
    assert "wyz-private-note" not in text
    assert "[redacted]" in text


def test_duplicate_provider_names_rejected(make_engine) -> None:
    with pytest.raises(ConfigurationError):
        make_engine([FakeProvider("a"), FakeProvider("a")])


def test_duplicate_question_names_rejected(make_engine) -> None:
    with pytest.raises(ConfigurationError):
        make_engine([FakeProvider("a")]).decide_many("x", [INTENT, INTENT])


def test_async_api(make_engine) -> None:
    engine = make_engine([FakeProvider("p", {"urgent": yes_answer(0.99)})])

    async def main() -> bool:
        decision = await engine.adecide("x", URGENT)
        return bool(decision.value)

    assert asyncio.run(main()) is True


def test_summary_counts_escalations_and_cost(make_engine) -> None:
    fast = FakeProvider(
        "fast", {"intent": choice_answer("refund", {"refund": 0.4, "status": 0.3, "other": 0.3})}
    )
    llm = FakeProvider("llm", {"intent": Answer(value="status")}, plane=Plane.LLM, calibrated=False)
    engine = make_engine([fast, llm])
    with engine.run("t") as run:
        engine.decide("x", INTENT)
    summary = summarize(run.spans)
    assert summary.escalations == 1
    assert summary.llm_calls == 1
    assert summary.decisions_by_plane == {"llm": 1}


def test_abstaining_rule_is_not_an_escalation(make_engine) -> None:
    from thinkless.providers import Rules

    rules = Rules()
    rules.match("intent", r"never matches", "refund")
    model = FakeProvider(
        "model", {"intent": choice_answer("status", {"refund": 0.0, "status": 1.0, "other": 0.0})}
    )
    low = FakeProvider(
        "low", {"intent": choice_answer("status", {"refund": 0.3, "status": 0.4, "other": 0.3})}
    )

    passed_on = make_engine([rules, model]).decide("x", INTENT)
    assert [a.provider for a in passed_on.attempts] == ["rules", "model"]
    assert not passed_on.escalated

    escalated = make_engine([low, model]).decide("x", INTENT)
    assert escalated.escalated
    assert escalated.summary()["escalated"] is True


def test_per_provider_thresholds(make_engine) -> None:
    answer = choice_answer(
        "refund", {"refund": 0.9, "status": 0.05, "other": 0.05}
    )  # confidence 0.85
    strict = FakeProvider("strict", {"intent": answer})
    lenient = FakeProvider("lenient", {"intent": answer})
    engine = make_engine([strict, lenient], threshold=0.8, thresholds={"intent@strict": 0.95})
    decision = engine.decide("x", INTENT)
    assert decision.provider == "lenient"
    assert [a.reason for a in decision.attempts] == ["below_threshold", "met_threshold"]
    assert decision.threshold == 0.8


def test_warmup_runs_real_questions_but_never_the_llm(make_engine) -> None:
    model = FakeProvider(
        "model", {"intent": choice_answer("refund", {"refund": 1.0, "status": 0.0, "other": 0.0})}
    )
    llm = FakeProvider("llm", {"intent": Answer(value="other")}, plane=Plane.LLM, calibrated=False)
    engine = make_engine([model, llm])
    engine.warmup([INTENT], rounds=2)
    assert model.calls == [["intent"], ["intent"]]
    assert llm.calls == []
    engine.warmup()
    assert len(model.calls) == 2
