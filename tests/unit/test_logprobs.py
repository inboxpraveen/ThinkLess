"""LLM confidence from token log probabilities, on a recorded Qwen 3.7 Flash reply."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from thinkless import Choice, Engine, Extract, Score, Status, YesNo
from thinkless.llm.base import LLM, Completion, Message
from thinkless.providers import LLMDecider
from thinkless.providers.logprobs import option_probabilities, span_confidence, value_span
from thinkless.tracing import Tracer

RECORDED = json.loads(
    (Path(__file__).parent / "data" / "logprobs_qwen3.7_flash.json").read_text(encoding="utf-8")
)
QUESTIONS = {
    "intent": Choice(
        "What does the customer want?",
        options={
            "refund": "money back",
            "status": "where is the order",
            "cancel": "stop the order before it ships",
            "other": None,
        },
    ),
    "urgent": YesNo("Is the customer upset or in a hurry?"),
    "tone": Score("How polite is the message?", levels=["rude", "neutral", "polite"]),
    "order": Extract(
        "Find the order details.", fields={"order_id": "the order number", "item": "the product"}
    ),
}


class Replay(LLM):
    """Returns a recorded completion, with or without its token log probabilities."""

    provider = "openrouter"
    model = "qwen/qwen3.7-flash"

    def __init__(self, completion: dict[str, Any], *, keep_logprobs: bool = True) -> None:
        self.completion = dict(completion)
        if not keep_logprobs:
            self.completion["logprobs"] = None
        self.asked: list[int | None] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
        logprobs: int | None = None,
    ) -> Completion:
        self.asked.append(logprobs)
        return Completion.model_validate(self.completion)


def _decide(threshold: float, **kwargs: Any) -> dict:
    llm = Replay(RECORDED["completion"], **kwargs)
    engine = Engine([LLMDecider(llm, logprobs=5)], threshold=threshold, tracer=Tracer())
    return engine.decide_many(RECORDED["state"], QUESTIONS), llm


def test_recorded_reply_gives_a_distribution_per_question() -> None:
    decisions, llm = _decide(0.8)
    assert llm.asked == [5]
    intent = decisions["intent"]
    assert intent.value == "cancel"
    assert sum(intent.probabilities.values()) == pytest.approx(1.0)
    assert max(intent.probabilities, key=intent.probabilities.get) == "cancel"
    # The message asks to cancel or for a status: the model's doubt shows.
    assert 0.5 < intent.probabilities["cancel"] < 0.9
    assert intent.probabilities["status"] > 0.1
    assert intent.status is Status.UNCERTAIN

    urgent = decisions["urgent"]
    assert urgent.value is True
    assert urgent.probabilities["yes"] == pytest.approx(1 - urgent.probabilities["no"])
    assert urgent.probability == pytest.approx(urgent.probabilities["yes"])

    tone = decisions["tone"]
    assert tone.level == "neutral"
    assert set(tone.probabilities) == {"rude", "neutral", "polite"}

    order = decisions["order"]
    assert order.value == {"order_id": "4471", "item": "blue kettle"}
    assert all(0.9 < c <= 1.0 for c in order.fields.values())
    assert order.status is Status.ACCEPTED


def test_a_lower_threshold_accepts_the_llm_answers() -> None:
    decisions, _ = _decide(0.0)
    assert all(d.status is Status.ACCEPTED for d in decisions.values())
    assert all(d.confidence is not None for d in decisions.values())


def test_without_logprobs_from_the_backend_answers_have_no_confidence(caplog) -> None:
    decisions, _ = _decide(0.0, keep_logprobs=False)
    assert decisions["intent"].value == "cancel"
    assert decisions["intent"].confidence is None
    assert decisions["intent"].status is Status.UNCERTAIN  # calibrated, but no number
    assert "no token log probabilities" in caplog.text


def test_plain_decider_never_asks_for_logprobs() -> None:
    llm = Replay(RECORDED["completion"])
    decider = LLMDecider(llm)
    assert not decider.calibrated
    decider.answer(RECORDED["state"], QUESTIONS)
    assert llm.asked == [None]
    with pytest.raises(ValueError):
        LLMDecider(llm, logprobs=50)


def _tokens(*parts: tuple[str, float, list[tuple[str, float]]]) -> list[dict]:
    return [
        {"token": t, "logprob": math.log(p), "top": [[a, math.log(q)] for a, q in alts]}
        for t, p, alts in parts
    ]


def test_options_that_share_a_first_token_split_where_they_diverge() -> None:
    text = '{"intent": "card arrival"}'
    tokens = _tokens(
        ('{"', 1.0, []),
        ("intent", 1.0, []),
        ('":', 1.0, []),
        (' "', 1.0, [(" null", 1e-12)]),
        ("card", 0.9, [("lost", 0.1)]),
        (" arrival", 0.6, [(" delivery", 0.4)]),
        ('"}', 1.0, []),
    )
    literals = {
        "card arrival": ['"card arrival"'],
        "card delivery estimate": ['"card delivery estimate"'],
        "lost or stolen card": ['"lost or stolen card"'],
        "refund": ['"refund"'],
    }
    probs = option_probabilities(text, tokens, "intent", literals)
    assert probs == pytest.approx(
        {
            "card arrival": 0.54,
            "card delivery estimate": 0.36,
            "lost or stolen card": 0.10,
            "refund": 0.0,
        }
    )
    assert value_span(text, "intent") == (11, 25)
    assert span_confidence(text, tokens, "intent") == pytest.approx(0.54)


def test_unaligned_tokens_give_no_probabilities() -> None:
    tokens = _tokens(("something else", 1.0, []))
    assert (
        option_probabilities('{"a": true}', tokens, "a", {"yes": ["true"], "no": ["false"]}) is None
    )
    assert option_probabilities('{"b": true}', _tokens(('{"b": true}', 1.0, [])), "a", {}) is None


def test_client_retries_at_the_hosts_top_logprobs_limit() -> None:
    from types import SimpleNamespace

    from thinkless.llm.openai_compat import OpenAICompatibleLLM

    sent: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            sent.append(kwargs)
            if kwargs.get("top_logprobs", 0) > 5:
                raise RuntimeError("Range of top_logprobs should be [0, 5]")
            item = SimpleNamespace(
                token="hi", logprob=-0.1, top_logprobs=[SimpleNamespace(token="hi", logprob=-0.1)]
            )
            choice = SimpleNamespace(
                message=SimpleNamespace(content="hi"),
                finish_reason="stop",
                logprobs=SimpleNamespace(content=[item]),
            )
            return SimpleNamespace(
                choices=[choice],
                model="m",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm = OpenAICompatibleLLM("m", client=client)
    first = llm.complete([{"role": "user", "content": "x"}], logprobs=10)
    assert [s["top_logprobs"] for s in sent] == [10, 5]
    assert first.logprobs == [{"token": "hi", "logprob": -0.1, "top": [["hi", -0.1]]}]
    llm.complete([{"role": "user", "content": "x"}], logprobs=10)
    assert sent[-1]["top_logprobs"] == 5  # the limit is remembered
    plain = llm.complete([{"role": "user", "content": "x"}])
    assert "logprobs" not in sent[-1]
    assert plain.logprobs is None
