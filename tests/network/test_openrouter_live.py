"""Live calls to OpenRouter. Opt in with ``pytest -m network``.

Needs ``OPENROUTER_API_KEY`` (a ``.env`` in the repository root works). Each
run costs a small fraction of a cent on Qwen 3.7 Flash.
"""

from __future__ import annotations

import os

import pytest

from thinkless import Engine, MemorySink, Tracer, load_env
from thinkless.demo.support import load_scenarios
from thinkless.demo.support.questions import TRIAGE

pytestmark = pytest.mark.network

load_env()
MODEL = os.environ.get("THINKLESS_TEST_OPENROUTER_MODEL", "qwen/qwen3.7-flash")


@pytest.fixture(scope="module")
def llm():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")
    from thinkless.llm import OpenRouterLLM

    return OpenRouterLLM(MODEL, reasoning={"enabled": False})


def test_triage_batch_is_valid_and_billed(llm) -> None:
    from thinkless.providers import LLMDecider

    ticket = next(s for s in load_scenarios() if s["id"] == "T-001")
    memory = MemorySink()
    engine = Engine([LLMDecider(llm)], llm=llm, tracer=Tracer([memory]))
    with engine.run("live") as run:
        decisions = engine.decide_many(
            {"subject": ticket["subject"], "message": ticket["message"]}, TRIAGE
        )
        reply = engine.generate("Say hello in five words.", max_tokens=40)
    assert decisions["intent"].value == "refund_duplicate_charge"
    assert decisions["order"].value == {"order_id": "4471"}
    assert reply.text
    attempt = next(s for s in memory.spans if s.kind == "attempt")
    assert attempt.attributes["invalid"] == []
    assert attempt.attributes["cost_source"] == "reported"
    assert run.summary().cost_usd > 0
