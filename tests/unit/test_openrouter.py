from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from thinkless import Choice, Engine, MemorySink, Tracer, Usage, load_env
from thinkless.llm import LLM, Completion, Message, OpenRouterLLM, from_spec
from thinkless.llm.openai_compat import OpenAICompatibleLLM, reported_cost
from thinkless.providers import LLMDecider


def _response(text: str = '{"intent": "refund"}', cost: float | None = 0.00042):
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=8)
    if cost is not None:
        usage.cost = cost
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=usage,
        model="qwen/qwen3.7-flash",
    )


class _FakeChat:
    def __init__(self, error: Exception | None = None, cost: float | None = 0.00042) -> None:
        self.calls: list[dict] = []
        self.error = error
        self.cost = cost
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None and len(self.calls) == 1:
            raise self.error
        return _response(cost=self.cost)


class _BadRequestError(Exception):
    status_code = 400


def test_spec_builds_openrouter_client(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = from_spec("openrouter:qwen/qwen3.7-flash")
    assert isinstance(llm, OpenRouterLLM)
    assert llm.provider == "openrouter"
    assert llm.model == "qwen/qwen3.7-flash"
    assert str(llm._client.base_url).startswith("https://openrouter.ai/api/v1")
    with pytest.raises(ValueError):
        from_spec("openrouter")


def test_missing_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM("qwen/qwen3.7-flash")


def test_request_carries_routing_and_reasoning_options() -> None:
    fake = _FakeChat()
    llm = OpenRouterLLM(
        "openai/gpt-5.6-luna",
        client=fake,
        reasoning={"effort": "low"},
        provider_preferences={"sort": "price"},
    )
    completion = llm.complete([{"role": "user", "content": "hi"}], max_tokens=50, json_mode=True)
    sent = fake.calls[0]
    assert sent["max_tokens"] == 50
    assert sent["extra_body"] == {"reasoning": {"effort": "low"}, "provider": {"sort": "price"}}
    assert sent["response_format"] == {"type": "json_object"}
    assert completion.cost_usd == pytest.approx(0.00042)


def test_reported_cost_reads_both_sdk_shapes() -> None:
    assert reported_cost(SimpleNamespace(cost=0.5)) == 0.5
    assert reported_cost(SimpleNamespace(model_extra={"cost": "0.25"})) == 0.25
    assert reported_cost(SimpleNamespace(prompt_tokens=3)) is None
    assert reported_cost(None) is None


def test_json_mode_rejection_falls_back_once() -> None:
    fake = _FakeChat(
        error=_BadRequestError("response_format json_object is not supported for this model")
    )
    llm = OpenAICompatibleLLM("some/model", base_url="http://x", client=fake)
    first = llm.complete([{"role": "user", "content": "hi"}], json_mode=True)
    assert first.text
    assert "response_format" in fake.calls[0]
    assert "response_format" not in fake.calls[1]
    llm.complete([{"role": "user", "content": "again"}], json_mode=True)
    assert "response_format" not in fake.calls[2]


def test_other_errors_propagate() -> None:
    fake = _FakeChat(error=_BadRequestError("context length exceeded"))
    llm = OpenAICompatibleLLM("some/model", base_url="http://x", client=fake)
    with pytest.raises(_BadRequestError):
        llm.complete([{"role": "user", "content": "hi"}], json_mode=True)


class _BilledLLM(LLM):
    provider = "openrouter"
    model = "billed"

    def complete(self, messages: Sequence[Message], **kwargs) -> Completion:
        return Completion(
            text='{"intent": "refund"}',
            model="billed",
            usage=Usage(input_tokens=100, output_tokens=5),
            cost_usd=0.0021,
        )


def test_engine_prefers_reported_cost() -> None:
    memory = MemorySink()
    llm = _BilledLLM()
    engine = Engine([LLMDecider(llm)], llm=llm, tracer=Tracer([memory]))
    with engine.run("t") as run:
        engine.decide("x", Choice("Intent?", options=["refund", "other"], name="intent"))
        engine.generate("reply")
    for span in memory.spans:
        if span.kind in ("attempt", "llm"):
            assert span.attributes["cost_source"] == "reported"
            assert span.attributes["cost_usd"] == pytest.approx(0.0021)
    assert run.summary().cost_usd == pytest.approx(0.0042)


def test_price_table_and_unknown_cost_sources(make_engine, memory) -> None:
    from thinkless.llm import ScriptedLLM

    engine = make_engine([], llm=ScriptedLLM(["hello"]))
    engine.generate("hi")
    span = next(s for s in memory.spans if s.kind == "llm")
    assert span.attributes["cost_source"] == "price_table"


def test_load_env(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "export THINKLESS_TEST_A='quoted value'\n"
        'THINKLESS_TEST_B="double"\n'
        "THINKLESS_TEST_EMPTY=\n"
        "THINKLESS_TEST_KEEP=from-file\n"
        "not a variable line\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THINKLESS_TEST_KEEP", "from-environment")
    for name in ("THINKLESS_TEST_A", "THINKLESS_TEST_B", "THINKLESS_TEST_EMPTY"):
        monkeypatch.delenv(name, raising=False)
    loaded = load_env(env)
    import os

    assert sorted(loaded) == ["THINKLESS_TEST_A", "THINKLESS_TEST_B"]
    assert os.environ["THINKLESS_TEST_A"] == "quoted value"
    assert os.environ["THINKLESS_TEST_B"] == "double"
    assert os.environ["THINKLESS_TEST_KEEP"] == "from-environment"
    assert "THINKLESS_TEST_EMPTY" not in os.environ
    assert load_env(tmp_path / "missing.env") == []
    monkeypatch.delenv("THINKLESS_TEST_A")
    monkeypatch.delenv("THINKLESS_TEST_B")


def test_reasoning_maps_to_each_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    off = from_spec("openrouter:qwen/qwen3.7-flash", reasoning="off")
    low = from_spec("openrouter:openai/gpt-5.6-luna", reasoning="low")
    default = from_spec("openrouter:google/gemini-2.5-flash-lite", reasoning="default")
    assert off._extra_body == {"reasoning": {"enabled": False}}
    assert low._extra_body == {"reasoning": {"effort": "low"}}
    assert default._extra_body == {}
    with pytest.raises(ValueError):
        from_spec("openrouter:x/y", reasoning="extreme")


class _TruncatingLLM(LLM):
    provider = "openrouter"
    model = "thinker"

    def __init__(self) -> None:
        self.budgets: list[int] = []

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int = 512, **kwargs
    ) -> Completion:
        self.budgets.append(max_tokens)
        if len(self.budgets) == 1:
            return Completion(
                text="",
                model="thinker",
                stop_reason="length",
                usage=Usage(input_tokens=500, output_tokens=max_tokens),
                cost_usd=0.001,
            )
        return Completion(
            text='{"intent": "refund"}',
            model="thinker",
            stop_reason="stop",
            usage=Usage(input_tokens=500, output_tokens=40),
            cost_usd=0.0005,
        )


def test_truncated_reply_is_retried_with_a_larger_budget() -> None:
    llm = _TruncatingLLM()
    result = LLMDecider(llm).answer("x", {"intent": Choice("Intent?", options=["refund", "other"])})
    assert result.answers["intent"].value == "refund"
    assert llm.budgets[1] == llm.budgets[0] * 3
    assert result.meta["truncated"] is True
    assert result.meta["retried"] is True
    assert result.cost_usd == pytest.approx(0.0015)
    assert result.usage.input_tokens == 1000


def test_truncation_retry_can_be_disabled() -> None:
    llm = _TruncatingLLM()
    decider = LLMDecider(llm, retry_on_truncation=False)
    result = decider.answer("x", {"intent": Choice("Intent?", options=["refund", "other"])})
    assert result.answers["intent"] is None
    assert result.meta == {
        "invalid": ["intent"],
        "stop_reason": "length",
        "truncated": True,
        "retried": False,
    }
    assert len(llm.budgets) == 1
