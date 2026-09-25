from __future__ import annotations

from types import SimpleNamespace

import pytest

from thinkless import Usage
from thinkless.llm import AnthropicLLM, ScriptedLLM
from thinkless.pricing import PriceTable


def test_bundled_table_has_sources() -> None:
    table = PriceTable.load()
    assert table.as_of
    assert all(price.source for price in table.prices)


def test_cost_lookup() -> None:
    table = PriceTable.load()
    cost, known = table.cost(
        "anthropic", "claude-haiku-4-5", Usage(input_tokens=1_000_000, output_tokens=100_000)
    )
    assert known
    assert cost == pytest.approx(1.0 + 0.5)
    jev, known = table.cost("typesafe", "jev-1.13.0", Usage(input_tokens=1_000_000))
    assert known
    assert jev == pytest.approx(0.042)
    free, known = table.cost("ollama", "qwen3:8b", Usage(input_tokens=10**9))
    assert (free, known) == (0.0, True)
    unknown, known = table.cost("acme", "mystery", Usage(input_tokens=5))
    assert (unknown, known) == (0.0, False)


def test_override_table(tmp_path, monkeypatch) -> None:
    path = tmp_path / "prices.toml"
    path.write_text(
        '[[price]]\nprovider = "openai"\nmodel = "gpt-x"\ninput = 2.0\noutput = 8.0\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("THINKLESS_PRICING", str(path))
    table = PriceTable.load()
    assert table.cost("openai", "gpt-x", Usage(input_tokens=10**6))[0] == pytest.approx(2.0)


def test_scripted_llm_cycles_and_counts_tokens() -> None:
    llm = ScriptedLLM(["one", "two"])
    assert llm.complete([{"role": "user", "content": "hi"}]).text == "one"
    assert llm.complete([{"role": "user", "content": "hi"}]).text == "two"
    last = llm.complete([{"role": "user", "content": "hi"}])
    assert last.text == "two"
    assert last.usage.input_tokens >= 1


class _FakeAnthropic:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.calls: list[tuple[str, dict]] = []
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Hello"),
            ],
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
            model="claude-opus-5",
            _request_id="req_1",
        )

        def create(path: str):
            def inner(**kwargs):
                self.calls.append((path, kwargs))
                return response

            return inner

        self.messages = SimpleNamespace(create=create("messages"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create("beta")))


def test_anthropic_uses_server_side_fallbacks_and_no_sampling() -> None:
    client = _FakeAnthropic()
    llm = AnthropicLLM(client=client, effort="low")
    completion = llm.complete(
        [{"role": "user", "content": "hi"}], system="Be brief", temperature=0.7
    )
    path, kwargs = client.calls[0]
    assert path == "beta"
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["output_config"] == {"effort": "low"}
    assert "temperature" not in kwargs
    assert completion.text == "Hello"
    assert completion.usage.input_tokens == 12


def test_anthropic_without_fallbacks_and_refusal() -> None:
    client = _FakeAnthropic(stop_reason="refusal")
    llm = AnthropicLLM(client=client, fallbacks=False)
    completion = llm.complete([{"role": "user", "content": "hi"}])
    assert client.calls[0][0] == "messages"
    assert completion.text == ""
    assert completion.stop_reason == "refusal"


def test_anthropic_fallbacks_follow_the_model_family() -> None:
    for model, expected in [
        ("claude-opus-5", "beta"),
        ("claude-opus-5-5", "beta"),
        ("claude-fable-5-1", "beta"),
        ("claude-haiku-4-5", "messages"),
        ("claude-sonnet-5", "messages"),
    ]:
        client = _FakeAnthropic()
        AnthropicLLM(model, client=client).complete([{"role": "user", "content": "hi"}])
        assert client.calls[0][0] == expected, model
    forced = _FakeAnthropic()
    AnthropicLLM("claude-haiku-4-5", client=forced, fallbacks=True).complete(
        [{"role": "user", "content": "hi"}]
    )
    assert forced.calls[0][0] == "beta"
