"""Batched decisions: Engine.decide_batch, provider answer_batch and the Batched wrapper."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence

import pytest

from tests.conftest import FakeProvider
from thinkless import Answer, Choice, Engine, Status, YesNo
from thinkless.providers import Batched, BatchItem, ProviderResult
from thinkless.providers.base import group_by_questions
from thinkless.tracing import Tracer
from thinkless.tracing.summary import summarize

INTENT = Choice("What?", options=["refund", "status", "other"], name="intent")
URGENT = YesNo("Urgent?", name="urgent")


class BatchingFake(FakeProvider):
    """A FakeProvider with a real answer_batch that records each batch's size."""

    def __init__(self, *args, fail_batch: bool = False, delay_s: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_sizes: list[int] = []
        self.fail_batch = fail_batch
        self.delay_s = delay_s
        self.threads: set[str] = set()

    def answer_batch(self, items: Sequence[BatchItem]) -> list[ProviderResult]:
        self.batch_sizes.append(len(items))
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay_s)
        if self.fail_batch:
            raise RuntimeError("batch failed")
        return [FakeProvider.answer(self, state, questions) for state, questions in items]

    def answer(self, state, questions):
        # A lone call is as slow as a batch, like a forward pass on a GPU.
        time.sleep(self.delay_s)
        return super().answer(state, questions)


def _intent(state, question):
    text = str(state)
    if "refund" in text:
        return Answer(value="refund", probabilities={"refund": 0.95, "status": 0.03, "other": 0.02})
    return Answer(value="status", probabilities={"refund": 0.3, "status": 0.4, "other": 0.3})


def _urgent(state, question):
    return Answer(value="now" in str(state), probabilities={"yes": 0.9, "no": 0.1})


STATES = ["refund please", "where is it", "refund now", "hello", "where now"]


def _engine(local, fallback, **kwargs) -> Engine:
    return Engine([local, fallback], threshold=0.8, tracer=Tracer(), **kwargs)


def _fallback() -> FakeProvider:
    return FakeProvider(
        "llm",
        {"intent": Answer(value="other"), "urgent": Answer(value=False)},
        calibrated=False,
    )


def _comparable(decisions):
    return {
        k: (d.value, d.status, d.provider, d.confidence, [a.reason for a in d.attempts])
        for k, d in decisions.items()
    }


def test_decide_batch_matches_one_by_one() -> None:
    sequential = _engine(FakeProvider("local", {"intent": _intent, "urgent": _urgent}), _fallback())
    expected = [_comparable(sequential.decide_many(s, [INTENT, URGENT])) for s in STATES]

    local = BatchingFake("local", {"intent": _intent, "urgent": _urgent})
    fallback = _fallback()
    batched = _engine(local, fallback)
    got = batched.decide_batch(STATES, [INTENT, URGENT])
    assert [_comparable(d) for d in got] == expected
    # The local model saw every state in one call; the fallback got only the
    # states whose intent stayed open, one call each.
    assert local.batch_sizes == [5]
    assert len(fallback.calls) == 3
    assert got[0]["intent"].status is Status.ACCEPTED
    assert got[1]["intent"].provider == "llm"


def test_decide_batch_asks_a_hosted_provider_concurrently() -> None:
    from thinkless.tracing.span import current_span

    seen: list[str | None] = []
    running = 0
    most = 0
    lock = threading.Lock()

    def slow(state, question):
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        span = current_span()
        seen.append(span.kind if span is not None else None)
        time.sleep(0.15)
        with lock:
            running -= 1
        return Answer(value="other")

    states = [f"ticket {i}" for i in range(8)]
    serial = _engine(FakeProvider("local", {}), FakeProvider("llm", {"intent": slow}))
    expected = [_comparable(d) for d in serial.decide_batch(states, [INTENT])]
    assert most == 1
    most = 0
    seen.clear()
    engine = _engine(FakeProvider("local", {}), FakeProvider("llm", {"intent": slow}))
    started = time.perf_counter()
    got = engine.decide_batch(states, [INTENT], concurrency=4)
    assert time.perf_counter() - started < 0.15 * 8 / 2
    assert most == 4
    assert [_comparable(d) for d in got] == expected
    # Every call ran inside the attempt span, as it does one at a time.
    assert seen == ["attempt"] * 8
    with pytest.raises(ValueError, match="concurrency"):
        engine.decide_batch(states, [INTENT], concurrency=0)


def test_concurrent_decide_batch_raises_the_first_error() -> None:
    def broken(state, question):
        raise RuntimeError("upstream down")

    engine = _engine(
        FakeProvider("local", {}), FakeProvider("llm", {"intent": broken}), on_error="raise"
    )
    with pytest.raises(RuntimeError, match="upstream down"):
        engine.decide_batch(["a", "b", "c"], [INTENT], concurrency=3)


def test_decide_batch_traces_every_decision_once() -> None:
    tracer = Tracer()
    engine = Engine(
        [BatchingFake("local", {"intent": _intent, "urgent": _urgent}), _fallback()],
        tracer=tracer,
    )
    with engine.run("bulk") as run:
        engine.decide_batch(STATES, [INTENT, URGENT])
    summary = summarize(run.spans)
    assert summary.decisions == 10
    attempts = [s for s in run.spans if s["kind"] == "attempt"]
    assert attempts[0]["attributes"]["items"] == 5
    assert engine.decide_batch([], [INTENT]) == []


def test_a_failed_batch_moves_every_item_down_the_cascade() -> None:
    local = BatchingFake("local", {"intent": _intent}, fail_batch=True)
    engine = _engine(local, _fallback())
    got = engine.decide_batch(STATES[:3], [INTENT])
    assert all(d["intent"].provider == "llm" for d in got)
    assert all(d["intent"].attempts[0].reason == "error: RuntimeError" for d in got)

    raising = _engine(
        BatchingFake("local", {"intent": _intent}, fail_batch=True), _fallback(), on_error="raise"
    )
    with pytest.raises(RuntimeError, match="batch failed"):
        raising.decide_batch(STATES[:3], [INTENT])


def test_providers_without_answer_batch_are_asked_per_state() -> None:
    local = FakeProvider("local", {"intent": _intent})
    assert not local.batches
    engine = _engine(local, _fallback())
    engine.decide_batch(STATES, [INTENT])
    assert len(local.calls) == 5


def test_group_by_questions() -> None:
    other = Choice("What?", options=["a", "b"], name="intent")
    items = [("x", {"intent": INTENT}), ("y", {"intent": other}), ("z", {"intent": INTENT})]
    assert group_by_questions(items) == [[0, 2], [1]]


def test_batched_wrapper_coalesces_concurrent_calls() -> None:
    inner = BatchingFake("local", {"intent": _intent}, delay_s=0.05)
    wrapped = Batched(inner, max_batch=8)
    assert wrapped.name == "local"
    assert wrapped.batches
    engine = Engine([wrapped], threshold=0.0, tracer=Tracer())
    results: dict[int, str] = {}

    def ask(i: int) -> None:
        results[i] = engine.decide(f"refund {i}", INTENT).value

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == dict.fromkeys(range(12), "refund")
    # The first call went alone; the rest piled up while it ran and shared batches.
    assert sum(inner.batch_sizes) <= 12
    assert len(inner.batch_sizes) < 12
    assert max(inner.batch_sizes) <= 8
    assert inner.threads == {"thinkless-batch-local"}
    wrapped.close()
    with pytest.raises(RuntimeError, match="closed"):
        wrapped.answer("x", {"intent": INTENT})


def test_batched_wrapper_reports_errors_to_every_caller() -> None:
    inner = BatchingFake("local", {"intent": _intent}, fail_batch=True, delay_s=0.05)
    wrapped = Batched(inner)
    errors: list[BaseException] = []

    def ask() -> None:
        try:
            wrapped.answer("refund", {"intent": INTENT})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=ask) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    # A lone first call goes through answer(), which works; batched ones fail together.
    assert all(isinstance(e, RuntimeError) for e in errors)
    wrapped.stop()
    assert not wrapped._worker.is_alive()


def test_batched_wrapper_waits_for_company_when_asked() -> None:
    inner = BatchingFake("local", {"intent": _intent})
    wrapped = Batched(inner, max_wait_ms=200)
    barrier = threading.Barrier(4)

    def ask() -> None:
        barrier.wait()
        wrapped.answer("refund", {"intent": INTENT})

    threads = [threading.Thread(target=ask) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert inner.batch_sizes == [4]
    wrapped.close()


def test_stop_fails_calls_still_queued() -> None:
    gate = threading.Event()

    class Slow(BatchingFake):
        def answer(self, state, questions: Mapping[str, object]) -> ProviderResult:
            gate.wait(5)
            return super().answer(state, questions)

    wrapped = Batched(Slow("local", {"intent": _intent}), timeout_s=5)
    outcomes: list[str] = []

    def ask() -> None:
        try:
            wrapped.answer("refund", {"intent": INTENT})
            outcomes.append("ok")
        except RuntimeError:
            outcomes.append("closed")

    first = threading.Thread(target=ask)
    first.start()
    time.sleep(0.1)  # the worker is now inside the first call
    stopper = threading.Thread(target=wrapped.stop)
    stopper.start()
    time.sleep(0.1)
    gate.set()
    first.join(5)
    stopper.join(5)
    assert outcomes == ["ok"]
    assert not wrapped._worker.is_alive()


def test_server_batches_local_providers_and_restores_them() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from thinkless.server.app import create_app

    local = BatchingFake("local", {"intent": _intent})
    plain = FakeProvider("rules", {})
    engine = Engine([plain, local], threshold=0.0, tracer=Tracer())
    app = create_app(engine, [INTENT])
    with TestClient(app) as client:
        assert isinstance(engine.providers[1], Batched)
        assert engine.providers[0] is plain  # does not batch: left alone
        health = client.get("/healthz").json()
        assert [p["batched"] for p in health["providers"]] == [False, True]
        body = client.post("/v1/decide", json={"state": "refund", "questions": ["intent"]}).json()
        assert body["decisions"]["intent"]["value"] == "refund"
    assert engine.providers == [plain, local]

    unbatched = create_app(engine, [INTENT], batching=False)
    with TestClient(unbatched):
        assert engine.providers[1] is local


@pytest.mark.parametrize(
    ("factory", "attribute"),
    [
        ("thinkless.providers.gliner:GLiNER", "_model"),
        ("thinkless.providers.laya:Laya", "_agent"),
        ("thinkless.providers.hf:HFClassifier", "_pipeline"),
        ("thinkless.providers.hf:HFZeroShot", "_pipeline"),
        ("thinkless.llm.local:TransformersLLM", "_model"),
    ],
)
def test_close_waits_for_a_call_in_progress(factory: str, attribute: str) -> None:
    import importlib
    import threading
    import time

    module, name = factory.split(":")
    cls = getattr(importlib.import_module(module), name)
    kwargs = {"answers": {"q": {"yes": "Y"}}} if name == "HFClassifier" else {}
    provider = cls("org/model", **kwargs) if name != "GLiNER" else cls()
    provider.close()  # nothing loaded yet: a no-op
    setattr(provider, attribute, object())
    closed = threading.Event()
    with provider._lock:  # a call holding the model
        thread = threading.Thread(target=lambda: (provider.close(), closed.set()))
        thread.start()
        time.sleep(0.05)
        assert not closed.is_set()
        assert getattr(provider, attribute) is not None
    thread.join(5)
    assert closed.is_set()
    assert getattr(provider, attribute) is None
