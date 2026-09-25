"""The engine: routes questions through a cascade of providers and traces everything."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from .confidence import from_distribution, from_yes_probability
from .decision import Answer, Attempt, Decision, Plane, Status
from .errors import ConfigurationError
from .llm.base import LLM, Completion, Message, as_messages
from .logs import get_logger
from .pricing import PriceTable, default_prices
from .providers.base import DecisionProvider, ProviderResult, State
from .questions import Extract, Kind, Question, Score
from .tracing.span import Span
from .tracing.summary import TraceSummary, summarize
from .tracing.tracer import Tracer

__all__ = ["Engine", "Run"]

logger = get_logger("engine")

WARMUP_STATE = {
    "subject": "Question about my order",
    "message": "Hello, I placed order 4471 last Tuesday and the tracking page has not changed "
    "since. Could you tell me when it will arrive, or refund it if it is lost? Thanks.",
}


class _RunCollector:
    """Sink that keeps the spans of one trace while a run is open."""

    def __init__(self) -> None:
        self.trace_id: str | None = None
        self.spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def on_end(self, span: Span) -> None:
        if span.trace_id == self.trace_id:
            with self._lock:
                self.spans.append(span.to_dict())


class Run:
    """Handle for a traced run, returned by :meth:`Engine.run`."""

    def __init__(self, span: Span, collector: _RunCollector) -> None:
        self.span = span
        self._collector = collector

    @property
    def trace_id(self) -> str:
        return self.span.trace_id

    @property
    def spans(self) -> list[dict[str, Any]]:
        return list(self._collector.spans)

    def set(self, **attributes: Any) -> None:
        """Attach attributes to the run's root span (outcome, labels, ids)."""
        self.span.set(**attributes)

    def summary(self) -> TraceSummary:
        """Roll-up of the run. Complete once the ``with`` block has exited."""
        return summarize(self.spans)


class Engine:
    """Answers typed questions with the cheapest provider that is confident enough.

    Providers are tried in the order given. Each provider receives, in one
    call, every still-open question it supports. An answer whose normalized
    confidence meets the question's threshold is accepted; the rest move on to
    the next provider. When every provider has been tried, questions that are
    still open come back ``uncertain`` with the best answer seen, or
    ``abstained`` if nobody answered.

    Args:
        providers: The decision cascade, cheapest first. A typical order is
            rules, then small local models, then a hosted decision model,
            then an LLM.
        llm: The reasoning plane, used by :meth:`generate`.
        threshold: Default minimum confidence to accept an answer.
        thresholds: Overrides keyed by question name, or by
            ``"<question>@<provider>"`` for a threshold that applies to one
            provider only (providers calibrate differently, so the same
            question often needs a different bar per model). Resolution
            order: ``question@provider``, then ``Question(threshold=...)``,
            then ``thresholds[question]``, then ``threshold``.
        trust_uncalibrated: Accept answers from providers without calibrated
            probabilities (a prompted LLM) as final. Set it to ``False`` to
            have such answers come back ``uncertain`` instead.
        tracer: Where spans go. Defaults to a tracer with no sinks.
        prices: Price table for cost estimates. Defaults to the bundled table,
            or ``$THINKLESS_PRICING`` when set.
        on_error: ``continue`` logs a failing provider and moves down the
            cascade; ``raise`` propagates the exception.
        escalation_context: Tell providers that accept it (the LLM decider)
            which questions of the same batch are already settled, and how.
            An escalated question otherwise reaches the LLM stripped of its
            siblings, and on the support benchmark that changed answers: asked
            alone whether a request for a human is an injection, one model
            said yes to 5 of 10 such messages. The cost is one short line per
            settled question.

    Example:
        >>> engine = Engine([Rules(), GLiNER(), Laya(), LLMDecider(llm)], llm=llm)
        >>> intent = engine.decide(ticket, Choice("What does the customer want?", options=[...]))
        >>> if intent.is_("refund"):
        ...     ...
    """

    def __init__(
        self,
        providers: Sequence[DecisionProvider] = (),
        llm: LLM | None = None,
        *,
        threshold: float = 0.8,
        thresholds: Mapping[str, float] | None = None,
        trust_uncalibrated: bool = True,
        tracer: Tracer | None = None,
        prices: PriceTable | None = None,
        on_error: Literal["continue", "raise"] = "continue",
        escalation_context: bool = True,
    ) -> None:
        names = [p.name for p in providers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ConfigurationError(f"provider names must be unique, got duplicates: {duplicates}")
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("threshold must be between 0 and 1")
        if on_error not in ("continue", "raise"):
            raise ConfigurationError("on_error must be 'continue' or 'raise'")
        self.providers: list[DecisionProvider] = list(providers)
        self.llm = llm
        self.threshold = threshold
        self.thresholds = dict(thresholds or {})
        self.trust_uncalibrated = trust_uncalibrated
        self.tracer = tracer or Tracer()
        self.prices = prices or default_prices()
        self.on_error = on_error
        self.escalation_context = escalation_context

    # ------------------------------------------------------------------ runs

    @contextmanager
    def run(self, name: str, *, input: Any = None, **attributes: Any) -> Iterator[Run]:
        """Open a root span for one unit of work (a ticket, a request, a task).

        Everything the engine does inside the block, including decorated tool
        calls, becomes part of this trace.
        """
        collector = _RunCollector()
        self.tracer.add_sink(collector)
        try:
            with self.tracer.span("run", name, **attributes) as span:
                collector.trace_id = span.trace_id
                if input is not None:
                    span.set(input=self.tracer.content(input))
                yield Run(span, collector)
        finally:
            self.tracer.sinks.remove(collector)

    @contextmanager
    def step(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Group related work under a named phase of the trace."""
        with self.tracer.span("step", name, **attributes) as span:
            yield span

    def rule(self, name: str, value: Any, **attributes: Any) -> Any:
        """Record a deterministic check made in application code, and return its value.

        Example:
            >>> if not engine.rule("authenticated", ticket.customer_id is not None):
            ...     return ask_to_sign_in()
        """
        with self.tracer.span("rule", name, plane="rule", value=value, **attributes):
            pass
        return value

    # ------------------------------------------------------------- decisions

    def decide(self, state: State, question: Question, *, name: str | None = None) -> Decision:
        """Answer one question. See :meth:`decide_many` for batching."""
        key = name or question.key
        return self.decide_many(state, {key: question}, label=key)[key]

    def decide_many(
        self,
        state: State,
        questions: Mapping[str, Question] | Sequence[Question],
        *,
        label: str | None = None,
    ) -> dict[str, Decision]:
        """Answer several questions about the same state.

        Questions travel through the cascade together: each provider gets one
        call with all the open questions it supports, which is how System One
        models are meant to be used and what keeps LLM-only baselines fair.

        Args:
            state: Text, a JSON-like object, or a list of either.
            questions: A mapping of name to question, or a sequence of
                questions keyed by their ``key``.
            label: Span name. Defaults to the joined question names.

        Returns:
            Decisions keyed by question name, in input order.
        """
        batch = self._as_mapping(questions)
        with self.tracer.span("decide", label or ", ".join(batch)[:80]) as span:
            span.set(
                questions={
                    k: {"kind": q.kind.value, "threshold": self._threshold(k, q)}
                    for k, q in batch.items()
                }
            )
            if self.tracer.capture_content:
                span.set(state=self.tracer.content(state))
            decisions = self._cascade(state, batch)
            for decision in decisions.values():
                decision.span_id = span.span_id
            span.set(decisions=[self._trace_summary(d) for d in decisions.values()])
        return decisions

    def extract(
        self,
        state: State,
        fields: Mapping[str, str] | Extract,
        *,
        required: Sequence[str] = (),
        name: str | None = None,
        threshold: float | None = None,
    ) -> Decision:
        """Extract fields from ``state``. Shorthand for deciding an :class:`Extract`."""
        question = (
            fields
            if isinstance(fields, Extract)
            else Extract(
                fields=dict(fields), required=tuple(required), name=name, threshold=threshold
            )
        )
        return self.decide(state, question)

    # ------------------------------------------------------------ generation

    def generate(
        self,
        prompt: str | Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        name: str = "generate",
        llm: LLM | None = None,
    ) -> Completion:
        """Generate text with the reasoning plane and record it as an ``llm`` span."""
        model = llm or self.llm
        if model is None:
            raise ConfigurationError(
                "generate() needs an LLM: pass llm= to Engine or to generate()"
            )
        messages = as_messages(prompt)
        with self.tracer.span(
            "llm", name, plane="llm", provider=model.provider, model=model.model
        ) as span:
            if self.tracer.capture_content:
                span.set(system=self.tracer.content(system), messages=self.tracer.content(messages))
            completion = model.complete(
                messages, system=system, max_tokens=max_tokens, temperature=temperature
            )
            cost, source = self._cost(
                model.provider, completion.model, completion.usage, completion.cost_usd
            )
            span.set(
                model=completion.model,
                usage=completion.usage.model_dump(),
                cost_usd=cost,
                cost_known=source != "unknown",
                cost_source=source,
                stop_reason=completion.stop_reason,
            )
            if self.tracer.capture_content:
                span.set(completion=completion.text)
        return completion

    # ----------------------------------------------------------------- async

    async def adecide(
        self, state: State, question: Question, *, name: str | None = None
    ) -> Decision:
        return await asyncio.to_thread(self.decide, state, question, name=name)

    async def adecide_many(
        self,
        state: State,
        questions: Mapping[str, Question] | Sequence[Question],
        *,
        label: str | None = None,
    ) -> dict[str, Decision]:
        return await asyncio.to_thread(self.decide_many, state, questions, label=label)

    async def agenerate(self, prompt: str | Sequence[Message], **kwargs: Any) -> Completion:
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    # ------------------------------------------------------------- lifecycle

    def warmup(
        self,
        questions: Mapping[str, Question] | Sequence[Question] | None = None,
        *,
        state: State = WARMUP_STATE,
        rounds: int = 2,
    ) -> None:
        """Load every local model now instead of on the first request.

        Args:
            questions: The questions the application will ask. When given,
                every non-LLM provider answers them ``rounds`` times on
                ``state``, so GPU kernels are specialized for the real
                shapes before traffic arrives. LLM providers are never
                called here.
            state: Sample input for the warmup rounds.
            rounds: Warmup passes per provider.
        """
        for provider in self.providers:
            provider.warmup()
        if self.llm is not None:
            self.llm.warmup()
        if questions is None:
            return
        batch = self._as_mapping(questions)
        for provider in self.providers:
            if provider.plane is Plane.LLM:
                continue
            asked = {k: q for k, q in batch.items() if provider.supports(q)}
            for _ in range(rounds if asked else 0):
                provider.answer(state, asked)

    def close(self) -> None:
        for provider in self.providers:
            provider.close()
        if self.llm is not None:
            self.llm.close()
        self.tracer.flush()

    # -------------------------------------------------------------- internals

    @staticmethod
    def _as_mapping(questions: Mapping[str, Question] | Sequence[Question]) -> dict[str, Question]:
        if isinstance(questions, Mapping):
            batch = dict(questions)
        else:
            batch = {}
            for question in questions:
                if question.key in batch:
                    raise ConfigurationError(f"duplicate question name {question.key!r}")
                batch[question.key] = question
        if not batch:
            raise ConfigurationError("at least one question is required")
        return batch

    def _cost(
        self, provider: str, model: str | None, usage: Any, reported: float | None
    ) -> tuple[float, str]:
        """Cost of one call and where the number came from.

        A cost reported by the backend wins over the price table: it is what
        was actually billed.
        """
        if reported is not None:
            return reported, "reported"
        cost, known = self.prices.cost(provider, model, usage)
        return cost, "price_table" if known else "unknown"

    def _threshold(self, key: str, question: Question, provider: str | None = None) -> float:
        # Most specific first: a calibrated threshold for this question on this
        # provider, then the question's own, then the engine-wide settings.
        if provider is not None and f"{key}@{provider}" in self.thresholds:
            return self.thresholds[f"{key}@{provider}"]
        if question.threshold is not None:
            return question.threshold
        return self.thresholds.get(key, self.threshold)

    @staticmethod
    def _confidence(question: Question, answer: Answer) -> float | None:
        if answer.probabilities:
            if question.kind is Kind.YES_NO:
                return from_yes_probability(answer.probabilities.get("yes", 0.0))
            return from_distribution(answer.probabilities)
        if (
            question.kind is Kind.EXTRACT
            and answer.fields is not None
            and isinstance(question, Extract)
        ):
            values = answer.value if isinstance(answer.value, Mapping) else {}
            if any(values.get(field) is None for field in question.required):
                return 0.0
            found = [answer.fields.get(f) for f, v in values.items() if v is not None]
            if any(c is None for c in found):
                return None
            return min((float(c) for c in found if c is not None), default=1.0)
        if answer.confidence is not None:
            return max(0.0, min(1.0, float(answer.confidence)))
        return None

    def _build(
        self,
        key: str,
        question: Question,
        answer: Answer,
        confidence: float | None,
        threshold: float,
        provider: DecisionProvider,
        result: ProviderResult,
        cost: float,
        status: Status,
    ) -> Decision:
        probability = None
        level = None
        probs = answer.probabilities
        if probs:
            if question.kind is Kind.YES_NO:
                p_yes = probs.get("yes", 0.0)
                probability = p_yes if answer.value else 1.0 - p_yes
            elif question.kind is Kind.CHOICE:
                probability = probs.get(str(answer.value))
            elif question.kind is Kind.SCORE:
                level = max(probs, key=probs.__getitem__)
                probability = probs[level]
        if isinstance(question, Score) and level is None and answer.value is not None:
            index = min(max(round(float(answer.value)), 0), len(question.levels) - 1)
            level = question.levels[index]
        return Decision(
            name=key,
            kind=question.kind,
            value=answer.value,
            status=status,
            confidence=confidence,
            probability=probability,
            probabilities=probs,
            level=level,
            fields=answer.fields,
            threshold=threshold,
            provider=provider.name,
            plane=provider.plane,
            model=result.model,
            usage=result.usage,
            cost_usd=cost,
            raw=answer.raw,
        )

    def _trace_value(self, kind: Kind, value: Any) -> Any:
        # Choice labels, levels and booleans come from the question itself;
        # extracted values are user content.
        return self.tracer.content(value) if kind is Kind.EXTRACT else value

    def _trace_summary(self, decision: Decision) -> dict[str, Any]:
        summary = decision.summary()
        summary["value"] = self._trace_value(decision.kind, decision.value)
        return summary

    def _cascade(self, state: State, batch: dict[str, Question]) -> dict[str, Decision]:
        pending = dict(batch)
        resolved: dict[str, Decision] = {}
        best: dict[str, Decision] = {}
        attempts: dict[str, list[Attempt]] = defaultdict(list)
        latency: dict[str, float] = defaultdict(float)

        for provider in self.providers:
            if not pending:
                break
            asked = {k: q for k, q in pending.items() if provider.supports(q)}
            if not asked:
                continue
            with self.tracer.span(
                "attempt",
                provider.name,
                plane=provider.plane.value,
                provider=provider.name,
                questions=list(asked),
            ) as span:
                started = time.perf_counter()
                try:
                    if self.escalation_context and provider.accepts_context and resolved:
                        settled = {k: (batch[k], d) for k, d in resolved.items()}
                        span.set(context_from=list(settled))
                        result = provider.answer(state, asked, context=settled)  # type: ignore[call-arg]
                    else:
                        result = provider.answer(state, asked)
                except Exception as exc:
                    elapsed = (time.perf_counter() - started) * 1000.0
                    span.fail(exc)
                    logger.warning("provider %s failed: %s", provider.name, exc)
                    for key in asked:
                        latency[key] += elapsed
                        attempts[key].append(
                            Attempt(
                                provider=provider.name,
                                plane=provider.plane,
                                latency_ms=elapsed,
                                reason=f"error: {type(exc).__name__}",
                            )
                        )
                    if self.on_error == "raise":
                        raise
                    continue
                elapsed = (time.perf_counter() - started) * 1000.0
                cost, source = self._cost(
                    provider.price_key, result.model, result.usage, result.cost_usd
                )
                span.set(
                    model=result.model,
                    usage=result.usage.model_dump(),
                    cost_usd=cost,
                    cost_known=source != "unknown",
                    cost_source=source,
                    **result.meta,
                )
                if self.tracer.capture_content and result.content:
                    span.set(**{k: self.tracer.content(v) for k, v in result.content.items()})

                accepted: list[str] = []
                outcomes: list[dict[str, Any]] = []
                for key, question in asked.items():
                    latency[key] += elapsed
                    answer = result.answers.get(key)
                    if answer is None:
                        attempts[key].append(
                            Attempt(
                                provider=provider.name,
                                plane=provider.plane,
                                latency_ms=elapsed,
                                reason="abstained",
                            )
                        )
                        outcomes.append({"name": key, "accepted": False, "reason": "abstained"})
                        continue
                    confidence = self._confidence(question, answer)
                    threshold = self._threshold(key, question, provider.name)
                    if confidence is not None:
                        ok = confidence >= threshold
                        reason = "met_threshold" if ok else "below_threshold"
                    else:
                        ok = (not provider.calibrated) and self.trust_uncalibrated
                        reason = "trusted_uncalibrated" if ok else "no_confidence"
                    attempts[key].append(
                        Attempt(
                            provider=provider.name,
                            plane=provider.plane,
                            value=self._trace_value(question.kind, answer.value),
                            confidence=confidence,
                            latency_ms=elapsed,
                            accepted=ok,
                            reason=reason,
                        )
                    )
                    outcomes.append(
                        {
                            "name": key,
                            "value": self._trace_value(question.kind, answer.value),
                            "confidence": None if confidence is None else round(confidence, 4),
                            "threshold": threshold,
                            "accepted": ok,
                            "reason": reason,
                        }
                    )
                    candidate = self._build(
                        key,
                        question,
                        answer,
                        confidence,
                        threshold,
                        provider,
                        result,
                        cost,
                        Status.ACCEPTED if ok else Status.UNCERTAIN,
                    )
                    if ok:
                        resolved[key] = candidate
                        del pending[key]
                        accepted.append(key)
                    else:
                        previous = best.get(key)
                        if previous is None or (confidence or 0.0) > (previous.confidence or 0.0):
                            best[key] = candidate
                span.set(accepted=accepted, results=outcomes)

        for key, question in pending.items():
            resolved[key] = best.get(key) or Decision(
                name=key,
                kind=question.kind,
                status=Status.ABSTAINED,
                threshold=self._threshold(key, question),
            )
        for key, decision in resolved.items():
            decision.attempts = attempts[key]
            decision.latency_ms = latency[key]
        return {key: resolved[key] for key in batch}
