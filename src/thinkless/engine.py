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
from .decision import Answer, Attempt, Decision, Plane, Status, Usage
from .errors import ConfigurationError
from .limits import SpendLimit, SpendLimitError, _run_limit
from .llm.base import LLM, Completion, Message, as_messages
from .logs import get_logger
from .pricing import LOCAL_PROVIDERS, PriceTable, default_prices
from .providers.base import BatchItem, DecisionProvider, ProviderResult, State
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
    """Sink that keeps the spans of one run while it is open.

    It keeps the run's root span and its descendants, not every span of the
    trace: two runs opened side by side under one outer span share a trace,
    and each must count only its own work.
    """

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._ids: set[str] = set()
        self._lock = threading.Lock()

    def adopt(self, span: Span) -> None:
        """Make ``span`` the run's root."""
        with self._lock:
            self._ids.add(span.span_id)

    def on_start(self, span: Span) -> None:
        with self._lock:
            if span.parent_id in self._ids:
                self._ids.add(span.span_id)

    def on_end(self, span: Span) -> None:
        with self._lock:
            if span.span_id in self._ids:
                self.spans.append(span.to_dict())


class _Item:
    """The cascade's bookkeeping for one state."""

    __slots__ = ("attempts", "best", "latency", "pending", "resolved", "state")

    def __init__(self, state: State, batch: Mapping[str, Question]) -> None:
        self.state = state
        self.pending: dict[str, Question] = dict(batch)
        self.resolved: dict[str, Decision] = {}
        self.best: dict[str, Decision] = {}
        self.attempts: dict[str, list[Attempt]] = defaultdict(list)
        self.latency: dict[str, float] = defaultdict(float)


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
        deadline_ms: Time budget for one ``decide`` or ``decide_many`` call.
            It is checked before each provider is asked: once it has passed,
            the remaining providers are skipped and open questions come back
            ``uncertain`` with the best answer so far. A provider call that
            is already running is not interrupted, so give hosted clients
            their own timeout.
        spend_limit: A :class:`~thinkless.SpendLimit` shared by every call on
            this engine. Once it is used up, paid providers are skipped and
            :meth:`generate` raises :class:`~thinkless.SpendLimitError`;
            rules and local models keep answering. ``run(max_cost_usd=...)``
            adds a cap for one run.

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
        deadline_ms: float | None = None,
        spend_limit: SpendLimit | None = None,
    ) -> None:
        names = [p.name for p in providers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ConfigurationError(f"provider names must be unique, got duplicates: {duplicates}")
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("threshold must be between 0 and 1")
        if on_error not in ("continue", "raise"):
            raise ConfigurationError("on_error must be 'continue' or 'raise'")
        if deadline_ms is not None and deadline_ms <= 0:
            raise ConfigurationError("deadline_ms must be positive")
        self.providers: list[DecisionProvider] = list(providers)
        self.llm = llm
        self.threshold = threshold
        self.thresholds = dict(thresholds or {})
        self.trust_uncalibrated = trust_uncalibrated
        self.tracer = tracer or Tracer()
        self.prices = prices or default_prices()
        self.on_error = on_error
        self.escalation_context = escalation_context
        self.deadline_ms = deadline_ms
        self.spend_limit = spend_limit

    # ------------------------------------------------------------------ runs

    @contextmanager
    def run(
        self,
        name: str,
        *,
        input: Any = None,
        max_cost_usd: float | None = None,
        **attributes: Any,
    ) -> Iterator[Run]:
        """Open a root span for one unit of work (a ticket, a request, a task).

        Everything the engine does inside the block, including decorated tool
        calls, becomes part of this trace.

        Args:
            name: Span name.
            input: Recorded on the root span when content capture is on.
            max_cost_usd: Spend cap for this run. Past it, paid providers are
                skipped and ``generate`` raises
                :class:`~thinkless.SpendLimitError`, as with the engine-wide
                ``spend_limit``.
            attributes: Recorded on the root span.
        """
        # A nested run keeps the limits of the runs around it, and adds its own.
        outer = _run_limit.get()
        limits = (*outer, SpendLimit(max_cost_usd)) if max_cost_usd is not None else outer
        collector = _RunCollector()
        self.tracer.add_sink(collector)
        token = _run_limit.set(limits)
        try:
            with self.tracer.span("run", name, **attributes) as span:
                collector.adopt(span)
                if input is not None:
                    span.set(input=self.tracer.content(input))
                if max_cost_usd is not None:
                    span.set(max_cost_usd=max_cost_usd)
                yield Run(span, collector)
        finally:
            _run_limit.reset(token)
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

    def decide(
        self,
        state: State,
        question: Question,
        *,
        name: str | None = None,
        deadline_ms: float | None = None,
    ) -> Decision:
        """Answer one question. See :meth:`decide_many` for batching."""
        key = name or question.key
        return self.decide_many(state, {key: question}, label=key, deadline_ms=deadline_ms)[key]

    def decide_many(
        self,
        state: State,
        questions: Mapping[str, Question] | Sequence[Question],
        *,
        label: str | None = None,
        deadline_ms: float | None = None,
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
            deadline_ms: Time budget for this call. Defaults to the engine's
                ``deadline_ms``.

        Returns:
            Decisions keyed by question name, in input order.
        """
        batch = self._as_mapping(questions)
        deadline = deadline_ms if deadline_ms is not None else self.deadline_ms
        with self.tracer.span("decide", label or ", ".join(batch)[:80]) as span:
            span.set(
                questions={
                    k: {"kind": q.kind.value, "threshold": self._threshold(k, q)}
                    for k, q in batch.items()
                }
            )
            if self.tracer.capture_content:
                span.set(state=self.tracer.content(state))
            decisions = self._cascade(state, batch, deadline)
            for decision in decisions.values():
                decision.span_id = span.span_id
            span.set(decisions=[self._trace_summary(d) for d in decisions.values()])
        return decisions

    def decide_batch(
        self,
        states: Sequence[State],
        questions: Mapping[str, Question] | Sequence[Question],
        *,
        label: str | None = None,
        deadline_ms: float | None = None,
    ) -> list[dict[str, Decision]]:
        """Answer the same questions about many states, sharing each provider's calls.

        The cascade runs one provider level at a time across every state.
        Providers that batch (GLiNER, Laya, the Hugging Face providers) answer
        all the states still open at their level in one call, which on a GPU
        is several times faster per decision than asking one state at a time.
        Other providers are asked once per state, as with :meth:`decide_many`.
        Every state gets the decisions it would have got alone, apart from
        the small numeric differences batching can make in a model's scores.

        Use it for bulk work: backfills, evaluations, queues. For concurrent
        requests in a server, wrap local providers in
        :class:`~thinkless.providers.Batched` instead.

        Args:
            states: The inputs, each text, a JSON-like object, or a list.
            questions: As for :meth:`decide_many`.
            label: Span name. Defaults to the joined question names.
            deadline_ms: Time budget for the whole batch. Defaults to the
                engine's ``deadline_ms``.

        Returns:
            One mapping of decisions per state, in the order of ``states``.
        """
        states = list(states)
        if not states:
            return []
        batch = self._as_mapping(questions)
        deadline = deadline_ms if deadline_ms is not None else self.deadline_ms
        with self.tracer.span("decide", label or ", ".join(batch)[:80]) as span:
            span.set(
                items=len(states),
                questions={
                    k: {"kind": q.kind.value, "threshold": self._threshold(k, q)}
                    for k, q in batch.items()
                },
            )
            results = self._cascade_items(states, batch, deadline)
            summaries = []
            for index, decisions in enumerate(results):
                for decision in decisions.values():
                    decision.span_id = span.span_id
                    summaries.append({"item": index, **self._trace_summary(decision)})
            span.set(decisions=summaries)
        return results

    async def adecide_batch(
        self,
        states: Sequence[State],
        questions: Mapping[str, Question] | Sequence[Question],
        *,
        label: str | None = None,
        deadline_ms: float | None = None,
    ) -> list[dict[str, Decision]]:
        return await asyncio.to_thread(
            self.decide_batch, states, questions, label=label, deadline_ms=deadline_ms
        )

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
        paid = model.provider not in LOCAL_PROVIDERS
        if paid:
            limit = self._exhausted_limit()
            if limit is not None:
                raise SpendLimitError(f"generate() refused: {limit!r} is used up")
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
            self._charge(cost)
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
        self,
        state: State,
        question: Question,
        *,
        name: str | None = None,
        deadline_ms: float | None = None,
    ) -> Decision:
        return await asyncio.to_thread(
            self.decide, state, question, name=name, deadline_ms=deadline_ms
        )

    async def adecide_many(
        self,
        state: State,
        questions: Mapping[str, Question] | Sequence[Question],
        *,
        label: str | None = None,
        deadline_ms: float | None = None,
    ) -> dict[str, Decision]:
        return await asyncio.to_thread(
            self.decide_many, state, questions, label=label, deadline_ms=deadline_ms
        )

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
            # An unnamed question takes the name it was given here, so rules,
            # classifiers, thresholds and traces all know it by that name.
            batch = {
                key: q if q.name is not None or q.key == key else q.model_copy(update={"name": key})
                for key, q in questions.items()
            }
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

    def _limits(self) -> list[SpendLimit]:
        engine_limit = [self.spend_limit] if self.spend_limit is not None else []
        return [*engine_limit, *_run_limit.get()]

    def _exhausted_limit(self) -> SpendLimit | None:
        return next((lim for lim in self._limits() if lim.exhausted), None)

    def _charge(self, cost: float) -> None:
        for limit in self._limits():
            limit.add(cost)

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

    def _skip_reason(
        self, provider: DecisionProvider, started: float, deadline_ms: float | None
    ) -> str | None:
        if deadline_ms is not None and (time.perf_counter() - started) * 1000.0 >= deadline_ms:
            return "deadline"
        if provider.price_key not in LOCAL_PROVIDERS and self._exhausted_limit() is not None:
            return "spend_limit"
        return None

    def _cascade(
        self, state: State, batch: dict[str, Question], deadline_ms: float | None = None
    ) -> dict[str, Decision]:
        return self._cascade_items([state], batch, deadline_ms)[0]

    def _cascade_items(
        self, states: Sequence[State], batch: dict[str, Question], deadline_ms: float | None = None
    ) -> list[dict[str, Decision]]:
        """Run the cascade for every state at once, one provider level at a time.

        Each provider is asked once per level for every state that still has
        open questions it supports: in one ``answer_batch`` call when it
        batches, otherwise one call per state. With one state this is exactly
        the cascade of ``decide_many``.
        """
        items = [_Item(state, batch) for state in states]
        many = len(items) > 1
        cascade_started = time.perf_counter()

        for provider in self.providers:
            if all(not item.pending for item in items):
                break
            active = []
            for index, item in enumerate(items):
                asked = {k: q for k, q in item.pending.items() if provider.supports(q)}
                if asked:
                    active.append((index, item, asked))
            if not active:
                continue
            names = list(dict.fromkeys(k for _, _, asked in active for k in asked))
            extra = {"items": len(active)} if many else {}
            skip = self._skip_reason(provider, cascade_started, deadline_ms)
            if skip is not None:
                logger.info("skipping %s for %s: %s", provider.name, names, skip)
                with self.tracer.span(
                    "attempt",
                    provider.name,
                    plane=provider.plane.value,
                    provider=provider.name,
                    questions=names,
                    skipped=skip,
                    **extra,
                ):
                    pass
                for _, item, asked in active:
                    for key in asked:
                        item.attempts[key].append(
                            Attempt(provider=provider.name, plane=provider.plane, reason=skip)
                        )
                continue
            with self.tracer.span(
                "attempt",
                provider.name,
                plane=provider.plane.value,
                provider=provider.name,
                questions=names,
                **extra,
            ) as span:
                calls = self._ask(
                    provider, [(item, asked) for _, item, asked in active], batch, span
                )
                failures = [error for _, error, _ in calls if error is not None]
                if failures:
                    span.fail(failures[0])
                    for failure in dict.fromkeys(failures):
                        logger.warning("provider %s failed: %s", provider.name, failure)
                costs: list[float] = []
                sources: list[str] = []
                for result, _, _ in calls:
                    if result is None:
                        costs.append(0.0)
                        continue
                    cost, source = self._cost(
                        provider.price_key, result.model, result.usage, result.cost_usd
                    )
                    self._charge(cost)
                    costs.append(cost)
                    sources.append(source)
                results = [r for r, _, _ in calls if r is not None]
                if results:
                    usage = results[0].usage
                    if len(results) > 1:
                        usage = Usage(
                            input_tokens=sum(r.usage.input_tokens for r in results),
                            output_tokens=sum(r.usage.output_tokens for r in results),
                        )
                    span.set(
                        model=results[0].model,
                        usage=usage.model_dump(),
                        cost_usd=sum(costs),
                        cost_known=all(src != "unknown" for src in sources),
                        cost_source=sources[0] if len(set(sources)) == 1 else "mixed",
                        **results[0].meta,
                    )
                    if self.tracer.capture_content and results[0].content and not many:
                        span.set(
                            **{k: self.tracer.content(v) for k, v in results[0].content.items()}
                        )

                accepted: list[str] = []
                outcomes: list[dict[str, Any]] = []
                for (index, item, asked), (result, error, elapsed), cost in zip(
                    active, calls, costs, strict=True
                ):
                    tag: dict[str, Any] = {"item": index} if many else {}
                    if result is None:
                        for key in asked:
                            item.latency[key] += elapsed
                            item.attempts[key].append(
                                Attempt(
                                    provider=provider.name,
                                    plane=provider.plane,
                                    latency_ms=elapsed,
                                    reason=f"error: {type(error).__name__}",
                                )
                            )
                        continue
                    for key, question in asked.items():
                        item.latency[key] += elapsed
                        answer = result.answers.get(key)
                        if answer is None:
                            item.attempts[key].append(
                                Attempt(
                                    provider=provider.name,
                                    plane=provider.plane,
                                    latency_ms=elapsed,
                                    reason="abstained",
                                )
                            )
                            outcomes.append(
                                {"name": key, **tag, "accepted": False, "reason": "abstained"}
                            )
                            continue
                        confidence = self._confidence(question, answer)
                        threshold = self._threshold(key, question, provider.name)
                        if confidence is not None:
                            ok = confidence >= threshold
                            reason = "met_threshold" if ok else "below_threshold"
                        else:
                            ok = (not provider.calibrated) and self.trust_uncalibrated
                            reason = "trusted_uncalibrated" if ok else "no_confidence"
                        item.attempts[key].append(
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
                                **tag,
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
                            item.resolved[key] = candidate
                            del item.pending[key]
                            accepted.append(f"{key}#{index}" if many else key)
                        else:
                            previous = item.best.get(key)
                            if previous is None or (confidence or 0.0) > (
                                previous.confidence or 0.0
                            ):
                                item.best[key] = candidate
                span.set(accepted=accepted, results=outcomes)
                if failures and self.on_error == "raise":
                    raise failures[0]

        out = []
        for item in items:
            for key, question in item.pending.items():
                item.resolved[key] = item.best.get(key) or Decision(
                    name=key,
                    kind=question.kind,
                    status=Status.ABSTAINED,
                    threshold=self._threshold(key, question),
                )
            for key, decision in item.resolved.items():
                decision.attempts = item.attempts[key]
                decision.latency_ms = item.latency[key]
            out.append({key: item.resolved[key] for key in batch})
        return out

    def _ask(
        self,
        provider: DecisionProvider,
        active: Sequence[tuple[_Item, dict[str, Question]]],
        batch: Mapping[str, Question],
        span: Span,
    ) -> list[tuple[ProviderResult | None, Exception | None, float]]:
        """Ask one provider for every active item: a result or an error, and the wall time."""
        calls: list[tuple[ProviderResult | None, Exception | None, float]] = [
            (None, None, 0.0)
        ] * len(active)
        shared: list[int] = []
        for position, (item, asked) in enumerate(active):
            context = bool(self.escalation_context and provider.accepts_context and item.resolved)
            if len(active) > 1 and provider.batches and not context:
                shared.append(position)
                continue
            started = time.perf_counter()
            try:
                if context:
                    settled = {k: (batch[k], d) for k, d in item.resolved.items()}
                    if len(active) == 1:
                        span.set(context_from=list(settled))
                    result = provider.answer(item.state, asked, context=settled)  # type: ignore[call-arg]
                else:
                    result = provider.answer(item.state, asked)
                calls[position] = (result, None, (time.perf_counter() - started) * 1000.0)
            except Exception as exc:
                calls[position] = (None, exc, (time.perf_counter() - started) * 1000.0)
                if self.on_error == "raise":
                    break
        if shared:
            requests: list[BatchItem] = [(active[p][0].state, active[p][1]) for p in shared]
            started = time.perf_counter()
            try:
                results = provider.answer_batch(requests)
                if len(results) != len(requests):
                    raise RuntimeError(
                        f"{provider.name}.answer_batch returned {len(results)} results "
                        f"for {len(requests)} items"
                    )
                elapsed = (time.perf_counter() - started) * 1000.0
                for position, result in zip(shared, results, strict=True):
                    calls[position] = (result, None, elapsed)
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                for position in shared:
                    calls[position] = (None, exc, elapsed)
        return calls
