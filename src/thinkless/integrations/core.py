"""Framework-neutral building blocks: route on a decision, gate a tool call."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any, Generic, Literal, TypeVar

from ..decision import Decision, Status
from ..engine import Engine
from ..errors import ThinkLessError
from ..questions import Question

__all__ = ["Router", "ToolBlockedError", "gate", "route"]

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
OnUncertain = Literal["block", "allow"]


class ToolBlockedError(ThinkLessError):
    """A gated tool call was not allowed. ``decision`` says why."""

    def __init__(self, tool: str, decision: Decision) -> None:
        self.tool = tool
        self.decision = decision
        super().__init__(
            f"{tool} was blocked: {decision.name} came back {decision.value!r} "
            f"({decision.status.value}, via {decision.provider})"
        )


class Router(Generic[T]):
    """Maps the answer to a question onto a destination.

    ``routes`` maps answers to destinations (node names, agents, handlers).
    Only an accepted answer follows its route; an uncertain or abstained
    decision, or an answer with no route, goes to ``default``. That default
    is usually the LLM path you run today, so routing never gets worse than
    the current behavior.

    Args:
        engine: The engine that answers.
        question: What to decide.
        routes: Answer to destination.
        default: Where everything else goes.
        state: Builds the question's input from whatever the router is called
            with. Defaults to passing it through unchanged.
    """

    def __init__(
        self,
        engine: Engine,
        question: Question,
        routes: Mapping[Any, T],
        default: T,
        *,
        state: Callable[[Any], Any] | None = None,
    ) -> None:
        self.engine = engine
        self.question = question
        self.routes = dict(routes)
        self.default = default
        self.state = state

    @property
    def destinations(self) -> list[T]:
        """Every place this router can send to, for graph declarations."""
        seen: list[T] = []
        for target in (*self.routes.values(), self.default):
            if target not in seen:
                seen.append(target)
        return seen

    def pick(self, decision: Decision) -> T:
        return self.pick_value(decision.value, accepted=decision.status is Status.ACCEPTED)

    def pick_value(self, value: Any, *, accepted: bool) -> T:
        """Route on an answer made earlier, for example one stored in graph state."""
        if accepted:
            for answer, target in self.routes.items():
                if value == answer:
                    return target
        return self.default

    def decide(self, value: Any) -> Decision:
        payload = self.state(value) if self.state is not None else value
        return self.engine.decide(payload, self.question)

    async def adecide(self, value: Any) -> Decision:
        payload = self.state(value) if self.state is not None else value
        return await self.engine.adecide(payload, self.question)

    def __call__(self, value: Any) -> T:
        return self.pick(self.decide(value))

    async def aroute(self, value: Any) -> T:
        return self.pick(await self.adecide(value))


def route(
    engine: Engine,
    state: Any,
    question: Question,
    routes: Mapping[Any, T],
    default: T,
) -> T:
    """Decide ``question`` on ``state`` and return the matching destination.

    Example:
        >>> handler = route(engine, ticket, INTENT, {"refund": refunds, "order_status": tracking}, agent)
    """
    return Router(engine, question, routes, default).pick(engine.decide(state, question))


def _arguments(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return {"args": list(args), **kwargs}
    bound.apply_defaults()
    return {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}


def gate(
    engine: Engine,
    question: Question,
    *,
    allow_when: Any = True,
    on_uncertain: OnUncertain = "block",
    state: Callable[..., Any] | None = None,
    on_block: Callable[[Decision], Any] | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """Check a tool call with a decision before it runs.

    The question is asked about ``{"tool": name, "arguments": {...}}`` by
    default, so rules can read the arguments directly and a model sees them
    as text. The call goes ahead only when the decision is accepted and
    equals ``allow_when``.

    Args:
        engine: The engine that answers.
        question: Usually a :class:`~thinkless.YesNo` such as "Is this refund
            within policy?".
        allow_when: The answer that lets the call through.
        on_uncertain: ``block`` (the default) or ``allow`` when no provider
            is confident.
        state: Builds the question's input from the call's arguments instead.
        on_block: Called with the decision when a call is blocked; its return
            value becomes the tool's result. Without it, :class:`ToolBlockedError`
            is raised.
        name: Tool name in the input and in errors. Defaults to the function
            name.

    Example:
        >>> @gate(engine, YesNo("Is this refund within policy?", name="refund_ok"))
        ... def refund(order_id: str, amount: float) -> str: ...
    """
    if on_uncertain not in ("block", "allow"):
        raise ValueError("on_uncertain must be 'block' or 'allow'")

    def decorate(func: F) -> F:
        tool = name or func.__name__
        signature = inspect.signature(func)

        def payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if state is not None:
                return state(*args, **kwargs)
            return {"tool": tool, "arguments": _arguments(signature, args, kwargs)}

        def allowed(decision: Decision) -> bool:
            if decision.status is Status.ACCEPTED:
                return bool(decision.value == allow_when)
            return on_uncertain == "allow"

        def blocked(decision: Decision) -> Any:
            if on_block is not None:
                return on_block(decision)
            raise ToolBlockedError(tool, decision)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                decision = await engine.adecide(payload(args, kwargs), question)
                if not allowed(decision):
                    return blocked(decision)
                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            decision = engine.decide(payload(args, kwargs), question)
            if not allowed(decision):
                return blocked(decision)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
