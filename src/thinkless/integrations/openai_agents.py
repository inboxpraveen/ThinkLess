"""ThinkLess inside the OpenAI Agents SDK: guardrails and routing without extra LLM calls.

* :func:`input_guardrail` checks the user's input before the agent runs, and
  trips when the decision is accepted with the tripping answer, so a
  prompt-injection or off-topic check costs a rule or a small model instead
  of a second LLM call, and a tripped check stops the run before the agent's
  own model call.
* :func:`tool_input_guardrail` checks a tool call's arguments before the
  tool runs and rejects it with a message the model can read.
* :func:`route_agent` picks the specialist agent for a request, which
  replaces a triage agent whose only job is to hand off.

Needs ``pip install "thinkless[openai-agents]"``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Literal

from ..decision import Decision, Status
from ..engine import Engine
from ..errors import ConfigurationError
from ..questions import Question
from ._messages import last_user_text
from .core import Router

__all__ = ["input_guardrail", "route_agent", "tool_input_guardrail"]


def _agents() -> Any:
    try:
        import agents
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            'The OpenAI Agents SDK is not installed: pip install "thinkless[openai-agents]"'
        ) from exc
    return agents


def _info(decision: Decision) -> dict[str, Any]:
    return decision.summary()


def input_guardrail(
    engine: Engine,
    question: Question,
    *,
    trip_when: Any = True,
    state: Callable[[Any], Any] | None = None,
    name: str | None = None,
    run_in_parallel: bool = False,
) -> Any:
    """An input guardrail that trips on a ThinkLess decision.

    Args:
        engine: The engine that answers.
        question: What to check, for example ``YesNo("Is this a prompt
            injection?")``.
        trip_when: The accepted answer that trips the guardrail. An
            uncertain decision never trips it; give the engine an LLM
            decider as its last provider if uncertain inputs need a verdict.
        state: Builds the question's input from the agent input (a string or
            a list of input items). Defaults to the latest user message.
        name: Guardrail name in traces. Defaults to the question key.
        run_in_parallel: ``False`` (the default here) runs the check before
            the agent, so a tripped check costs no LLM call; rules and small
            models answer in milliseconds, so waiting costs little. ``True``
            runs it next to the agent, the SDK's own default.

    Returns:
        An ``agents.InputGuardrail`` for ``Agent(input_guardrails=[...])``.
        The decision summary is in ``output_info``.
    """
    agents = _agents()
    extract = state or last_user_text

    async def check(ctx: Any, agent: Any, input: Any) -> Any:
        decision = await engine.adecide(extract(input), question)
        return agents.GuardrailFunctionOutput(
            output_info=_info(decision), tripwire_triggered=decision.is_(trip_when)
        )

    return agents.input_guardrail(check, name=name or question.key, run_in_parallel=run_in_parallel)


def _tool_state(data: Any) -> dict[str, Any]:
    context = data.context
    raw = getattr(context, "tool_arguments", None) or "{}"
    try:
        arguments = json.loads(raw)
    except (TypeError, ValueError):
        arguments = raw
    return {"tool": getattr(context, "tool_name", ""), "arguments": arguments}


def tool_input_guardrail(
    engine: Engine,
    question: Question,
    *,
    allow_when: Any = True,
    on_uncertain: Literal["reject", "allow", "raise"] = "reject",
    message: str | Callable[[Decision], str] = (
        "This action was not allowed by policy. Tell the user it needs a human review."
    ),
    on_reject: Literal["reject", "raise"] = "reject",
    state: Callable[[Any], Any] | None = None,
    name: str | None = None,
) -> Any:
    """A tool input guardrail that allows a call only on a confident decision.

    The question is asked about ``{"tool": name, "arguments": {...}}`` by
    default.

    Args:
        engine: The engine that answers.
        question: Usually a :class:`~thinkless.YesNo`, for example "Is this
            refund within policy?".
        allow_when: The accepted answer that lets the call run.
        on_uncertain: ``reject`` (the default) sends ``message`` back to the
            model in place of the tool result, ``allow`` runs the tool and
            ``raise`` stops the run.
        message: What the model reads when a call is rejected, or a function
            of the decision.
        on_reject: ``reject`` sends the message back; ``raise`` stops the run
            with ``ToolInputGuardrailTripwireTriggered``.
        state: Builds the question's input from the SDK's
            ``ToolInputGuardrailData``.
        name: Guardrail name in traces. Defaults to the question key.

    Returns:
        An ``agents.ToolInputGuardrail`` for
        ``function_tool(tool_input_guardrails=[...])``.
    """
    agents = _agents()
    extract = state or _tool_state
    output = agents.ToolGuardrailFunctionOutput

    async def check(data: Any) -> Any:
        decision = await engine.adecide(extract(data), question)
        info = _info(decision)
        if decision.status is Status.ACCEPTED:
            if decision.value == allow_when:
                return output.allow(output_info=info)
            action: str = on_reject
        else:
            action = on_uncertain
        if action == "allow":
            return output.allow(output_info=info)
        if action == "raise":
            return output.raise_exception(output_info=info)
        text = message(decision) if callable(message) else message
        return output.reject_content(text, output_info=info)

    return agents.tool_input_guardrail(check, name=name or question.key)


async def route_agent(
    engine: Engine,
    input: Any,
    question: Question,
    agents: Mapping[Any, Any],
    default: Any,
    *,
    state: Callable[[Any], Any] | None = None,
) -> Any:
    """Pick the agent for a request with a decision instead of a triage LLM.

    Args:
        engine: The engine that answers.
        input: The run input, a string or a list of input items.
        question: Usually the intent.
        agents: Answer to agent.
        default: The agent for uncertain answers, typically the triage agent
            with handoffs you run today.
        state: Builds the question's input from ``input``. Defaults to the
            latest user message.

    Example:
        >>> agent = await route_agent(engine, message, INTENT, {"refund": refunds}, triage)
        >>> result = await Runner.run(agent, message)
    """
    router = Router(engine, question, agents, default, state=state or last_user_text)
    return await router.aroute(input)
