"""A generative model as a decision provider.

This is the usual way agents make decisions today: describe the options in a
prompt and parse the reply. ThinkLess uses it as the last step of a cascade and
as the whole decision plane in LLM-only baselines, so benchmarks compare the
same questions answered two ways.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, ClassVar

from ..decision import Answer, Decision, Plane
from ..llm.base import LLM
from ..logs import get_logger
from ..questions import Choice, Extract, Kind, Question, Score, YesNo
from .base import DecisionProvider, ProviderResult, State, render_state

__all__ = ["LLMDecider", "build_prompt", "parse_reply"]

logger = get_logger("providers.llm")

# Stop reasons that mean the reply hit the token limit, across backends.
TRUNCATED = frozenset({"length", "max_tokens"})

SYSTEM_PROMPT = (
    "You answer typed questions about an input. Follow the allowed answers exactly. "
    "Reply with a single JSON object and nothing else."
)

_YES = {"yes", "true", "y", "1"}
_NO = {"no", "false", "n", "0"}


def _describe(key: str, question: Question, number: int) -> tuple[str, str]:
    """Prompt lines for one question and its JSON placeholder."""
    lines = []
    if isinstance(question, Choice):
        lines.append(f'{number}. "{key}": {question.instructions}')
        lines.append("   Answer with exactly one of these labels:")
        for label, description in question.options.items():
            lines.append(f"   - {label}" + (f": {description}" if description else ""))
        placeholder = '"<label>"'
    elif isinstance(question, Score):
        lines.append(f'{number}. "{key}": {question.instructions}')
        lines.append(
            "   Answer with one level, from lowest to highest: " + ", ".join(question.levels)
        )
        placeholder = '"<level>"'
    elif isinstance(question, YesNo):
        lines.append(f'{number}. "{key}": {question.instructions}')
        if question.yes_means:
            lines.append(f"   Yes means: {question.yes_means}")
        if question.no_means:
            lines.append(f"   No means: {question.no_means}")
        lines.append("   Answer true or false.")
        placeholder = "<true or false>"
    elif isinstance(question, Extract):
        lines.append(f'{number}. "{key}": {question.instructions}')
        for name, description in question.fields.items():
            lines.append(f"   - {name}: {description}")
        lines.append("   Copy values from the input. Use null for a field that is not present.")
        placeholder = (
            "{" + ", ".join(f'"{name}": <string or null>' for name in question.fields) + "}"
        )
    else:  # pragma: no cover - guarded by kinds
        raise TypeError(type(question).__name__)
    return "\n".join(lines), f'"{key}": {placeholder}'


def _settled_value(question: Question, decision: Decision) -> str:
    if isinstance(question, YesNo):
        return "yes" if decision.value else "no"
    if isinstance(question, Score):
        return str(decision.level)
    if isinstance(question, Extract):
        return json.dumps(decision.value, ensure_ascii=False)
    return str(decision.value)


def build_prompt(
    state: State,
    questions: Mapping[str, Question],
    context: Mapping[str, tuple[Question, Decision]] | None = None,
) -> str:
    """The user prompt for a batch of questions.

    ``context`` lists decisions already settled for the same input by other
    providers. They are shown as established facts so the model reads each
    open question with the same information it would have had in one batch.
    """
    blocks = []
    shapes = []
    for number, (key, question) in enumerate(questions.items(), start=1):
        block, shape = _describe(key, question, number)
        blocks.append(block)
        shapes.append(shape)
    known = ""
    if context:
        lines = [
            f'- "{key}" ({question.instructions}): {_settled_value(question, decision)}'
            for key, (question, decision) in context.items()
        ]
        known = "Already established by other checks:\n" + "\n".join(lines) + "\n\n"
    return (
        "Input:\n<<<\n"
        + render_state(state)
        + "\n>>>\n\n"
        + known
        + "Questions:\n"
        + "\n".join(blocks)
        + "\n\nReply with JSON in this shape:\n{"
        + ", ".join(shapes)
        + "}"
    )


def _json_object(text: str) -> dict[str, Any] | None:
    """Find the first top-level JSON object in a reply."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                        except json.JSONDecodeError:
                            break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def _coerce(question: Question, value: Any) -> Answer | None:
    if value is None:
        return None
    if isinstance(question, Choice):
        wanted = _normalize(str(value))
        for label in question.options:
            if _normalize(label) == wanted:
                return Answer(value=label)
        return None
    if isinstance(question, Score):
        levels = list(question.levels)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            index = round(float(value))
            return (
                Answer(value=float(index), raw={"level": levels[index]})
                if 0 <= index < len(levels)
                else None
            )
        wanted = _normalize(str(value))
        for index, level in enumerate(levels):
            if _normalize(level) == wanted:
                return Answer(value=float(index), raw={"level": level})
        return None
    if isinstance(question, YesNo):
        if isinstance(value, bool):
            return Answer(value=value)
        word = str(value).strip().lower()
        if word in _YES:
            return Answer(value=True)
        if word in _NO:
            return Answer(value=False)
        return None
    if isinstance(question, Extract):
        if not isinstance(value, Mapping):
            return None
        values = {}
        for name in question.fields:
            item = value.get(name)
            values[name] = None if item in (None, "", "null") else str(item)
        return Answer(value=values)
    return None


def parse_reply(text: str, questions: Mapping[str, Question]) -> dict[str, Answer | None]:
    """Map a model reply onto the questions. Invalid answers become ``None``."""
    payload = _json_object(text) or {}
    lowered = {str(k).lower(): v for k, v in payload.items()}
    return {
        key: _coerce(q, payload.get(key, lowered.get(key.lower()))) for key, q in questions.items()
    }


class LLMDecider(DecisionProvider):
    """Answers any question kind by prompting a generative model.

    Answers carry no probabilities, so the engine treats them as uncalibrated:
    by default they are accepted as the final word of a cascade (see
    ``Engine(trust_uncalibrated=...)``).

    Args:
        llm: The model to prompt.
        name: Provider name. Defaults to ``llm``.
        max_tokens: Reply budget per batch, plus 48 tokens per question.
        retry_on_truncation: When a reply is cut off at the token limit and
            answers are missing, retry once with three times the budget.
            Reasoning models are the usual cause: hidden reasoning tokens
            count against the limit. The retry is recorded in the trace.
    """

    plane: ClassVar[Plane] = Plane.LLM
    kinds: ClassVar[frozenset[Kind]] = frozenset(Kind)
    calibrated: ClassVar[bool] = False
    accepts_context: ClassVar[bool] = True

    def __init__(
        self,
        llm: LLM,
        *,
        name: str = "llm",
        max_tokens: int = 256,
        retry_on_truncation: bool = True,
    ) -> None:
        self.llm = llm
        self.name = name
        self.price_key = llm.provider
        self._max_tokens = max_tokens
        self._retry_on_truncation = retry_on_truncation

    def warmup(self) -> None:
        self.llm.warmup()

    def answer(
        self,
        state: State,
        questions: Mapping[str, Question],
        *,
        context: Mapping[str, tuple[Question, Decision]] | None = None,
    ) -> ProviderResult:
        prompt = build_prompt(state, questions, context)
        budget = self._max_tokens + 48 * len(questions)
        messages = [{"role": "user", "content": prompt}]
        completion = self.llm.complete(
            messages, system=SYSTEM_PROMPT, max_tokens=budget, json_mode=True
        )
        answers = parse_reply(completion.text, questions)
        invalid = [key for key, answer in answers.items() if answer is None]
        usage = completion.usage
        cost = completion.cost_usd
        truncated = completion.stop_reason in TRUNCATED and bool(invalid)
        retried = False
        if truncated:
            logger.warning(
                "%s stopped at the %d token limit with %d of %d answers missing; if the model "
                "reasons, lower its reasoning effort (for example --reasoning off)",
                completion.model,
                budget,
                len(invalid),
                len(questions),
            )
            if self._retry_on_truncation:
                retried = True
                completion = self.llm.complete(
                    messages, system=SYSTEM_PROMPT, max_tokens=budget * 3, json_mode=True
                )
                answers = parse_reply(completion.text, questions)
                invalid = [key for key, answer in answers.items() if answer is None]
                usage = usage + completion.usage
                if cost is not None and completion.cost_usd is not None:
                    cost += completion.cost_usd
                else:
                    cost = None
        return ProviderResult(
            answers=answers,
            model=completion.model,
            usage=usage,
            cost_usd=cost,
            meta={
                "invalid": invalid,
                "stop_reason": completion.stop_reason,
                "truncated": truncated,
                "retried": retried,
            },
            content={"system": SYSTEM_PROMPT, "prompt": prompt, "completion": completion.text},
        )
