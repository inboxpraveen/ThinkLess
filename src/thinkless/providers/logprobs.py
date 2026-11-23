"""Turn an LLM's token log probabilities into a probability for each answer.

A prompted LLM writes its answers as JSON, one token at a time. When the
backend returns the top alternatives for every generated token, the
probability of each option can be read off the reply:

* the chosen option gets the product of the probabilities of its tokens;
* at every token of the chosen value, each alternative token that would
  have started a different option gives that option the probability of the
  tokens before it times the alternative's own probability.

Options the model never considered within the top alternatives get 0, and
the result is normalized over the options. Mass on tokens that start no
option (a malformed reply) is left out. This is an estimate: the
continuation after a diverging token is assumed to finish the option it
started, which holds for labels that differ early, as most do.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..questions import Choice, Extract, Question, Score, YesNo

__all__ = [
    "TokenLogprob",
    "field_confidences",
    "literals_for",
    "option_probabilities",
    "span_confidence",
    "value_span",
]

TokenLogprob = Mapping[str, Any]
"""``{"token": str, "logprob": float, "top": [[token, logprob], ...]}``."""

_DECODER = json.JSONDecoder()


def _offsets(tokens: Sequence[TokenLogprob], text: str) -> list[int] | None:
    """Character offset of each token in ``text``; ``None`` if they do not spell it."""
    offsets = []
    position = 0
    for token in tokens:
        offsets.append(position)
        position += len(str(token["token"]))
    joined = "".join(str(t["token"]) for t in tokens)
    if joined != text:
        # Some servers strip surrounding whitespace from the message.
        lead = len(joined) - len(joined.lstrip())
        if joined.strip() != text.strip():
            return None
        shift = lead - (len(text) - len(text.lstrip()))
        offsets = [o - shift for o in offsets]
    return offsets


def value_span(text: str, key: str) -> tuple[int, int] | None:
    """Where the JSON value of ``key`` is in ``text``: start and end offsets."""
    match = re.search(r'"' + re.escape(key) + r'"\s*:\s*', text) or re.search(
        r'"' + re.escape(key) + r'"\s*:\s*', text, re.IGNORECASE
    )
    if match is None:
        return None
    try:
        _, end = _DECODER.raw_decode(text, match.end())
    except json.JSONDecodeError:
        return None
    return match.end(), end


def _span_tokens(
    offsets: list[int], tokens: Sequence[TokenLogprob], start: int, end: int
) -> list[int]:
    return [
        k
        for k, (offset, token) in enumerate(zip(offsets, tokens, strict=True))
        if offset < end and offset + len(str(token["token"])) > start
    ]


def span_confidence(text: str, tokens: Sequence[TokenLogprob], key: str) -> float | None:
    """Probability of the whole value written for ``key``: the product of its tokens."""
    offsets = _offsets(tokens, text)
    span = value_span(text, key)
    if offsets is None or span is None:
        return None
    picked = _span_tokens(offsets, tokens, *span)
    if not picked:
        return None
    return math.exp(sum(float(tokens[k]["logprob"]) for k in picked))


def option_probabilities(
    text: str,
    tokens: Sequence[TokenLogprob],
    key: str,
    literals: Mapping[str, Sequence[str]],
) -> dict[str, float] | None:
    """A distribution over options for the value of ``key``.

    Args:
        text: The reply.
        tokens: The reply's tokens with their top alternatives.
        key: The JSON key whose value answers the question.
        literals: For each option, the JSON texts that would mean it, for
            example ``{"yes": ["true", '"yes"'], "no": ["false", '"no"']}``.

    Returns:
        Probabilities over the options, or ``None`` when the reply cannot be
        lined up with its tokens or does not contain the key.
    """
    offsets = _offsets(tokens, text)
    span = value_span(text, key)
    if offsets is None or span is None:
        return None
    start, end = span
    chosen = text[start:end].lower()
    options = {name: [lit.lower() for lit in lits] for name, lits in literals.items()}
    picked_option = next((n for n, lits in options.items() if chosen in lits), None)
    if picked_option is None:
        return None
    probs = dict.fromkeys(literals, 0.0)
    mass = 1.0
    for k in _span_tokens(offsets, tokens, start, end):
        token = str(tokens[k]["token"])
        offset = offsets[k]
        before = text[offset:start] if offset < start else ""
        emitted = chosen[: max(0, offset - start)]
        for alternative, logprob in tokens[k].get("top") or []:
            alternative = str(alternative)
            if alternative == token or not alternative.startswith(before):
                continue
            rest = alternative[len(before) :].lower()
            if not rest:
                continue
            matches = [
                name
                for name, lits in options.items()
                if any(
                    lit.startswith(emitted)
                    and (
                        lit[len(emitted) :].startswith(rest) or rest.startswith(lit[len(emitted) :])
                    )
                    for lit in lits
                )
            ]
            for name in matches:
                probs[name] += mass * math.exp(float(logprob)) / len(matches)
        mass *= math.exp(float(tokens[k]["logprob"]))
    probs[picked_option] += mass
    total = sum(probs.values())
    if total <= 0:
        return None
    return {name: p / total for name, p in probs.items()}


def literals_for(question: Question) -> dict[str, list[str]] | None:
    """The JSON texts that mean each option of a closed question."""
    if isinstance(question, Choice):
        return {label: [json.dumps(label, ensure_ascii=False)] for label in question.options}
    if isinstance(question, Score):
        return {level: [json.dumps(level, ensure_ascii=False)] for level in question.levels}
    if isinstance(question, YesNo):
        return {
            "yes": ["true", '"yes"', '"true"'],
            "no": ["false", '"no"', '"false"'],
        }
    return None


def field_confidences(
    text: str, tokens: Sequence[TokenLogprob], key: str, question: Extract
) -> dict[str, float | None]:
    """Per extracted field, the probability of the value the model wrote."""
    offsets = _offsets(tokens, text)
    span = value_span(text, key)
    out: dict[str, float | None] = dict.fromkeys(question.fields)
    if offsets is None or span is None:
        return out
    inner = text[span[0] : span[1]]
    for name in question.fields:
        found = value_span(inner, name)
        if found is None:
            continue
        start, end = span[0] + found[0], span[0] + found[1]
        picked = _span_tokens(offsets, tokens, start, end)
        if picked:
            out[name] = math.exp(sum(float(tokens[k]["logprob"]) for k in picked))
    return out
