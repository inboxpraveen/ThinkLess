"""Plug your own model into the cascade.

A provider is one class: declare which questions it answers and return
answers with probabilities. This one wraps a keyword scorer so the example
runs without downloads; swap in a fine-tuned classifier, a vector search or a
call to an internal service the same way.

    python examples/04_custom_provider.py
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from thinkless import Answer, Choice, Engine, Kind, MemorySink, Plane, Question, Tracer
from thinkless.llm import ScriptedLLM
from thinkless.providers import DecisionProvider, LLMDecider, ProviderResult, render_state


class KeywordClassifier(DecisionProvider):
    """Scores each option by keyword hits and turns the scores into a softmax."""

    name = "keywords"
    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset({Kind.CHOICE})

    def __init__(self, keywords: Mapping[str, list[str]], sharpness: float = 3.0) -> None:
        self.keywords = keywords
        self.sharpness = sharpness

    def supports(self, question: Question) -> bool:
        return super().supports(question) and set(question.labels) <= set(self.keywords)  # type: ignore[attr-defined]

    def answer(self, state: Any, questions: Mapping[str, Question]) -> ProviderResult:
        text = render_state(state).lower()
        answers = {}
        for key, question in questions.items():
            hits = {
                label: sum(word in text for word in self.keywords[label])
                for label in question.labels
            }  # type: ignore[attr-defined]
            weights = {label: math.exp(self.sharpness * n) for label, n in hits.items()}
            total = sum(weights.values())
            probabilities = {label: w / total for label, w in weights.items()}
            best = max(probabilities, key=probabilities.__getitem__)
            answers[key] = Answer(value=best, probabilities=probabilities)
        return ProviderResult(answers=answers, model="keywords-v1")


TOPIC = Choice(
    "Which team should handle this?", name="team", options=["billing", "shipping", "other"]
)

classifier = KeywordClassifier(
    {
        "billing": ["charge", "invoice", "refund", "card", "billed"],
        "shipping": ["delivery", "tracking", "package", "arrive", "shipped"],
        "other": [],
    }
)
fallback = LLMDecider(ScriptedLLM(['{"team": "other"}'], model="stand-in"))
memory = MemorySink()
engine = Engine([classifier, fallback], threshold=0.8, tracer=Tracer([memory]))

for text in (
    "I was billed twice and want a refund on my card",
    "Where is my package? Tracking hasn't updated",
    "Do you have a store in Berlin?",
):
    decision = engine.decide(text, TOPIC)
    conf = "n/a" if decision.confidence is None else f"{decision.confidence:.2f}"
    print(f"{decision.value:9} conf {conf:5} via {decision.provider:9} <- {text}")
