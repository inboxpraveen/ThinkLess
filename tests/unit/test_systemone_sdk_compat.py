"""The System One wire format, checked against TypeSafe's official Python SDK.

No API key is needed: the SDK's own request and response models are the
reference for what Jev accepts and returns.
"""

from __future__ import annotations

import json

import pytest

from thinkless import Choice, Score, YesNo
from thinkless.providers.wire import from_wire, to_wire

sdk = pytest.importorskip("typesafe_sdk")

QUESTIONS = {
    "team": Choice(
        "Which team should handle this?", options={"billing": "payments", "technical": None}
    ),
    "urgency": Score("How urgent is it?", levels=["low", "medium", "high"]),
    "urgent": YesNo("Does this need attention now?"),
    "outage": YesNo(
        "Is there an outage?",
        yes_means="customers cannot use the service",
        no_means="anything else",
    ),
}


def test_encoded_questions_match_the_sdk() -> None:
    theirs = {
        "team": sdk.Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "payments", "technical": None},
        ),
        "urgency": sdk.Score(instructions="How urgent is it?", criteria=["low", "medium", "high"]),
        "urgent": sdk.Noul(instructions="Does this need attention now?"),
        "outage": sdk.Noul(
            instructions="Is there an outage?",
            criteria={"true": "customers cannot use the service", "false": "anything else"},
        ),
    }
    expected = {key: q.model_dump(exclude_none=True, mode="json") for key, q in theirs.items()}
    assert to_wire(QUESTIONS) == expected


def test_sdk_response_decodes() -> None:
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.88,
                "probabilities": {"billing": 0.94, "technical": 0.06},
            },
            "urgency": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.5,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75},
            },
            "urgent": {"type": "noul", "noul": 0.97},
            "outage": {"type": "noul", "noul": 0.12},
        },
        "usage": {"input_tokens": 296, "output_tokens": 20},
    }
    # Parsed from JSON text, as the SDK parses a real HTTP response.
    validated = sdk.SystemOneResponse.model_validate_json(json.dumps(payload)).model_dump(
        mode="json"
    )
    answers = from_wire(validated["answers"], QUESTIONS)
    assert answers["team"].value == "billing"
    assert answers["urgency"].probabilities == {"low": 0.05, "medium": 0.2, "high": 0.75}
    assert answers["urgent"].value is True
    assert answers["outage"].value is False
