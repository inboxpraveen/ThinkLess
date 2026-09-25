from __future__ import annotations

import pytest
from pydantic import ValidationError

from thinkless import Choice, Extract, Kind, Score, YesNo


def test_choice_accepts_list_or_mapping() -> None:
    from_list = Choice("Pick one", options=["a", "b"])
    from_map = Choice("Pick one", options={"a": "first", "b": None})
    assert from_list.options == {"a": None, "b": None}
    assert from_map.labels == ["a", "b"]
    assert from_list.kind is Kind.CHOICE


def test_choice_needs_two_options() -> None:
    with pytest.raises(ValidationError):
        Choice("Pick one", options=["only"])


def test_key_is_derived_from_instructions_or_name() -> None:
    assert (
        Choice("What does the customer want?", options=["a", "b"]).key
        == "what_does_the_customer_want"
    )
    assert Choice("Anything", options=["a", "b"], name="intent").key == "intent"


def test_questions_are_frozen() -> None:
    question = YesNo("Is it urgent?")
    with pytest.raises(ValidationError):
        question.instructions = "changed"  # type: ignore[misc]


def test_score_levels_and_threshold_bounds() -> None:
    score = Score("How urgent?", levels=["low", "medium", "high"])
    assert score.levels == ("low", "medium", "high")
    with pytest.raises(ValidationError):
        Score("How urgent?", levels=["only"])
    with pytest.raises(ValidationError):
        YesNo("Urgent?", threshold=1.5)


def test_extract_required_must_be_declared() -> None:
    extract = Extract(fields={"order_id": "order number"}, required=["order_id"])
    assert extract.key == "extract_order_id"
    assert extract.instructions
    with pytest.raises(ValidationError):
        Extract(fields={"order_id": "order number"}, required=["email"])
    with pytest.raises(ValidationError):
        Extract(fields={})


def test_provider_allowlist() -> None:
    question = YesNo("Approve the refund?", providers=["rules", "jev"])
    assert question.allows("jev")
    assert not question.allows("gliner")
    assert YesNo("Anything?").allows("gliner")


def test_spec_is_json_ready() -> None:
    spec = Choice("Pick", options=["a", "b"], name="pick").spec()
    assert spec["kind"] == "choice"
    assert spec["options"] == {"a": None, "b": None}


def test_instructions_can_be_positional_or_keyword() -> None:
    positional = Choice("Pick one", ["a", "b"], name="pick")
    keyword = Choice(instructions="Pick one", options=["a", "b"], name="pick")
    assert positional == keyword
    assert YesNo(instructions="Urgent?").instructions == "Urgent?"
    assert Score("Level?", ["low", "high"]).levels == ("low", "high")
    assert Extract("Pull the id", fields={"id": "the id"}).instructions == "Pull the id"
    assert Extract(fields={"id": "the id"}).instructions.startswith("Extract")
