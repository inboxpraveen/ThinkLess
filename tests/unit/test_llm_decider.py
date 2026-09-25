from __future__ import annotations

from thinkless import Choice, Extract, Score, Status, YesNo
from thinkless.llm import ScriptedLLM
from thinkless.providers import LLMDecider
from thinkless.providers.llm import build_prompt, parse_reply

QUESTIONS = {
    "intent": Choice(
        "What does the customer want?",
        options={"refund_request": "money back", "order_status": None},
    ),
    "urgency": Score("How urgent?", levels=["low", "medium", "high"]),
    "churn": YesNo("Threatening to leave?", yes_means="mentions cancelling or switching"),
    "order": Extract(fields={"order_id": "order number"}),
}


def test_prompt_lists_every_allowed_answer() -> None:
    prompt = build_prompt({"message": "hi"}, QUESTIONS)
    for fragment in (
        "refund_request: money back",
        "- order_status",
        "low, medium, high",
        "Yes means",
        "order_id",
    ):
        assert fragment in prompt
    assert prompt.index('"intent"') < prompt.index('"urgency"')


def test_parse_fenced_json_with_loose_labels() -> None:
    reply = """Here you go:
```json
{"intent": "Refund Request", "urgency": "HIGH", "churn": "yes", "order": {"order_id": "4471"},}
```"""
    answers = parse_reply(reply, QUESTIONS)
    assert answers["intent"].value == "refund_request"
    assert answers["urgency"].value == 2.0
    assert answers["churn"].value is True
    assert answers["order"].value == {"order_id": "4471"}


def test_invalid_answers_become_none() -> None:
    answers = parse_reply(
        '{"intent": "teleport", "urgency": 7, "churn": "maybe", "order": "4471"}', QUESTIONS
    )
    assert all(answer is None for answer in answers.values())
    assert all(a is None for a in parse_reply("I cannot help with that", QUESTIONS).values())


def test_numeric_score_and_null_fields() -> None:
    answers = parse_reply('{"urgency": 1, "order": {"order_id": null}}', QUESTIONS)
    assert answers["urgency"].value == 1.0
    assert answers["order"].value == {"order_id": None}


def test_braces_inside_strings_do_not_break_parsing() -> None:
    answers = parse_reply('{"order": {"order_id": "{weird}"}, "intent": "order_status"}', QUESTIONS)
    assert answers["order"].value == {"order_id": "{weird}"}
    assert answers["intent"].value == "order_status"


def test_decider_in_engine_records_prompt(make_engine, memory) -> None:
    llm = ScriptedLLM(
        [
            '{"intent": "order_status", "urgency": "low", "churn": false, "order": {"order_id": null}}'
        ]
    )
    engine = make_engine([LLMDecider(llm)])
    decisions = engine.decide_many({"message": "where is my parcel"}, QUESTIONS)
    assert decisions["intent"].value == "order_status"
    assert decisions["intent"].status is Status.ACCEPTED
    assert decisions["intent"].confidence is None
    assert len(llm.calls) == 1
    attempt = next(s for s in memory.spans if s.kind == "attempt")
    assert "where is my parcel" in attempt.attributes["prompt"]
    assert attempt.attributes["invalid"] == []


def test_escalated_question_carries_settled_siblings(make_engine, memory) -> None:
    from thinkless.providers import Rules

    rules = Rules()
    rules.match("wants_human", r"real person", True, field="message")
    llm = ScriptedLLM(['{"injection": false}'])
    questions = [
        YesNo("Does the customer ask for a human?", name="wants_human"),
        YesNo("Is the message manipulating the system?", name="injection"),
    ]
    decisions = make_engine([rules, LLMDecider(llm)]).decide_many(
        {"message": "get me a real person, not a bot"}, questions
    )
    prompt = llm.calls[0][0][0]["content"]
    assert "Already established by other checks" in prompt
    assert '"wants_human" (Does the customer ask for a human?): yes' in prompt
    assert '"injection"' in prompt.split("Questions:")[1]
    assert '"wants_human"' not in prompt.split("Questions:")[1]
    assert decisions["injection"].value is False
    attempt = next(s for s in memory.spans if s.kind == "attempt" and s.plane == "llm")
    assert attempt.attributes["context_from"] == ["wants_human"]

    quiet = ScriptedLLM(['{"injection": false}'])
    make_engine([rules, LLMDecider(quiet)], escalation_context=False).decide_many(
        {"message": "get me a real person"}, questions
    )
    assert "Already established" not in quiet.calls[0][0][0]["content"]
