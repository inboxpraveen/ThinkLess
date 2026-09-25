"""A customer support agent built on the ThinkLess decision plane.

The agent is ordinary code. It asks typed questions, reads the decisions,
calls tools and writes a reply. Which plane answers each question (rules, a
small model or an LLM) is decided by the engine it is given, so the same agent
runs unchanged in every benchmark mode.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from ...decision import Decision, Status
from ...engine import Engine
from ...tracing.summary import TraceSummary
from .questions import KB_MATCH, TRIAGE
from .world import World, find_duplicate_payment

__all__ = ["ACTIONS", "Outcome", "SupportAgent"]

ACTIONS = (
    "ask_login",
    "ask_order_id",
    "refund_issued",
    "refund_needs_approval",
    "no_duplicate_found",
    "escalate_returns",
    "order_status",
    "order_not_found",
    "subscription_cancelled",
    "no_subscription_found",
    "kb_answer",
    "escalate_support",
    "escalate_technical",
    "escalate_human",
    "escalate_security",
    "escalate_triage",
)

# Replies used when no generation is needed, or when a generated reply fails
# the grounding check. They promise nothing specific.
TEMPLATES = {
    "ask_login": "Thanks for getting in touch. Please sign in to your account and send your message again so we can look into it securely.",
    "ask_order_id": "Thanks for reaching out. Could you reply with your order number? You can find it in your confirmation email or on the Orders page.",
    "default": "Thanks for reaching out. We have your message and a member of our team will follow up with you by email.",
}

SLA = "a specialist replies within 24 hours"

_HANDOFF = "Say a specialist will reply within 24 hours. Do not promise a refund, a fix or any other outcome."

# What the reply should say for each outcome. Small models follow a concrete
# instruction far better than a general policy.
GUIDANCE = {
    "refund_issued": "Confirm the refund amount and when it will appear on the card. Apologize briefly for the double charge.",
    "refund_needs_approval": "Say the duplicate charge is confirmed and a manager must approve the refund. Do not promise a date.",
    "no_duplicate_found": "Say we found only one charge for this order and a billing specialist will double check. Do not promise a refund.",
    "escalate_returns": "Say the returns team will help with the return or refund. " + _HANDOFF,
    "order_status": "Give the order status, carrier, tracking number and expected date from the facts.",
    "order_not_found": "Say we could not find that order on their account and ask them to check the number.",
    "subscription_cancelled": "Confirm the cancellation and the date the membership stays active until.",
    "no_subscription_found": "Say there is no active subscription on the account, so there is nothing to cancel.",
    "kb_answer": "Answer the question using only the article.",
    "escalate_security": "Say the request needs a manual review by our team. Do not promise a refund or any other action.",
}

SYSTEM_PROMPT = (
    "You write replies for the customer support team at {store}, an online home goods store. "
    "Write two to four short sentences in a warm, plain tone. Use only the facts provided and "
    "never invent amounts, dates, order numbers or policies. Reply in the language the customer "
    "wrote in. Do not add a subject line. Sign off as {store} Support."
)

_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?")
_IDENTIFIER = re.compile(r"\b\d{4,6}\b")


class Outcome(BaseModel):
    """What the agent did with one ticket."""

    ticket_id: str
    action: str
    reply: str
    reply_source: str
    grounded: bool = True
    order_id: str | None = None
    queue: str | None = None
    priority: str = "normal"
    refund: dict[str, Any] | None = None
    decisions: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


def _decision_view(decision: Decision) -> dict[str, Any]:
    return {
        "value": decision.level if decision.level else decision.value,
        "status": decision.status.value,
        "provider": decision.provider,
        "confidence": None if decision.confidence is None else round(decision.confidence, 3),
    }


def _canonical(amount: str) -> str:
    value = re.sub(r"[\s$,]", "", amount)
    return value.rstrip("0").rstrip(".") if "." in value else value


def _unsupported_numbers(reply: str, facts: str, message: str) -> list[str]:
    """Money amounts and identifiers in a reply that the facts do not back up.

    Amounts must come from the facts: echoing a figure the customer claimed is
    exactly the mistake this check exists to catch. Order numbers and similar
    identifiers may also come from the customer's own message.
    """
    fact_amounts = {_canonical(m) for m in _MONEY.findall(facts)}
    fact_amounts |= {_canonical(n) for n in re.findall(r"\d+(?:\.\d+)?", facts)}
    known_ids = set(_IDENTIFIER.findall(facts)) | set(_IDENTIFIER.findall(message))
    unsupported = {m.strip() for m in _MONEY.findall(reply) if _canonical(m) not in fact_amounts}
    unsupported |= {n for n in _IDENTIFIER.findall(reply) if n not in known_ids}
    return sorted(unsupported)


class SupportAgent:
    """Handles one ticket at a time.

    Args:
        engine: Supplies the decision plane (``decide``) and the reasoning
            plane (``generate``).
        reply_max_tokens: Budget for a generated reply.
    """

    def __init__(self, engine: Engine, *, reply_max_tokens: int = 160) -> None:
        self.engine = engine
        self.reply_max_tokens = reply_max_tokens

    def handle(
        self, ticket: Mapping[str, Any], world: World | None = None, **run_attributes: Any
    ) -> tuple[Outcome, TraceSummary]:
        """Process a ticket inside a traced run and return the outcome and its roll-up."""
        world = world or World()
        with self.engine.run(
            "support_ticket", input=dict(ticket), ticket_id=ticket.get("id"), **run_attributes
        ) as run:
            outcome = self._handle(ticket, world)
            run.set(
                action=outcome.action, reply_source=outcome.reply_source, grounded=outcome.grounded
            )
        outcome.trace_id = run.trace_id
        return outcome, run.summary()

    # ------------------------------------------------------------------ flow

    def _handle(self, ticket: Mapping[str, Any], world: World) -> Outcome:
        engine = self.engine
        ticket_id = str(ticket.get("id") or "ticket")
        customer_id = ticket.get("customer_id")

        with engine.step("authenticate"):
            signed_in = engine.rule("signed_in", customer_id is not None)
        if not signed_in:
            return self._template(ticket_id, "ask_login", {})

        state = {"subject": ticket.get("subject", ""), "message": ticket["message"]}
        with engine.step("triage"):
            decisions = engine.decide_many(state, TRIAGE, label="triage")
        views = {name: _decision_view(d) for name, d in decisions.items()}

        order_id = self._order_id(decisions["order"])
        urgent = decisions["urgency"].accepted and decisions["urgency"].level == "high"
        priority = "high" if urgent or decisions["churn_risk"].is_(True) else "normal"
        intent = decisions["intent"]
        injection = decisions["injection"]

        with engine.step("act"):
            facts: dict[str, Any] = {}
            queue: str | None
            # Only a confident answer routes to security. An unsure answer that
            # leans yes still fails closed: higher priority, a trace flag, and
            # no money moves without a person approving it.
            suspicious = injection.status is Status.UNCERTAIN and injection.value is True
            if suspicious:
                priority = "high"
                engine.rule("injection_suspected", True, confidence=injection.confidence)
            if injection.is_(True):
                action, queue = "escalate_security", "security"
            elif decisions["wants_human"].is_(True):
                action, queue = "escalate_human", "human"
            elif not intent.accepted:
                action, queue = "escalate_triage", "triage"
            else:
                action, queue, facts = self._route(
                    str(intent.value),
                    ticket,
                    world,
                    order_id,
                    priority,
                    state,
                    allow_auto_refund=not suspicious,
                )

            if queue is not None:
                facts["ticket"] = world.create_ticket(
                    queue, priority, f"{action}: {ticket.get('subject', '')}", str(customer_id)
                )
                facts["next_step"] = SLA

        outcome = self._reply(ticket_id, action, ticket, world, facts)
        outcome.order_id = order_id
        outcome.queue = queue
        outcome.priority = priority
        outcome.refund = facts.get("refund")
        outcome.decisions = views
        return outcome

    def _route(
        self,
        intent: str,
        ticket: Mapping[str, Any],
        world: World,
        order_id: str | None,
        priority: str,
        state: Mapping[str, Any],
        *,
        allow_auto_refund: bool = True,
    ) -> tuple[str, str | None, dict[str, Any]]:
        engine = self.engine
        customer_id = str(ticket["customer_id"])
        facts: dict[str, Any] = {}

        if intent in ("refund_duplicate_charge", "order_status"):
            if not engine.rule("order_id_known", order_id is not None):
                return "ask_order_id", None, facts
            order = world.lookup_order(str(order_id), customer_id)
            if not engine.rule("order_belongs_to_customer", order is not None, order_id=order_id):
                return "order_not_found", None, {"order_id": order_id}
            assert order is not None
            if intent == "order_status":
                keys = (
                    "order_id",
                    "items",
                    "status",
                    "carrier",
                    "tracking",
                    "eta",
                    "delivered_at",
                    "note",
                )
                return "order_status", None, {"order": {k: order[k] for k in keys if k in order}}

            duplicate = find_duplicate_payment(world.list_payments(str(order_id)))
            if not engine.rule("duplicate_charge_found", duplicate is not None, order_id=order_id):
                facts["order"] = {"order_id": order_id, "total": order["total"]}
                facts["finding"] = "only one charge was found for this order"
                return "no_duplicate_found", "billing", facts
            assert duplicate is not None
            limit = world.auto_refund_limit
            within_limit = engine.rule(
                "within_auto_refund_limit",
                duplicate["amount"] <= limit,
                amount=duplicate["amount"],
                limit=limit,
            )
            if within_limit and allow_auto_refund:
                refund = world.refund_payment(
                    duplicate["id"], duplicate["amount"], "duplicate charge"
                )
                facts["refund"] = refund
                facts["refund_timeline"] = "the refund appears on the card in 3 to 5 business days"
                return "refund_issued", None, facts
            facts["duplicate"] = {"amount": duplicate["amount"], "order_id": order_id}
            facts["finding"] = (
                "a duplicate charge was confirmed and needs a manager's approval to refund"
            )
            return "refund_needs_approval", "billing_approval", facts

        if intent == "refund_other":
            if order_id:
                facts["order_id"] = order_id
            return "escalate_returns", "returns", facts

        if intent == "cancel_subscription":
            subscription = world.get_subscription(customer_id)
            if not engine.rule(
                "has_active_subscription", bool(subscription and subscription["status"] == "active")
            ):
                return (
                    "no_subscription_found",
                    None,
                    {"finding": "there is no active subscription on this account"},
                )
            record = world.cancel_subscription(customer_id)
            return "subscription_cancelled", None, {"cancellation": record}

        if intent == "product_question":
            articles = world.search_kb(str(state["message"]))
            if not engine.rule("kb_candidate_found", bool(articles)):
                return "escalate_support", "support", facts
            article = articles[0]
            match = engine.decide(
                {"question": state["message"], "article": f"{article['title']}. {article['body']}"},
                KB_MATCH,
            )
            if match.is_(True):
                return (
                    "kb_answer",
                    None,
                    {"article": {"title": article["title"], "body": article["body"]}},
                )
            return "escalate_support", "support", facts

        if intent == "technical_issue":
            return "escalate_technical", "technical", facts
        return "escalate_support", "support", facts

    # ----------------------------------------------------------------- reply

    def _template(self, ticket_id: str, action: str, facts: Mapping[str, Any]) -> Outcome:
        return Outcome(
            ticket_id=ticket_id,
            action=action,
            reply=TEMPLATES.get(action, TEMPLATES["default"]),
            reply_source="template",
        )

    def _reply(
        self,
        ticket_id: str,
        action: str,
        ticket: Mapping[str, Any],
        world: World,
        facts: dict[str, Any],
    ) -> Outcome:
        if action in ("ask_login", "ask_order_id"):
            return self._template(ticket_id, action, facts)

        engine = self.engine
        with engine.step("reply"):
            fact_lines = "\n".join(
                f"- {key}: {value}" for key, value in {"outcome": action, **facts}.items()
            )
            guidance = GUIDANCE.get(action, _HANDOFF)
            prompt = (
                f"Customer message:\n{ticket['message']}\n\nFacts:\n{fact_lines}\n\n"
                f"What to tell the customer: {guidance}\n\nWrite the reply to the customer."
            )
            completion = engine.generate(
                prompt,
                system=SYSTEM_PROMPT.format(store=world.store),
                max_tokens=self.reply_max_tokens,
                name="reply",
            )
            reply = completion.text.strip()
            unsupported = _unsupported_numbers(reply, fact_lines, str(ticket["message"]))
            grounded = engine.rule(
                "reply_grounded", bool(reply) and not unsupported, unsupported=unsupported
            )
        if not grounded:
            return Outcome(
                ticket_id=ticket_id,
                action=action,
                reply=TEMPLATES["default"],
                reply_source="template_fallback",
                grounded=False,
            )
        return Outcome(ticket_id=ticket_id, action=action, reply=reply, reply_source="llm")

    @staticmethod
    def _order_id(decision: Decision) -> str | None:
        if decision.status is Status.ABSTAINED or not isinstance(decision.value, Mapping):
            return None
        raw = decision.value.get("order_id")
        if raw is None:
            return None
        digits = re.sub(r"\D", "", str(raw))
        return digits if 4 <= len(digits) <= 6 else None
