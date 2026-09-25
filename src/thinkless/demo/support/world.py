"""An in-memory store with orders, payments and a help center.

Every ticket runs against a fresh copy, so refunds and cancellations from one
ticket never leak into the next. Tool methods are decorated with
:func:`thinkless.tool`, which records them as spans inside a traced run.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from functools import lru_cache
from importlib import resources
from itertools import pairwise
from typing import Any

from ...tracing import tool

__all__ = ["World", "find_duplicate_payment", "load_scenarios"]


@lru_cache(maxsize=1)
def _fixtures() -> dict[str, Any]:
    text = (
        resources.files("thinkless.demo.support")
        .joinpath("data/world.json")
        .read_text(encoding="utf-8")
    )
    data: dict[str, Any] = json.loads(text)
    return data


def load_scenarios() -> list[dict[str, Any]]:
    """The bundled support tickets with their expected outcomes."""
    path = resources.files("thinkless.demo.support").joinpath("data/scenarios.jsonl")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def find_duplicate_payment(
    payments: list[dict[str, Any]], window_minutes: float = 10.0
) -> dict[str, Any] | None:
    """The later of two captured payments with the same amount inside the window."""
    captured = sorted(
        (p for p in payments if p.get("status") == "captured"), key=lambda p: p["captured_at"]
    )
    for earlier, later in pairwise(captured):
        same_amount = abs(earlier["amount"] - later["amount"]) < 0.005
        gap = (
            _parse_time(later["captured_at"]) - _parse_time(earlier["captured_at"])
        ).total_seconds()
        if same_amount and gap <= window_minutes * 60:
            return later
    return None


_WORD = re.compile(r"[a-z0-9']+")


class World:
    """The store behind the support agent."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = copy.deepcopy(data or _fixtures())
        self.refunds: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []
        self.cancellations: list[dict[str, Any]] = []

    @property
    def store(self) -> str:
        return str(self.data["store"])

    @property
    def auto_refund_limit(self) -> float:
        return float(self.data["auto_refund_limit"])

    def customer(self, customer_id: str) -> dict[str, Any] | None:
        customer: dict[str, Any] | None = self.data["customers"].get(customer_id)
        return customer

    @tool
    def lookup_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None:
        """The order, only if it belongs to the customer asking."""
        order = self.data["orders"].get(order_id)
        if order is None or order["customer_id"] != customer_id:
            return None
        return {"order_id": order_id, **order}

    @tool
    def list_payments(self, order_id: str) -> list[dict[str, Any]]:
        return [dict(p) for p in self.data["payments"] if p["order_id"] == order_id]

    @tool
    def refund_payment(self, payment_id: str, amount: float, reason: str) -> dict[str, Any]:
        for payment in self.data["payments"]:
            if payment["id"] == payment_id:
                if payment["status"] != "captured":
                    raise ValueError(f"payment {payment_id} is {payment['status']}")
                payment["status"] = "refunded"
                refund = {
                    "refund_id": f"R-{len(self.refunds) + 1:04d}",
                    "payment_id": payment_id,
                    "amount": amount,
                    "card": payment["card"],
                    "reason": reason,
                }
                self.refunds.append(refund)
                return refund
        raise KeyError(payment_id)

    @tool
    def get_subscription(self, customer_id: str) -> dict[str, Any] | None:
        subscription = self.data["subscriptions"].get(customer_id)
        return dict(subscription) if subscription else None

    @tool
    def cancel_subscription(self, customer_id: str) -> dict[str, Any]:
        subscription = self.data["subscriptions"][customer_id]
        subscription["status"] = "cancelled"
        record = {
            "customer_id": customer_id,
            "plan": subscription["plan"],
            "active_until": subscription["renews_at"],
        }
        self.cancellations.append(record)
        return record

    @tool
    def search_kb(self, query: str, limit: int = 1) -> list[dict[str, Any]]:
        """Keyword search over help articles. Deterministic and cheap on purpose."""
        text = query.lower()
        words = set(_WORD.findall(text))
        scored = []
        for article in self.data["knowledge_base"]:
            score = 0.0
            for keyword in article["keywords"]:
                if " " in keyword:
                    score += 2.0 if keyword in text else 0.0
                elif keyword in words:
                    score += 1.0
            if score > 0:
                scored.append((score, article))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"score": score, **article} for score, article in scored[:limit]]

    @tool
    def create_ticket(
        self, queue: str, priority: str, summary: str, customer_id: str
    ) -> dict[str, Any]:
        ticket = {
            "ticket_id": f"H-{len(self.tickets) + 1:04d}",
            "queue": queue,
            "priority": priority,
            "summary": summary,
            "customer_id": customer_id,
        }
        self.tickets.append(ticket)
        return ticket
