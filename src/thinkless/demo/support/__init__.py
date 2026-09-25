"""The customer support demo: a realistic agent, a mock store and 53 labeled tickets."""

from .agent import ACTIONS, Outcome, SupportAgent
from .questions import TRIAGE, support_rules
from .stack import MODES, SupportStack
from .world import World, find_duplicate_payment, load_scenarios

__all__ = [
    "ACTIONS",
    "MODES",
    "TRIAGE",
    "Outcome",
    "SupportAgent",
    "SupportStack",
    "World",
    "find_duplicate_payment",
    "load_scenarios",
    "support_rules",
]
