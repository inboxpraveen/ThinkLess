"""Habits: ThinkLess installed under the LLM calls an agent already makes.

One line puts it under an existing client::

    from openai import OpenAI
    from thinkless import habits

    client = habits.wrap(OpenAI())

From then on ThinkLess records every chat completion, finds the call sites
whose answers are decisions (a route, a yes or no, a label, a value copied
from the input), and trains a small local student for each from the LLM's
own answers. A site is answered locally only after its student has agreed
with the LLM on held-out calls at the target rate, as a lower confidence
bound; a share of local answers is still checked against the LLM, and a
site that stops agreeing goes back to the LLM on its own.

Nothing about the calling code changes: the response has the same shape the
LLM would have returned. ``thinkless habits scan`` reports what a call log
holds before anything is switched on, and ``thinkless habits proxy`` does
the same for clients in any language, through an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from .calls import CallLog, Observation, read_calls
from .gate import GateResult, binomial_lower_bound, binomial_upper_bound, select_threshold
from .learn import LearnOptions, learn, load_plan
from .runtime import MODES, Habits, Ticket, default_habits
from .sites import FieldProfile, SiteProfile, discover
from .student import SiteModel
from .transport import async_http_client, http_client, sdk_http_library, transport_classes, wrap

__all__ = [
    "MODES",
    "CallLog",
    "FieldProfile",
    "GateResult",
    "Habits",
    "LearnOptions",
    "Observation",
    "SiteModel",
    "SiteProfile",
    "Ticket",
    "async_http_client",
    "binomial_lower_bound",
    "binomial_upper_bound",
    "default_habits",
    "discover",
    "http_client",
    "learn",
    "load_plan",
    "read_calls",
    "sdk_http_library",
    "select_threshold",
    "transport_classes",
    "wrap",
]
