"""Use ThinkLess decisions inside existing agent code.

The helpers here work with any framework: :class:`Router` and :func:`route`
send a request down a path chosen by a decision, and :func:`gate` checks a
tool call before it runs. Framework adapters build on them:

* :mod:`thinkless.integrations.langgraph` for LangGraph graphs,
* :mod:`thinkless.integrations.openai_agents` for the OpenAI Agents SDK.
"""

from ._messages import content_text, last_user_text
from .core import Router, ToolBlockedError, gate, route

__all__ = ["Router", "ToolBlockedError", "content_text", "gate", "last_user_text", "route"]
