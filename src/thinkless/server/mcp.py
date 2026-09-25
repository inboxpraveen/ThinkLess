"""Expose an engine's questions as MCP tools.

Each registered question becomes one tool, ``decide_<question>``, that takes
the text to decide on and returns the decision. An MCP client such as an IDE
assistant or a desktop agent can then make routine decisions for the price of
a rule or a small model, and see how confident the answer is.

Works with the MCP Python SDK 1.x (``FastMCP``) and 2.x (``MCPServer``).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..engine import Engine
from ..errors import ConfigurationError
from ..questions import Choice, Question, Score, YesNo

__all__ = ["create_mcp_server", "tool_description"]


def _server_class() -> Any:
    # 2.x renamed FastMCP to MCPServer; try the new name first.
    for module, attr in (("mcp.server.mcpserver", "MCPServer"), ("mcp.server.fastmcp", "FastMCP")):
        try:
            return getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            continue
    raise ConfigurationError('The MCP SDK is not installed: pip install "thinkless[mcp]"')


def tool_description(question: Question) -> str:
    """What an MCP client reads about a question tool."""
    lines = [question.instructions]
    if isinstance(question, Choice):
        options = [
            f"{label} ({desc})" if desc else label for label, desc in question.options.items()
        ]
        lines.append("Answers: " + ", ".join(options) + ".")
    elif isinstance(question, Score):
        lines.append("Levels, lowest first: " + ", ".join(question.levels) + ".")
    elif isinstance(question, YesNo):
        lines.append("Answers true or false.")
    lines.append(
        "Returns value, status (accepted, uncertain or abstained), confidence and the "
        "provider that answered. Treat anything but an accepted status as unknown."
    )
    return " ".join(lines)


def _tool(engine: Engine, key: str, question: Question) -> Callable[[str], dict[str, Any]]:
    def decide(text: str) -> dict[str, Any]:
        decision = engine.decide(text, question, name=key)
        return {
            "question": key,
            "value": decision.value,
            "status": decision.status.value,
            "confidence": decision.confidence,
            "plane": None if decision.plane is None else decision.plane.value,
            "provider": decision.provider,
            "latency_ms": round(decision.latency_ms, 3),
        }

    decide.__name__ = f"decide_{key}"
    decide.__doc__ = tool_description(question)
    return decide


def create_mcp_server(
    engine: Engine,
    questions: Mapping[str, Question] | Sequence[Question],
    *,
    name: str = "thinkless",
) -> Any:
    """An MCP server with one tool per question.

    Args:
        engine: The engine that answers.
        questions: The questions to expose.
        name: Server name shown to clients.

    Returns:
        The SDK's server object. Call ``.run()`` for stdio, or
        ``.run("streamable-http")`` to serve over HTTP.

    Example:
        >>> create_mcp_server(engine, [INTENT, INJECTION]).run()
    """
    registry = dict(questions) if isinstance(questions, Mapping) else {q.key: q for q in questions}
    if not registry:
        raise ConfigurationError("an MCP server needs at least one question")
    server = _server_class()(
        name=name,
        instructions=(
            "Fast typed decisions from rules and small models, with an LLM only for unsure "
            "cases. Call a decide_ tool instead of reasoning about routine classifications."
        ),
    )
    for key, question in registry.items():
        server.add_tool(
            _tool(engine, key, question),
            name=f"decide_{key}",
            description=tool_description(question),
        )
    return server
