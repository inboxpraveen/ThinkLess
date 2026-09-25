"""ThinkLess inside LangGraph: route edges and fill state with typed decisions.

Two pieces cover most graphs:

* :func:`router` returns a function for ``add_conditional_edges``. It decides
  one question on the latest user message and returns the next node. An
  uncertain decision goes to ``default``, usually the LLM agent node you
  already have.
* :func:`decision_node` returns a node that answers several questions in one
  batch and writes plain results into the state, where later nodes and
  edges read them.

Tool calls can be checked before they run with :func:`thinkless.integrations.gate`
on the tool function.

Example:
    >>> intent = router(engine, INTENT, {"order_status": "tracking"}, default="agent")
    >>> builder.add_conditional_edges(START, intent, intent.destinations)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ..decision import Decision
from ..engine import Engine
from ..questions import Question
from ._messages import last_user_text
from .core import Router

__all__ = ["GraphRouter", "adecision_node", "decision_node", "decisions_update", "router"]


class GraphRouter(Router[str]):
    """A :class:`~thinkless.integrations.Router` that reads decisions already in the state.

    When a :func:`decision_node` upstream has answered the question, the
    router routes on that answer instead of asking again.
    """

    def __init__(self, *args: Any, key: str | None = "decisions", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.key = key
        # LangGraph names a branch after its callable; one name per question
        # keeps several routers on the same source node apart.
        self.__name__ = f"route_{self.question.key}"

    def _stored(self, graph_state: Any) -> Mapping[str, Any] | None:
        if self.key is None or not isinstance(graph_state, Mapping):
            return None
        stored = (graph_state.get(self.key) or {}).get(self.question.key)
        return stored if isinstance(stored, Mapping) and "accepted" in stored else None

    def __call__(self, graph_state: Any) -> str:
        stored = self._stored(graph_state)
        if stored is not None:
            return self.pick_value(stored.get("value"), accepted=bool(stored["accepted"]))
        return super().__call__(graph_state)

    async def aroute(self, graph_state: Any) -> str:
        stored = self._stored(graph_state)
        if stored is not None:
            return self.pick_value(stored.get("value"), accepted=bool(stored["accepted"]))
        return await super().aroute(graph_state)


def router(
    engine: Engine,
    question: Question,
    routes: Mapping[Any, str],
    default: str,
    *,
    state: Callable[[Any], Any] | None = None,
    key: str | None = "decisions",
) -> GraphRouter:
    """A conditional-edge function that routes on one decision.

    Args:
        engine: The engine that answers.
        question: What to decide, for example the intent.
        routes: Answer to node name. ``END`` works as a node name.
        default: The node for uncertain answers and answers with no route.
        state: Builds the question's input from the graph state. Defaults to
            the text of the latest user message in ``state["messages"]``.
        key: Where a :func:`decision_node` stores its results. If the
            question was already answered there, the router uses that answer
            and asks nothing. ``None`` always asks.

    Returns:
        A callable router. Pass ``router.destinations`` as the ``path_map``
        so the graph drawing shows every edge.
    """
    return GraphRouter(engine, question, routes, default, state=state or last_user_text, key=key)


def decisions_update(decisions: Mapping[str, Decision]) -> dict[str, dict[str, Any]]:
    """Plain, checkpoint-safe summaries of decisions, keyed by question name."""
    return {
        name: {
            "value": d.value,
            "status": d.status.value,
            "accepted": d.accepted,
            "confidence": d.confidence,
            "plane": None if d.plane is None else d.plane.value,
            "provider": d.provider,
            "level": d.level,
        }
        for name, d in decisions.items()
    }


def decision_node(
    engine: Engine,
    questions: Mapping[str, Question] | Sequence[Question],
    *,
    state: Callable[[Any], Any] | None = None,
    key: str = "decisions",
) -> Callable[[Any], dict[str, Any]]:
    """A node that answers several questions and writes them to ``state[key]``.

    Each entry holds ``value``, ``status``, ``accepted``, ``confidence``,
    ``plane``, ``provider`` and ``level``. Declare the key in the state
    schema, for example ``decisions: dict[str, dict]``.

    Args:
        engine: The engine that answers.
        questions: The questions, answered in one batch.
        state: Builds the input from the graph state. Defaults to the text of
            the latest user message.
        key: The state key to write.
    """
    extract = state or last_user_text

    def decide(graph_state: Any) -> dict[str, Any]:
        decisions = engine.decide_many(extract(graph_state), questions)
        return {key: decisions_update(decisions)}

    decide.__name__ = f"decide_{key}"
    return decide


def adecision_node(
    engine: Engine,
    questions: Mapping[str, Question] | Sequence[Question],
    *,
    state: Callable[[Any], Any] | None = None,
    key: str = "decisions",
) -> Callable[[Any], Awaitable[dict[str, Any]]]:
    """The async form of :func:`decision_node`, for graphs run with ``ainvoke``.

    The engine runs on a worker thread, so local model inference does not
    block the event loop.
    """
    extract = state or last_user_text

    async def decide(graph_state: Any) -> dict[str, Any]:
        decisions = await engine.adecide_many(extract(graph_state), questions)
        return {key: decisions_update(decisions)}

    decide.__name__ = f"decide_{key}"
    return decide
