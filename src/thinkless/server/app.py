"""An HTTP decision server: one engine, shared by every agent that needs decisions.

Endpoints:

* ``POST /v1/decide``: ThinkLess questions in, full decisions out.
* ``POST /v1/systemone``: the System One wire format used by Jev, so any
  System One client, including TypeSafe's SDK and ThinkLess's own
  :class:`~thinkless.providers.SystemOne` provider, can use this engine as its
  model.
* ``GET /v1/questions``: the questions registered on the server.
* ``GET /healthz``: liveness, version and the provider cascade.

Needs ``pip install "thinkless[server]"``.
"""

from __future__ import annotations

import hmac
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .._version import __version__
from ..engine import Engine
from ..errors import ConfigurationError
from ..providers.wire import decisions_to_wire, questions_from_wire
from ..questions import Question, question_from_spec

__all__ = ["create_app"]


class DecideRequest(BaseModel):
    """Body of ``POST /v1/decide``.

    ``questions`` is a list of registered question names, or a mapping of
    name to question spec (``{"kind": "choice", "instructions": ...,
    "options": {...}}``) for questions the server does not know.
    """

    state: Any
    questions: list[str] | dict[str, dict[str, Any]]
    deadline_ms: float | None = Field(default=None, gt=0)


class SystemOneRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]]
    model: str | None = None


def _registry(questions: Mapping[str, Question] | Sequence[Question] | None) -> dict[str, Question]:
    if questions is None:
        return {}
    if isinstance(questions, Mapping):
        return dict(questions)
    return {q.key: q for q in questions}


def create_app(
    engine: Engine,
    questions: Mapping[str, Question] | Sequence[Question] | None = None,
    *,
    api_key: str | None = None,
    allow_ad_hoc: bool = True,
    max_questions: int = 64,
    warmup: bool = True,
    model_name: str = "thinkless",
) -> FastAPI:
    """Build the FastAPI app around an engine.

    Args:
        engine: The engine that answers every request.
        questions: Questions clients can ask by name.
        api_key: When set, every ``/v1`` request needs
            ``Authorization: Bearer <api_key>``.
        allow_ad_hoc: Accept question specs in the request as well as
            registered names. Turn it off to limit clients to the registry.
        max_questions: Upper bound on questions per request.
        warmup: Load local models (and warm them on the registered
            questions) at startup rather than on the first request.
        model_name: The ``model`` reported in System One responses.

    Example:
        >>> app = create_app(engine, [INTENT, URGENCY], api_key=os.environ["THINKLESS_API_KEY"])
        >>> # uvicorn mymodule:app --workers 1
    """
    registry = _registry(questions)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if warmup:
            engine.warmup(registry or None)
        yield
        engine.close()

    app = FastAPI(
        title="ThinkLess decision server",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
    )

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        expected = f"Bearer {api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    def resolve(requested: list[str] | dict[str, dict[str, Any]]) -> dict[str, Question]:
        if len(requested) == 0:
            raise HTTPException(status_code=422, detail="ask at least one question")
        if len(requested) > max_questions:
            raise HTTPException(
                status_code=422, detail=f"at most {max_questions} questions per request"
            )
        if isinstance(requested, list):
            unknown = [name for name in requested if name not in registry]
            if unknown:
                raise HTTPException(status_code=404, detail=f"unknown questions: {unknown}")
            return {name: registry[name] for name in requested}
        if not allow_ad_hoc:
            raise HTTPException(
                status_code=422, detail="this server only answers registered questions"
            )
        try:
            return {name: question_from_spec(spec, name=name) for name, spec in requested.items()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "providers": [{"name": p.name, "plane": p.plane.value} for p in engine.providers],
            "questions": list(registry),
        }

    @app.get("/v1/questions", dependencies=[Depends(authorize)])
    def list_questions() -> dict[str, Any]:
        return {name: q.spec() for name, q in registry.items()}

    @app.post("/v1/decide", dependencies=[Depends(authorize)])
    def decide(request: DecideRequest) -> dict[str, Any]:
        batch = resolve(request.questions)
        started = time.perf_counter()
        try:
            with engine.run("decide", source="http") as run:
                decisions = engine.decide_many(
                    request.state, batch, deadline_ms=request.deadline_ms
                )
        except ConfigurationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "decisions": {
                name: d.model_dump(mode="json", exclude={"raw", "span_id"})
                for name, d in decisions.items()
            },
            "cost_usd": sum(d.cost_usd for d in decisions.values()),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "trace_id": run.trace_id,
        }

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    def systemone(request: SystemOneRequest) -> dict[str, Any]:
        if not request.questions:
            raise HTTPException(status_code=422, detail="ask at least one question")
        if len(request.questions) > max_questions:
            raise HTTPException(
                status_code=422, detail=f"at most {max_questions} questions per request"
            )
        try:
            batch = questions_from_wire(request.questions)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with engine.run("systemone", source="http"):
            decisions = engine.decide_many(request.state, batch)
        return {
            "model": request.model or model_name,
            "answers": decisions_to_wire(decisions, batch),
            "usage": {
                "input_tokens": sum(d.usage.input_tokens for d in decisions.values()),
                "output_tokens": sum(d.usage.output_tokens for d in decisions.values()),
            },
            # Ignored by System One clients; ThinkLess clients can read it.
            "thinkless": {"decisions": {k: d.summary() for k, d in decisions.items()}},
        }

    return app
