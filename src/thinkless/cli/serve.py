"""``thinkless serve`` and ``thinkless mcp``: run an engine you defined in Python."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable, Mapping
from typing import Annotated, Any

import typer
from rich.console import Console

err = Console(stderr=True)

EngineArg = Annotated[
    str,
    typer.Argument(
        help="Where the engine is, as module:attribute, for example myapp.decisions:engine. "
        "The attribute can also be a function that returns an engine."
    ),
]
QuestionsOption = Annotated[
    str | None,
    typer.Option(
        help="Questions to register, as module:attribute (a list or a mapping of questions). "
        "Default: QUESTIONS in the engine's module, if it exists."
    ),
]


def _load(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise typer.BadParameter(f"expected module:attribute, got {path!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise typer.BadParameter(f"cannot import {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise typer.BadParameter(f"{module_name!r} has no attribute {attribute!r}") from exc


def load_engine_and_questions(engine_path: str, questions_path: str | None) -> tuple[Any, Any]:
    from ..engine import Engine
    from ..questions import Question

    engine = _load(engine_path)
    if callable(engine) and not isinstance(engine, Engine):
        engine = engine()
    if not isinstance(engine, Engine):
        raise typer.BadParameter(f"{engine_path} is not a thinkless.Engine")
    if questions_path:
        questions = _load(questions_path)
    else:
        module = sys.modules[engine_path.partition(":")[0]]
        questions = getattr(module, "QUESTIONS", None)
    if questions is not None:
        items = questions.values() if isinstance(questions, Mapping) else questions
        if not isinstance(items, Iterable) or not all(isinstance(q, Question) for q in items):
            raise typer.BadParameter("questions must be a list or a mapping of Question objects")
    return engine, questions


def serve(
    engine: EngineArg,
    questions: QuestionsOption = None,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8080,
    api_key_env: Annotated[
        str,
        typer.Option(help="Environment variable holding the bearer token clients must send."),
    ] = "THINKLESS_API_KEY",
    registered_only: Annotated[
        bool,
        typer.Option(
            "--registered-only", help="Refuse questions that are not registered on the server."
        ),
    ] = False,
) -> None:
    """Serve an engine over HTTP: /v1/decide, /v1/systemone, /v1/questions, /healthz."""
    try:
        import uvicorn

        from ..server.app import create_app
    except ImportError as exc:
        raise typer.BadParameter('the server needs: pip install "thinkless[server]"') from exc

    built, registered = load_engine_and_questions(engine, questions)
    api_key = os.environ.get(api_key_env) or None
    if api_key is None and host not in ("127.0.0.1", "localhost", "::1"):
        err.print(
            f"[yellow]Serving on {host} without authentication. Set {api_key_env} to require "
            "a bearer token.[/]"
        )
    app = create_app(built, registered, api_key=api_key, allow_ad_hoc=not registered_only)
    uvicorn.run(app, host=host, port=port)


def mcp(
    engine: EngineArg,
    questions: QuestionsOption = None,
    transport: Annotated[
        str, typer.Option(help="stdio (for desktop and IDE clients) or streamable-http.")
    ] = "stdio",
) -> None:
    """Serve an engine's questions as MCP tools, one decide_<question> tool each."""
    from ..server.mcp import create_mcp_server

    built, registered = load_engine_and_questions(engine, questions)
    if not registered:
        raise typer.BadParameter("MCP needs registered questions: pass --questions")
    create_mcp_server(built, registered).run(transport)
