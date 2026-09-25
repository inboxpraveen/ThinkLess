"""ThinkLess: the decision plane for AI agents.

Rules and small calibrated models make the routine decisions of an agent, a
generative model is reserved for the steps that need thought, and every step
is traced with its latency, confidence, tokens and cost.

Quick start::

    from thinkless import Choice, Engine
    from thinkless.providers import GLiNER, LLMDecider, Rules
    from thinkless.llm import TransformersLLM

    llm = TransformersLLM("Qwen/Qwen3-1.7B")
    engine = Engine([Rules(), GLiNER(), LLMDecider(llm)], llm=llm, threshold=0.8)

    intent = engine.decide(
        "I was charged twice for order #4471, please refund one of them.",
        Choice("What does the customer want?", options={
            "refund": "wants money back",
            "order_status": "asks where an order is",
            "other": "anything else",
        }),
    )
    print(intent.value, intent.confidence, intent.provider)
"""

from ._version import __version__
from .decision import Answer, Attempt, Decision, Plane, Status, Usage
from .engine import Engine, Run
from .errors import ConfigurationError, ThinkLessError
from .limits import SpendLimit, SpendLimitError
from .logs import configure_logging
from .questions import Choice, Extract, Kind, Question, Score, YesNo
from .settings import load_env
from .tracing import ConsoleSink, JSONLSink, MemorySink, Tracer, TraceSummary, summarize, tool

__all__ = [
    "Answer",
    "Attempt",
    "Choice",
    "ConfigurationError",
    "ConsoleSink",
    "Decision",
    "Engine",
    "Extract",
    "JSONLSink",
    "Kind",
    "MemorySink",
    "Plane",
    "Question",
    "Run",
    "Score",
    "SpendLimit",
    "SpendLimitError",
    "Status",
    "ThinkLessError",
    "TraceSummary",
    "Tracer",
    "Usage",
    "YesNo",
    "__version__",
    "configure_logging",
    "load_env",
    "summarize",
    "tool",
]
