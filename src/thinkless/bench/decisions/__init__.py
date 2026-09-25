"""The decision benchmark: a neutral test of decision models on public data.

Eight tasks cover the four question kinds on public datasets: support and
assistant intents, out-of-scope detection, five languages, a multi-turn
conversation, jailbreaks, toxicity, rating an assistant's reply, and entity
extraction. Every provider gets the same question and the same frozen rows.
Result files hold raw predictions, and every metric is recomputed from them
by :func:`verify_result`, so a result can be checked without trusting who
produced it.
"""

from .leaderboard import collect, leaderboard_markdown
from .run import BenchmarkResult, Submission, TaskResult, build_submission_provider, run_benchmark
from .score import TaskMetrics, score_task
from .tasks import TaskSpec, benchmark_version, load_rows, load_tasks, question_for
from .verify import Verification, verify_result

__all__ = [
    "BenchmarkResult",
    "Submission",
    "TaskMetrics",
    "TaskResult",
    "TaskSpec",
    "Verification",
    "benchmark_version",
    "build_submission_provider",
    "collect",
    "leaderboard_markdown",
    "load_rows",
    "load_tasks",
    "question_for",
    "run_benchmark",
    "score_task",
    "verify_result",
]
