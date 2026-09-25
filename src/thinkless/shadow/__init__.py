"""Shadow mode: measure what ThinkLess would decide, and save, before it decides anything.

A :class:`Shadow` runs a candidate engine on the inputs an existing system
already handles, logs both answers, and changes nothing the existing system
returns. :func:`build_report` (``thinkless shadow report``) turns the log into
agreement with confidence intervals, projected savings and a verdict per
question; :func:`export_labels` (``thinkless shadow export``) writes the
disagreements as rows to label and calibrate on.
"""

from .compare import agree, comparable
from .report import (
    PlaneAgreement,
    QuestionReport,
    ShadowReport,
    SideStats,
    ThresholdAdvice,
    build_report,
    export_labels,
    read_log,
)
from .runner import SCHEMA_VERSION, Shadow, ShadowLog, ShadowStats, side_from_decision

__all__ = [
    "SCHEMA_VERSION",
    "PlaneAgreement",
    "QuestionReport",
    "Shadow",
    "ShadowLog",
    "ShadowReport",
    "ShadowStats",
    "SideStats",
    "ThresholdAdvice",
    "agree",
    "build_report",
    "comparable",
    "export_labels",
    "read_log",
    "side_from_decision",
]
