"""Report renderers: terminal (Rich), JSON, and Markdown."""

from __future__ import annotations

from cuda_doctor.reporters.base import Reporter, ReportInputs
from cuda_doctor.reporters.json_report import SCHEMA_VERSION, JsonReporter
from cuda_doctor.reporters.markdown import MarkdownReporter
from cuda_doctor.reporters.terminal import TerminalReporter

__all__ = [
    "SCHEMA_VERSION",
    "JsonReporter",
    "MarkdownReporter",
    "ReportInputs",
    "Reporter",
    "TerminalReporter",
]
