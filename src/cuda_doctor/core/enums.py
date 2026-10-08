"""Shared enums used across collectors, checks, and reporters."""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    """Severity of a diagnostic issue."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnvironmentStatus(str, Enum):
    """Overall health of the development environment."""

    HEALTHY = "HEALTHY"
    USABLE_WITH_WARNINGS = "USABLE WITH WARNINGS"
    DEGRADED = "DEGRADED"
    BROKEN = "BROKEN"


class Platform(str, Enum):
    """Operating-system platform, as relevant to CUDA development."""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    OTHER = "other"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    TERMINAL = "terminal"
    JSON = "json"
    MARKDOWN = "markdown"
