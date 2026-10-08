"""System information collector."""

from __future__ import annotations

import platform
import sys

from cuda_doctor.core.models import SystemInfo
from cuda_doctor.utils.platform import current_platform


class SystemCollector:
    """Collects operating system and Python runtime facts.

    Deliberately excludes hostname and username: they are personal data the
    tool does not need.
    """

    def collect(self) -> SystemInfo:
        return SystemInfo(
            os_name=platform.system() or "Unknown",
            os_version=self._os_version(),
            kernel_version=platform.release() or "unknown",
            architecture=platform.machine() or "unknown",
            platform=current_platform(),
            python_version=platform.python_version(),
            python_executable=sys.executable or "unknown",
        )

    def _os_version(self) -> str:
        try:
            if sys.platform.startswith("linux"):
                return self._linux_pretty_name() or platform.version()
            if sys.platform.startswith("win"):
                return f"{platform.release()} ({platform.version()})"
            if sys.platform.startswith("darwin"):
                return platform.mac_ver()[0] or platform.version()
        except Exception:
            pass
        return platform.version() or "unknown"

    @staticmethod
    def _linux_pretty_name() -> str | None:
        try:
            with open("/etc/os-release", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            return None
        return None
