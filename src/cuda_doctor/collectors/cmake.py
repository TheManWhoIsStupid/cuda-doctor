"""CMake collector."""

from __future__ import annotations

from cuda_doctor.collectors import probe_tool
from cuda_doctor.core.models import ToolInfo
from cuda_doctor.utils.commands import CommandRunner


class CMakeCollector:
    """Collects the CMake version and executable path."""

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner()

    def collect(self) -> ToolInfo:
        return probe_tool(self.runner, "cmake", ("--version",))
