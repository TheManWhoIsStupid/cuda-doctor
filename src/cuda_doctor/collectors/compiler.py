"""Host compiler collector.

Linux: gcc / g++ / clang / clang++.
Windows: Visual Studio via vswhere (read-only, no registry) and MSVC ``cl``
when it happens to be on PATH (i.e. inside a developer prompt).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from cuda_doctor.collectors import probe_tool
from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import CompilerInfo, ToolInfo
from cuda_doctor.utils.commands import CommandRunner
from cuda_doctor.utils.platform import current_platform
from cuda_doctor.utils.versions import format_version, parse_version

LINUX_COMPILERS = ("gcc", "g++", "clang", "clang++")
VSWHERE_RELATIVE_PATH = r"Microsoft Visual Studio\Installer\vswhere.exe"


class CompilerCollector:
    """Collects host compiler toolchain facts."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        env: Mapping[str, str] | None = None,
        platform: Platform | None = None,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.env: dict[str, str] = dict(env if env is not None else os.environ)
        self.platform = platform or current_platform()

    def collect(self) -> CompilerInfo:
        if self.platform is Platform.WINDOWS:
            return CompilerInfo(compilers=self._windows_compilers())
        return CompilerInfo(
            compilers=[probe_tool(self.runner, name, ("--version",)) for name in LINUX_COMPILERS]
        )

    def _windows_compilers(self) -> list[ToolInfo]:
        tools: list[ToolInfo] = []
        vswhere = self._vswhere_path()
        if vswhere is not None:
            tools.append(self._probe_visual_studio(vswhere))
        tools.append(probe_tool(self.runner, "cl", ("/?",)))
        return tools

    def _vswhere_path(self) -> str | None:
        program_files_x86 = self.env.get("ProgramFiles(x86)") or self.env.get(
            "PROGRAMFILES(X86)"
        )
        if not program_files_x86:
            return None
        candidate = os.path.join(program_files_x86, VSWHERE_RELATIVE_PATH)
        return candidate if os.path.isfile(candidate) else None

    def _probe_visual_studio(self, vswhere: str) -> ToolInfo:
        result = self.runner.run(
            (vswhere, "-latest", "-products", "*", "-property", "installationVersion")
        )
        version = format_version(parse_version(result.stdout)) if result.success else None
        return ToolInfo(
            name="Visual Studio (vswhere)",
            found=bool(version),
            path=vswhere,
            version=version,
        )
