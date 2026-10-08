"""Environment collectors.

Collectors gather facts only — no diagnostic policy lives here. All external
interaction goes through the injected :class:`CommandRunner` and environment
mapping so collectors are fully testable without real hardware.
"""

from __future__ import annotations

from cuda_doctor.core.models import ToolInfo
from cuda_doctor.utils.commands import CommandRunner
from cuda_doctor.utils.paths import find_executable
from cuda_doctor.utils.versions import format_version, parse_version

__all__ = ["probe_tool"]


def probe_tool(
    runner: CommandRunner,
    name: str,
    version_args: tuple[str, ...] = ("--version",),
) -> ToolInfo:
    """Locate an executable and ask it for its version.

    ``found`` means the executable exists on PATH, even if the version query
    fails or its output cannot be parsed.
    """
    path = find_executable(name)
    if path is None:
        return ToolInfo(name=name, found=False)
    result = runner.run((path, *version_args))
    version = None
    for text in (result.stdout, result.stderr):
        version = format_version(parse_version(text))
        if version:
            break
    return ToolInfo(name=name, found=True, path=path, version=version)
