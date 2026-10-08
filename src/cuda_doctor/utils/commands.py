"""Safe subprocess execution abstraction.

Every external tool (nvidia-smi, nvcc, gcc, cl, cmake, ninja, ...) is invoked
through :class:`CommandRunner`. A missing, failing, or hanging command is a
normal diagnostic fact represented in :class:`CommandResult` — it never
propagates as an exception and never crashes the application.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = 15.0

ERROR_NOT_FOUND = "executable-not-found"
ERROR_TIMEOUT = "timeout"
ERROR_OS = "os-error"

# Hide the console window when spawning subprocesses on Windows.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass(frozen=True)
class CommandResult:
    """Structured outcome of one external command invocation."""

    command: tuple[str, ...]
    return_code: int | None
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.return_code == 0

    @property
    def timed_out(self) -> bool:
        return self.error == ERROR_TIMEOUT

    @property
    def display(self) -> str:
        return " ".join(self.command)

    @classmethod
    def not_found(cls, command: Sequence[str]) -> CommandResult:
        return cls(tuple(command), None, "", "", ERROR_NOT_FOUND)


def _to_text(data: object) -> str:
    """Best-effort decode; TimeoutExpired may carry bytes even in text mode."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


class CommandRunner:
    """Runs external commands with timeouts and structured failure handling."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout

    def run(self, command: Sequence[str], timeout: float | None = None) -> CommandResult:
        """Execute ``command`` (no shell) and return a structured result."""
        cmd = tuple(str(part) for part in command)
        if not cmd:
            return CommandResult(cmd, None, "", "", f"{ERROR_OS}:empty-command")
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout if timeout is not None else self.timeout,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                cmd, None, _to_text(exc.stdout), _to_text(exc.stderr), ERROR_TIMEOUT
            )
        except FileNotFoundError:
            return CommandResult.not_found(cmd)
        except PermissionError:
            return CommandResult(cmd, None, "", "", f"{ERROR_OS}:permission-denied")
        except OSError as exc:
            return CommandResult(cmd, None, "", "", f"{ERROR_OS}:{exc.errno or 'unknown'}")
        return CommandResult(
            cmd,
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
            None,
        )
