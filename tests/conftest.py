"""Shared test fixtures and helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from cuda_doctor.utils.commands import CommandResult


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def fake_home(monkeypatch):
    """Pin home-directory detection at /home/secretuser on any host.

    Redaction goes through ``redact.default_home``; on Windows that reads
    USERPROFILE (not HOME), so plain ``setenv("HOME", ...)`` would leave
    the fake home unredacted and the assertions host-dependent.
    """
    monkeypatch.setattr(
        "cuda_doctor.utils.redact.default_home", lambda: "/home/secretuser"
    )
    monkeypatch.setenv("HOME", "/home/secretuser")
    monkeypatch.setenv("USERPROFILE", "/home/secretuser")
    return "/home/secretuser"


def load_fixture(fixtures_dir: Path, relative: str) -> str:
    return (fixtures_dir / relative).read_text(encoding="utf-8")


class FakeRunner:
    """Command runner that answers from a lookup table.

    Keys are matched against the command tail (argv without the executable
    path) so tests stay independent of where an executable resolves.
    """

    def __init__(
        self,
        responses: Mapping[Sequence[str], CommandResult] | None = None,
        default: CommandResult | None = None,
    ) -> None:
        self.responses = {tuple(key): value for key, value in (responses or {}).items()}
        self.default = default
        self.calls: list[tuple[str, ...]] = []

    def run(self, command: Sequence[str], timeout: float | None = None) -> CommandResult:
        cmd = tuple(str(part) for part in command)
        self.calls.append(cmd)
        for tail in (cmd[1:], cmd):
            if tail in self.responses:
                return self.responses[tail]
        if self.default is not None:
            return self.default
        return CommandResult(cmd, 1, "", "", "os-error:no-fake-response")


def ok(stdout: str = "", stderr: str = "") -> CommandResult:
    """A successful fake command result."""
    return CommandResult(("<fake>",), 0, stdout, stderr, None)


def failed(stderr: str = "", return_code: int = 1) -> CommandResult:
    """A failing-but-executed fake command result."""
    return CommandResult(("<fake>",), return_code, "", stderr, None)
