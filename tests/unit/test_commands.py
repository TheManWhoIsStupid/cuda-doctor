"""Tests for the safe command runner (uses real subprocesses, no GPU needed)."""

from __future__ import annotations

import sys

from cuda_doctor.utils.commands import (
    ERROR_NOT_FOUND,
    ERROR_TIMEOUT,
    CommandResult,
    CommandRunner,
)


class TestHappyPaths:
    def test_success(self):
        result = CommandRunner().run([sys.executable, "-c", "print('hello')"])
        assert result.success is True
        assert result.return_code == 0
        assert "hello" in result.stdout

    def test_stderr_captured(self):
        result = CommandRunner().run(
            [sys.executable, "-c", "import sys; print('bad', file=sys.stderr)"]
        )
        assert result.success is True
        assert "bad" in result.stderr

    def test_nonzero_exit_is_not_an_error_state(self):
        result = CommandRunner().run([sys.executable, "-c", "raise SystemExit(3)"])
        assert result.success is False
        assert result.return_code == 3
        assert result.error is None


class TestFailures:
    def test_executable_not_found(self):
        result = CommandRunner().run(["cuda-doctor-definitely-missing-cmd-xyz"])
        assert result.success is False
        assert result.error == ERROR_NOT_FOUND
        assert result.return_code is None

    def test_timeout(self):
        result = CommandRunner().run(
            [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.5
        )
        assert result.timed_out is True
        assert result.error == ERROR_TIMEOUT
        assert result.return_code is None

    def test_empty_command(self):
        result = CommandRunner().run([])
        assert result.success is False
        assert result.error is not None
        assert result.error.startswith("os-error")


class TestCommandResult:
    def test_display(self):
        result = CommandResult(("a", "b c"), 0, "", "")
        assert result.display == "a b c"

    def test_not_found_constructor(self):
        result = CommandResult.not_found(["missing"])
        assert result.error == ERROR_NOT_FOUND
        assert result.command == ("missing",)
