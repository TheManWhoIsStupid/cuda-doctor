"""CLI behavior tests (hermetic: collection is monkeypatched)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest
from tests.factories import base_snapshot
from typer.testing import CliRunner

from cuda_doctor import __main__
from cuda_doctor.cli import app

runner = CliRunner()


@pytest.fixture()
def fake_collect(monkeypatch):
    """Point the CLI at a deterministic snapshot instead of the real machine."""

    def _install(snapshot):
        monkeypatch.setattr("cuda_doctor.cli._collect", lambda: snapshot)

    return _install


class TestVersionAndHelp:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "cuda-doctor 0.1.2" in result.output

    def test_help_lists_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for word in ("diagnose", "info"):
            assert word in result.output


class TestDiagnose:
    def test_bare_invocation_runs_diagnose(self, fake_collect):
        fake_collect(base_snapshot())
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "CUDA Doctor" in result.output
        assert "HEALTHY" in result.output

    def test_terminal_report_sections(self, fake_collect):
        fake_collect(base_snapshot())
        result = runner.invoke(app, ["diagnose"])
        assert result.exit_code == 0
        assert "NVIDIA Driver" in result.output
        assert "Potential Issues" in result.output

    def test_exit_zero_with_findings(self, fake_collect):
        # Findings never turn into a non-zero exit (flutter-doctor style).
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        fake_collect(snapshot)
        result = runner.invoke(app, ["diagnose"])
        assert result.exit_code == 0
        assert "DEGRADED" in result.output

    def test_json_format(self, fake_collect):
        fake_collect(base_snapshot())
        result = runner.invoke(app, ["diagnose", "--format", "json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["schema_version"] == 1
        assert payload["summary"]["status"] == "HEALTHY"

    def test_markdown_format(self, fake_collect):
        fake_collect(base_snapshot())
        result = runner.invoke(app, ["diagnose", "--format", "markdown"])
        assert result.exit_code == 0
        assert "# CUDA Doctor Report" in result.output
        assert "## Summary" in result.output

    def test_unwritable_output_path_exits_cleanly(self, fake_collect, tmp_path):
        # A bad --output must degrade to a friendly message + exit 2,
        # never a traceback (safe failure over crashes).
        fake_collect(base_snapshot())
        missing_dir = tmp_path / "no" / "such" / "dir" / "report.md"
        result = runner.invoke(
            app, ["diagnose", "--format", "markdown", "--output", str(missing_dir)]
        )
        assert result.exit_code == 2
        assert "Internal error while producing the report" in result.output
        # A traceback would surface as exit code 1, not this clean exit 2.

    def test_output_writes_file(self, fake_collect, tmp_path):
        fake_collect(base_snapshot())
        target = tmp_path / "report.md"
        result = runner.invoke(app, ["diagnose", "--format", "markdown", "--output", str(target)])
        assert result.exit_code == 0
        assert "Report written to" in result.output
        assert target.read_text(encoding="utf-8").startswith("# CUDA Doctor Report")

    def test_verbose_shows_info_notes(self, fake_collect):
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, cuda_home=None, cuda_path=None)
        fake_collect(snapshot)
        terse = runner.invoke(app, ["diagnose"])
        verbose = runner.invoke(app, ["diagnose", "--verbose"])
        assert "CUDA002" not in terse.output
        assert "CUDA002" in verbose.output
        assert "--verbose" in terse.output  # hint for hidden notes

    def test_internal_fatal_exits_two(self, monkeypatch):
        def explode():
            raise RuntimeError("unrecoverable")

        monkeypatch.setattr("cuda_doctor.cli._collect", explode)
        result = runner.invoke(app, ["diagnose"])
        assert result.exit_code == 2
        assert "Internal error" in result.output


class TestInfo:
    def test_info_output(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "cuda-doctor 0.1.2" in result.output
        assert "Diagnostic checks: 24" in result.output
        assert "Read-only" in result.output


class TestModuleInvocation:
    def test_python_dash_m_matches_entry_point(self, fake_collect):
        fake_collect(base_snapshot())
        proc = subprocess.run(
            [sys.executable, "-m", "cuda_doctor", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0
        assert "cuda-doctor 0.1.2" in proc.stdout

    def test_main_module_exports_main(self):
        assert callable(__main__.main)
