"""Tests for the three report renderers."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest
from rich.console import Console
from tests.factories import base_snapshot

from cuda_doctor.core.enums import EnvironmentStatus, Severity
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.reporters import (
    SCHEMA_VERSION,
    JsonReporter,
    MarkdownReporter,
    ReportInputs,
    TerminalReporter,
)

GENERATED_AT = "2026-10-08T12:00:00+00:00"


def inputs_for(snapshot) -> ReportInputs:
    return ReportInputs(
        snapshot=snapshot, result=DiagnosisEngine().run(snapshot), generated_at=GENERATED_AT
    )


def broken_inputs() -> ReportInputs:
    """A snapshot with several real issues (import failure + CPU build path)."""
    snapshot = base_snapshot()
    snapshot.driver = replace(snapshot.driver, version="535.104.05", cuda_version="12.2")
    snapshot.cuda = replace(snapshot.cuda, toolkit_version="12.6")
    return inputs_for(snapshot)


def torch_import_error_inputs() -> ReportInputs:
    snapshot = base_snapshot()
    snapshot.pytorch = replace(
        snapshot.pytorch,
        installed=True,
        import_error="RuntimeError: libcuda.so.1: cannot open shared object file",
        is_cuda_build=None,
        cuda_available=None,
        device_count=None,
        devices=[],
    )
    return inputs_for(snapshot)


class TestTerminalReporter:
    def test_renders_all_sections(self):
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        for section in (
            "System",
            "GPU",
            "NVIDIA Driver",
            "CUDA Toolkit",
            "Python",
            "PyTorch",
            "C++ Toolchain",
            "Environment",
            "Compatibility",
            "Potential Issues",
            "Summary",
        ):
            assert section in text, section

    def test_healthy_summary_line(self):
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        assert "HEALTHY" in text
        assert "no issues found" in text

    def test_issues_rendered_with_codes_and_recommendations(self):
        text = TerminalReporter().render(broken_inputs())
        assert "TORCH006" in text  # torch cu124 > driver max 12.2
        assert "DRV002" in text  # toolkit 12.6 > driver max 12.2
        assert "→" in text  # recommendation marker
        assert "DEGRADED" in text

    def test_gpu_listing(self):
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        assert "NVIDIA H20-3e" in text
        assert "140.4 GiB" in text
        assert "compute 9.0" in text

    def test_torch_section_healthy(self):
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        assert "2.6.0+cu124" in text
        assert "available, 1 device(s)" in text

    def test_ascii_fallback_when_not_utf8(self):
        class Cp1252File(io.StringIO):
            encoding = "cp1252"  # consoles derive encoding from their file

        console = Console(
            record=True, width=100, file=Cp1252File(), force_terminal=False
        )
        text = TerminalReporter(console).render(broken_inputs())
        assert "ERR" in text
        assert "✗" not in text

    def test_home_paths_redacted(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/secretuser")
        snapshot = base_snapshot()
        snapshot.system = replace(
            snapshot.system, python_executable="/home/secretuser/venv/bin/python"
        )
        text = TerminalReporter().render(inputs_for(snapshot))
        assert "secretuser" not in text

    def test_torch_import_error_rendered(self):
        text = TerminalReporter().render(torch_import_error_inputs())
        assert "installed but cannot be imported" in text
        assert "Import error" in text
        assert "libcuda.so.1" in text
        assert "TORCH005" in text  # issue section picks it up too

    def test_internal_errors_surfaced(self):
        snapshot = base_snapshot()
        snapshot.collection_errors = {"pytorch": "Boom: broken"}
        text = TerminalReporter().render(inputs_for(snapshot))
        assert "Internal Diagnostics" in text
        assert "pytorch" in text


class TestJsonReporter:
    def test_valid_json_with_schema(self):
        payload = json.loads(JsonReporter().render(inputs_for(base_snapshot())))
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["tool"]["name"] == "cuda-doctor"
        assert payload["generated_at"] == GENERATED_AT

    def test_environment_section_structure(self):
        payload = JsonReporter.build(inputs_for(base_snapshot()))
        env = payload["environment"]
        assert env["driver"]["version"] == "580.126.09"
        assert env["cuda"]["toolkit_version"] == "12.4"
        assert env["gpus"][0]["name"] == "NVIDIA H20-3e"

    def test_issues_and_summary(self):
        payload = JsonReporter.build(broken_inputs())
        codes = [issue["code"] for issue in payload["issues"]]
        assert "TORCH006" in codes and "DRV002" in codes
        assert payload["summary"]["status"] == "DEGRADED"
        assert payload["summary"]["errors"] >= 2

    def test_snapshot_redacted(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/secretuser")
        snapshot = base_snapshot()
        snapshot.system = replace(
            snapshot.system, python_executable="/home/secretuser/venv/bin/python"
        )
        rendered = JsonReporter().render(inputs_for(snapshot))
        assert "secretuser" not in rendered
        assert "~" in rendered

    def test_check_errors_included(self):
        inputs = inputs_for(base_snapshot())
        inputs.result.check_errors["compatibility-data"] = "unreadable"
        payload = JsonReporter.build(inputs)
        assert payload["check_errors"] == {"compatibility-data": "unreadable"}

    def test_full_path_and_ld_library_path_never_dumped(self):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            path_entries=[(0, "/home/secretuser/.npm-global/bin"), (1, "/usr/bin")],
            ld_library_path="/home/secretuser/.local/lib:/usr/lib",
        )
        payload = JsonReporter.build(inputs_for(snapshot))
        env = payload["environment"]["environment"]
        assert "path_entries" not in env
        assert "ld_library_path" not in env
        assert ".npm-global" not in JsonReporter().render(inputs_for(snapshot))
        # The CUDA-relevant subsets remain.
        assert "cuda_path_entries" in env

    def test_check_errors_are_redacted(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/secretuser")
        inputs = inputs_for(base_snapshot())
        inputs.result.check_errors["ExplodingCheck"] = (
            "RuntimeError: boom at /home/secretuser/venv/lib/libtorch.so"
        )
        rendered = JsonReporter().render(inputs)
        assert "secretuser" not in rendered
        assert "<user>" in rendered or "~" in rendered


class TestMarkdownReporter:
    def test_document_structure(self):
        text = MarkdownReporter().render(inputs_for(base_snapshot()))
        assert text.startswith("# CUDA Doctor Report")
        for header in ("## Environment Overview", "## Potential Issues", "## Summary"):
            assert header in text
        assert "No issues found" in text
        assert "HEALTHY" in text

    def test_issue_sections(self):
        text = MarkdownReporter().render(broken_inputs())
        assert "`TORCH006`" in text
        assert "**Recommendations**" in text
        assert "**Evidence**" in text
        assert "DEGRADED" in text

    def test_redacted(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/secretuser")
        snapshot = base_snapshot()
        snapshot.system = replace(
            snapshot.system, python_executable="/home/secretuser/venv/bin/python"
        )
        text = MarkdownReporter().render(inputs_for(snapshot))
        assert "secretuser" not in text


@pytest.mark.parametrize(
    "reporter_cls", [TerminalReporter, JsonReporter, MarkdownReporter]
)
def test_all_reporters_never_crash_on_empty_environment(reporter_cls):
    """Bare-bones snapshot (nothing installed) must still render."""
    snapshot = base_snapshot(
        gpus=[], driver=None, nvidia_smi=None, python=None, cmake=None, ninja=None
    )
    snapshot.cuda = replace(
        snapshot.cuda, nvcc_found=False, nvcc_path=None, toolkit_version=None, installations=[]
    )
    snapshot.pytorch = replace(snapshot.pytorch, installed=False)
    snapshot.compiler = replace(snapshot.compiler, compilers=[])
    snapshot.environment = replace(
        snapshot.environment, variables={}, cuda_path_entries=[], cuda_ld_library_entries=[]
    )
    output = reporter_cls().render(inputs_for(snapshot))
    assert output.strip()
    assert "not installed" in output or "not found" in output or "none detected" in output


def test_status_enum_values_are_strings_in_json():
    payload = JsonReporter.build(inputs_for(base_snapshot()))
    assert isinstance(payload["summary"]["status"], str)
    assert EnvironmentStatus.HEALTHY.value == "HEALTHY"
    assert Severity.INFO.value == "INFO"
