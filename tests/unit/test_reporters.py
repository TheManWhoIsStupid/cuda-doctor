"""Tests for the three report renderers."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace

import pytest
from rich.console import Console
from tests.factories import base_snapshot

from cuda_doctor.collectors.cuda import VERSION_SOURCE_DIRECT
from cuda_doctor.collectors.runtime_libraries import ORIGIN_LD_LIBRARY_PATH
from cuda_doctor.core.enums import EnvironmentStatus, Severity
from cuda_doctor.core.models import (
    CUDAInstallation,
    CUDASelectorObservation,
    RuntimeLibraryCandidate,
    RuntimeLibraryInventory,
)
from cuda_doctor.diagnosis.engine import DiagnosisEngine, DiagnosisResult
from cuda_doctor.diagnosis.issue import Issue, Summary
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    DiagnosisRelationship,
    DiagnosisRole,
    EvidenceItem,
    ImpactItem,
    RemediationSafety,
    RemediationStep,
)
from cuda_doctor.reporters import (
    SCHEMA_VERSION,
    JsonReporter,
    MarkdownReporter,
    ReportInputs,
    TerminalReporter,
)
from cuda_doctor.reporters.base import suppressed_issue_codes
from cuda_doctor.version import __version__

GENERATED_AT = "2026-10-08T12:00:00+00:00"

DIAGNOSIS_FIELD_NAMES = {
    "code",
    "title",
    "severity",
    "confidence",
    "role",
    "summary",
    "cause_class",
    "evidence",
    "impact",
    "remediation",
    "verification",
    "related_issues",
    "relationships",
    "exclusions",
    "presentation_suppressed_issue_codes",
}


def inputs_for(snapshot) -> ReportInputs:
    return ReportInputs(
        snapshot=snapshot, result=DiagnosisEngine().run(snapshot), generated_at=GENERATED_AT
    )


def broken_inputs() -> ReportInputs:
    """A snapshot with several real issues (driver a generation too old)."""
    snapshot = base_snapshot()
    # CUDA 11-generation driver with a CUDA 12.6 toolkit and a CUDA-12 torch
    # build whose availability probe failed (cuda_available None): DRV002 +
    # TORCH006 (the static verdict — TORCH002 stays silent because CUDA was
    # not *observed* unavailable).
    snapshot.driver = replace(snapshot.driver, version="470.42.01", cuda_version="11.4")
    snapshot.cuda = replace(snapshot.cuda, toolkit_version="12.6")
    snapshot.pytorch = replace(snapshot.pytorch, cuda_available=None, devices=[])
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


# -- v0.2 diagnosis fixtures ------------------------------------------------

def _selector(name: str, root: str, version: str) -> CUDASelectorObservation:
    """A valid, direct, canonical-rooted selector observation."""
    return CUDASelectorObservation(
        name=name,
        raw_value=root,
        resolved_path=f"{root}/bin/nvcc",
        canonical_path=f"{root}/bin/nvcc",
        canonical_root=root,
        toolkit_version=version,
        exists=True,
        valid=True,
        version_source=VERSION_SOURCE_DIRECT,
    )


def _row_c_inventory() -> RuntimeLibraryInventory:
    """The canonical Revision-3 Class-A row-C D004 inventory (LDP shadowing)."""
    return RuntimeLibraryInventory(
        candidates=[
            RuntimeLibraryCandidate(
                family="libcudart",
                path="/opt/inv/libcudart.so.12",
                canonical_path="/opt/inventory/libcudart.so.11.8.89",
                version="12",
                version_source="filename",
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_order=0,
            ),
            RuntimeLibraryCandidate(
                family="libcudart",
                path="/opt/inv2/libcudart.so.12",
                canonical_path="/opt/inventory/libcudart.so.12",
                version="12",
                version_source="filename",
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_order=1,
            ),
        ],
        scan_errors={},
    )


def d001_snapshot():
    """Two participating direct selectors resolving to different toolkits."""
    snapshot = base_snapshot()
    snapshot.cuda = replace(
        snapshot.cuda,
        selector_observations=[
            _selector("cuda_home", "/opt/cuda-11.8", "11.8"),
            _selector("nvcc", "/usr/local/cuda-12.4", "12.4"),
        ],
    )
    return snapshot


def d002_torch004_snapshot():
    """nvcc reference 11.8 vs the torch cu124 build: D002 + TORCH004 (INFO).

    The toolkit itself reports 11.8 so the TORCH004 issue fires from the
    same underlying mismatch D002 explains (truth table §17.9).
    """
    snapshot = base_snapshot()
    snapshot.cuda = replace(
        snapshot.cuda,
        toolkit_version="11.8",
        nvcc_path="/usr/local/cuda-11.8/bin/nvcc",
        cuda_home="/usr/local/cuda-11.8",
        cuda_home_exists=True,
        cuda_home_has_nvcc=True,
        installations=[CUDAInstallation("/usr/local/cuda-11.8", "11.8")],
        selector_observations=[_selector("nvcc", "/usr/local/cuda-11.8", "11.8")],
    )
    return snapshot


def d003_snapshot():
    """CUDA build with cuda_available=False: D003 lands on UNKNOWN (LOW)."""
    snapshot = base_snapshot()
    snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
    return snapshot


def d004_snapshot():
    snapshot = base_snapshot()
    snapshot.runtime_libraries = _row_c_inventory()
    return snapshot


def critical_snapshot():
    """The D004 environment with torch CUDA unavailable: D004 PRIMARY and
    D003 SECONDARY (RUNTIME_LIBRARY_CONFLICT) with explains -> D004."""
    snapshot = d004_snapshot()
    snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
    return snapshot


def make_issue(code: str, severity: Severity = Severity.ERROR) -> Issue:
    return Issue(code=code, severity=severity, title=f"Issue {code}")


def carrier_diagnosis(
    code: str = "D00X",
    role: DiagnosisRole | None = DiagnosisRole.PRIMARY,
    suppressed: tuple[str, ...] = (),
) -> Diagnosis:
    return Diagnosis(
        code=code,
        title=f"Synthetic {code}",
        severity=Severity.WARNING,
        confidence=Confidence.MEDIUM,
        summary=f"Synthetic summary for {code}.",
        role=role,
        presentation_suppressed_issue_codes=list(suppressed),
    )


def synthetic_result(
    diagnoses: list[Diagnosis] | None = None,
    issues: list[Issue] | None = None,
    diagnosis_errors: dict[str, str] | None = None,
) -> DiagnosisResult:
    """A finalized result assembled directly for presentation tests."""
    result = DiagnosisResult()
    result.issues = issues if issues is not None else []
    result.diagnoses = diagnoses if diagnoses is not None else []
    result.summary = Summary.from_issues(result.issues)
    result.diagnosis_errors = diagnosis_errors if diagnosis_errors is not None else {}
    return result


def synthetic_inputs(result: DiagnosisResult) -> ReportInputs:
    return ReportInputs(
        snapshot=base_snapshot(), result=result, generated_at=GENERATED_AT
    )


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
        assert "TORCH006" in text  # torch cu124 vs CUDA 11-generation driver
        assert "DRV002" in text  # toolkit 12.6 vs CUDA 11-generation driver
        assert "→" in text  # recommendation marker
        assert "DEGRADED" in text

    def test_driver_cuda_labeled_reported_not_max(self):
        # The nvidia-smi CUDA version must not be presented as a maximum:
        # minor-version compatibility makes "Max CUDA" misleading.
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        assert "Reported CUDA" in text
        assert "Max CUDA" not in text
        assert "max CUDA" not in text

    def test_gpu_listing(self):
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        assert "NVIDIA H20-3e" in text
        assert "140.4 GiB" in text
        assert "compute 9.0" in text

    def test_torch_section_healthy(self):
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        assert "2.6.0+cu124" in text
        assert "available, 1 device(s)" in text

    def test_same_family_minor_gap_marked_info_not_error(self):
        # Minor-version compatibility: toolkit 12.6 on a driver reporting
        # CUDA 12.2 must carry the info mark, not the error mark.
        snapshot = base_snapshot()
        snapshot.driver = replace(snapshot.driver, version="535.216.01", cuda_version="12.2")
        snapshot.cuda = replace(snapshot.cuda, toolkit_version="12.6")
        text = TerminalReporter().render(inputs_for(snapshot))
        line = next(ln for ln in text.splitlines() if "toolkit 12.6 vs driver" in ln)
        assert "✗" not in line
        assert line.strip().startswith(("i ", "INFO "))

    def test_generation_gap_marked_error(self):
        snapshot = base_snapshot()
        snapshot.driver = replace(snapshot.driver, version="470.42.01", cuda_version="11.4")
        snapshot.cuda = replace(snapshot.cuda, toolkit_version="12.6")
        text = TerminalReporter().render(inputs_for(snapshot))
        line = next(ln for ln in text.splitlines() if "toolkit 12.6 vs driver" in ln)
        assert "✗" in line

    def test_torch_row_green_when_cuda_observed_working(self):
        # Observed runtime success outranks the static comparison: even a
        # generation gap in the numbers must not show the error mark.
        snapshot = base_snapshot()
        snapshot.driver = replace(snapshot.driver, version="470.42.01", cuda_version="11.4")
        snapshot.pytorch = replace(snapshot.pytorch, cuda_version="12.6")
        text = TerminalReporter().render(inputs_for(snapshot))
        line = next(
            ln for ln in text.splitlines() if "torch runtime 12.6 vs driver" in ln
        )
        assert "✗" not in line
        assert ("✓" in line) or ("OK" in line)

    def test_ascii_fallback_when_not_utf8(self):
        class Cp1252File(io.StringIO):
            encoding = "cp1252"  # consoles derive encoding from their file

        console = Console(
            record=True, width=100, file=Cp1252File(), force_terminal=False
        )
        text = TerminalReporter(console).render(broken_inputs())
        assert "ERR" in text
        assert "✗" not in text

    def test_home_paths_redacted(self, fake_home):
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

    def test_snapshot_redacted(self, fake_home):
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
            ld_library_path_entries=[(0, "/home/secretuser/.local/lib"), (1, "/usr/lib")],
        )
        # The raw runtime-library inventory (v0.2 Phase 3) joins the same
        # privacy class: internal state, never serialized into any report.
        snapshot.runtime_libraries = RuntimeLibraryInventory(
            candidates=[
                RuntimeLibraryCandidate(
                    family="libcudart",
                    path="/home/secretuser/.local/lib/libcudart.so.12",
                    version="12",
                    origin="ld_library_path",
                    search_order=0,
                ),
                RuntimeLibraryCandidate(
                    family="libcudnn",
                    path="/secret/diagnosis/libcudnn.so.9",
                    soname="libcudnn.so.9",
                    origin="ldconfig",
                ),
            ],
            scan_errors={"/home/secretuser/.local/stale": "FileNotFoundError: gone"},
        )
        payload = JsonReporter.build(inputs_for(snapshot))
        env = payload["environment"]["environment"]
        assert "path_entries" not in env
        assert "ld_library_path" not in env
        assert "ld_library_path_entries" not in env
        assert "runtime_libraries" not in payload["environment"]
        rendered = JsonReporter().render(inputs_for(snapshot))
        assert ".npm-global" not in rendered
        # The ordered entries are popped before redaction, so neither the raw
        # path nor a "~"-prefixed reconstruction can leak.
        assert ".local/lib" not in rendered
        # Neither do inventory candidate paths, scan-error keys, family
        # listings, or the LD_LIBRARY_PATH roots they would reconstruct.
        assert "libcudart" not in rendered
        assert "libcudnn" not in rendered
        assert ".local/stale" not in rendered
        assert "/secret/diagnosis" not in rendered
        assert "ld_library_path" not in env
        # The CUDA-relevant subsets remain.
        assert "cuda_path_entries" in env

    def test_check_errors_are_redacted(self, fake_home):
        inputs = inputs_for(base_snapshot())
        inputs.result.check_errors["ExplodingCheck"] = (
            "RuntimeError: boom at /home/secretuser/venv/lib/libtorch.so"
        )
        rendered = JsonReporter().render(inputs)
        assert "secretuser" not in rendered
        assert "<user>" in rendered or "~" in rendered

    def test_internal_cuda_probe_error_never_serialized(self):
        # v0.2 pre-Phase-6 follow-up: the bounded probe-failure observation
        # is internal D003 evidence. JSON schema v1 must not gain the field
        # — not even as null in a healthy environment — and the private
        # path inside the captured error must not surface anywhere.
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch,
            cuda_available=None,
            cuda_probe_error="RuntimeError: /home/secretuser/private/path",
        )
        payload = JsonReporter.build(inputs_for(snapshot))
        pytorch = payload["environment"]["pytorch"]
        assert "cuda_probe_error" not in pytorch
        assert "import_error" in pytorch  # sibling bounded error stays public
        rendered = JsonReporter().render(inputs_for(snapshot))
        assert "cuda_probe_error" not in rendered
        assert "secretuser" not in rendered


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

    def test_driver_line_reports_cuda_version_neutrally(self):
        text = MarkdownReporter().render(inputs_for(base_snapshot()))
        assert "reported CUDA 13.0" in text
        assert "max CUDA" not in text

    def test_redacted(self, fake_home):
        snapshot = base_snapshot()
        snapshot.system = replace(
            snapshot.system, python_executable="/home/secretuser/venv/bin/python"
        )
        text = MarkdownReporter().render(inputs_for(snapshot))
        assert "secretuser" not in text


class TestJsonSchemaV2:
    """Schema v2: additive diagnosis fields over an unchanged v1 body."""

    def test_schema_version_is_two(self):
        payload = JsonReporter.build(inputs_for(base_snapshot()))
        assert SCHEMA_VERSION == 2
        assert payload["schema_version"] == 2

    def test_top_level_keys_exact_and_ordered(self):
        payload = JsonReporter.build(inputs_for(critical_snapshot()))
        assert list(payload) == [
            "schema_version",
            "tool",
            "generated_at",
            "environment",
            "issues",
            "diagnoses",
            "summary",
            "check_errors",
            "diagnosis_errors",
        ]

    def test_legacy_v1_keys_keep_shape_and_meaning(self):
        inputs = broken_inputs()
        payload = JsonReporter.build(inputs)
        assert payload["tool"] == {"name": "cuda-doctor", "version": __version__}
        assert payload["generated_at"] == GENERATED_AT
        assert payload["environment"]["driver"]["version"] == "470.42.01"
        codes = [issue["code"] for issue in payload["issues"]]
        assert codes == [issue.code for issue in inputs.result.issues]
        assert payload["summary"] == inputs.result.summary.to_dict()
        assert payload["check_errors"] == {}

    def test_healthy_result_has_empty_diagnosis_layers(self):
        payload = JsonReporter.build(inputs_for(base_snapshot()))
        assert payload["diagnoses"] == []
        assert payload["diagnosis_errors"] == {}

    @pytest.mark.parametrize(
        ("snapshot", "code", "severity", "confidence"),
        [
            (d001_snapshot(), "D001", "WARNING", "high"),
            (d002_torch004_snapshot(), "D002", "WARNING", "high"),
            (d003_snapshot(), "D003", "ERROR", "low"),
            (d004_snapshot(), "D004", "WARNING", "medium"),
        ],
    )
    def test_single_diagnosis_serialization(
        self, snapshot, code, severity, confidence
    ):
        inputs = inputs_for(snapshot)
        payload = JsonReporter.build(inputs)
        diagnoses = payload["diagnoses"]
        assert len(diagnoses) == 1
        dx = diagnoses[0]
        assert dx["code"] == code
        assert dx["role"] == "primary"
        assert dx["severity"] == severity
        assert dx["confidence"] == confidence
        # No reporter-side reclassification: the payload mirrors the
        # finalized objects field by field.
        assert dx == inputs.result.diagnoses[0].to_dict(redact=True)

    def test_multi_diagnosis_critical_case(self):
        inputs = inputs_for(critical_snapshot())
        payload = JsonReporter.build(inputs)
        assert [(d["code"], d["role"]) for d in payload["diagnoses"]] == [
            ("D004", "primary"),
            ("D003", "secondary"),
        ]
        assert payload["diagnoses"][0]["severity"] == "WARNING"
        assert payload["diagnoses"][1]["severity"] == "ERROR"

    def test_every_diagnosis_field_serialized(self):
        payload = JsonReporter.build(inputs_for(critical_snapshot()))
        for dx in payload["diagnoses"]:
            assert set(dx) == DIAGNOSIS_FIELD_NAMES

    def test_relationship_preserved_exactly(self):
        payload = JsonReporter.build(inputs_for(critical_snapshot()))
        d003 = next(d for d in payload["diagnoses"] if d["code"] == "D003")
        assert d003["relationships"] == [{"kind": "explains", "target_code": "D004"}]

    def test_remediation_step_representation(self):
        payload = JsonReporter.build(inputs_for(critical_snapshot()))
        for dx in payload["diagnoses"]:
            for step in dx["remediation"]:
                assert set(step) == {"action", "why", "safety", "command"}
                assert step["safety"] in ("safe", "caution", "destructive")
        # v0.2 rules emit SAFE display-only guidance: commands may be null
        # but the safety classification always serializes.
        d004 = payload["diagnoses"][0]
        assert all(step["safety"] == "safe" for step in d004["remediation"])

    def test_exclusions_serialized_separately_from_evidence(self):
        payload = JsonReporter.build(inputs_for(d003_snapshot()))
        dx = payload["diagnoses"][0]
        assert dx["exclusions"]
        assert dx["exclusions"] != dx["evidence"]
        assert all(isinstance(line, str) for line in dx["exclusions"])

    def test_presentation_suppressed_issue_codes_represented(self):
        payload = JsonReporter.build(inputs_for(d003_snapshot()))
        assert payload["diagnoses"][0]["presentation_suppressed_issue_codes"] == [
            "TORCH002"
        ]

    def test_diagnosis_errors_populated_and_bounded(self):
        result = DiagnosisEngine().run(base_snapshot())
        result.diagnosis_errors["D001"] = "RuntimeError: diagnosis component failed"
        result.diagnosis_errors["resolver"] = "ValueError: diagnosis component failed"
        result.check_errors["ExplodingCheck"] = "boom"
        payload = JsonReporter.build(synthetic_inputs(result))
        assert payload["diagnosis_errors"] == {
            "D001": "RuntimeError: diagnosis component failed",
            "resolver": "ValueError: diagnosis component failed",
        }
        # A separate layer: never merged into check_errors.
        assert payload["check_errors"] == {"ExplodingCheck": "boom"}

    def test_private_material_redacted(self, fake_home):
        secret = "/home/secretuser/private/project"
        diagnosis = Diagnosis(
            code="D00X",
            title="Leaky diagnosis",
            severity=Severity.WARNING,
            confidence=Confidence.HIGH,
            summary=f"conflict under {secret}",
            role=DiagnosisRole.PRIMARY,
            cause_class=f"under {secret}",
            evidence=[EvidenceItem(key="root", value=secret, source="env")],
            impact=[ImpactItem(scope="build", status=f"broken under {secret}")],
            remediation=[
                RemediationStep(
                    action=f"inspect {secret}",
                    why=f"because {secret}",
                    safety=RemediationSafety.CAUTION,
                    command=f"ls {secret}",
                )
            ],
            verification=[f"check {secret}"],
            related_issues=["TORCH002"],
            relationships=[DiagnosisRelationship(kind="explains", target_code="D004")],
            exclusions=[f"nothing healthy in {secret}"],
            presentation_suppressed_issue_codes=["TORCH002"],
        )
        rendered = JsonReporter().render(
            synthetic_inputs(synthetic_result(diagnoses=[diagnosis]))
        )
        assert "secretuser" not in rendered
        assert "~" in rendered or "<user>" in rendered

    def test_deterministic_output(self):
        inputs = inputs_for(critical_snapshot())
        assert JsonReporter().render(inputs) == JsonReporter().render(inputs)

    def test_no_candidate_hint_fields_leak(self):
        payload = JsonReporter.build(inputs_for(critical_snapshot()))
        for dx in payload["diagnoses"]:
            for hint in ("explains_active_failure", "risk_only", "causal_parent_codes"):
                assert hint not in dx

    def test_roleless_diagnosis_never_reaches_json_output(self):
        # Frozen finalized-report boundary: role=None is internal
        # pre-resolver state, not a fifth role. JSON skips the entry
        # exactly like the human reporters — never serialized as
        # "role": null, never given an invented role.
        diagnosis = carrier_diagnosis(role=None, suppressed=("TORCH002",))
        payload = JsonReporter.build(synthetic_inputs(synthetic_result([diagnosis])))
        assert payload["diagnoses"] == []
        assert diagnosis.role is None  # the stored Diagnosis is untouched

    def test_roleless_diagnosis_skipped_finalized_peers_unchanged(self):
        roleless = carrier_diagnosis("D00L", role=None)
        finalized = carrier_diagnosis("D00F", DiagnosisRole.PRIMARY)
        payload = JsonReporter.build(
            synthetic_inputs(synthetic_result([roleless, finalized]))
        )
        assert [dx["code"] for dx in payload["diagnoses"]] == ["D00F"]
        assert payload["diagnoses"][0] == finalized.to_dict(redact=True)


class TestTerminalDiagnoses:
    def test_healthy_output_has_no_diagnoses_section(self):
        text = TerminalReporter().render(inputs_for(base_snapshot()))
        assert "Diagnoses" not in text
        assert "no issues found" in text

    def test_diagnoses_section_sits_before_potential_issues(self):
        text = TerminalReporter().render(inputs_for(critical_snapshot()))
        assert 0 < text.index("Diagnoses") < text.index("Potential Issues")

    def test_primary_diagnosis_first_resolver_order_preserved(self):
        text = TerminalReporter().render(inputs_for(critical_snapshot()))
        assert text.index("CUDA runtime library conflict") < text.index(
            "PyTorch CUDA unavailable"
        )

    def test_role_severity_confidence_metadata_line(self):
        text = TerminalReporter().render(inputs_for(critical_snapshot()))
        assert "PRIMARY · WARNING · medium confidence" in text
        assert "SECONDARY · ERROR · medium confidence" in text

    def test_structured_sections_rendered(self):
        text = TerminalReporter().render(inputs_for(critical_snapshot()))
        for label in ("Evidence", "Impact", "Remediation", "Verification", "Exclusions"):
            assert label in text
        assert "→ " in text  # remediation marker
        assert "Related issues: TORCH002" in text

    def test_relationship_rendered_without_reversal(self):
        text = TerminalReporter().render(inputs_for(critical_snapshot()))
        assert "explains -> D004" in text

    def test_cause_class_rendered_where_present(self):
        text = TerminalReporter().render(inputs_for(critical_snapshot()))
        assert "Cause: RUNTIME_LIBRARY_CONFLICT" in text

    def test_remediation_command_and_safety_label(self):
        diagnosis = Diagnosis(
            code="D00X",
            title="Commanded diagnosis",
            severity=Severity.WARNING,
            confidence=Confidence.MEDIUM,
            summary="Synthetic.",
            role=DiagnosisRole.PRIMARY,
            remediation=[
                RemediationStep(
                    action="Review the change",
                    why="Careful review first",
                    safety=RemediationSafety.DESTRUCTIVE,
                    command="echo review before changing anything",
                )
            ],
        )
        text = TerminalReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "(destructive)" in text  # safety label never dropped
        assert "$ echo review before changing anything" in text

    def test_unknown_d003_wording_is_conservative(self):
        text = TerminalReporter().render(inputs_for(d003_snapshot()))
        assert "Unable to determine the root cause" in text
        assert "PRIMARY · ERROR · low confidence" in text
        assert "Cause: UNKNOWN" in text

    def test_all_four_roles_distinguished_in_input_order(self):
        diagnoses = [
            carrier_diagnosis("D00P", DiagnosisRole.POSSIBLE),
            carrier_diagnosis("D00C", DiagnosisRole.CONTRIBUTING),
            carrier_diagnosis("D00S", DiagnosisRole.SECONDARY),
            carrier_diagnosis("D00R", DiagnosisRole.PRIMARY),
        ]
        text = TerminalReporter().render(synthetic_inputs(synthetic_result(diagnoses)))
        positions = [text.index(f"Synthetic {d.code}") for d in diagnoses]
        # No reporter-side sorting: the resolver-provided order is king.
        assert positions == sorted(positions)
        for role in ("PRIMARY", "SECONDARY", "CONTRIBUTING", "POSSIBLE"):
            assert role in text

    def test_info_severity_primary_still_rendered_non_verbose(self):
        diagnosis = carrier_diagnosis("D00I", DiagnosisRole.PRIMARY)
        diagnosis = replace(diagnosis, severity=Severity.INFO)
        text = TerminalReporter(verbose=False).render(
            synthetic_inputs(synthetic_result([diagnosis]))
        )
        assert "D00I" in text and "PRIMARY" in text

    def test_roleless_diagnosis_skipped_not_invented(self):
        diagnosis = carrier_diagnosis(role=None)
        text = TerminalReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "D00X" not in text
        assert "Diagnoses" not in text

    def test_torch002_suppressed_from_issue_list_only(self):
        inputs = inputs_for(critical_snapshot())
        text = TerminalReporter().render(inputs)
        # Exactly one presentation: the diagnosis's related-issues line.
        assert text.count("TORCH002") == 1
        issues_block = text[text.index("Potential Issues"):text.index("Summary")]
        assert "TORCH002" not in issues_block
        assert "no issues found" in issues_block
        # The stored layer is untouched by rendering.
        assert [issue.code for issue in inputs.result.issues] == ["TORCH002"]
        assert inputs.result.summary.errors == 1

    def test_torch004_suppressed_when_d002_rendered(self):
        inputs = inputs_for(d002_torch004_snapshot())
        for verbose in (True, False):
            text = TerminalReporter(verbose=verbose).render(inputs)
            assert text.count("TORCH004") == 1, verbose  # related-issues line only
            issues_block = text[text.index("Potential Issues"):text.index("Summary")]
            assert "TORCH004" not in issues_block
        assert [i.code for i in inputs.result.issues] == ["TORCH004"]

    def test_suppression_set_is_union_and_duplicate_carriers_inert(self):
        diagnoses = [
            carrier_diagnosis("D00A", DiagnosisRole.PRIMARY, ("TORCH002", "GHOST999")),
            carrier_diagnosis("D00B", DiagnosisRole.SECONDARY, ("TORCH002",)),
        ]
        issues = [make_issue("TORCH002"), make_issue("OTHER001", Severity.WARNING)]
        result = synthetic_result(diagnoses, issues)
        text = TerminalReporter().render(synthetic_inputs(result))
        assert "TORCH002" not in text[text.index("Potential Issues"):]
        assert "OTHER001" in text  # unrelated issues are never suppressed
        # Absent issue codes cause no error; carriers stay unmutated.
        assert diagnoses[0].presentation_suppressed_issue_codes == [
            "TORCH002",
            "GHOST999",
        ]

    def test_suppression_requires_a_rendered_carrier(self):
        diagnosis = carrier_diagnosis(role=None, suppressed=("TORCH002",))
        result = synthetic_result([diagnosis], [make_issue("TORCH002")])
        text = TerminalReporter().render(synthetic_inputs(result))
        assert "TORCH002" in text[text.index("Potential Issues"):]

    def test_diagnosis_errors_verbose_only(self):
        result = synthetic_result(
            diagnosis_errors={"D001": "RuntimeError: diagnosis component failed"}
        )
        result.check_errors["ExplodingCheck"] = "boom"
        terse = TerminalReporter(verbose=False).render(synthetic_inputs(result))
        assert "diagnosis component failed" not in terse
        verbose = TerminalReporter(verbose=True).render(synthetic_inputs(result))
        assert "Internal Diagnostics" in verbose
        assert "D001" in verbose
        assert "diagnosis component failed" in verbose
        assert "ExplodingCheck" in verbose  # check layer stays distinct but co-shown

    def test_private_diagnosis_title_redacted(self, fake_home):
        # Maintainer-review blocker regression: the TITLE itself carries
        # the private path — the presentation boundary must redact it
        # like every other free-form diagnosis string.
        secret = "/home/secretuser/private/project"
        diagnosis = Diagnosis(
            code="D00X",
            title=f"Conflict under {secret}",
            severity=Severity.WARNING,
            confidence=Confidence.HIGH,
            summary="Synthetic.",
            role=DiagnosisRole.PRIMARY,
        )
        text = TerminalReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "secretuser" not in text
        assert "D00X" in text  # redaction never hides the diagnosis itself
        assert "~" in text or "<user>" in text

    def test_private_diagnosis_text_redacted(self, fake_home):
        secret = "/home/secretuser/private/project"
        diagnosis = Diagnosis(
            code="D00X",
            title="Leaky",
            severity=Severity.WARNING,
            confidence=Confidence.HIGH,
            summary=f"conflict under {secret}",
            role=DiagnosisRole.PRIMARY,
            evidence=[EvidenceItem(key="root", value=secret, source="env")],
            remediation=[
                RemediationStep(
                    action=f"inspect {secret}",
                    why="why",
                    safety=RemediationSafety.SAFE,
                    command=f"ls {secret}",
                )
            ],
            verification=[f"check {secret}"],
            exclusions=[f"nothing healthy in {secret}"],
        )
        text = TerminalReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "secretuser" not in text

    def test_render_is_pure_with_respect_to_the_result(self):
        inputs = inputs_for(critical_snapshot())
        before = copy.deepcopy(inputs.result)
        TerminalReporter(verbose=True).render(inputs)
        assert inputs.result == before


class TestMarkdownDiagnoses:
    def test_healthy_output_has_no_diagnoses_section(self):
        text = MarkdownReporter().render(inputs_for(base_snapshot()))
        assert "Diagnoses" not in text
        assert "No issues found" in text

    def test_section_hierarchy_and_order(self):
        text = MarkdownReporter().render(inputs_for(critical_snapshot()))
        assert (
            text.index("## Environment Overview")
            < text.index("## Diagnoses")
            < text.index("## Potential Issues")
            < text.index("## Summary")
        )
        assert text.index("`D004` — CUDA runtime library conflict") < text.index(
            "`D003` — PyTorch CUDA unavailable"
        )

    def test_role_headings(self):
        text = MarkdownReporter().render(inputs_for(critical_snapshot()))
        assert "### PRIMARY · `D004` — CUDA runtime library conflict" in text
        assert "### SECONDARY · `D003` — PyTorch CUDA unavailable" in text

    def test_structured_content(self):
        text = MarkdownReporter().render(inputs_for(critical_snapshot()))
        for header in (
            "**Evidence**",
            "**Impact**",
            "**Remediation**",
            "**Verification**",
            "**Related issues**",
            "**Relationships**",
            "**Exclusions**",
        ):
            assert header in text
        assert "confidence: medium" in text
        assert "cause: `RUNTIME_LIBRARY_CONFLICT`" in text
        assert "`explains` -> `D004`" in text

    def test_remediation_safety_label_and_command(self):
        diagnosis = Diagnosis(
            code="D00X",
            title="Commanded",
            severity=Severity.WARNING,
            confidence=Confidence.MEDIUM,
            summary="Synthetic.",
            role=DiagnosisRole.PRIMARY,
            remediation=[
                RemediationStep(
                    action="Review the change",
                    why="Careful review first",
                    safety=RemediationSafety.CAUTION,
                    command="echo review before changing anything",
                )
            ],
        )
        text = MarkdownReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "*(caution)*" in text
        assert "Command: `echo review before changing anything`" in text

    def test_unknown_d003_wording(self):
        text = MarkdownReporter().render(inputs_for(d003_snapshot()))
        assert "Unable to determine the root cause" in text
        assert "confidence: low" in text

    def test_roles_in_input_order(self):
        diagnoses = [
            carrier_diagnosis("D00P", DiagnosisRole.POSSIBLE),
            carrier_diagnosis("D00C", DiagnosisRole.CONTRIBUTING),
            carrier_diagnosis("D00S", DiagnosisRole.SECONDARY),
            carrier_diagnosis("D00R", DiagnosisRole.PRIMARY),
        ]
        text = MarkdownReporter().render(synthetic_inputs(synthetic_result(diagnoses)))
        positions = [text.index(f"Synthetic {d.code}") for d in diagnoses]
        assert positions == sorted(positions)

    def test_roleless_diagnosis_skipped(self):
        diagnosis = carrier_diagnosis(role=None)
        text = MarkdownReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "D00X" not in text
        assert "## Diagnoses" not in text

    def test_suppression_requires_a_rendered_carrier(self):
        # A skipped role-less diagnosis is not rendered, so its carrier
        # never activates human issue suppression (§25.2) — the twin of
        # the terminal regression.
        diagnosis = carrier_diagnosis(role=None, suppressed=("TORCH002",))
        result = synthetic_result([diagnosis], [make_issue("TORCH002")])
        text = MarkdownReporter().render(synthetic_inputs(result))
        assert "### 🔴 ERROR · `TORCH002`" in text

    def test_torch002_suppressed_from_issue_list_only(self):
        inputs = inputs_for(critical_snapshot())
        text = MarkdownReporter().render(inputs)
        assert text.count("TORCH002") == 1  # the diagnosis related-issues entry
        assert "### 🔴 ERROR · `TORCH002`" not in text
        assert [issue.code for issue in inputs.result.issues] == ["TORCH002"]
        assert "**Status: DEGRADED**" in text  # Summary stays issue-derived

    def test_torch004_suppressed_when_d002_rendered(self):
        inputs = inputs_for(d002_torch004_snapshot())
        text = MarkdownReporter().render(inputs)
        assert text.count("TORCH004") == 1
        assert "### 🔵 INFO · `TORCH004`" not in text
        assert [i.code for i in inputs.result.issues] == ["TORCH004"]

    def test_private_diagnosis_title_redacted(self, fake_home):
        # Maintainer-review blocker regression: the TITLE itself carries
        # the private path — the shareable-output boundary must redact it
        # like every other free-form diagnosis string.
        secret = "/home/secretuser/private/project"
        diagnosis = Diagnosis(
            code="D00X",
            title=f"Conflict under {secret}",
            severity=Severity.WARNING,
            confidence=Confidence.HIGH,
            summary="Synthetic.",
            role=DiagnosisRole.PRIMARY,
        )
        text = MarkdownReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "secretuser" not in text
        assert "`D00X`" in text  # redaction never hides the diagnosis itself
        assert "~" in text or "<user>" in text

    def test_private_diagnosis_text_redacted(self, fake_home):
        secret = "/home/secretuser/private/project"
        diagnosis = Diagnosis(
            code="D00X",
            title="Leaky",
            severity=Severity.WARNING,
            confidence=Confidence.HIGH,
            summary=f"conflict under {secret}",
            role=DiagnosisRole.PRIMARY,
            evidence=[EvidenceItem(key="root", value=secret, source="env")],
            verification=[f"check {secret}"],
            exclusions=[f"nothing healthy in {secret}"],
        )
        text = MarkdownReporter().render(synthetic_inputs(synthetic_result([diagnosis])))
        assert "secretuser" not in text

    def test_render_is_pure(self):
        inputs = inputs_for(critical_snapshot())
        before = copy.deepcopy(inputs.result)
        MarkdownReporter().render(inputs)
        assert inputs.result == before


class TestSuppressionIsPresentationOnly:
    """The frozen invariant trio: issues, carrier, and Summary survive."""

    def test_all_reporters_leave_stored_state_untouched(self):
        inputs = inputs_for(critical_snapshot())
        before = copy.deepcopy(inputs.result)
        TerminalReporter().render(inputs)
        MarkdownReporter().render(inputs)
        JsonReporter().render(inputs)
        assert inputs.result == before
        carrier = inputs.result.diagnoses[1].presentation_suppressed_issue_codes
        assert carrier == ["TORCH002"]
        assert inputs.result.summary == Summary.from_issues(inputs.result.issues)

    def test_json_keeps_suppressed_issue_machine_readable(self):
        inputs = inputs_for(critical_snapshot())
        json_text = JsonReporter().render(inputs)
        payload = json.loads(json_text)
        assert [issue["code"] for issue in payload["issues"]] == ["TORCH002"]
        assert payload["summary"]["errors"] == 1

    def test_suppressed_issue_codes_helper_reads_only_the_carrier(self):
        d003 = carrier_diagnosis("D003", DiagnosisRole.PRIMARY, ("TORCH002",))
        assert suppressed_issue_codes([d003]) == {"TORCH002"}
        # Inference-free: none of the other diagnosis dimensions suppress.
        inference_only = Diagnosis(
            code="D009",
            title="TORCH002 everywhere",
            severity=Severity.ERROR,
            confidence=Confidence.HIGH,
            summary="TORCH002",
            role=DiagnosisRole.PRIMARY,
            related_issues=["TORCH002"],
        )
        assert suppressed_issue_codes([inference_only]) == set()


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
