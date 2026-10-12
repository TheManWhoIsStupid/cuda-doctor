"""Tests for the v0.2 diagnosis output models (passive data containers)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, replace

import pytest
from tests.factories import base_snapshot
from typer.testing import CliRunner

from cuda_doctor.cli import app
from cuda_doctor.core.enums import EnvironmentStatus, Severity
from cuda_doctor.diagnosis.engine import DiagnosisEngine, DiagnosisResult
from cuda_doctor.diagnosis.issue import Issue, Summary
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    DiagnosisCandidate,
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

GENERATED_AT = "2026-10-12T12:00:00+00:00"

CLI = CliRunner()

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


def make_diagnosis(**overrides) -> Diagnosis:
    params: dict = {
        "code": "D00X",
        "title": "Example diagnosis",
        "severity": Severity.WARNING,
        "confidence": Confidence.MEDIUM,
        "role": DiagnosisRole.SECONDARY,
        "summary": "Example summary text.",
    }
    params.update(overrides)
    return Diagnosis(**params)


def make_full_diagnosis() -> Diagnosis:
    return Diagnosis(
        code="D00X",
        title="Example conflict",
        severity=Severity.WARNING,
        confidence=Confidence.MEDIUM,
        role=DiagnosisRole.PRIMARY,
        summary="Two selectors disagree.",
        cause_class="selector_conflict",
        evidence=[EvidenceItem(key="cudacxx_version", value="12.4", source="CUDACXX")],
        impact=[ImpactItem(scope="extension_build", status="may compile against 12.4")],
        remediation=[
            RemediationStep(
                action="Unset CUDACXX",
                why="Let nvcc resolve the toolkit",
                safety=RemediationSafety.SAFE,
                command="echo review before changing anything",
            )
        ],
        verification=["cuda-doctor diagnose"],
        related_issues=["CUDA001"],
        relationships=[DiagnosisRelationship(kind="explains", target_code="D004")],
        exclusions=["Driver usable"],
        presentation_suppressed_issue_codes=["TORCH002"],
    )


def inputs_for(snapshot, result) -> ReportInputs:
    return ReportInputs(snapshot=snapshot, result=result, generated_at=GENERATED_AT)


class TestConfidence:
    def test_members_and_values_are_frozen(self):
        assert [(m.name, m.value) for m in Confidence] == [
            ("HIGH", "high"),
            ("MEDIUM", "medium"),
            ("LOW", "low"),
        ]

    def test_no_unknown_member(self):
        assert not hasattr(Confidence, "UNKNOWN")


class TestDiagnosisRole:
    def test_members_and_values_are_frozen(self):
        assert [(m.name, m.value) for m in DiagnosisRole] == [
            ("PRIMARY", "primary"),
            ("SECONDARY", "secondary"),
            ("CONTRIBUTING", "contributing"),
            ("POSSIBLE", "possible"),
        ]


class TestRemediationSafety:
    def test_members_and_values_are_frozen(self):
        assert [(m.name, m.value) for m in RemediationSafety] == [
            ("SAFE", "safe"),
            ("CAUTION", "caution"),
            ("DESTRUCTIVE", "destructive"),
        ]


class TestSeverityReuse:
    def test_severity_enum_unchanged_uppercase(self):
        # Phase 4 reuses the low-level Severity verbatim (uppercase values,
        # same member order) instead of introducing a diagnosis severity.
        assert [(m.name, m.value) for m in Severity] == [
            ("INFO", "INFO"),
            ("WARNING", "WARNING"),
            ("ERROR", "ERROR"),
            ("CRITICAL", "CRITICAL"),
        ]

    def test_no_second_severity_enum_in_models(self):
        from cuda_doctor.diagnosis import models

        assert not hasattr(models, "DiagnosisSeverity")


class TestEvidenceItem:
    def test_fields_exactly_key_value_source(self):
        assert {f.name for f in fields(EvidenceItem)} == {"key", "value", "source"}

    def test_all_fields_are_plain_strings(self):
        # Bounded, presentation-ready evidence: no snapshot, raw collector
        # object, or arbitrary nested dict can live in an EvidenceItem.
        assert {f.type for f in fields(EvidenceItem)} == {"str"}

    def test_frozen(self):
        item = EvidenceItem(key="k", value="v", source="s")
        with pytest.raises(FrozenInstanceError):
            item.value = "other"


class TestImpactItem:
    def test_fields_exactly_scope_status(self):
        assert {f.name for f in fields(ImpactItem)} == {"scope", "status"}

    def test_all_fields_are_plain_strings(self):
        assert {f.type for f in fields(ImpactItem)} == {"str"}

    def test_frozen(self):
        item = ImpactItem(scope="pytorch_runtime", status="unavailable")
        with pytest.raises(FrozenInstanceError):
            item.scope = "other"


class TestRemediationStep:
    def test_fields_exactly_the_frozen_set(self):
        assert {f.name for f in fields(RemediationStep)} == {
            "action",
            "why",
            "safety",
            "command",
        }

    def test_command_defaults_to_none(self):
        step = RemediationStep(action="a", why="w", safety=RemediationSafety.SAFE)
        assert step.command is None

    def test_safety_is_the_dedicated_enum(self):
        step = RemediationStep(
            action="a", why="w", safety=RemediationSafety.DESTRUCTIVE, command="rm -rf old"
        )
        assert step.safety is RemediationSafety.DESTRUCTIVE

    def test_read_only_no_execution_surface(self):
        step = RemediationStep(
            action="a", why="w", safety=RemediationSafety.CAUTION, command="echo inspect first"
        )
        # command is a display string only; the model is inert data.
        assert isinstance(step.command, str)
        assert not callable(step)
        assert not hasattr(step, "execute")
        assert not hasattr(step, "run")


class TestDiagnosisDefaults:
    def test_minimal_construction_safe_defaults(self):
        diagnosis = make_diagnosis()
        assert diagnosis.cause_class is None
        assert diagnosis.evidence == []
        assert diagnosis.impact == []
        assert diagnosis.remediation == []
        assert diagnosis.verification == []
        assert diagnosis.related_issues == []
        assert diagnosis.relationships == []
        assert diagnosis.exclusions == []
        assert diagnosis.presentation_suppressed_issue_codes == []

    def test_no_shared_mutable_state_between_instances(self):
        a = make_diagnosis()
        b = make_diagnosis()
        a.evidence.append(EvidenceItem(key="k", value="v", source="s"))
        a.impact.append(ImpactItem(scope="s", status="t"))
        a.remediation.append(RemediationStep(action="a", why="w", safety=RemediationSafety.SAFE))
        a.relationships.append(DiagnosisRelationship(kind="explains", target_code="D004"))
        a.exclusions.append("driver usable")
        a.presentation_suppressed_issue_codes.append("TORCH002")
        for empty in (
            b.evidence,
            b.impact,
            b.remediation,
            b.relationships,
            b.exclusions,
            b.presentation_suppressed_issue_codes,
        ):
            assert empty == []


class TestExclusions:
    def test_default_empty_and_list_valued(self):
        diagnosis = make_diagnosis()
        assert diagnosis.exclusions == []
        assert isinstance(diagnosis.exclusions, list)

    def test_carries_strings_separate_from_evidence(self):
        diagnosis = make_diagnosis(
            exclusions=["CUDA-enabled PyTorch build confirmed", "GPU independently visible"],
            evidence=[EvidenceItem(key="k", value="v", source="s")],
        )
        assert diagnosis.exclusions == [
            "CUDA-enabled PyTorch build confirmed",
            "GPU independently visible",
        ]
        assert all(isinstance(item, str) for item in diagnosis.exclusions)
        assert diagnosis.exclusions is not diagnosis.evidence


class TestRelationships:
    def test_relationship_fields_exactly_kind_and_target(self):
        assert {f.name for f in fields(DiagnosisRelationship)} == {"kind", "target_code"}

    def test_all_fields_are_plain_strings(self):
        assert {f.type for f in fields(DiagnosisRelationship)} == {"str"}

    def test_relationship_is_frozen(self):
        relationship = DiagnosisRelationship(kind="explains", target_code="D004")
        with pytest.raises(FrozenInstanceError):
            relationship.kind = "related_to"

    def test_diagnosis_relationships_default_empty(self):
        assert make_diagnosis().relationships == []

    def test_no_graph_machinery_on_models(self):
        diagnosis = make_diagnosis()
        for name in ("traverse", "detect_cycles", "select_parent"):
            assert not hasattr(diagnosis, name), name


class TestPresentationSuppressedIssueCodes:
    def test_default_empty(self):
        assert make_diagnosis().presentation_suppressed_issue_codes == []

    def test_carrier_touches_no_issues(self):
        issues = [Issue(code="TORCH002", severity=Severity.ERROR, title="torch.cuda unavailable")]
        diagnosis = make_diagnosis(presentation_suppressed_issue_codes=["TORCH002"])
        assert diagnosis.presentation_suppressed_issue_codes == ["TORCH002"]
        # The carrier removes and mutates nothing: the issue objects and
        # the issue list are exactly as they were.
        assert [issue.code for issue in issues] == ["TORCH002"]
        assert issues[0].severity is Severity.ERROR
        assert issues[0].title == "torch.cuda unavailable"

    def test_carrier_survives_serialization(self):
        diagnosis = make_diagnosis(presentation_suppressed_issue_codes=["TORCH002"])
        assert diagnosis.to_dict()["presentation_suppressed_issue_codes"] == ["TORCH002"]


class TestDiagnosisSerialization:
    def test_to_dict_field_set(self):
        data = make_full_diagnosis().to_dict(redact=False)
        assert set(data) == DIAGNOSIS_FIELD_NAMES
        assert data["cause_class"] == "selector_conflict"
        assert data["evidence"] == [
            {"key": "cudacxx_version", "value": "12.4", "source": "CUDACXX"}
        ]
        assert data["remediation"][0]["command"] == "echo review before changing anything"

    def test_enum_casing_follows_repository_conventions(self):
        data = make_full_diagnosis().to_dict()
        # Reused low-level Severity keeps its uppercase serialization...
        assert data["severity"] == "WARNING"
        # ...while the new v0.2 enums serialize their declared lowercase values.
        assert data["confidence"] == "medium"
        assert data["role"] == "primary"
        assert data["remediation"][0]["safety"] == "safe"

    def test_redacts_home_paths(self, fake_home):
        diagnosis = make_diagnosis(
            summary="Selector resolves under /home/secretuser/cuda.",
            evidence=[
                EvidenceItem(
                    key="resolved", value="/home/secretuser/cuda/bin/nvcc", source="CUDACXX"
                )
            ],
        )
        data = diagnosis.to_dict()
        assert "secretuser" not in str(data)
        assert data["evidence"][0]["value"] == "~/cuda/bin/nvcc"

    def test_redact_false_normalizes_only(self):
        diagnosis = make_diagnosis(
            evidence=[EvidenceItem(key="root", value="/home/alice/cuda", source="CUDA_HOME")],
        )
        data = diagnosis.to_dict(redact=False)
        assert data["evidence"][0]["value"] == "/home/alice/cuda"
        assert data["severity"] == "WARNING"
        assert data["confidence"] == "medium"


class TestIssueSerializationUntouched:
    def test_issue_field_set_unchanged(self):
        assert {f.name for f in fields(Issue)} == {
            "code",
            "severity",
            "title",
            "description",
            "evidence",
            "recommendations",
            "category",
        }

    def test_issue_to_dict_shape_unchanged(self):
        data = Issue(code="T", severity=Severity.WARNING, title="t").to_dict()
        assert data == {
            "code": "T",
            "severity": "WARNING",
            "title": "t",
            "description": "",
            "evidence": [],
            "recommendations": [],
            "category": "general",
        }


class TestPassiveModelGuarantee:
    FORBIDDEN_METHODS = (
        "resolve_role",
        "compute_confidence",
        "suppress_issues",
        "choose_primary",
        "rank",
        "evaluate",
        "remediate",
        "execute",
    )

    def test_diagnosis_has_no_policy_methods(self):
        diagnosis = make_diagnosis()
        for name in self.FORBIDDEN_METHODS:
            assert not callable(getattr(diagnosis, name, None)), name

    def test_diagnosis_has_no_policy_properties(self):
        diagnosis = make_diagnosis()
        for name in (
            "is_primary",
            "effective_severity",
            "effective_confidence",
            "suppressed",
            "causal_priority",
        ):
            assert not hasattr(diagnosis, name), name

    def test_diagnosis_fields_exactly_the_frozen_set(self):
        assert {f.name for f in fields(Diagnosis)} == DIAGNOSIS_FIELD_NAMES


class TestDiagnosisCandidate:
    def test_defaults(self):
        candidate = DiagnosisCandidate(diagnosis=make_diagnosis())
        assert candidate.explains_active_failure is False
        assert candidate.risk_only is False
        assert candidate.causal_parent_codes == ()

    def test_carries_diagnosis_and_hints(self):
        diagnosis = make_diagnosis()
        candidate = DiagnosisCandidate(
            diagnosis=diagnosis,
            explains_active_failure=True,
            causal_parent_codes=("D004",),
        )
        assert candidate.diagnosis is diagnosis
        assert candidate.causal_parent_codes == ("D004",)

    def test_not_part_of_the_public_result(self):
        # Candidates are internal resolver inputs; the public result
        # carries concluded Diagnosis objects only.
        assert "candidates" not in {f.name for f in fields(DiagnosisResult)}


class TestDiagnosisResultIntegration:
    def test_fields_exactly_the_architecture_set(self):
        assert {f.name for f in fields(DiagnosisResult)} == {
            "issues",
            "diagnoses",
            "summary",
            "check_errors",
            "diagnosis_errors",
        }

    def test_legacy_no_argument_construction(self):
        result = DiagnosisResult()
        assert result.issues == []
        assert result.summary.status is EnvironmentStatus.HEALTHY
        assert result.check_errors == {}
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}

    def test_no_shared_mutable_state(self):
        a = DiagnosisResult()
        b = DiagnosisResult()
        a.issues.append(Issue(code="T", severity=Severity.INFO, title="t"))
        a.diagnoses.append(make_diagnosis())
        a.check_errors["X"] = "boom"
        a.diagnosis_errors["Y"] = "boom"
        assert b.issues == []
        assert b.diagnoses == []
        assert b.check_errors == {}
        assert b.diagnosis_errors == {}

    def test_engine_run_healthy_leaves_diagnosis_fields_empty(self):
        result = DiagnosisEngine().run(base_snapshot())
        assert result.issues == []
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}
        assert result.summary.status is EnvironmentStatus.HEALTHY

    def test_engine_run_degraded_legacy_fields_intact(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        result = DiagnosisEngine().run(snapshot)
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}
        assert any(issue.code == "TORCH002" for issue in result.issues)
        assert result.summary.status is EnvironmentStatus.DEGRADED


class TestSummaryCompatibility:
    def test_summary_derived_only_from_issues(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        result = DiagnosisEngine().run(snapshot)
        before = result.summary.to_dict()
        # Even a CRITICAL/HIGH/PRIMARY diagnosis must not move the summary.
        result.diagnoses.append(
            make_diagnosis(
                severity=Severity.CRITICAL,
                confidence=Confidence.HIGH,
                role=DiagnosisRole.PRIMARY,
            )
        )
        assert result.summary.to_dict() == before
        assert result.summary == Summary.from_issues(result.issues)


class TestPublicSurfaceCompatibility:
    """Phase-4 boundary: carrying diagnoses must not change public output."""

    def test_cli_exit_code_and_json_unchanged(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.cli._collect", lambda: base_snapshot())
        real_run = DiagnosisEngine.run

        def run_with_diagnoses(engine_self, snapshot):
            result = real_run(engine_self, snapshot)
            result.diagnoses.append(make_diagnosis(role=DiagnosisRole.PRIMARY))
            result.diagnosis_errors["D00X"] = "RuntimeError: simulated"
            return result

        monkeypatch.setattr(DiagnosisEngine, "run", run_with_diagnoses)
        outcome = CLI.invoke(app, ["diagnose", "--format", "json"])
        assert outcome.exit_code == 0
        payload = json.loads(outcome.output)
        assert payload["schema_version"] == 1
        assert "diagnoses" not in payload
        assert "diagnosis_errors" not in payload
        assert payload["summary"]["status"] == "HEALTHY"

    @pytest.mark.parametrize(
        "reporter_cls", [TerminalReporter, MarkdownReporter, JsonReporter]
    )
    def test_renderers_identical_with_and_without_diagnoses(self, reporter_cls):
        snapshot = base_snapshot()
        plain = DiagnosisEngine().run(snapshot)
        loaded = DiagnosisEngine().run(snapshot)
        loaded.diagnoses.append(make_full_diagnosis())
        loaded.diagnosis_errors["D00X"] = "RuntimeError: simulated"
        without = reporter_cls().render(inputs_for(snapshot, plain))
        with_diagnoses = reporter_cls().render(inputs_for(snapshot, loaded))
        assert with_diagnoses == without

    def test_json_payload_key_set_unchanged(self):
        snapshot = base_snapshot()
        result = DiagnosisEngine().run(snapshot)
        result.diagnoses.append(make_full_diagnosis())
        result.diagnosis_errors["D00X"] = "RuntimeError: simulated"
        payload = JsonReporter.build(inputs_for(snapshot, result))
        assert set(payload) == {
            "schema_version",
            "tool",
            "generated_at",
            "environment",
            "issues",
            "summary",
            "check_errors",
        }
        assert SCHEMA_VERSION == 1
        assert payload["schema_version"] == SCHEMA_VERSION
