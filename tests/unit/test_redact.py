"""Tests for privacy redaction."""

from __future__ import annotations

from typing import ClassVar

from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.models import (
    Confidence,
    DiagnosisRole,
    RemediationSafety,
)
from cuda_doctor.utils.redact import redact_text, redact_value

HOME = "/home/alice"


class TestRedactText:
    def test_home_replaced(self):
        assert redact_text("/home/alice/project", HOME) == "~/project"

    def test_exact_home(self):
        assert redact_text("/home/alice", HOME) == "~"

    def test_username_leftover_masked(self):
        assert redact_text("/data2/alice/workspace", HOME) == "/data2/<user>/workspace"

    def test_unrelated_text_unchanged(self):
        assert redact_text("/usr/local/cuda-12.4", HOME) == "/usr/local/cuda-12.4"

    def test_empty(self):
        assert redact_text("", HOME) == ""

    def test_windows_style(self):
        assert redact_text(r"C:\Users\alice\project", r"C:\Users\alice") == r"~\project"

    def test_empty_home_is_noop(self):
        assert redact_text("/home/alice/x", "") == "/home/alice/x"


class TestRedactValue:
    def test_nested_structures(self):
        data = {
            "path": "/home/alice/venv",
            "items": ["/home/alice/a", ("plain", 42)],
            "nested": {"deep": "/home/alice/deep"},
        }
        out = redact_value(data, home=HOME)
        assert out["path"] == "~/venv"
        assert out["items"][0] == "~/a"
        assert out["items"][1] == ["plain", 42]
        assert out["nested"]["deep"] == "~/deep"

    def test_enums_become_values(self):
        assert redact_value(Severity.INFO, home=HOME) == "INFO"

    def test_non_strings_untouched(self):
        assert redact_value(7, home=HOME) == 7
        assert redact_value(None, home=HOME) is None
        assert redact_value(True, home=HOME) is True

    def test_empty_home_normalizes_only(self):
        out = redact_value({"s": "/home/alice/x", "sev": Severity.ERROR}, home="")
        assert out == {"s": "/home/alice/x", "sev": "ERROR"}


class TestV02VariableRedaction:
    def test_new_environment_variable_values_redacted(self):
        # CUDACXX / CONDA_PREFIX carry home paths once collected; an empty
        # CUDA_VISIBLE_DEVICES must survive redaction unchanged.
        out = redact_value(
            {
                "variables": {
                    "CUDACXX": "/home/alice/toolkits/cuda/bin/nvcc",
                    "CONDA_PREFIX": "/home/alice/miniconda3",
                    "CUDA_VISIBLE_DEVICES": "",
                }
            },
            home=HOME,
        )
        assert out["variables"]["CUDACXX"] == "~/toolkits/cuda/bin/nvcc"
        assert out["variables"]["CONDA_PREFIX"] == "~/miniconda3"
        assert out["variables"]["CUDA_VISIBLE_DEVICES"] == ""


class TestRuntimeLibraryInventoryShapedRedaction:
    """Defense in depth: were an inventory-shaped dict ever routed through
    redaction (the snapshot excludes it earlier), home paths inside
    candidates AND scan-error keys would still redact recursively."""

    def test_candidate_paths_and_scan_error_keys_redact(self):
        data = {
            "runtime_libraries": {
                "candidates": [
                    {
                        "family": "libcudart",
                        "path": "/home/alice/.local/lib/libcudart.so.12",
                        "canonical_path": "/home/alice/.local/lib/libcudart.so.12",
                    }
                ],
                "scan_errors": {"/home/alice/.local/stale": "FileNotFoundError: gone"},
            }
        }
        out = redact_value(data, home=HOME)
        inventory = out["runtime_libraries"]
        assert inventory["candidates"][0]["path"] == "~/.local/lib/libcudart.so.12"
        assert inventory["candidates"][0]["canonical_path"] == "~/.local/lib/libcudart.so.12"
        assert list(inventory["scan_errors"]) == ["~/.local/stale"]


class TestDiagnosisShapedRedaction:
    """v0.2 Phase 12: serialized diagnoses carry free-form rule text, so
    every human-authored surface (summaries, evidence values, remediation
    prose and commands, verification, exclusions) must redact home paths
    recursively — this is the dict shape Diagnosis.to_dict feeds in."""

    DIAGNOSIS: ClassVar[dict] = {
        "code": "D00X",
        "title": "Conflict under /home/alice/project",
        "severity": Severity.WARNING,
        "confidence": Confidence.MEDIUM,
        "summary": "Selector resolves under /home/alice/project.",
        "role": DiagnosisRole.PRIMARY,
        "cause_class": "/home/alice/project mismatch",
        "evidence": [
            {
                "key": "resolved",
                "value": "/home/alice/project/bin/nvcc",
                "source": "CUDACXX",
            }
        ],
        "impact": [{"scope": "build", "status": "broken under /home/alice/project"}],
        "remediation": [
            {
                "action": "Inspect /home/alice/project",
                "why": "the conflict lives under /home/alice/project",
                "safety": RemediationSafety.SAFE,
                "command": "ls /home/alice/project",
            }
        ],
        "verification": ["re-run under /home/alice/project"],
        "related_issues": ["TORCH002"],
        "relationships": [{"kind": "explains", "target_code": "D004"}],
        "exclusions": ["nothing healthy in /home/alice/project"],
        "presentation_suppressed_issue_codes": ["TORCH002"],
    }

    def test_every_free_form_diagnosis_field_redacts(self):
        out = redact_value(self.DIAGNOSIS, home=HOME)
        rendered = str(out)
        assert "alice" not in rendered
        assert out["summary"] == "Selector resolves under ~/project."
        assert out["evidence"][0]["value"] == "~/project/bin/nvcc"
        assert out["remediation"][0]["command"] == "ls ~/project"
        assert out["exclusions"] == ["nothing healthy in ~/project"]
        # Code-like fields are structural, not prose; redaction leaves them
        # intact so issue/diagnosis codes never get mangled.
        assert out["presentation_suppressed_issue_codes"] == ["TORCH002"]
        assert out["relationships"] == [{"kind": "explains", "target_code": "D004"}]

    def test_enum_values_survive_alongside_redaction(self):
        data = {
            "severity": Severity.ERROR,
            "confidence": Confidence.LOW,
            "role": DiagnosisRole.SECONDARY,
            "remediation": [
                {
                    "action": "a",
                    "why": "/home/alice/why",
                    "safety": RemediationSafety.CAUTION,
                }
            ],
        }
        out = redact_value(data, home=HOME)
        # Redaction never rewrites enum serialization (repo conventions:
        # uppercase Severity, lowercase v0.2 diagnosis enums).
        assert out["severity"] == "ERROR"
        assert out["confidence"] == "low"
        assert out["role"] == "secondary"
        assert out["remediation"][0]["safety"] == "caution"
        assert out["remediation"][0]["why"] == "~/why"
