"""D003 truth-table suite (Phase 9, plan §23-§30).

Pins the frozen PyTorch-CUDA-unavailable contract over normalized
``DiagnosisFacts`` plus synthetic ``prior_candidates``:

- applicability: the four PyTorch observation states never collapse
  (clean False evaluates; probe exception, genuine unknown, and import
  failure are not evaluated), plus the platform gate;
- causes: every frozen cause family with full-field assertions and the
  exact confidence/severity matrix (HIGH/ERROR for the directly
  observed causes, MEDIUM/ERROR for the D004-based cause, LOW/ERROR for
  UNKNOWN);
- precedence: the frozen deterministic order (§18.3) pairwise and the
  three §18.12 combined examples;
- D004 integration: prior-candidate presence semantics (absence is
  never negative evidence), the MEDIUM static-evidence ceiling, and the
  "explains" relationship only when D004 is the selected cause;
- visibility/driver matrices over normalized states;
- TORCH002 linkage/suppression carriers;
- the §29 probe-error privacy regression and structural purity.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
from tests.factories import base_snapshot

from cuda_doctor.compatibility import load_compatibility_data
from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.core.models import DriverInfo, EnvironmentInfo
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.facts import (
    CUDAVisibleDevicesState,
    DiagnosisFactBuilder,
    DiagnosisFacts,
    DriverDiagnosisFacts,
    GPUDiagnosisFacts,
    PlatformApplicability,
    TorchDiagnosisFacts,
    Trinary,
    VisibilityFacts,
)
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    DiagnosisCandidate,
    DiagnosisRole,
    RemediationSafety,
)
from cuda_doctor.diagnosis.rules import (
    D003PyTorchCudaUnavailable,
    DiagnosisRule,
)
from cuda_doctor.diagnosis.rules import d003 as d003_module

COMPATIBILITY = load_compatibility_data()

RULE = D003PyTorchCudaUnavailable()

CPU_ONLY = "CPU_ONLY_PYTORCH"
GPU_NOT_VISIBLE = "GPU_NOT_VISIBLE"
DRIVER_UNUSABLE = "DRIVER_UNUSABLE"
DRIVER_INCOMPATIBLE = "DRIVER_INCOMPATIBLE"
RUNTIME_LIBRARY_CONFLICT = "RUNTIME_LIBRARY_CONFLICT"
EXPLICIT_VISIBILITY_MASK = "EXPLICIT_VISIBILITY_MASK"
UNKNOWN = "UNKNOWN"

PROBE_ERROR = "RuntimeError: /home/alice/private/libcuda.so failed"


# -- helpers -----------------------------------------------------------------


def base_facts(platform: Platform = Platform.LINUX) -> DiagnosisFacts:
    snapshot = base_snapshot()
    if platform is not Platform.LINUX:
        snapshot.system = replace(snapshot.system, platform=platform)
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


def torch_symptom(**overrides) -> TorchDiagnosisFacts:
    """The observed symptom: importable CUDA build, clean available=False."""
    fields = {
        "importable": True,
        "cuda_build": True,
        "cuda_version": "12.4",
        "cuda_available": False,
        "cuda_probe_error": None,
        "cudnn_version": "90100",
    }
    fields.update(overrides)
    return TorchDiagnosisFacts(**fields)


def facts(
    torch_f: TorchDiagnosisFacts | None = None,
    *,
    gpu: Trinary = Trinary.TRUE,
    driver_usable: Trinary = Trinary.TRUE,
    driver_compatible: Trinary = Trinary.TRUE,
    visibility: CUDAVisibleDevicesState = CUDAVisibleDevicesState.UNSET,
    issue_codes: frozenset[str] = frozenset(),
    platform: Platform = Platform.LINUX,
    applicability: PlatformApplicability | None = None,
) -> DiagnosisFacts:
    """Symptom facts over a healthy independent environment by default.

    Every knob is an already-normalized fact: no raw environment value,
    driver output, or probe text is ever constructed here.
    """
    base = base_facts(platform)
    mask = {
        CUDAVisibleDevicesState.MASK_ALL: Trinary.TRUE,
        CUDAVisibleDevicesState.UNPARSEABLE: Trinary.UNKNOWN,
    }.get(visibility, Trinary.FALSE)
    return replace(
        base,
        torch=torch_f if torch_f is not None else torch_symptom(),
        gpu=GPUDiagnosisFacts(visible=gpu),
        driver=DriverDiagnosisFacts(
            version="580.126.09",
            usable=driver_usable,
            compatible_with_torch_runtime=driver_compatible,
        ),
        visibility=VisibilityFacts(
            cuda_visible_devices=visibility,
            explicit_all_gpu_mask=mask,
        ),
        issue_codes=issue_codes,
        applicability=(
            applicability if applicability is not None else base.applicability
        ),
    )


def prior(code: str, confidence: Confidence = Confidence.MEDIUM) -> DiagnosisCandidate:
    """A synthetic prior candidate (plan §26: construct synthetically)."""
    return DiagnosisCandidate(
        diagnosis=Diagnosis(
            code=code,
            title=f"synthetic {code}",
            severity=Severity.WARNING,
            confidence=confidence,
            summary=f"synthetic {code} summary",
        ),
        risk_only=True,
    )


def d004_prior() -> dict[str, DiagnosisCandidate]:
    """A present D004 candidate in its accepted MEDIUM/WARNING envelope."""
    return {"D004": prior("D004", Confidence.MEDIUM)}


def evaluate(
    facts_value: DiagnosisFacts, priors: dict[str, DiagnosisCandidate] | None = None
) -> DiagnosisCandidate | None:
    return RULE.evaluate(facts_value, [], priors or {})


def cause_of(
    facts_value: DiagnosisFacts, priors: dict[str, DiagnosisCandidate] | None = None
) -> str:
    candidate = evaluate(facts_value, priors)
    assert candidate is not None, "expected a D003 candidate"
    return candidate.diagnosis.cause_class or ""


def rendered(candidate: DiagnosisCandidate) -> str:
    """Every presentation field joined for privacy scans (plan §29)."""
    diagnosis = candidate.diagnosis
    parts = [diagnosis.title, diagnosis.summary, diagnosis.cause_class or ""]
    parts += [f"{item.key}={item.value}" for item in diagnosis.evidence]
    parts += [f"{item.scope}={item.status}" for item in diagnosis.impact]
    parts += [f"{step.action} {step.why}" for step in diagnosis.remediation]
    parts += diagnosis.verification
    parts += diagnosis.exclusions
    parts += diagnosis.related_issues
    parts += [f"{rel.kind}:{rel.target_code}" for rel in diagnosis.relationships]
    return " ".join(parts)


def built_facts(snapshot) -> DiagnosisFacts:
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


# -- rule contract -----------------------------------------------------------


class TestRuleContract:
    def test_satisfies_the_shared_protocol(self):
        assert isinstance(RULE, DiagnosisRule)

    def test_code_is_d003(self):
        assert RULE.code == "D003"

    def test_at_most_one_candidate(self):
        for candidate in (
            evaluate(facts()),
            evaluate(base_facts()),  # healthy: no symptom
            evaluate(facts(torch_symptom(cuda_available=None))),
        ):
            assert candidate is None or isinstance(candidate, DiagnosisCandidate)

    def test_same_facts_produce_equal_candidates(self):
        assert evaluate(facts()) == evaluate(facts())

    def test_host_environment_changes_nothing(self, monkeypatch):
        before = evaluate(facts())
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/host/legacy/cuda-99/lib64")
        monkeypatch.setenv("CUDA_HOME", "/host/cuda-99.0")
        assert evaluate(facts()) == before

    def test_issue_prose_never_classifies(self):
        # issues stay unconsumed: identical facts with a TORCH002 issue
        # present select the identical cause (issue codes only feed the
        # related-issues linkage, never the cause tree).
        without = evaluate(facts())
        with_issue = evaluate(facts(issue_codes=frozenset({"TORCH002"})))
        assert without is not None and with_issue is not None
        assert without.diagnosis.cause_class == with_issue.diagnosis.cause_class
        assert without.diagnosis.summary == with_issue.diagnosis.summary


# -- applicability (plan §23 A-J) ---------------------------------------------


class TestApplicability:
    def test_a_b_import_failure_is_not_evaluated(self):
        # A/B: torch not importable — never CUDA unavailability, whatever
        # the GPU/driver/visibility state (N05/§22.7).
        for gpu in (Trinary.TRUE, Trinary.FALSE, Trinary.UNKNOWN):
            assert (
                evaluate(
                    facts(
                        torch_symptom(importable=False, cuda_available=None),
                        gpu=gpu,
                        visibility=CUDAVisibleDevicesState.MASK_ALL,
                    )
                )
                is None
            )

    def test_c_healthy_true_is_not_emitted(self):
        assert (
            evaluate(
                facts(
                    torch_symptom(cuda_available=True),
                    visibility=CUDAVisibleDevicesState.MASK_ALL,
                )
            )
            is None
        )

    def test_c_cpu_build_with_available_true_is_not_emitted(self):
        # N07: a mask-looking value with no symptom emits nothing.
        assert (
            evaluate(
                facts(
                    torch_symptom(cuda_build=False, cuda_version=None,
                                  cuda_available=True),
                    visibility=CUDAVisibleDevicesState.MASK_ALL,
                )
            )
            is None
        )

    def test_d_clean_false_evaluates(self):
        candidate = evaluate(facts(torch_symptom()))
        assert candidate is not None
        assert candidate.diagnosis.cause_class == UNKNOWN

    @pytest.mark.parametrize(
        "probe_error", [PROBE_ERROR, None], ids=["raised", "genuine-unknown"]
    )
    def test_e_f_none_states_are_not_evaluated(self, probe_error):
        # E (probe exception) and F (genuine unknown) both carry
        # cuda_available=None and both stay unevaluated (U04/§18.8) —
        # None never collapses onto False, and the direct probe-failure
        # evidence is never thrown into the cause tree.
        assert (
            evaluate(
                facts(
                    torch_symptom(cuda_available=None, cuda_probe_error=probe_error),
                    visibility=CUDAVisibleDevicesState.MASK_ALL,
                )
            )
            is None
        )

    def test_g_confirmed_cpu_build_selects_cpu_only(self):
        assert cause_of(facts(torch_symptom(cuda_build=False, cuda_version=None))) == (
            CPU_ONLY
        )

    def test_h_unknown_build_state_is_not_cpu_only(self):
        # cuda_build=None is UNKNOWN, never a CPU-only inference from an
        # absent cuda_version (plan §7).
        assert (
            cause_of(facts(torch_symptom(cuda_build=None, cuda_version=None))) == UNKNOWN
        )

    @pytest.mark.parametrize(
        "platform", [Platform.LINUX, Platform.WINDOWS], ids=["linux", "windows"]
    )
    def test_i_supported_platforms_evaluate(self, platform):
        assert evaluate(facts(platform=platform)) is not None

    def test_j_other_platform_is_not_evaluated(self):
        other = facts(
            platform=Platform.OTHER,
            applicability=PlatformApplicability(
                d001=False, d002=False, d003=False, d004=False
            ),
        )
        assert evaluate(other) is None


# -- observation states never collapse (plan §5) ------------------------------


class TestObservationStates:
    def test_states_are_distinguished(self):
        states = {
            "clean-false": facts(torch_symptom()),
            "probe-error": facts(
                torch_symptom(cuda_available=None, cuda_probe_error=PROBE_ERROR)
            ),
            "genuine-unknown": facts(torch_symptom(cuda_available=None)),
            "import-failure": facts(torch_symptom(importable=False)),
        }
        results = {name: evaluate(value) for name, value in states.items()}
        assert results["clean-false"] is not None
        assert results["probe-error"] is None
        assert results["genuine-unknown"] is None
        assert results["import-failure"] is None

    def test_runtime_initialization_failure_is_not_fabricated(self):
        # §18.2: the direct runtime-init cause may stay unreachable in
        # v0.2.0 — a raised probe never produces it because the state
        # never enters the tree.
        candidate = evaluate(
            facts(torch_symptom(cuda_available=None, cuda_probe_error=PROBE_ERROR))
        )
        assert candidate is None


# -- cause families with full-field assertions (plan §24) ---------------------


class TestCauses:
    def test_cpu_only_full_fields(self):
        candidate = evaluate(
            facts(
                torch_symptom(cuda_build=False, cuda_version=None),
                issue_codes=frozenset({"TORCH002"}),
            )
        )
        assert candidate is not None
        diagnosis = candidate.diagnosis
        assert diagnosis.code == "D003"
        assert diagnosis.title == "PyTorch CUDA unavailable"
        assert diagnosis.severity is Severity.ERROR
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.cause_class == CPU_ONLY
        assert diagnosis.role is None
        assert [item.key for item in diagnosis.evidence] == [
            "cuda_available",
            "torch_cuda_build",
        ]
        assert diagnosis.evidence[1].value == (
            "CPU-only build (torch.version.cuda is None)"
        )
        assert {item.source for item in diagnosis.evidence} == {"torch"}
        assert "CPU-only" in diagnosis.summary
        assert all(
            step.safety is RemediationSafety.SAFE for step in diagnosis.remediation
        )
        assert all(step.command is None for step in diagnosis.remediation)
        assert any("CUDA-capable PyTorch build" in step.action
                   for step in diagnosis.remediation)
        assert diagnosis.related_issues == ["TORCH002"]
        assert diagnosis.presentation_suppressed_issue_codes == ["TORCH002"]
        assert diagnosis.relationships == []
        assert candidate.risk_only is False
        assert candidate.explains_active_failure is True
        assert candidate.causal_parent_codes == ()
        # P01: a healthy GPU/driver environment does not outrank the build.
        assert diagnosis.exclusions == [
            "GPU independently visible",
            "driver usable",
            "driver compatible",
            "no explicit CUDA visibility mask",
        ]

    def test_cpu_only_wins_with_unknown_environment(self):
        # P02: unknown GPU/driver states never block branch 1.
        assert (
            cause_of(
                facts(
                    torch_symptom(cuda_build=False, cuda_version=None),
                    gpu=Trinary.UNKNOWN,
                    driver_usable=Trinary.UNKNOWN,
                    driver_compatible=Trinary.UNKNOWN,
                )
            )
            == CPU_ONLY
        )

    def test_gpu_not_visible_full_fields(self):
        candidate = evaluate(facts(gpu=Trinary.FALSE))
        assert candidate is not None
        diagnosis = candidate.diagnosis
        assert diagnosis.cause_class == GPU_NOT_VISIBLE
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.severity is Severity.ERROR
        assert [item.key for item in diagnosis.evidence] == [
            "cuda_available",
            "torch_cuda_build",
            "gpu_visible",
        ]
        assert diagnosis.evidence[2].value == "no NVIDIA GPU independently visible"
        assert diagnosis.evidence[2].source == "gpu"
        assert "no NVIDIA GPU" in diagnosis.summary
        assert candidate.causal_parent_codes == ()
        assert diagnosis.relationships == []

    def test_gpu_unknown_never_claims_not_visible(self):
        # U01: UNKNOWN visibility is not FALSE.
        assert cause_of(facts(gpu=Trinary.UNKNOWN)) == UNKNOWN

    def test_driver_unusable_full_fields(self):
        candidate = evaluate(facts(driver_usable=Trinary.FALSE))
        assert candidate is not None
        diagnosis = candidate.diagnosis
        assert diagnosis.cause_class == DRIVER_UNUSABLE
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.severity is Severity.ERROR
        assert [item.key for item in diagnosis.evidence] == [
            "cuda_available",
            "torch_cuda_build",
            "driver_usable",
        ]
        assert diagnosis.evidence[2].source == "driver"
        assert any("driver health" in step.action.lower()
                   for step in diagnosis.remediation)

    def test_driver_incompatible_full_fields(self):
        candidate = evaluate(facts(driver_compatible=Trinary.FALSE))
        assert candidate is not None
        diagnosis = candidate.diagnosis
        assert diagnosis.cause_class == DRIVER_INCOMPATIBLE
        assert diagnosis.confidence is Confidence.HIGH  # semantics §22.10
        assert diagnosis.severity is Severity.ERROR
        assert [item.key for item in diagnosis.evidence] == [
            "cuda_available",
            "torch_cuda_build",
            "driver_compatible_with_torch_runtime",
        ]
        assert diagnosis.evidence[2].source == "compatibility"
        assert "documented minimum" in diagnosis.summary
        assert any(
            "documented minimum" in step.action for step in diagnosis.remediation
        )

    def test_driver_compatibility_unknown_never_claims_incompatible(self):
        # U03: UNKNOWN compatibility is not incompatibility.
        assert cause_of(facts(driver_compatible=Trinary.UNKNOWN)) == UNKNOWN

    def test_runtime_library_conflict_full_fields(self):
        candidate = evaluate(facts(), d004_prior())
        assert candidate is not None
        diagnosis = candidate.diagnosis
        assert diagnosis.cause_class == RUNTIME_LIBRARY_CONFLICT
        assert diagnosis.confidence is Confidence.MEDIUM  # D004 static ceiling
        assert diagnosis.severity is Severity.ERROR
        assert diagnosis.role is None
        assert [rel.kind for rel in diagnosis.relationships] == ["explains"]
        assert [rel.target_code for rel in diagnosis.relationships] == ["D004"]
        assert candidate.causal_parent_codes == ("D004",)
        assert candidate.risk_only is False
        assert candidate.explains_active_failure is True
        evidence = {item.key: item for item in diagnosis.evidence}
        assert evidence["runtime_library_conflict"].value == (
            "D004 present (medium confidence, static inventory evidence)"
        )
        assert evidence["runtime_library_conflict"].source == "prior_diagnosis"
        assert "plausible contributing cause" in diagnosis.summary
        assert "not observed" in diagnosis.summary

    def test_explicit_visibility_mask_full_fields(self):
        candidate = evaluate(facts(visibility=CUDAVisibleDevicesState.MASK_ALL))
        assert candidate is not None
        diagnosis = candidate.diagnosis
        assert diagnosis.cause_class == EXPLICIT_VISIBILITY_MASK
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.severity is Severity.ERROR
        evidence = {item.key: item for item in diagnosis.evidence}
        assert evidence["cuda_visible_devices"].value == (
            "explicit mask-all (empty value or -1)"
        )
        assert evidence["cuda_visible_devices"].source == "visibility"
        assert "CUDA_VISIBLE_DEVICES" in diagnosis.summary
        assert any(
            "without the CUDA_VISIBLE_DEVICES mask" in step.action
            for step in diagnosis.remediation
        )
        assert candidate.causal_parent_codes == ()

    def test_unknown_full_fields(self):
        candidate = evaluate(facts())
        assert candidate is not None
        diagnosis = candidate.diagnosis
        assert diagnosis.cause_class == UNKNOWN
        assert diagnosis.confidence is Confidence.LOW
        assert diagnosis.severity is Severity.ERROR
        assert diagnosis.role is None
        assert (
            "Unable to determine the root cause from available evidence."
            in diagnosis.summary
        )
        assert [item.key for item in diagnosis.evidence] == [
            "cuda_available",
            "torch_cuda_build",
        ]
        assert diagnosis.relationships == []
        assert candidate.causal_parent_codes == ()
        assert candidate.risk_only is False
        assert candidate.explains_active_failure is True
        assert all(
            step.safety is RemediationSafety.SAFE for step in diagnosis.remediation
        )
        assert diagnosis.exclusions == [
            "CUDA-enabled PyTorch build confirmed",
            "GPU independently visible",
            "driver usable",
            "driver compatible",
            "no explicit CUDA visibility mask",
        ]

    def test_unknown_never_invents_a_library_explanation(self):
        # §18.13/plan §17.8: no "probably runtime library conflict".
        diagnosis = evaluate(facts()).diagnosis
        assert diagnosis is not None
        assert "probably" not in diagnosis.summary
        assert "likely runtime" not in diagnosis.summary

    @pytest.mark.parametrize(
        "cause,confidence",
        [
            (CPU_ONLY, Confidence.HIGH),
            (GPU_NOT_VISIBLE, Confidence.HIGH),
            (DRIVER_UNUSABLE, Confidence.HIGH),
            (DRIVER_INCOMPATIBLE, Confidence.HIGH),
            (RUNTIME_LIBRARY_CONFLICT, Confidence.MEDIUM),
            (EXPLICIT_VISIBILITY_MASK, Confidence.HIGH),
            (UNKNOWN, Confidence.LOW),
        ],
    )
    def test_confidence_severity_matrix(self, cause, confidence):
        matrix: dict[str, tuple[DiagnosisFacts, object]] = {
            CPU_ONLY: (facts(torch_symptom(cuda_build=False, cuda_version=None)), None),
            GPU_NOT_VISIBLE: (facts(gpu=Trinary.FALSE), None),
            DRIVER_UNUSABLE: (facts(driver_usable=Trinary.FALSE), None),
            DRIVER_INCOMPATIBLE: (facts(driver_compatible=Trinary.FALSE), None),
            RUNTIME_LIBRARY_CONFLICT: (facts(), d004_prior()),
            EXPLICIT_VISIBILITY_MASK: (
                facts(visibility=CUDAVisibleDevicesState.MASK_ALL),
                None,
            ),
            UNKNOWN: (facts(), None),
        }
        facts_value, priors = matrix[cause]
        candidate = evaluate(facts_value, priors)  # type: ignore[arg-type]
        assert candidate is not None
        assert candidate.diagnosis.cause_class == cause
        assert candidate.diagnosis.confidence is confidence
        assert candidate.diagnosis.severity is Severity.ERROR


# -- frozen cause precedence (plan §25, §18.12) -------------------------------


class TestPrecedence:
    def test_example_a_cpu_only_outranks_mask(self):
        candidate = evaluate(
            facts(
                torch_symptom(cuda_build=False, cuda_version=None),
                visibility=CUDAVisibleDevicesState.MASK_ALL,
            )
        )
        assert candidate is not None
        assert candidate.diagnosis.cause_class == CPU_ONLY
        # The mask is reported as a contributing fact, not the cause.
        evidence = {item.key: item for item in candidate.diagnosis.evidence}
        assert "contributing fact" in evidence["cuda_visible_devices"].value

    def test_cpu_only_outranks_driver_unusable(self):
        assert (
            cause_of(
                facts(
                    torch_symptom(cuda_build=False, cuda_version=None),
                    driver_usable=Trinary.FALSE,
                )
            )
            == CPU_ONLY
        )

    def test_gpu_not_visible_outranks_driver_incompatible(self):
        assert (
            cause_of(
                facts(gpu=Trinary.FALSE, driver_compatible=Trinary.FALSE)
            )
            == GPU_NOT_VISIBLE
        )

    def test_gpu_not_visible_outranks_mask(self):
        assert (
            cause_of(
                facts(
                    gpu=Trinary.FALSE,
                    visibility=CUDAVisibleDevicesState.MASK_ALL,
                )
            )
            == GPU_NOT_VISIBLE
        )

    def test_driver_unusable_outranks_incompatible(self):
        assert (
            cause_of(
                facts(driver_usable=Trinary.FALSE, driver_compatible=Trinary.FALSE)
            )
            == DRIVER_UNUSABLE
        )

    def test_driver_unusable_outranks_d004(self):
        assert cause_of(facts(driver_usable=Trinary.FALSE), d004_prior()) == (
            DRIVER_UNUSABLE
        )

    def test_driver_incompatible_outranks_d004(self):
        assert cause_of(facts(driver_compatible=Trinary.FALSE), d004_prior()) == (
            DRIVER_INCOMPATIBLE
        )

    def test_example_b_driver_incompatible_outranks_mask(self):
        assert (
            cause_of(
                facts(
                    driver_compatible=Trinary.FALSE,
                    visibility=CUDAVisibleDevicesState.MASK_ALL,
                )
            )
            == DRIVER_INCOMPATIBLE
        )

    def test_example_c_d004_outranks_mask(self):
        assert (
            cause_of(
                facts(visibility=CUDAVisibleDevicesState.MASK_ALL), d004_prior()
            )
            == RUNTIME_LIBRARY_CONFLICT
        )

    def test_d004_outranks_unknown_fallback(self):
        assert cause_of(facts(), d004_prior()) == RUNTIME_LIBRARY_CONFLICT

    def test_higher_cause_with_all_unknown_conditions(self):
        # P02 shape: an established higher cause stands even when every
        # lower-branch fact is UNKNOWN.
        assert (
            cause_of(
                facts(
                    torch_symptom(cuda_build=False, cuda_version=None),
                    gpu=Trinary.UNKNOWN,
                    driver_usable=Trinary.UNKNOWN,
                    driver_compatible=Trinary.UNKNOWN,
                    visibility=CUDAVisibleDevicesState.UNPARSEABLE,
                )
            )
            == CPU_ONLY
        )


# -- D004 integration (plan §26 items 1-9) ------------------------------------


class TestD004Integration:
    def test_1_no_d004_prior_falls_through(self):
        candidate = evaluate(facts(), {})
        assert candidate is not None
        assert candidate.diagnosis.cause_class == UNKNOWN
        assert candidate.diagnosis.relationships == []
        assert candidate.causal_parent_codes == ()

    def test_2_d004_prior_selects_the_conflict_cause(self):
        candidate = evaluate(facts(), d004_prior())
        assert candidate is not None
        assert candidate.diagnosis.cause_class == RUNTIME_LIBRARY_CONFLICT

    @pytest.mark.parametrize("code", ["D001", "D002"], ids=["d001", "d002"])
    def test_3_4_unrelated_prior_candidates_are_ignored(self, code):
        # D001/D002 are never ad-hoc CUDA-runtime causes (plan §12).
        with_unrelated = evaluate(facts(), {code: prior(code)})
        without = evaluate(facts(), {})
        assert with_unrelated is not None and without is not None
        assert with_unrelated == without

    def test_5_d004_with_higher_priority_cause(self):
        candidate = evaluate(
            facts(driver_usable=Trinary.FALSE), d004_prior()
        )
        assert candidate is not None
        assert candidate.diagnosis.cause_class == DRIVER_UNUSABLE
        assert candidate.diagnosis.relationships == []
        assert candidate.causal_parent_codes == ()

    def test_6_d004_with_lower_priority_mask(self):
        assert (
            cause_of(
                facts(visibility=CUDAVisibleDevicesState.MASK_ALL), d004_prior()
            )
            == RUNTIME_LIBRARY_CONFLICT
        )

    def test_7_d004_evidence_never_becomes_loaded_object_certainty(self):
        candidate = evaluate(facts(), d004_prior())
        assert candidate is not None
        text = rendered(candidate)
        for forbidden in (
            "definitely",
            "will load",
            "is loaded",
            "proves",
            "certainly",
            "confirmed loaded",
            "guaranteed",
        ):
            assert forbidden not in text, forbidden
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_8_d003_never_calls_d004_itself(self):
        # Structural: the rule module never imports or instantiates the
        # D004 rule; D004 state arrives only through prior_candidates.
        tree = ast.parse(Path(d003_module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith(".d004"), node.module
                assert (node.module or "") != "d004"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.endswith("d004"), alias.name
        source = Path(d003_module.__file__).read_text()
        assert "D004RuntimeLibraryConflict" not in source
        # Behavioral: with no priors the D004 branch cannot fire at all.
        baseline = evaluate(facts(), {})
        assert baseline is not None
        assert baseline.diagnosis.cause_class == UNKNOWN

    @pytest.mark.parametrize(
        "priors",
        [
            {},
            {"D001": prior("D001")},
            {"D001": prior("D001"), "D002": prior("D002")},
        ],
        ids=["empty", "d001-only", "d001-d002"],
    )
    def test_9_decision_stable_regardless_of_unrelated_ordering(self, priors):
        baseline = evaluate(facts(), {})
        assert evaluate(facts(), priors) == baseline

    def test_9_d004_decision_stable_with_extra_unrelated_priors(self):
        baseline = evaluate(facts(), d004_prior())
        noisy = evaluate(
            facts(),
            {"D002": prior("D002"), "D004": prior("D004"), "D001": prior("D001")},
        )
        assert noisy == baseline

    def test_d004_confidence_is_carried_not_recomputed(self):
        high = {"D004": prior("D004", Confidence.HIGH)}
        candidate = evaluate(facts(), high)
        assert candidate is not None
        # The D003 cause stays MEDIUM regardless of the carried label —
        # the frozen ceiling, not the prior's own confidence, governs.
        assert candidate.diagnosis.confidence is Confidence.MEDIUM
        evidence = {item.key: item for item in candidate.diagnosis.evidence}
        assert "high confidence" in evidence["runtime_library_conflict"].value


# -- visibility matrix (plan §27) ----------------------------------------------


class TestVisibilityMatrix:
    def test_unset_is_not_a_cause(self):
        candidate = evaluate(facts(visibility=CUDAVisibleDevicesState.UNSET))
        assert candidate is not None
        assert candidate.diagnosis.cause_class == UNKNOWN
        assert "no explicit CUDA visibility mask" in candidate.diagnosis.exclusions

    def test_mask_all_normalized_state_is_the_mask_cause(self):
        assert (
            cause_of(facts(visibility=CUDAVisibleDevicesState.MASK_ALL))
            == EXPLICIT_VISIBILITY_MASK
        )

    @pytest.mark.parametrize("raw", ["", "-1"], ids=["empty", "minus-one"])
    def test_both_verified_mask_all_tokens_build_to_the_mask_cause(self, raw):
        # Builder integration: the two Phase-0-verified mask-all tokens
        # normalize to MASK_ALL and select the mask cause end-to-end.
        snapshot = base_snapshot()
        snapshot.environment = EnvironmentInfo(
            variables={
                "CUDA_HOME": "/usr/local/cuda-12.4",
                "CUDA_VISIBLE_DEVICES": raw,
            },
            path_entries=[(0, "/usr/bin")],
            cuda_path_entries=[],
        )
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        assert cause_of(built_facts(snapshot)) == EXPLICIT_VISIBILITY_MASK

    @pytest.mark.parametrize("raw", ["0", "0,1"], ids=["single", "multi-index"])
    def test_numeric_restrictions_are_not_mask_all(self, raw):
        # N06: restricted-nonempty proceeds toward UNKNOWN.
        snapshot = base_snapshot()
        snapshot.environment = EnvironmentInfo(
            variables={
                "CUDA_HOME": "/usr/local/cuda-12.4",
                "CUDA_VISIBLE_DEVICES": raw,
            },
            path_entries=[(0, "/usr/bin")],
            cuda_path_entries=[],
        )
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        assert cause_of(built_facts(snapshot)) == UNKNOWN

    def test_nonexistent_gpu_index_is_not_mask_all(self):
        # N08: "3" on a 1-GPU machine stays RESTRICTED_NONEMPTY —
        # deliberate conservatism, no GPU-count reinterpretation.
        snapshot = base_snapshot()
        snapshot.environment = EnvironmentInfo(
            variables={
                "CUDA_HOME": "/usr/local/cuda-12.4",
                "CUDA_VISIBLE_DEVICES": "3",
            },
            path_entries=[(0, "/usr/bin")],
            cuda_path_entries=[],
        )
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        assert cause_of(built_facts(snapshot)) == UNKNOWN

    def test_unparseable_stays_unknown(self):
        # U05: unparseable resolves to UNKNOWN, never a guessed cause.
        candidate = evaluate(facts(visibility=CUDAVisibleDevicesState.UNPARSEABLE))
        assert candidate is not None
        assert candidate.diagnosis.cause_class == UNKNOWN
        assert "no explicit CUDA visibility mask" not in candidate.diagnosis.exclusions

    def test_gpu_invisible_outranks_a_restriction(self):
        assert (
            cause_of(
                facts(
                    gpu=Trinary.FALSE,
                    visibility=CUDAVisibleDevicesState.RESTRICTED_NONEMPTY,
                )
            )
            == GPU_NOT_VISIBLE
        )

    def test_gpu_visible_true_does_not_block_the_mask(self):
        # The mask branch is independent of the GPU observation.
        assert (
            cause_of(facts(visibility=CUDAVisibleDevicesState.MASK_ALL))
            == EXPLICIT_VISIBILITY_MASK
        )


# -- driver matrix (plan §28: usable x compatible) -----------------------------


class TestDriverMatrix:
    @pytest.mark.parametrize(
        "usable,compatible,expected",
        [
            (Trinary.FALSE, Trinary.FALSE, DRIVER_UNUSABLE),
            (Trinary.FALSE, Trinary.TRUE, DRIVER_UNUSABLE),
            (Trinary.FALSE, Trinary.UNKNOWN, DRIVER_UNUSABLE),
            (Trinary.TRUE, Trinary.FALSE, DRIVER_INCOMPATIBLE),
            (Trinary.UNKNOWN, Trinary.FALSE, DRIVER_INCOMPATIBLE),
            (Trinary.TRUE, Trinary.TRUE, UNKNOWN),
            (Trinary.TRUE, Trinary.UNKNOWN, UNKNOWN),
            (Trinary.UNKNOWN, Trinary.TRUE, UNKNOWN),
            (Trinary.UNKNOWN, Trinary.UNKNOWN, UNKNOWN),
        ],
        ids=[
            "unusable-incompatible",
            "unusable-compatible",
            "unusable-unknown",
            "usable-incompatible",
            "unknown-incompatible",
            "usable-compatible",
            "usable-unknown",
            "unknown-compatible",
            "unknown-unknown",
        ],
    )
    def test_usable_by_compatible_grid(self, usable, compatible, expected):
        # U02: an unknown driver never fabricates a cause; compatibility
        # FALSE never overrides the higher-priority unusable branch.
        assert (
            cause_of(facts(driver_usable=usable, driver_compatible=compatible))
            == expected
        )

    def test_incompatible_driver_does_not_override_invisible_gpu(self):
        assert (
            cause_of(facts(gpu=Trinary.FALSE, driver_compatible=Trinary.FALSE))
            == GPU_NOT_VISIBLE
        )


# -- TORCH002 linkage and suppression (plan §17) -------------------------------


class TestTorch002Linkage:
    def test_related_issues_listed_only_when_present(self):
        assert evaluate(facts()).diagnosis.related_issues == []
        with_issue = evaluate(facts(issue_codes=frozenset({"TORCH002"})))
        assert with_issue is not None
        assert with_issue.diagnosis.related_issues == ["TORCH002"]

    def test_other_issue_codes_are_not_linked(self):
        candidate = evaluate(facts(issue_codes=frozenset({"TORCH004", "ENV001"})))
        assert candidate is not None
        assert candidate.diagnosis.related_issues == []

    @pytest.mark.parametrize(
        "facts_value,priors,cause",
        [
            (facts(torch_symptom(cuda_build=False, cuda_version=None)), None, CPU_ONLY),
            (facts(gpu=Trinary.FALSE), None, GPU_NOT_VISIBLE),
            (facts(driver_usable=Trinary.FALSE), None, DRIVER_UNUSABLE),
            (facts(driver_compatible=Trinary.FALSE), None, DRIVER_INCOMPATIBLE),
            (facts(), d004_prior(), RUNTIME_LIBRARY_CONFLICT),
            (facts(visibility=CUDAVisibleDevicesState.MASK_ALL), None,
             EXPLICIT_VISIBILITY_MASK),
            (facts(), None, UNKNOWN),
        ],
        ids=[
            "cpu-only",
            "gpu-not-visible",
            "driver-unusable",
            "driver-incompatible",
            "runtime-library-conflict",
            "visibility-mask",
            "unknown",
        ],
    )
    def test_every_emitted_d003_carries_the_suppression(
        self, facts_value, priors, cause
    ):
        # §25.2: suppression applies whenever the D003 diagnosis —
        # including the UNKNOWN-cause variant — is rendered; the carrier
        # is present on every emission, and it is presentation metadata
        # only (the machine-readable issue is never touched).
        candidate = evaluate(facts_value, priors)
        assert candidate is not None
        assert candidate.diagnosis.cause_class == cause
        assert candidate.diagnosis.presentation_suppressed_issue_codes == ["TORCH002"]

    def test_suppression_is_carrier_only(self):
        candidate = evaluate(facts(issue_codes=frozenset({"TORCH002"})))
        assert candidate is not None
        # The carrier removes nothing by itself: TORCH002 stays listed
        # among related issues and the field is exactly the frozen map.
        assert candidate.diagnosis.related_issues == ["TORCH002"]
        assert candidate.diagnosis.presentation_suppressed_issue_codes == ["TORCH002"]


# -- probe-error privacy (plan §29, mandatory) ---------------------------------


class TestProbeErrorPrivacy:
    def test_raised_probe_state_emits_nothing(self):
        # The mandatory regression input: with cuda_available=None the
        # raw exception exists in the facts and nothing is rendered.
        facts_value = facts(
            torch_symptom(cuda_available=None, cuda_probe_error=PROBE_ERROR)
        )
        assert evaluate(facts_value) is None

    def test_emitted_candidate_never_renders_probe_error_text(self):
        # Defensive: even a (collector-impossible) False-with-error fact
        # state must not leak the raw text — no branch reads the field.
        facts_value = facts(torch_symptom(cuda_available=False,
                                          cuda_probe_error=PROBE_ERROR))
        candidate = evaluate(facts_value)
        assert candidate is not None
        text = rendered(candidate)
        for forbidden in (
            "alice",
            "/home/",
            "private",
            "libcuda.so",
            PROBE_ERROR,
            "RuntimeError",
        ):
            assert forbidden not in text, forbidden

    @pytest.mark.parametrize(
        "facts_value,priors",
        [
            (facts(torch_symptom(cuda_build=False, cuda_version=None)), None),
            (facts(gpu=Trinary.FALSE), None),
            (facts(driver_usable=Trinary.FALSE), None),
            (facts(driver_compatible=Trinary.FALSE), None),
            (facts(), d004_prior()),
            (facts(visibility=CUDAVisibleDevicesState.MASK_ALL), None),
            (facts(), None),
        ],
        ids=[
            "cpu-only",
            "gpu-not-visible",
            "driver-unusable",
            "driver-incompatible",
            "runtime-library-conflict",
            "visibility-mask",
            "unknown",
        ],
    )
    def test_no_cause_reveals_private_detail(self, facts_value, priors):
        candidate = evaluate(facts_value, priors)
        assert candidate is not None
        text = rendered(candidate)
        for forbidden in ("alice", "/home/", "private", "libcuda.so", ".so"):
            assert forbidden not in text, forbidden
        # No filesystem-path separators beyond the single frozen " / "
        # separator inside the UNKNOWN next-diagnostic-area sentence.
        assert text.count("/") == text.count(" / ")

    def test_rule_source_never_reads_the_probe_error_field(self):
        source = Path(d003_module.__file__).read_text()
        assert "cuda_probe_error" not in source


# -- purity (plan §30) ---------------------------------------------------------


class TestPurity:
    def test_imports_no_host_probes_or_collector_surfaces(self):
        tree = ast.parse(Path(d003_module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert imported.isdisjoint(
            {"os", "subprocess", "shutil", "site", "time", "random", "torch",
             "pathlib"}
        )
        for name in imported:
            assert not name.startswith("cuda_doctor.collectors"), name
            assert name != "cuda_doctor.core.models", name

    def test_source_names_no_collector_or_environment_surfaces(self):
        source = Path(d003_module.__file__).read_text()
        for forbidden in (
            "EnvironmentSnapshot",
            "DiagnosisFactBuilder",
            "LD_LIBRARY_PATH",
            "readelf",
            "objdump",
            "LD_DEBUG",
            "dlopen",
            "ctypes",
            "DT_NEEDED",
            "DT_SONAME",
            "RPATH",
            "RUNPATH",
        ):
            assert forbidden not in source, forbidden

    def test_runtime_libraries_are_never_consumed(self):
        # D003's causes come from torch/gpu/driver/visibility facts and
        # prior-candidate presence only: the runtime-library inventory is
        # irrelevant to the decision (D004 owns it).
        baseline = facts()
        mutated = replace(
            baseline,
            runtime_libraries=replace(
                baseline.runtime_libraries, scan_incomplete=True
            ),
        )
        assert evaluate(baseline) == evaluate(mutated)


# -- D001/D002 independence (plan §12) -----------------------------------------


class TestD001D002Independence:
    def test_neither_becomes_an_ad_hoc_runtime_cause(self):
        # A D001/D002 conflict is never selected as the D003 cause and
        # never changes the selected cause, whatever it is.
        priors = {"D001": prior("D001"), "D002": prior("D002")}
        assert cause_of(facts(), priors) == UNKNOWN
        assert cause_of(
            facts(driver_usable=Trinary.FALSE), priors
        ) == DRIVER_UNUSABLE

    @pytest.mark.parametrize(
        "facts_value",
        [
            facts(torch_symptom(cuda_build=False, cuda_version=None)),
            facts(gpu=Trinary.FALSE),
            facts(driver_usable=Trinary.FALSE),
            facts(visibility=CUDAVisibleDevicesState.MASK_ALL),
            facts(),
        ],
        ids=["cpu-only", "gpu", "driver", "mask", "unknown"],
    )
    def test_no_d001_d002_parents_or_relationships(self, facts_value):
        candidate = evaluate(
            facts_value, {"D001": prior("D001"), "D002": prior("D002")}
        )
        assert candidate is not None
        assert candidate.causal_parent_codes == ()
        assert candidate.diagnosis.relationships == []


# -- verification / platform ----------------------------------------------------


class TestVerification:
    def test_linux_verification_is_the_frozen_triple(self):
        candidate = evaluate(facts())
        assert candidate is not None
        assert candidate.diagnosis.verification == [
            "python -c \"import torch; print(torch.version.cuda); "
            "print(torch.cuda.is_available())\"",
            "nvidia-smi",
            "echo $CUDA_VISIBLE_DEVICES",
        ]

    def test_windows_verification_uses_windows_syntax(self):
        candidate = evaluate(facts(platform=Platform.WINDOWS))
        assert candidate is not None
        assert candidate.diagnosis.verification == [
            "python -c \"import torch; print(torch.version.cuda); "
            "print(torch.cuda.is_available())\"",
            "nvidia-smi",
            "echo %CUDA_VISIBLE_DEVICES%",
        ]

    def test_no_forbidden_verification_commands(self):
        # Plan §21: no ldd/readelf/objdump/LD_DEBUG, no filesystem or
        # package-manager mutation in any cause's verification.
        variants = [
            (facts(torch_symptom(cuda_build=False, cuda_version=None)), None),
            (facts(gpu=Trinary.FALSE), None),
            (facts(driver_usable=Trinary.FALSE), None),
            (facts(driver_compatible=Trinary.FALSE), None),
            (facts(), d004_prior()),
            (facts(visibility=CUDAVisibleDevicesState.MASK_ALL), None),
            (facts(), None),
        ]
        for facts_value, priors in variants:
            candidate = evaluate(facts_value, priors)
            assert candidate is not None
            for forbidden in ("ldd", "readelf", "objdump", "LD_DEBUG", "pip install",
                              "apt", "rm "):
                assert forbidden not in " ".join(candidate.diagnosis.verification), (
                    forbidden
                )


# -- builder end-to-end ----------------------------------------------------------


class TestBuilderEndToEnd:
    def test_cpu_only_snapshot(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch,
            is_cuda_build=False,
            cuda_version=None,
            cuda_available=False,
            device_count=0,
            devices=[],
        )
        assert cause_of(built_facts(snapshot)) == CPU_ONLY

    def test_driver_below_documented_minimum_snapshot(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        snapshot.driver = DriverInfo("450.51.06", "11.0")
        facts_value = built_facts(snapshot)
        assert facts_value.driver.compatible_with_torch_runtime is Trinary.FALSE
        assert cause_of(facts_value) == DRIVER_INCOMPATIBLE

    def test_no_gpu_snapshot(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        snapshot.gpus = []
        assert cause_of(built_facts(snapshot)) == GPU_NOT_VISIBLE

    def test_healthy_unavailable_environment_is_unknown(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        assert cause_of(built_facts(snapshot)) == UNKNOWN


# -- no engine integration (plan §32) ---------------------------------------------


class TestEngineIntegration:
    def test_engine_emits_the_resolved_diagnosis(self):
        # Phase 11 wired the accepted rules into the engine: the rule fires
        # on the built facts and its candidate reaches
        # DiagnosisResult.diagnoses finalized by the resolver. No D004
        # prior exists here, so D003 lands on the frozen UNKNOWN cause.
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        result = DiagnosisEngine().run(snapshot)
        assert [(d.code, d.role) for d in result.diagnoses] == [
            ("D003", DiagnosisRole.PRIMARY)
        ]
        assert result.diagnoses[0].confidence is Confidence.LOW
        assert result.diagnosis_errors == {}
        # The observed failure stays reported at the issue layer.
        assert [issue.code for issue in result.issues] == ["TORCH002"]
        assert evaluate(built_facts(snapshot)) is not None
