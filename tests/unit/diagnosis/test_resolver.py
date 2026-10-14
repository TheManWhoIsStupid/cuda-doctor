"""Resolver truth-table suite (Phase 10, plan §4-§34).

Pins the frozen diagnosis-candidate resolver contract over synthetic
``DiagnosisCandidate`` factories plus one real-rule contract test:

- input/output: candidates only in, finalized ``Diagnosis`` objects out,
  every output carrying a concrete role, nothing discarded;
- the frozen §12.1-§12.6 PRIMARY ranking with each dimension proven to
  decide only in its frozen place (causal parent, active-failure
  explanation, confidence, severity, SAFE remediation, code order);
- the role matrix (PRIMARY/SECONDARY/CONTRIBUTING/POSSIBLE are not
  synonyms), LOW-confidence eligibility, risk-only and active-failure
  hint semantics, missing parents, duplicate codes, cycles;
- deterministic permutation-invariant output ordering;
- payload preservation, input immutability, relationship pass-through,
  suppression-carrier boundary, purity, and no engine integration.
"""

from __future__ import annotations

import ast
import copy
import itertools
from dataclasses import fields, replace
from pathlib import Path

import pytest
from tests.factories import base_snapshot

from cuda_doctor.compatibility import load_compatibility_data
from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis import resolver as resolver_module
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.facts import (
    ORIGIN_LD_LIBRARY_PATH,
    DiagnosisFactBuilder,
    IdentityReliability,
    RuntimeLibraryCandidateFact,
    RuntimeLibraryFacts,
    Trinary,
)
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
from cuda_doctor.diagnosis.resolver import DiagnosisResolver
from cuda_doctor.diagnosis.rules import (
    D003PyTorchCudaUnavailable,
    D004RuntimeLibraryConflict,
)

COMPATIBILITY = load_compatibility_data()
RESOLVER = DiagnosisResolver()


# -- synthetic candidate factories (plan §33) ---------------------------------


def candidate(
    code: str = "D001",
    *,
    confidence: Confidence = Confidence.HIGH,
    severity: Severity = Severity.WARNING,
    explains: bool = False,
    risk_only: bool = True,
    parents: tuple[str, ...] = (),
    title: str | None = None,
    steps: list[RemediationStep] | None = None,
    relationships: list[DiagnosisRelationship] | None = None,
    role: DiagnosisRole | None = None,
    summary: str | None = None,
    cause_class: str | None = None,
    related_issues: list[str] | None = None,
) -> DiagnosisCandidate:
    """A fully populated synthetic candidate; only knobs the frozen
    resolver contract actually consumes vary by default."""
    return DiagnosisCandidate(
        diagnosis=Diagnosis(
            code=code,
            title=title or f"synthetic {code}",
            severity=severity,
            confidence=confidence,
            summary=summary or f"synthetic {code} summary",
            role=role,
            cause_class=cause_class,
            evidence=[EvidenceItem(key="fact", value="observed", source="synthetic")],
            impact=[ImpactItem(scope="scope", status="affected")],
            remediation=steps
            if steps is not None
            else [RemediationStep(action="act", why="because",
                                  safety=RemediationSafety.SAFE)],
            verification=["verify --synthetic"],
            related_issues=related_issues or [],
            relationships=relationships or [],
            exclusions=["an excluded alternative"],
            presentation_suppressed_issue_codes=["TORCH002"],
        ),
        explains_active_failure=explains,
        risk_only=risk_only,
        causal_parent_codes=parents,
    )


def d001(confidence: Confidence = Confidence.HIGH) -> DiagnosisCandidate:
    """The accepted D001 envelope: risk-only, never explains, no parents."""
    return candidate("D001", confidence=confidence)


def d002(confidence: Confidence = Confidence.HIGH) -> DiagnosisCandidate:
    """The accepted D002 envelope: risk-only, never explains, no parents."""
    return candidate("D002", confidence=confidence)


def d004(confidence: Confidence = Confidence.MEDIUM) -> DiagnosisCandidate:
    """The accepted D004 envelope: risk-only MEDIUM, no parents."""
    return candidate("D004", confidence=confidence)


def d003(
    confidence: Confidence = Confidence.MEDIUM,
    parents: tuple[str, ...] = (),
) -> DiagnosisCandidate:
    """The accepted D003 envelope: active-failure explanation, ERROR,
    parents only in the D004-backed conflict branch."""
    return candidate(
        "D003",
        confidence=confidence,
        severity=Severity.ERROR,
        explains=True,
        risk_only=False,
        parents=parents,
        title="PyTorch CUDA unavailable",
        cause_class="RUNTIME_LIBRARY_CONFLICT" if parents else "UNKNOWN",
        relationships=[
            DiagnosisRelationship(kind="explains", target_code="D004")
        ]
        if parents
        else [],
    )


def d003_unknown() -> DiagnosisCandidate:
    """The D003 UNKNOWN variant: ERROR severity, LOW confidence, active."""
    return candidate(
        "D003",
        confidence=Confidence.LOW,
        severity=Severity.ERROR,
        explains=True,
        risk_only=False,
        title="PyTorch CUDA unavailable",
        cause_class="UNKNOWN",
    )


def safe_command_step(action: str = "run the safe check") -> RemediationStep:
    return RemediationStep(
        action=action, why="why not", safety=RemediationSafety.SAFE, command="echo ok"
    )


def plain_step(action: str = "act") -> RemediationStep:
    return RemediationStep(action=action, why="because", safety=RemediationSafety.SAFE)


def resolve(*candidates: DiagnosisCandidate) -> list[Diagnosis]:
    return RESOLVER.resolve(list(candidates))


def roles(out: list[Diagnosis]) -> dict[str, DiagnosisRole]:
    return {diagnosis.code: diagnosis.role for diagnosis in out}


def order(out: list[Diagnosis]) -> list[tuple[str, DiagnosisRole]]:
    return [(diagnosis.code, diagnosis.role) for diagnosis in out]


# -- input/output contract (plan §4/§11) ---------------------------------------


class TestResolverContract:
    def test_empty_input_yields_empty_list(self):
        assert RESOLVER.resolve([]) == []

    def test_accepts_any_sequence(self):
        assert RESOLVER.resolve(()) == []
        assert len(RESOLVER.resolve((d001(),))) == 1

    def test_every_output_carries_a_concrete_role(self):
        out = resolve(d001(), d002(), d003(), d004())
        assert all(diagnosis.role is not None for diagnosis in out)
        assert all(isinstance(diagnosis.role, DiagnosisRole) for diagnosis in out)

    def test_no_candidate_is_discarded(self):
        for out in (
            resolve(d001()),
            resolve(d001(), d002(), d004()),
            resolve(d003(parents=("D004",)), d004()),
            resolve(d001(), candidate("D001", title="alpha")),
        ):
            assert len(out) >= 1
        assert len(resolve(d001(), d002(), d004())) == 3
        assert len(resolve(candidate("D001", title="alpha"),
                           candidate("D001", title="beta"))) == 2

    def test_instance_is_reusable(self):
        first = resolve(d001())
        second = resolve(d001())
        assert first == second

    def test_same_input_produces_equal_output(self):
        assert resolve(d003(parents=("D004",)), d004()) == resolve(
            d003(parents=("D004",)), d004()
        )


# -- global PRIMARY invariant (plan §5) -----------------------------------------


class TestPrimaryInvariant:
    @pytest.mark.parametrize(
        "case",
        [
            (d001(),),
            (d001(), d002()),
            (d001(), d002(), d004()),
            (d003(), d004()),
            (d003(parents=("D004",)), d004(), d001(), d002()),
            (d003_unknown(), d001()),
            (candidate("D001", confidence=Confidence.LOW),),
            (candidate("D001", confidence=Confidence.LOW), d004()),
        ],
        ids=[
            "single-risk",
            "two-risk",
            "three-risk",
            "active-plus-risk",
            "all-four",
            "unknown-plus-high",
            "single-low",
            "low-pair",
        ],
    )
    def test_at_most_one_primary(self, case):
        out = resolve(*case)
        assert sum(d.role is DiagnosisRole.PRIMARY for d in out) <= 1

    def test_no_primary_is_forced_when_none_eligible(self):
        # LOW-confidence risk-only findings are not PRIMARY-eligible:
        # the resolver reports POSSIBLE findings instead of promoting one
        # to make the report "look better" (plan §5).
        out = resolve(candidate("D001", confidence=Confidence.LOW))
        assert order(out) == [("D001", DiagnosisRole.POSSIBLE)]

    def test_two_low_candidates_yield_no_primary(self):
        out = resolve(
            candidate("D001", confidence=Confidence.LOW),
            candidate("D002", confidence=Confidence.LOW),
        )
        assert order(out) == [
            ("D001", DiagnosisRole.POSSIBLE),
            ("D002", DiagnosisRole.POSSIBLE),
        ]


# -- single-candidate cases (plan §12) -------------------------------------------


class TestSingleCandidateMatrix:
    @pytest.mark.parametrize(
        "single,expected_role",
        [
            (d001(Confidence.HIGH), DiagnosisRole.PRIMARY),
            (d001(Confidence.MEDIUM), DiagnosisRole.PRIMARY),
            (d002(Confidence.HIGH), DiagnosisRole.PRIMARY),
            (d002(Confidence.MEDIUM), DiagnosisRole.PRIMARY),
            (d003(Confidence.HIGH), DiagnosisRole.PRIMARY),
            (d003(Confidence.MEDIUM, parents=("D004",)), DiagnosisRole.PRIMARY),
            (d003_unknown(), DiagnosisRole.PRIMARY),
            (d004(Confidence.MEDIUM), DiagnosisRole.PRIMARY),
        ],
        ids=[
            "d001-high",
            "d001-medium",
            "d002-high",
            "d002-medium",
            "d003-high-active",
            "d003-medium-d004-backed",
            "d003-low-unknown",
            "d004-medium",
        ],
    )
    def test_single_candidate_role(self, single, expected_role):
        out = resolve(single)
        assert order(out) == [(single.diagnosis.code, expected_role)]

    def test_low_risk_only_single_candidate_is_not_primary(self):
        # Not every single candidate becomes PRIMARY: LOW confidence
        # without an active-failure explanation stays POSSIBLE.
        out = resolve(candidate("D004", confidence=Confidence.LOW))
        assert order(out) == [("D004", DiagnosisRole.POSSIBLE)]


# -- D003 + D004 causal case (plan §13, mandatory) --------------------------------


class TestD003D004CausalCase:
    def test_d004_wins_as_the_causal_parent(self):
        # The exact accepted Phase-9 shape: D003 explains the observed
        # failure and names D004; D004 is risk-only MEDIUM. Truth tables
        # §20.5: PRIMARY = D004 (the causal explanation); D003 provides
        # symptom context as a SECONDARY consequence in the chain.
        out = resolve(d003(parents=("D004",)), d004())
        assert order(out) == [
            ("D004", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
        ]

    def test_causal_parent_outranks_error_severity(self):
        # The ERROR/High-looking symptom diagnosis must NOT win just by
        # looking severe: the explicit causal-parent hint decides first.
        out = resolve(d003(Confidence.HIGH, parents=("D004",)), d004())
        assert order(out)[0] == ("D004", DiagnosisRole.PRIMARY)

    def test_causal_parent_outranks_critical_severity(self):
        scary = candidate(
            "D003",
            confidence=Confidence.HIGH,
            severity=Severity.CRITICAL,
            explains=True,
            risk_only=False,
            parents=("D004",),
        )
        out = resolve(scary, d004())
        assert order(out)[0] == ("D004", DiagnosisRole.PRIMARY)

    def test_permutation_invariance(self):
        baseline = resolve(d003(parents=("D004",)), d004())
        for perm in itertools.permutations((d003(parents=("D004",)), d004())):
            assert resolve(*perm) == baseline


# -- causal-parent semantics (plan §7) ---------------------------------------------


class TestCausalParentSemantics:
    def test_parentage_requires_an_explicit_name(self):
        # D001 + D002 with no naming: neither is a parent, the full tie
        # falls through to code order (truth tables §20.1).
        out = resolve(d001(), d002())
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D002", DiagnosisRole.SECONDARY),
        ]

    def test_d004_coexistence_alone_is_not_parentage(self):
        # A D003 that does NOT name D004 leaves D004 a plain risk-only
        # candidate: D004 never becomes a parent (or PRIMARY) merely by
        # existing next to the symptom diagnosis.
        out = resolve(d003(), d004())
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D004", DiagnosisRole.SECONDARY),
        ]

    def test_title_summary_issues_and_severity_never_infer_parentage(self):
        scary = candidate(
            "D002",
            confidence=Confidence.MEDIUM,
            severity=Severity.CRITICAL,
            title="root cause of the CUDA failure",
            summary="explains why torch.cuda.is_available is false",
            cause_class="SOME_CAUSE",
            related_issues=["TORCH002"],
        )
        genuine = candidate("D004", confidence=Confidence.MEDIUM, explains=True)
        out = resolve(scary, genuine)
        assert order(out)[0] == ("D004", DiagnosisRole.PRIMARY)

    def test_code_numerics_never_infer_parentage(self):
        # D003 does not parent D001/D002 and vice versa through numbering.
        out = resolve(d001(), d002(), d003())
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
            ("D002", DiagnosisRole.SECONDARY),
        ]

    def test_self_reference_never_establishes_parentage(self):
        self_parent = candidate("D001", parents=("D001",))
        other = candidate("D002", confidence=Confidence.MEDIUM)
        out = resolve(self_parent, other)
        # Without the self-name counting, both are plain risk-only HIGH
        # vs MEDIUM: confidence decides, not the bogus self-edge.
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D002", DiagnosisRole.SECONDARY),
        ]

    def test_duplicate_parent_codes_behave_like_one(self):
        once = resolve(d003(parents=("D004",)), d004())
        twice = resolve(d003(parents=("D004", "D004")), d004())
        assert order(once) == order(twice)


# -- missing parent (plan §8) --------------------------------------------------------


class TestMissingParent:
    def test_absent_d004_parent_does_not_crash_or_fabricate(self):
        out = resolve(d003(parents=("D004",)))
        assert order(out) == [("D003", DiagnosisRole.PRIMARY)]
        assert len(out) == 1  # no fabricated D004

    def test_stale_parent_code_is_ignored(self):
        out = resolve(d003(parents=("D009",)), d001())
        # The stale name grants nothing; ranking proceeds normally.
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
        ]

    def test_relationship_to_an_absent_code_passes_through(self):
        out = resolve(d003(parents=("D004",)))
        assert out[0].relationships == [
            DiagnosisRelationship(kind="explains", target_code="D004")
        ]

    def test_child_is_never_reinterpreted_as_proving_the_parent(self):
        # The resolver must not treat the child's naming as evidence that
        # the absent parent exists: the output contains exactly the child.
        out = resolve(d003(parents=("D004",)), d001())
        assert [d.code for d in out] == ["D003", "D001"]


# -- risk-only coexistence (plan §15) --------------------------------------------------


class TestRiskOnlyCoexistence:
    def test_d001_plus_d002_both_high(self):
        out = resolve(d001(Confidence.HIGH), d002(Confidence.HIGH))
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D002", DiagnosisRole.SECONDARY),
        ]

    def test_confidence_beats_code_order(self):
        out = resolve(d001(Confidence.MEDIUM), d002(Confidence.HIGH))
        assert order(out) == [
            ("D002", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
        ]

    def test_d001_plus_d004(self):
        out = resolve(d001(Confidence.HIGH), d004(Confidence.MEDIUM))
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D004", DiagnosisRole.SECONDARY),
        ]

    def test_d002_plus_d004(self):
        out = resolve(d002(Confidence.HIGH), d004(Confidence.MEDIUM))
        assert order(out) == [
            ("D002", DiagnosisRole.PRIMARY),
            ("D004", DiagnosisRole.SECONDARY),
        ]

    def test_all_three_risk_only(self):
        out = resolve(d001(Confidence.HIGH), d002(Confidence.HIGH), d004())
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D002", DiagnosisRole.SECONDARY),
            ("D004", DiagnosisRole.SECONDARY),
        ]
        assert len(out) == 3  # overlapping remediation never collapses them


# -- confidence order (plan §16) ---------------------------------------------------------


class TestConfidenceOrder:
    def test_medium_outranks_low(self):
        # Lexicographic enum values would reverse this ("low" < "medium").
        out = resolve(d001(Confidence.MEDIUM), candidate("D002", confidence=Confidence.LOW))
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D002", DiagnosisRole.POSSIBLE),
        ]

    def test_high_outranks_medium(self):
        out = resolve(d001(Confidence.HIGH), d002(Confidence.MEDIUM))
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)

    def test_confidence_outranks_severity(self):
        out = resolve(
            candidate("D001", confidence=Confidence.HIGH, severity=Severity.WARNING),
            candidate("D002", confidence=Confidence.MEDIUM, severity=Severity.ERROR),
        )
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)

    def test_resolver_never_upgrades_confidence(self):
        out = resolve(d003_unknown(), d001(Confidence.HIGH))
        by_code = {d.code: d for d in out}
        assert by_code["D003"].confidence is Confidence.LOW
        assert by_code["D001"].confidence is Confidence.HIGH


# -- severity order (plan §17) -------------------------------------------------------------


class TestSeverityOrder:
    @pytest.mark.parametrize(
        "higher,lower",
        [
            (Severity.CRITICAL, Severity.ERROR),
            (Severity.ERROR, Severity.WARNING),
            (Severity.WARNING, Severity.INFO),
        ],
        ids=["critical-over-error", "error-over-warning", "warning-over-info"],
    )
    def test_frozen_severity_order(self, higher, lower):
        # WARNING outranking INFO is the lexicographic trap ("info" sorts
        # before "warning"); the resolver-local rank decides instead.
        out = resolve(
            candidate("D001", severity=higher, explains=True),
            candidate("D002", severity=lower, explains=True),
        )
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)

    def test_severity_never_mutated(self):
        out = resolve(
            candidate("D001", severity=Severity.ERROR, explains=True),
            candidate("D002", severity=Severity.INFO, explains=True),
        )
        assert {d.code: d.severity for d in out} == {
            "D001": Severity.ERROR,
            "D002": Severity.INFO,
        }


# -- SAFE-remediation tie-break (plan §18) ----------------------------------------------------


class TestSafeRemediationTieBreak:
    def test_qualifying_safe_remediation_wins(self):
        # Identical ranking dimensions except §12.5: the candidate with an
        # explicit SAFE command outranks code order.
        with_command = candidate("D002", steps=[safe_command_step()])
        without = candidate("D001", steps=[plain_step()])
        out = resolve(with_command, without)
        assert order(out)[0] == ("D002", DiagnosisRole.PRIMARY)

    def test_caution_command_does_not_qualify(self):
        caution = RemediationStep(
            action="careful act",
            why="because",
            safety=RemediationSafety.CAUTION,
            command="echo risky",
        )
        out = resolve(
            candidate("D001", steps=[caution]),
            candidate("D002", steps=[safe_command_step()]),
        )
        assert order(out)[0] == ("D002", DiagnosisRole.PRIMARY)

    def test_safe_without_command_does_not_qualify(self):
        out = resolve(
            candidate("D001", steps=[plain_step()]),
            candidate("D002", steps=[safe_command_step()]),
        )
        assert order(out)[0] == ("D002", DiagnosisRole.PRIMARY)

    def test_step_count_never_ranks(self):
        many = candidate("D001", steps=[plain_step(f"step {i}") for i in range(5)])
        one = candidate("D002", steps=[plain_step()])
        out = resolve(many, one)
        # Neither has a command: §12.5 ties and code order decides.
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)

    def test_prose_length_never_ranks(self):
        verbose = candidate(
            "D001",
            steps=[RemediationStep(
                action="a" * 500, why="w" * 500, safety=RemediationSafety.SAFE
            )],
        )
        terse = candidate("D002", steps=[plain_step("b")])
        out = resolve(verbose, terse)
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)


# -- code-order tie-break and permutations (plan §19/§27) ----------------------------------------


class TestDeterministicOrder:
    def test_fully_tied_candidates_fall_to_code_order(self):
        out = resolve(d001(), d002())
        assert order(out) == [("D001", DiagnosisRole.PRIMARY),
                              ("D002", DiagnosisRole.SECONDARY)]

    @pytest.mark.parametrize(
        "case",
        [
            (d001(Confidence.HIGH), d002(Confidence.HIGH), d004()),
            (d003(parents=("D004",)), d004()),
            (
                d001(Confidence.HIGH),
                d002(Confidence.MEDIUM),
                d003(parents=("D004",)),
                d004(),
            ),
        ],
        ids=["risk-trio", "causal-pair", "all-four"],
    )
    def test_permutation_invariance(self, case):
        baseline = resolve(*case)
        for perm in itertools.permutations(case):
            assert resolve(*perm) == baseline

    def test_all_four_expected_order(self):
        out = resolve(
            d001(Confidence.HIGH),
            d002(Confidence.MEDIUM),
            d003(parents=("D004",)),
            d004(),
        )
        assert order(out) == [
            ("D004", DiagnosisRole.PRIMARY),  # §12.1 causal parent
            ("D003", DiagnosisRole.SECONDARY),  # §12.2 explains the failure
            ("D001", DiagnosisRole.SECONDARY),  # §12.3 HIGH risk finding
            ("D002", DiagnosisRole.SECONDARY),  # §12.3 MEDIUM risk finding
        ]

    def test_host_environment_changes_nothing(self, monkeypatch):
        before = resolve(d003(parents=("D004",)), d004())
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/host/legacy/cuda-99/lib64")
        monkeypatch.setenv("CUDA_HOME", "/host/cuda-99.0")
        assert resolve(d003(parents=("D004",)), d004()) == before


# -- role assignment matrix (plan §20) -------------------------------------------------------------


class TestRoleAssignment:
    def test_all_four_roles_in_one_resolution(self):
        # Winner: the only eligible candidate (LOW but explains). Losers:
        # the code the winner names -> CONTRIBUTING; a MEDIUM risk finding
        # -> SECONDARY; a LOW risk finding -> POSSIBLE.
        out = resolve(
            candidate(
                "D003",
                confidence=Confidence.LOW,
                severity=Severity.ERROR,
                explains=True,
                risk_only=False,
                parents=("D004",),
            ),
            candidate("D004", confidence=Confidence.LOW),
            d001(Confidence.MEDIUM),
            candidate("D002", confidence=Confidence.LOW),
        )
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
            ("D004", DiagnosisRole.CONTRIBUTING),
            ("D002", DiagnosisRole.POSSIBLE),
        ]

    def test_primary_winner_does_not_make_every_loser_secondary(self):
        out = resolve(
            d003_unknown(),
            d001(Confidence.HIGH),
            candidate("D002", confidence=Confidence.LOW),
        )
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
            ("D002", DiagnosisRole.POSSIBLE),
        ]

    def test_named_consequence_gets_secondary_not_contributing(self):
        # The D003 symptom diagnosis BESIDE a primary D004 is a valid
        # consequence in the causal chain (semantics §12.2), not a
        # contributing condition (§12.3).
        out = resolve(d003(parents=("D004",)), d004())
        assert roles(out)["D003"] is DiagnosisRole.SECONDARY


# -- LOW-confidence behavior (plan §21) ------------------------------------------------------------


class TestLowConfidenceBehavior:
    def test_d003_unknown_alone(self):
        out = resolve(d003_unknown())
        assert order(out) == [("D003", DiagnosisRole.PRIMARY)]

    def test_d003_unknown_with_medium_risk(self):
        out = resolve(d003_unknown(), d004(Confidence.MEDIUM))
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D004", DiagnosisRole.SECONDARY),
        ]

    def test_d003_unknown_with_high_risk(self):
        # Truth tables §20.8 exactly: PRIMARY = D003, SECONDARY = D001.
        out = resolve(d003_unknown(), d001(Confidence.HIGH))
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
        ]

    def test_error_severity_alone_never_promotes_low(self):
        out = resolve(
            candidate("D002", confidence=Confidence.LOW, severity=Severity.ERROR)
        )
        assert order(out) == [("D002", DiagnosisRole.POSSIBLE)]

    def test_low_loser_stays_possible_beside_a_high_winner(self):
        out = resolve(
            d001(Confidence.HIGH),
            candidate("D004", confidence=Confidence.LOW),
        )
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D004", DiagnosisRole.POSSIBLE),
        ]


# -- worked examples PR-01..PR-06 (truth tables §21) -----------------------------------------------


class TestWorkedExamples:
    def test_pr01_d001_over_d002_by_code_order(self):
        out = resolve(d001(Confidence.HIGH), d002(Confidence.HIGH))
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)

    def test_pr02_d003_cpu_only_over_d001(self):
        out = resolve(d003(Confidence.HIGH), d001(Confidence.HIGH))
        assert order(out)[0] == ("D003", DiagnosisRole.PRIMARY)

    def test_pr03_d004_explaining_d003_wins(self):
        out = resolve(d003(parents=("D004",)), d004())
        assert order(out)[0] == ("D004", DiagnosisRole.PRIMARY)

    def test_pr04_d001_high_over_d004_medium(self):
        out = resolve(d001(Confidence.HIGH), d004(Confidence.MEDIUM))
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)

    def test_pr05_d003_unknown_over_d002_high(self):
        out = resolve(d002(Confidence.HIGH), d003_unknown())
        assert order(out)[0] == ("D003", DiagnosisRole.PRIMARY)

    def test_pr06_equal_independents_fall_to_lower_code(self):
        out = resolve(d002(Confidence.HIGH), d001(Confidence.HIGH))
        assert order(out)[0] == ("D001", DiagnosisRole.PRIMARY)


# -- risk-only semantics (plan §22) ----------------------------------------------------------------


class TestRiskOnlySemantics:
    def test_risk_only_is_not_a_public_field(self):
        out = resolve(d001())
        assert all("risk_only" not in {f.name for f in fields(d)} for d in out)

    def test_candidate_risk_only_hint_is_never_mutated(self):
        risk = d004()
        active = d003()
        resolve(active, risk)
        assert risk.risk_only is True
        assert active.risk_only is False


# -- active-failure semantics (plan §23) -----------------------------------------------------------


class TestActiveFailureSemantics:
    def test_hint_decides_not_severity_or_issues(self):
        looks_active = candidate(
            "D001",
            confidence=Confidence.MEDIUM,
            severity=Severity.ERROR,
            title="PyTorch CUDA unavailable",
            summary="torch.cuda.is_available() is false",
            related_issues=["TORCH002"],
        )
        plain = candidate("D004", confidence=Confidence.MEDIUM, explains=True)
        out = resolve(looks_active, plain)
        assert order(out)[0] == ("D004", DiagnosisRole.PRIMARY)

    def test_flipping_the_hint_flips_the_outcome(self):
        with_hint = candidate("D004", confidence=Confidence.MEDIUM, explains=True)
        without_hint = candidate("D004", confidence=Confidence.MEDIUM, explains=False)
        competitor = d001(Confidence.HIGH)
        assert order(resolve(with_hint, competitor))[0] == ("D004", DiagnosisRole.PRIMARY)
        assert order(resolve(without_hint, competitor))[0] == ("D001", DiagnosisRole.PRIMARY)

    def test_cause_class_and_title_never_imply_the_hint(self):
        disguised = candidate(
            "D001",
            confidence=Confidence.HIGH,
            cause_class="RUNTIME_LIBRARY_CONFLICT",
            title="PyTorch CUDA unavailable",
        )
        genuine = d003(Confidence.MEDIUM)
        out = resolve(disguised, genuine)
        assert order(out)[0] == ("D003", DiagnosisRole.PRIMARY)


# -- defensive validation (plan §24) ---------------------------------------------------------------


class TestCandidateValidation:
    def test_pre_set_role_is_overwritten(self):
        self_declared = candidate("D001", role=DiagnosisRole.PRIMARY)
        out = resolve(self_declared, d003(Confidence.MEDIUM))
        # Rules must never self-declare PRIMARY (architecture §6.4): the
        # resolver deterministically overwrites, neither honoring nor
        # raising on the malformed input.
        assert order(out) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
        ]

    def test_pre_set_role_on_a_loser_is_overwritten(self):
        presumptuous = candidate("D002", role=DiagnosisRole.PRIMARY)
        out = resolve(d001(Confidence.HIGH), presumptuous)
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D002", DiagnosisRole.SECONDARY),
        ]

    def test_validation_never_discards_candidates(self):
        noisy = [
            candidate("D001", parents=("D001",)),
            candidate("D002", parents=("D009", "D009")),
            d003(parents=("D004", "D004")),
            d004(),
        ]
        out = resolve(*noisy)
        assert len(out) == 4


# -- duplicate codes (plan §25) --------------------------------------------------------------------


class TestDuplicateCodes:
    def test_both_duplicates_are_kept(self):
        out = resolve(candidate("D001", title="alpha"),
                      candidate("D001", title="beta"))
        assert len(out) == 2
        assert order(out) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
        ]

    def test_input_order_never_decides_the_primary(self):
        alpha, beta = candidate("D001", title="alpha"), candidate(
            "D001", title="beta"
        )
        for perm in itertools.permutations((alpha, beta)):
            out = resolve(*perm)
            assert out[0].title == "alpha"
            assert out[0].role is DiagnosisRole.PRIMARY
            assert out[1].title == "beta"
            assert out[1].role is DiagnosisRole.SECONDARY

    def test_low_duplicates_yield_no_primary(self):
        out = resolve(
            candidate("D001", confidence=Confidence.LOW, title="alpha"),
            candidate("D001", confidence=Confidence.LOW, title="beta"),
        )
        assert order(out) == [
            ("D001", DiagnosisRole.POSSIBLE),
            ("D001", DiagnosisRole.POSSIBLE),
        ]

    def test_same_title_different_payload_is_permutation_invariant(self):
        # Maintainer-review regression: identical code, title, and every
        # frozen ranking dimension (same confidence, severity, and
        # SAFE-command remediation state), differing only in non-ranking
        # payload (summary + remediation prose). code+title alone is not
        # a total order — the payload fingerprint decides deterministically
        # so input order never picks which payload becomes PRIMARY.
        def twin(tag: str) -> DiagnosisCandidate:
            return candidate(
                "D001",
                title="same title",
                summary=f"payload {tag}",
                steps=[
                    RemediationStep(
                        action=f"act {tag}",
                        why="because",
                        safety=RemediationSafety.SAFE,
                    )
                ],
            )

        alpha, beta = twin("alpha"), twin("beta")
        forward = resolve(alpha, beta)
        reverse = resolve(beta, alpha)
        assert forward == reverse
        assert order(forward) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
        ]
        # The concrete payload receiving each role is fixed in both
        # orders: the lexically smaller serialized payload ("payload
        # alpha") is always the PRIMARY, never the input-first one.
        assert forward[0].summary == "payload alpha"
        assert forward[1].summary == "payload beta"
        assert reverse[0].summary == "payload alpha"
        assert reverse[1].summary == "payload beta"

    def test_fully_identical_duplicates_remain_deterministic(self):
        # Fully equal duplicates tie on the complete key including the
        # fingerprint: either object may be min()'s pick, but the
        # finalized outputs are equal either way — deterministic, stable,
        # and no crash.
        a = candidate("D001", title="same title")
        b = candidate("D001", title="same title")
        forward = resolve(a, b)
        reverse = resolve(b, a)
        assert forward == reverse
        assert order(forward) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D001", DiagnosisRole.SECONDARY),
        ]
        assert resolve(a, b) == forward  # stable across repeated resolution

    def test_different_hints_same_payload_is_permutation_invariant(self):
        # Maintainer follow-up regression: identical diagnosis payloads
        # (and code/title/confidence/severity/explains) but DIFFERENT
        # causal_parent_codes. The envelope hints feed §12.1 and the
        # loser-role matrix, so a diagnosis-only fingerprint still ties
        # and input order would decide the winner — flipping a third
        # candidate's role (CONTRIBUTING vs POSSIBLE). The candidate-
        # level fingerprint must decide deterministically.
        d003_payload = {
            "severity": Severity.ERROR,
            "summary": "same D003 summary",
            "title": "same D003 title",
        }
        named = candidate(
            "D003",
            confidence=Confidence.MEDIUM,
            explains=True,
            risk_only=False,
            parents=("D004",),
            **d003_payload,
        )
        unnamed = candidate(
            "D003",
            confidence=Confidence.MEDIUM,
            explains=True,
            risk_only=False,
            parents=(),
            **d003_payload,
        )
        ineligible_d004 = candidate("D004", confidence=Confidence.LOW)
        assert ineligible_d004.explains_active_failure is False
        # LOW confidence without an active-failure explanation keeps D004
        # out of the PRIMARY competition even though A names it (§12.1).
        forward = resolve(named, unnamed, ineligible_d004)
        reverse = resolve(unnamed, named, ineligible_d004)
        assert forward == reverse
        # Deterministic outcome: the parent-naming twin wins the
        # fingerprint tie (('D004',) sorts before ()), so its named D004
        # takes the matrix's CONTRIBUTING slot — in both input orders.
        assert order(forward) == [
            ("D003", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
            ("D004", DiagnosisRole.CONTRIBUTING),
        ]
        assert order(reverse)[2] == ("D004", DiagnosisRole.CONTRIBUTING)


# -- cycles (plan §26) -----------------------------------------------------------------------------


class TestCycles:
    def test_two_node_cycle_terminates_deterministically(self):
        a = candidate("D001", parents=("D002",))
        b = candidate("D002", parents=("D001",))
        baseline = resolve(a, b)
        # Parentage ties (both name each other), so the winner comes from
        # the frozen fallback order — never from the cycle — and the loser
        # the winner names takes the standard matrix's CONTRIBUTING slot.
        assert order(baseline) == [
            ("D001", DiagnosisRole.PRIMARY),
            ("D002", DiagnosisRole.CONTRIBUTING),
        ]
        for perm in itertools.permutations((a, b)):
            assert resolve(*perm) == baseline

    def test_no_causal_winner_is_invented_from_a_cycle(self):
        # Both sides name each other, so parentage ties and cannot crown a
        # "winner" by itself; the tie falls through to the frozen order.
        a = candidate("D001", parents=("D002",))
        b = candidate("D002", parents=("D001",))
        out = resolve(a, b)
        assert sum(d.role is DiagnosisRole.PRIMARY for d in out) == 1


# -- payload preservation / immutability (plan §10/§28/§29/§30) ------------------------------------


def assert_payload_preserved(original: Diagnosis, final: Diagnosis) -> None:
    assert final.code == original.code
    assert final.title == original.title
    assert final.severity is original.severity
    assert final.confidence is original.confidence
    assert final.summary == original.summary
    assert final.cause_class == original.cause_class
    assert final.evidence == original.evidence
    assert final.impact == original.impact
    assert final.remediation == original.remediation
    assert final.verification == original.verification
    assert final.related_issues == original.related_issues
    assert final.relationships == original.relationships
    assert final.exclusions == original.exclusions
    assert final.presentation_suppressed_issue_codes == (
        original.presentation_suppressed_issue_codes
    )
    # The serialized payload is byte-identical once the role is reset.
    assert replace(final, role=original.role).to_dict() == original.to_dict()


class TestPayloadPreservation:
    def test_every_rule_field_survives_except_role(self):
        inputs = [d003(parents=("D004",)), d004(), d001(), d002()]
        out = resolve(*inputs)
        by_code: dict[str, Diagnosis] = {}
        for diagnosis in out:
            by_code.setdefault(diagnosis.code, diagnosis)
        for original in inputs:
            assert_payload_preserved(original.diagnosis, by_code[original.diagnosis.code])

    def test_suppression_carrier_passes_through_untouched(self):
        out = resolve(d003(parents=("D004",)))
        assert out[0].presentation_suppressed_issue_codes == ["TORCH002"]

    def test_resolver_performs_no_issue_suppression(self):
        # The carrier removes nothing: related issues survive verbatim.
        with_issues = candidate("D001", related_issues=["TORCH002", "ENV001"])
        out = resolve(with_issues)
        assert out[0].related_issues == ["TORCH002", "ENV001"]

    def test_role_is_the_only_owned_transformation(self):
        original = d003(parents=("D004",))
        out = resolve(original, d004())
        winner = next(d for d in out if d.code == "D003")
        twin = replace(original.diagnosis, role=winner.role)
        assert winner == twin


class TestInputImmutability:
    def test_candidates_stay_role_none(self):
        inputs = [d003(parents=("D004",)), d004(), d001(), d002()]
        resolve(*inputs)
        assert all(c.diagnosis.role is None for c in inputs)

    def test_candidates_are_untouched_whole_objects(self):
        inputs = [d003(parents=("D004",)), d004(), d001()]
        before = copy.deepcopy(inputs)
        resolve(*inputs)
        assert inputs == before

    def test_resolving_twice_is_stable(self):
        inputs = [d003(parents=("D004",)), d004()]
        first = resolve(*inputs)
        second = resolve(*inputs)
        assert first == second
        assert all(c.diagnosis.role is None for c in inputs)


# -- relationships (plan §9) -----------------------------------------------------------------------


class TestRelationships:
    def test_rule_carried_relationship_passes_through_verbatim(self):
        out = resolve(d003(parents=("D004",)), d004())
        by_code = {d.code: d for d in out}
        assert by_code["D003"].relationships == [
            DiagnosisRelationship(kind="explains", target_code="D004")
        ]

    def test_no_reciprocal_edge_is_invented(self):
        out = resolve(d003(parents=("D004",)), d004())
        by_code = {d.code: d for d in out}
        assert by_code["D004"].relationships == []

    def test_exact_duplicates_collapse(self):
        duplicated = candidate(
            "D003",
            explains=True,
            risk_only=False,
            severity=Severity.ERROR,
            relationships=[
                DiagnosisRelationship(kind="explains", target_code="D004"),
                DiagnosisRelationship(kind="explains", target_code="D004"),
            ],
        )
        out = resolve(duplicated)
        assert out[0].relationships == [
            DiagnosisRelationship(kind="explains", target_code="D004")
        ]

    def test_distinct_relationships_are_all_kept(self):
        varied = candidate(
            "D003",
            relationships=[
                DiagnosisRelationship(kind="explains", target_code="D004"),
                DiagnosisRelationship(kind="related_to", target_code="D004"),
                DiagnosisRelationship(kind="explains", target_code="D001"),
            ],
        )
        out = resolve(varied)
        assert len(out[0].relationships) == 3

    def test_no_relationships_are_invented_from_coexistence(self):
        out = resolve(d001(), d002(), d004())
        assert all(d.relationships == [] for d in out)


# -- real-rule contract test (plan §34) ------------------------------------------------------------


def real_conflict_candidates() -> tuple[DiagnosisCandidate, DiagnosisCandidate]:
    """Real accepted rule candidates: a real D004 firing over a canonical
    Revision-3 same-name-shadowing inventory, then the real D003 conflict
    branch consuming that D004 as its prior candidate."""
    snapshot = base_snapshot()
    snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
    facts_value = DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)
    facts_value = replace(
        facts_value,
        runtime_libraries=RuntimeLibraryFacts(
            candidates=(
                RuntimeLibraryCandidateFact(
                    family="libcudart",
                    load_name="libcudart.so.12",
                    load_name_family_version="12",
                    target_family_version="11",
                    reliability=IdentityReliability.DIRECT,
                    origin=ORIGIN_LD_LIBRARY_PATH,
                    active_toolkit=Trinary.UNKNOWN,
                    modeled_search_order=0,
                ),
                RuntimeLibraryCandidateFact(
                    family="libcudart",
                    load_name="libcudart.so.12",
                    load_name_family_version="12",
                    target_family_version="12",
                    reliability=IdentityReliability.DIRECT,
                    origin=ORIGIN_LD_LIBRARY_PATH,
                    active_toolkit=Trinary.UNKNOWN,
                    modeled_search_order=1,
                ),
            ),
            torch_runtime_family="12",
            package_local_cudnn_family=None,
            scan_incomplete=False,
        ),
    )
    real_d004 = D004RuntimeLibraryConflict().evaluate(facts_value, [])
    assert real_d004 is not None, "expected a real D004 candidate"
    real_d003 = D003PyTorchCudaUnavailable().evaluate(
        facts_value, [], {"D004": real_d004}
    )
    assert real_d003 is not None, "expected a real D003 candidate"
    return real_d003, real_d004


class TestRealRuleContract:
    def test_real_candidate_hints_line_up(self):
        real_d003, real_d004 = real_conflict_candidates()
        assert real_d003.explains_active_failure is True
        assert real_d003.risk_only is False
        assert real_d003.causal_parent_codes == ("D004",)
        assert real_d003.diagnosis.confidence is Confidence.MEDIUM
        assert real_d003.diagnosis.severity is Severity.ERROR
        assert real_d004.explains_active_failure is False
        assert real_d004.risk_only is True
        # Cycle assumption pin (plan §26): the only parent edge any real
        # v0.2.0 rule carries is D003 -> D004; D004 carries none.
        assert real_d004.causal_parent_codes == ()

    def test_real_pair_resolves_to_the_frozen_roles(self):
        real_d003, real_d004 = real_conflict_candidates()
        out = RESOLVER.resolve([real_d003, real_d004])
        assert order(out) == [
            ("D004", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
        ]

    def test_real_payloads_survive_except_role(self):
        real_d003, real_d004 = real_conflict_candidates()
        out = RESOLVER.resolve([real_d003, real_d004])
        by_code = {d.code: d for d in out}
        assert_payload_preserved(real_d003.diagnosis, by_code["D003"])
        assert_payload_preserved(real_d004.diagnosis, by_code["D004"])

    def test_real_candidates_stay_role_none(self):
        real_d003, real_d004 = real_conflict_candidates()
        RESOLVER.resolve([real_d003, real_d004])
        assert real_d003.diagnosis.role is None
        assert real_d004.diagnosis.role is None

    def test_real_d003_relationship_direction_preserved(self):
        real_d003, _ = real_conflict_candidates()
        out = RESOLVER.resolve([real_d003])
        assert out[0].relationships == [
            DiagnosisRelationship(kind="explains", target_code="D004")
        ]

    def test_real_d004_alone_is_primary(self):
        _, real_d004 = real_conflict_candidates()
        out = RESOLVER.resolve([real_d004])
        assert order(out) == [("D004", DiagnosisRole.PRIMARY)]


# -- no engine integration (plan §31) --------------------------------------------------------------


class TestNoEngineIntegration:
    def test_engine_still_emits_no_diagnoses(self):
        result = DiagnosisEngine().run(base_snapshot())
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}

    def test_resolver_module_never_imports_the_engine(self):
        tree = ast.parse(Path(resolver_module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "") != "cuda_doctor.diagnosis.engine"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "cuda_doctor.diagnosis.engine"


# -- purity (plan §32) -----------------------------------------------------------------------------


class TestPurity:
    def test_imports_are_purely_local_policy(self):
        tree = ast.parse(Path(resolver_module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        allowed = {
            "__future__",
            "collections.abc",
            "dataclasses",
            "cuda_doctor.core.enums",
            "cuda_doctor.diagnosis.models",
        }
        assert imported <= allowed, imported - allowed
        assert imported.isdisjoint(
            {"os", "subprocess", "shutil", "site", "time", "random", "torch",
             "pathlib", "json"}
        )

    def test_source_names_no_host_or_fact_surfaces(self):
        source = Path(resolver_module.__file__).read_text()
        for forbidden in (
            "EnvironmentSnapshot",
            "DiagnosisFactBuilder",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "subprocess",
            "environ",
            "DiagnosisEngine",
            "load_compatibility",
            "collectors",
            "socket",
            "urlopen",
            "open(",
        ):
            assert forbidden not in source, forbidden

    def test_resolver_is_deterministic(self):
        inputs = [d003(parents=("D004",)), d004(), d001()]
        assert RESOLVER.resolve(inputs) == RESOLVER.resolve(list(inputs))
