"""Tests for the diagnosis engine: aggregation, ordering, and isolation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest
from tests.factories import base_snapshot, context_for

from cuda_doctor.checks import Check, default_checks
from cuda_doctor.checks.compiler_checks import NoHostCompiler
from cuda_doctor.collectors.cuda import VERSION_SOURCE_DIRECT
from cuda_doctor.collectors.runtime_libraries import ORIGIN_LD_LIBRARY_PATH
from cuda_doctor.compatibility import CompatibilityData, load_compatibility_data
from cuda_doctor.core.enums import EnvironmentStatus, Severity
from cuda_doctor.core.exceptions import CompatibilityDataError
from cuda_doctor.core.models import (
    CUDASelectorObservation,
    EnvironmentSnapshot,
    RuntimeLibraryCandidate,
    RuntimeLibraryInventory,
)
from cuda_doctor.diagnosis import engine as engine_module
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.facts import DiagnosisFactBuilder, DiagnosisFacts
from cuda_doctor.diagnosis.issue import Issue, Summary
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    DiagnosisCandidate,
    DiagnosisRelationship,
    DiagnosisRole,
)
from cuda_doctor.diagnosis.resolver import DiagnosisResolver
from cuda_doctor.diagnosis.rules import (
    D001ToolkitResolutionConflict,
    D002TorchToolkitMismatch,
    D003PyTorchCudaUnavailable,
    D004RuntimeLibraryConflict,
)
from cuda_doctor.diagnosis.rules.base import NO_PRIOR_CANDIDATES
from cuda_doctor.reporters.base import ReportInputs
from cuda_doctor.reporters.json_report import JsonReporter

COMPATIBILITY = load_compatibility_data()


class ExplodingCheck(Check):
    code = "XXX001"
    category = "test"

    def run(self, ctx) -> list[Issue]:
        raise RuntimeError("boom")


class CriticalCheck(Check):
    code = "XXX002"
    category = "test"

    def run(self, ctx) -> list[Issue]:
        return [
            Issue(
                code=self.code,
                severity=Severity.CRITICAL,
                title="catastrophe",
                category=self.category,
            )
        ]


class ErrorCheckA(Check):
    code = "AAA000"
    category = "test"

    def run(self, ctx) -> list[Issue]:
        return [Issue(code=self.code, severity=Severity.ERROR, title="a", category="test")]


class ErrorCheckZ(Check):
    code = "ZZZ000"
    category = "test"

    def run(self, ctx) -> list[Issue]:
        return [Issue(code=self.code, severity=Severity.ERROR, title="z", category="test")]


class WarningCheck(Check):
    code = "AAA001"
    category = "test"

    def run(self, ctx) -> list[Issue]:
        return [Issue(code=self.code, severity=Severity.WARNING, title="w", category="test")]


class TestEngine:
    def test_healthy_snapshot_yields_no_issues(self):
        result = DiagnosisEngine().run(base_snapshot())
        assert result.issues == []
        assert result.summary.status is EnvironmentStatus.HEALTHY
        assert result.check_errors == {}

    def test_error_leads_to_degraded(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        result = DiagnosisEngine().run(snapshot)
        codes = [issue.code for issue in result.issues]
        assert "TORCH002" in codes
        assert result.summary.status is EnvironmentStatus.DEGRADED
        assert result.summary.errors >= 1

    def test_warning_only_is_usable_with_warnings(self):
        snapshot = base_snapshot()
        snapshot.compiler = replace(snapshot.compiler, compilers=[])
        result = DiagnosisEngine().run(snapshot)
        assert result.summary.errors == 0
        assert result.summary.warnings >= 1
        assert result.summary.status is EnvironmentStatus.USABLE_WITH_WARNINGS

    def test_critical_leads_to_broken(self):
        result = DiagnosisEngine(checks=[CriticalCheck()]).run(base_snapshot())
        assert result.summary.status is EnvironmentStatus.BROKEN

    def test_issues_sorted_by_severity_then_code(self):
        engine = DiagnosisEngine(checks=[WarningCheck(), ErrorCheckZ(), ErrorCheckA()])
        result = engine.run(base_snapshot())
        assert [issue.code for issue in result.issues] == ["AAA000", "ZZZ000", "AAA001"]

    def test_summary_counts_match_issues(self):
        snapshot = base_snapshot()
        snapshot.compiler = replace(snapshot.compiler, compilers=[])
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        result = DiagnosisEngine().run(snapshot)
        counts = (
            result.summary.critical
            + result.summary.errors
            + result.summary.warnings
            + result.summary.info
        )
        assert counts == len(result.issues)

    def test_crashing_check_is_isolated(self):
        engine = DiagnosisEngine(checks=[ExplodingCheck(), NoHostCompiler()])
        snapshot = base_snapshot()
        snapshot.compiler = replace(snapshot.compiler, compilers=[])
        result = engine.run(snapshot)
        # Errors are keyed by check class name, not issue code.
        assert result.check_errors == {"ExplodingCheck": "RuntimeError: boom"}
        assert [issue.code for issue in result.issues] == ["CMP001"]

    def test_two_crashing_checks_sharing_a_code_are_both_recorded(self):
        class AlsoExploding(ExplodingCheck):
            def run(self, ctx) -> list[Issue]:
                raise ValueError("different failure")

        engine = DiagnosisEngine(checks=[ExplodingCheck(), AlsoExploding()])
        result = engine.run(base_snapshot())
        assert set(result.check_errors) == {"ExplodingCheck", "AlsoExploding"}
        assert "RuntimeError: boom" in result.check_errors["ExplodingCheck"]
        assert "ValueError: different failure" in result.check_errors["AlsoExploding"]

    def test_compatibility_load_failure_is_recorded_not_fatal(self, monkeypatch):
        def broken():
            raise CompatibilityDataError("unreadable")

        monkeypatch.setattr(engine_module, "load_compatibility_data", broken)
        snapshot = base_snapshot()
        snapshot.compiler = replace(
            snapshot.compiler, compilers=[]
        )  # CMP001 still runs fine
        result = DiagnosisEngine().run(snapshot)
        assert "compatibility-data" in result.check_errors
        assert [issue.code for issue in result.issues] == ["CMP001"]

    def test_injected_compatibility_skips_loading(self, monkeypatch):
        def broken():
            raise AssertionError("must not be called")

        monkeypatch.setattr(engine_module, "load_compatibility_data", broken)
        compat = context_for(base_snapshot()).compatibility
        assert compat is not None
        result = DiagnosisEngine(compatibility=compat).run(base_snapshot())
        assert result.check_errors == {}

    def test_default_checks_are_fresh_instances(self):
        first = default_checks()
        second = default_checks()
        assert len(first) == len(second)
        assert all(type(a) is type(b) for a, b in zip(first, second, strict=True))
        assert all(a is not b for a, b in zip(first, second, strict=True))
        assert all(isinstance(check, Check) for check in first)

    def test_real_machine_style_snapshot_end_to_end(self):
        """Rich snapshot: multiple toolkits + torch/runtime difference (this
        mirrors the development machine: 4 toolkits, torch cu124)."""
        from cuda_doctor.core.models import CUDAInstallation

        snapshot = base_snapshot()
        snapshot.cuda = replace(
            snapshot.cuda,
            toolkit_version="12.4",
            installations=[
                CUDAInstallation("/usr/local/cuda-12.9", "12.9"),
                CUDAInstallation("/usr/local/cuda-12.8", "12.8"),
                CUDAInstallation("/usr/local/cuda-12.6", "12.6"),
                CUDAInstallation("/usr/local/cuda-12.4", "12.4"),
            ],
        )
        snapshot.pytorch = replace(snapshot.pytorch, version="2.6.0+cu124", cuda_version="12.4")
        result = DiagnosisEngine().run(snapshot)
        codes = [issue.code for issue in result.issues]
        assert "CUDA004" in codes  # multiple toolkits -> info
        assert all(i.severity is Severity.INFO for i in result.issues)
        assert result.summary.status is EnvironmentStatus.HEALTHY


class TestSummary:
    def test_broken_on_critical(self):
        issue = Issue(code="X", severity=Severity.CRITICAL, title="t")
        assert Summary.from_issues([issue]).status is EnvironmentStatus.BROKEN

    def test_counts(self):
        issues = [
            Issue(code="A", severity=Severity.ERROR, title="t"),
            Issue(code="B", severity=Severity.WARNING, title="t"),
            Issue(code="C", severity=Severity.WARNING, title="t"),
            Issue(code="D", severity=Severity.INFO, title="t"),
        ]
        summary = Summary.from_issues(issues)
        assert (summary.errors, summary.warnings, summary.info) == (1, 2, 1)

    def test_to_dict_json_ready(self):
        data = Summary.from_issues([]).to_dict()
        assert data["status"] == "HEALTHY"
        assert set(data) == {"critical", "errors", "warnings", "info", "status"}


class TestIssueToDict:
    def test_redacts_home_paths(self, fake_home):
        issue = Issue(
            code="T",
            severity=Severity.INFO,
            title="t",
            evidence=["/home/secretuser/venv/lib/libcudart.so"],
        )
        data = issue.to_dict()
        joined = str(data)
        assert "secretuser" not in joined
        assert "~" in joined

    def test_severity_serialized_as_string(self):
        data = Issue(code="T", severity=Severity.WARNING, title="t").to_dict()
        assert data["severity"] == "WARNING"


# -- Phase-11 diagnosis pipeline integration -------------------------------------
#
# The engine now orchestrates the accepted pipeline additively after the
# check phase: one fact build -> D001 -> D002 -> D004 -> D003 (prior-candidate
# propagation) -> one resolver call. Everything below pins that orchestration
# and the boundary: issues/summary/exit semantics stay issue-only.


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


def _snapshot_with_selectors(*observations: CUDASelectorObservation) -> EnvironmentSnapshot:
    snapshot = base_snapshot()
    snapshot.cuda = replace(snapshot.cuda, selector_observations=list(observations))
    return snapshot


def _row_c_inventory() -> RuntimeLibraryInventory:
    """The canonical Revision-3 Class-A row-C inventory: the earliest LDP
    libcudart.so.12 (name claims family 12 == torch baseline) shadowing a
    canonical 11-series target, with a baseline-matching copy later."""
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


def d001_snapshot() -> EnvironmentSnapshot:
    """Two participating direct selectors resolving to different toolkits."""
    return _snapshot_with_selectors(
        _selector("cuda_home", "/opt/cuda-11.8", "11.8"),
        _selector("nvcc", "/usr/local/cuda-12.4", "12.4"),
    )


def d002_snapshot() -> EnvironmentSnapshot:
    """nvcc reference 11.8 vs the torch cu124 build -> MAJOR_INCOMPATIBLE."""
    return _snapshot_with_selectors(_selector("nvcc", "/usr/local/cuda-11.8", "11.8"))


def d004_snapshot() -> EnvironmentSnapshot:
    snapshot = base_snapshot()
    snapshot.runtime_libraries = _row_c_inventory()
    return snapshot


def d003_snapshot() -> EnvironmentSnapshot:
    """CUDA build with cuda_available=False and no D004: D003 lands on the
    frozen UNKNOWN cause (LOW confidence), never on the conflict branch."""
    snapshot = base_snapshot()
    snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
    return snapshot


def critical_snapshot() -> EnvironmentSnapshot:
    """The mandatory Phase-11 causal case: the row-C D004 environment with
    torch CUDA unavailable, so D003 selects D004 through prior_candidates."""
    snapshot = d004_snapshot()
    snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
    return snapshot


def dx_order(result) -> list[tuple[str, DiagnosisRole]]:
    return [(d.code, d.role) for d in result.diagnoses]


def _assert_issue_semantics_unchanged(enabled, disabled) -> None:
    """The diagnosis pipeline is additive: identical issues and summary."""
    assert enabled.issues == disabled.issues
    assert enabled.summary == disabled.summary
    assert enabled.check_errors == disabled.check_errors


class TestDiagnosisPipelineHealthy:
    def test_healthy_snapshot_yields_no_diagnoses_and_no_errors(self):
        result = DiagnosisEngine().run(base_snapshot())
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}
        assert result.check_errors == {}
        assert result.issues == []
        assert result.summary.status is EnvironmentStatus.HEALTHY


class TestRealRuleIntegration:
    """Each accepted rule reaches DiagnosisResult.diagnoses through the real
    pipeline (no direct rule invocation after engine entry), with the
    engine's output identical to the directly evaluated candidate once the
    resolver-assigned role is applied — the engine never reinterprets a
    rule payload."""

    def test_d001_reaches_diagnoses(self):
        snapshot = d001_snapshot()
        result = DiagnosisEngine().run(snapshot)
        assert dx_order(result) == [("D001", DiagnosisRole.PRIMARY)]
        assert result.diagnoses[0].confidence is Confidence.HIGH
        assert result.diagnoses[0].severity is Severity.WARNING
        assert result.diagnosis_errors == {}
        assert result.issues == []  # issue behavior stays independently correct

    def test_d002_reaches_diagnoses(self):
        snapshot = d002_snapshot()
        result = DiagnosisEngine().run(snapshot)
        assert dx_order(result) == [("D002", DiagnosisRole.PRIMARY)]
        assert result.diagnoses[0].confidence is Confidence.HIGH
        assert result.diagnoses[0].severity is Severity.WARNING
        assert result.diagnosis_errors == {}
        assert result.issues == []
        assert len(result.diagnoses) == 1  # no duplicate candidate

    def test_d004_reaches_diagnoses_with_exact_load_name_semantics(self):
        snapshot = d004_snapshot()
        result = DiagnosisEngine().run(snapshot)
        assert dx_order(result) == [("D004", DiagnosisRole.PRIMARY)]
        assert result.diagnoses[0].confidence is Confidence.MEDIUM
        assert result.diagnosis_errors == {}
        direct = D004RuntimeLibraryConflict().evaluate(
            DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY), []
        )
        assert direct is not None
        assert result.diagnoses[0] == replace(
            direct.diagnosis, role=DiagnosisRole.PRIMARY
        )

    def test_d003_non_d004_cause_reaches_diagnoses(self):
        snapshot = d003_snapshot()
        result = DiagnosisEngine().run(snapshot)
        assert dx_order(result) == [("D003", DiagnosisRole.PRIMARY)]
        assert result.diagnoses[0].confidence is Confidence.LOW
        assert result.diagnoses[0].cause_class == "UNKNOWN"
        assert result.diagnosis_errors == {}
        # The observed failure stays reported at the issue layer.
        assert [issue.code for issue in result.issues] == ["TORCH002"]

    def test_every_attached_diagnosis_carries_a_concrete_role(self):
        for snapshot in (
            base_snapshot(),
            d001_snapshot(),
            d002_snapshot(),
            d003_snapshot(),
            d004_snapshot(),
            critical_snapshot(),
        ):
            result = DiagnosisEngine().run(snapshot)
            assert all(
                isinstance(d.role, DiagnosisRole) for d in result.diagnoses
            ), snapshot


class TestCriticalD003D004Case:
    """The mandatory causal integration: snapshot -> checks -> facts -> D004
    -> prior candidate -> D003 -> resolver, all through the real engine."""

    def test_d004_primary_d003_secondary_in_resolver_order(self):
        result = DiagnosisEngine().run(critical_snapshot())
        assert dx_order(result) == [
            ("D004", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
        ]
        by_code = {d.code: d for d in result.diagnoses}
        assert by_code["D004"].confidence is Confidence.MEDIUM
        assert by_code["D004"].severity is Severity.WARNING
        assert by_code["D003"].confidence is Confidence.MEDIUM
        assert by_code["D003"].severity is Severity.ERROR
        # D003 selected the D004 runtime-library-conflict cause through the
        # prior-candidate mechanism, and the accepted relationship survives.
        assert by_code["D003"].cause_class == "RUNTIME_LIBRARY_CONFLICT"
        assert by_code["D003"].relationships == [
            DiagnosisRelationship(kind="explains", target_code="D004")
        ]

    def test_no_duplication_of_the_consumed_d004(self):
        result = DiagnosisEngine().run(critical_snapshot())
        assert len(result.diagnoses) == 2
        assert [d.code for d in result.diagnoses].count("D004") == 1

    def test_diagnoses_keep_resolver_output_order(self):
        # D003 sorts before D004 by code; the engine must keep the resolver's
        # order (PRIMARY first) instead of re-sorting by code or severity.
        result = DiagnosisEngine().run(critical_snapshot())
        assert result.diagnoses[0].code == "D004"

    def test_torch002_issue_is_not_suppressed(self):
        # presentation_suppressed_issue_codes is a carrier: Phase 11 executes
        # no suppression, the observed failure stays in result.issues.
        result = DiagnosisEngine().run(critical_snapshot())
        assert [issue.code for issue in result.issues] == ["TORCH002"]
        assert result.summary.errors == 1
        assert result.summary == Summary.from_issues(result.issues)


class TestRuleExecutionOrder:
    def test_injected_rules_run_in_provided_order_with_prior_propagation(self):
        journal: list[tuple[str, tuple[str, ...], int]] = []

        def fake(code: str) -> FakeRule:
            return FakeRule(code, journal=journal, candidate=fake_candidate(code))

        rules = [fake("D003"), fake("D001"), fake("D004"), fake("D002")]
        result = DiagnosisEngine(diagnosis_rules=rules).run(critical_snapshot())
        # Provided order is executed verbatim — the engine never reorders.
        assert [entry[0] for entry in journal] == ["D003", "D001", "D004", "D002"]
        # Each rule sees exactly the previously emitted candidates.
        assert journal[0][1] == ()
        assert journal[1][1] == ("D003",)
        assert journal[2][1] == ("D003", "D001")
        assert journal[3][1] == ("D003", "D001", "D004")
        # All four candidates survive into the resolver's output.
        assert len(result.diagnoses) == 4
        assert result.diagnosis_errors == {}

    def test_default_rules_execute_in_frozen_order(self, monkeypatch):
        executed: list[str] = []
        priors: dict[str, tuple[str, ...]] = {}
        for rule_cls in (
            D001ToolkitResolutionConflict,
            D002TorchToolkitMismatch,
            D004RuntimeLibraryConflict,
            D003PyTorchCudaUnavailable,
        ):
            original = rule_cls.evaluate

            def wrapper(
                self,
                facts: DiagnosisFacts,
                issues: Sequence[Issue],
                prior_candidates: Mapping[str, DiagnosisCandidate] = NO_PRIOR_CANDIDATES,
                _original=original,
            ):
                executed.append(self.code)
                priors[self.code] = tuple(prior_candidates)
                return _original(self, facts, issues, prior_candidates)

            monkeypatch.setattr(rule_cls, "evaluate", wrapper)

        result = DiagnosisEngine().run(critical_snapshot())
        # The frozen dependency order — architectural, not code order.
        assert executed == ["D001", "D002", "D004", "D003"]
        # D003 observes D004 only after D004 has run (its sole channel);
        # D001/D002 found nothing, so they are absent — not mapped to None.
        assert priors["D004"] == ()
        assert priors["D003"] == ("D004",)
        # D003 runs last and can never observe a D003 result.
        assert "D003" not in priors["D003"]
        assert dx_order(result) == [
            ("D004", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
        ]

    def test_default_registration_is_exactly_the_accepted_rules(self):
        engine = DiagnosisEngine()
        assert [rule.code for rule in engine._diagnosis_rules] == [
            "D001",
            "D002",
            "D004",
            "D003",
        ]
        assert all(
            isinstance(rule, cls)
            for rule, cls in zip(
                engine._diagnosis_rules,
                (
                    D001ToolkitResolutionConflict,
                    D002TorchToolkitMismatch,
                    D004RuntimeLibraryConflict,
                    D003PyTorchCudaUnavailable,
                ),
                strict=True,
            )
        )
        # Fresh instances per engine — no shared mutable rule state.
        assert all(
            a is not b
            for a, b in zip(
                engine._diagnosis_rules, DiagnosisEngine()._diagnosis_rules, strict=True
            )
        )


class FakeRule:
    """A journaling fake diagnosis rule (structural Protocol match)."""

    def __init__(
        self,
        code: str,
        *,
        journal: list[tuple[str, tuple[str, ...], int]] | None = None,
        candidate: DiagnosisCandidate | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.code = code
        self._journal = journal
        self._candidate = candidate
        self._exc = exc

    def evaluate(
        self,
        facts: DiagnosisFacts,
        issues: Sequence[Issue],
        prior_candidates: Mapping[str, DiagnosisCandidate] = NO_PRIOR_CANDIDATES,
    ) -> DiagnosisCandidate | None:
        if self._journal is not None:
            self._journal.append((self.code, tuple(prior_candidates), id(facts)))
        if self._exc is not None:
            raise self._exc
        return self._candidate


def fake_candidate(code: str) -> DiagnosisCandidate:
    return DiagnosisCandidate(
        diagnosis=Diagnosis(
            code=code,
            title=f"fake {code}",
            severity=Severity.WARNING,
            confidence=Confidence.HIGH,
            summary=f"fake {code} summary",
        )
    )


class ExplodingFactBuilder(DiagnosisFactBuilder):
    def build(self, snapshot, issues, compatibility) -> DiagnosisFacts:
        raise RuntimeError("fact builder boom")


class CountingFactBuilder(DiagnosisFactBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def build(self, snapshot, issues, compatibility) -> DiagnosisFacts:
        self.calls += 1
        return super().build(snapshot, issues, compatibility)


class ExplodingResolver(DiagnosisResolver):
    def resolve(self, candidates):
        raise RuntimeError("resolver boom")


class CountingResolver(DiagnosisResolver):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[DiagnosisCandidate]] = []

    def resolve(self, candidates):
        self.batches.append(list(candidates))
        return super().resolve(candidates)


class TestSingleFactsAndSingleResolve:
    def test_fact_builder_runs_once_and_rules_share_one_facts_object(self):
        builder = CountingFactBuilder()
        journal: list[tuple[str, tuple[str, ...], int]] = []
        rules = [
            FakeRule(code, journal=journal, candidate=fake_candidate(code))
            for code in ("D001", "D002", "D004", "D003")
        ]
        DiagnosisEngine(diagnosis_rules=rules, fact_builder=builder).run(
            critical_snapshot()
        )
        assert builder.calls == 1
        assert len({entry[2] for entry in journal}) == 1  # same facts identity

    def test_resolver_runs_exactly_once_with_all_candidates(self):
        resolver = CountingResolver()
        DiagnosisEngine(resolver=resolver).run(critical_snapshot())
        assert len(resolver.batches) == 1
        assert [c.diagnosis.code for c in resolver.batches[0]] == ["D004", "D003"]

    def test_resolver_runs_once_even_with_no_candidates(self):
        resolver = CountingResolver()
        DiagnosisEngine(resolver=resolver).run(base_snapshot())
        assert resolver.batches == [[]]
        assert resolver.batches[0] == []


class TestDiagnosisFailureIsolation:
    def test_fact_builder_failure_skips_rules_and_resolver(self):
        journal: list[tuple[str, tuple[str, ...], int]] = []
        resolver = CountingResolver()
        engine = DiagnosisEngine(
            fact_builder=ExplodingFactBuilder(),
            diagnosis_rules=[
                FakeRule(code, journal=journal, candidate=fake_candidate(code))
                for code in ("D001", "D002", "D004", "D003")
            ],
            resolver=resolver,
        )
        result = engine.run(critical_snapshot())
        # Bounded error under the frozen "facts" key; nothing fabricated.
        assert result.diagnosis_errors == {"facts": "RuntimeError: fact builder boom"}
        assert result.diagnoses == []
        assert journal == []  # no rule ever ran
        assert resolver.batches == []  # the resolver was never called
        # The issue layer is untouched by the diagnosis failure.
        assert [issue.code for issue in result.issues] == ["TORCH002"]
        assert result.summary.errors == 1
        assert result.check_errors == {}

    def test_rule_failure_is_isolated_and_later_rules_still_run(self):
        journal: list[tuple[str, tuple[str, ...], int]] = []
        rules = [
            FakeRule("D001", journal=journal, exc=ValueError("boom for /home/secretuser")),
            FakeRule("D002", journal=journal, candidate=fake_candidate("D002")),
            FakeRule("D004", journal=journal, candidate=fake_candidate("D004")),
            FakeRule("D003", journal=journal, candidate=fake_candidate("D003")),
        ]
        result = DiagnosisEngine(diagnosis_rules=rules).run(base_snapshot())
        assert set(result.diagnosis_errors) == {"D001"}
        assert result.diagnosis_errors["D001"] == (
            "ValueError: boom for /home/secretuser"
        )
        # The later rules all ran and their candidates were resolved.
        assert [entry[0] for entry in journal] == ["D001", "D002", "D004", "D003"]
        assert dx_order(result) == [
            ("D002", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
            ("D004", DiagnosisRole.SECONDARY),
        ]
        # The failed code is simply absent from the prior map — never
        # mapped to None, never treated as negative evidence.
        assert journal[1][1] == ()
        assert journal[2][1] == ("D002",)
        assert journal[3][1] == ("D002", "D004")

    def test_rule_exception_value_stays_out_of_public_json_output(self):
        # diagnosis_errors is engine-internal state in Phase 11: a private
        # path inside a recorded exception must not reach the v1 public
        # surface (the reporter/schema phase owns any later exposure).
        rules = [
            FakeRule("D001", exc=ValueError("boom for /home/secretuser/cuda")),
            FakeRule("D002", exc=ValueError("boom LD_LIBRARY_PATH=/home/secretuser")),
        ]
        snapshot = critical_snapshot()
        result = DiagnosisEngine(diagnosis_rules=rules).run(snapshot)
        assert set(result.diagnosis_errors) == {"D001", "D002"}
        payload = JsonReporter.build(
            ReportInputs(snapshot=snapshot, result=result, generated_at="2026-10-14T00:00:00+00:00")
        )
        assert "diagnosis_errors" not in payload
        assert "diagnoses" not in payload
        text = json.dumps(payload)
        assert "secretuser" not in text
        assert "LD_LIBRARY_PATH" not in text

    def test_d004_failure_leaves_d003_unavailable_not_acquitted(self, monkeypatch):
        # The dependency-failure pin (§23.1): with the real rules and a real
        # conflicting environment, a crashing D004 leaves D003's prior empty.
        # D003 must read that as "upstream unavailable" — landing on the
        # frozen UNKNOWN cause — never as proof no conflict exists.
        def explode(self, facts, issues, prior_candidates=NO_PRIOR_CANDIDATES):
            raise RuntimeError("d004 probe exploded")

        monkeypatch.setattr(D004RuntimeLibraryConflict, "evaluate", explode)
        result = DiagnosisEngine().run(critical_snapshot())
        assert set(result.diagnosis_errors) == {"D004"}
        assert dx_order(result) == [("D003", DiagnosisRole.PRIMARY)]
        d003 = result.diagnoses[0]
        assert d003.confidence is Confidence.LOW
        assert d003.cause_class == "UNKNOWN"
        assert d003.relationships == []
        # The issue layer is untouched by the rule failure.
        assert [issue.code for issue in result.issues] == ["TORCH002"]

    def test_resolver_failure_attaches_nothing_and_keeps_issues(self):
        reference = DiagnosisEngine().run(critical_snapshot())
        result = DiagnosisEngine(resolver=ExplodingResolver()).run(critical_snapshot())
        assert result.diagnosis_errors == {"resolver": "RuntimeError: resolver boom"}
        # No partially role-less candidates ever become diagnoses.
        assert result.diagnoses == []
        _assert_issue_semantics_unchanged(result, reference)

    def test_none_return_and_exception_stay_distinct(self):
        none_result = DiagnosisEngine(
            diagnosis_rules=[FakeRule("D001")]
        ).run(critical_snapshot())
        assert none_result.diagnoses == []
        assert none_result.diagnosis_errors == {}  # None is "not evaluated"

        raised = DiagnosisEngine(
            diagnosis_rules=[FakeRule("D001", exc=ValueError("boom"))]
        ).run(critical_snapshot())
        assert raised.diagnoses == []
        assert raised.diagnosis_errors == {"D001": "ValueError: boom"}

    def test_crashing_check_neither_aborts_diagnosis_nor_becomes_a_cause(self):
        result = DiagnosisEngine(checks=[ExplodingCheck()]).run(critical_snapshot())
        assert result.check_errors == {"ExplodingCheck": "RuntimeError: boom"}
        assert result.diagnosis_errors == {}
        assert dx_order(result) == [
            ("D004", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
        ]
        # A crashing check is error bookkeeping, never diagnosis evidence.
        assert {d.code for d in result.diagnoses} <= {"D001", "D002", "D003", "D004"}

    def test_diagnosis_and_check_errors_never_mix_keys(self):
        result = DiagnosisEngine(checks=[ExplodingCheck()]).run(
            critical_snapshot()
        )
        assert set(result.check_errors) == {"ExplodingCheck"}
        assert result.diagnosis_errors == {}

    def test_compatibility_failure_does_not_abort_or_duplicate_loading(self, monkeypatch):
        calls = []

        def flaky():
            calls.append(1)
            raise CompatibilityDataError("unreadable")

        monkeypatch.setattr(engine_module, "load_compatibility_data", flaky)
        result = DiagnosisEngine().run(critical_snapshot())
        assert calls == [1]  # loaded exactly once for the whole run
        assert "compatibility-data" in result.check_errors
        assert result.diagnosis_errors == {}
        # Driver/toolkit compatibility is UNKNOWN, not fatal: the diagnosis
        # pipeline still runs and still finds the library conflict.
        assert dx_order(result) == [
            ("D004", DiagnosisRole.PRIMARY),
            ("D003", DiagnosisRole.SECONDARY),
        ]


class TestAdditivityAndCompatibility:
    @pytest.mark.parametrize(
        "snapshot",
        [d001_snapshot(), critical_snapshot()],
        ids=["d001", "critical-d003-d004"],
    )
    def test_diagnosis_pipeline_changes_no_issue_or_summary_semantics(self, snapshot):
        enabled = DiagnosisEngine().run(snapshot)
        disabled = DiagnosisEngine(diagnosis_rules=()).run(snapshot)
        _assert_issue_semantics_unchanged(enabled, disabled)
        assert disabled.diagnoses == []
        assert disabled.diagnosis_errors == {}
        assert enabled.diagnoses != []  # the default pipeline really fires

    def test_summary_stays_derived_from_issues_only(self):
        result = DiagnosisEngine().run(critical_snapshot())
        assert result.summary == Summary.from_issues(result.issues)
        assert result.summary == Summary.from_issues(
            DiagnosisEngine(diagnosis_rules=()).run(critical_snapshot()).issues
        )

    def test_existing_construction_patterns_still_work(self):
        compat: CompatibilityData = context_for(base_snapshot()).compatibility
        assert compat is not None
        for engine in (
            DiagnosisEngine(),
            DiagnosisEngine(checks=[CriticalCheck()]),
            DiagnosisEngine(compatibility=compat),
        ):
            assert engine.checks is not None

    def test_explicitly_injected_production_pipeline_matches_the_default(self):
        snapshot = critical_snapshot()
        explicit = DiagnosisEngine(
            diagnosis_rules=engine_module._default_diagnosis_rules(),
            fact_builder=DiagnosisFactBuilder(),
            resolver=DiagnosisResolver(),
        ).run(snapshot)
        assert explicit == DiagnosisEngine().run(snapshot)
