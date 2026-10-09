"""Tests for the diagnosis engine: aggregation, ordering, and isolation."""

from __future__ import annotations

from dataclasses import replace

from tests.factories import base_snapshot, context_for

from cuda_doctor.checks import Check, default_checks
from cuda_doctor.checks.compiler_checks import NoHostCompiler
from cuda_doctor.core.enums import EnvironmentStatus, Severity
from cuda_doctor.core.exceptions import CompatibilityDataError
from cuda_doctor.diagnosis import engine as engine_module
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.issue import Issue, Summary


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
