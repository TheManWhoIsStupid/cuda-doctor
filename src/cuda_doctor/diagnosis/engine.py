"""The diagnosis engine: snapshot in, issues and diagnoses out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from cuda_doctor.checks import Check, default_checks
from cuda_doctor.compatibility import CompatibilityData, load_compatibility_data
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.core.exceptions import CompatibilityDataError
from cuda_doctor.core.models import EnvironmentSnapshot
from cuda_doctor.diagnosis.facts import DiagnosisFactBuilder
from cuda_doctor.diagnosis.issue import Issue, Summary
from cuda_doctor.diagnosis.models import Diagnosis, DiagnosisCandidate
from cuda_doctor.diagnosis.resolver import DiagnosisResolver
from cuda_doctor.diagnosis.rules import (
    D001ToolkitResolutionConflict,
    D002TorchToolkitMismatch,
    D003PyTorchCudaUnavailable,
    D004RuntimeLibraryConflict,
)
from cuda_doctor.diagnosis.rules.base import DiagnosisRule

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.ERROR: 1,
    Severity.WARNING: 2,
    Severity.INFO: 3,
}


def _bounded_diagnosis_error(exc: Exception) -> str:
    """Bounded, privacy-safe value for a diagnosis-pipeline failure.

    Retains only the code-defined exception type name plus a stable
    generic message. ``str(exc)`` is deliberately dropped: exception
    text can embed arbitrary user paths, usernames, environment values
    such as LD_LIBRARY_PATH/CUDA_VISIBLE_DEVICES, and raw probe
    details, none of which may reach ``diagnosis_errors`` (§23 error
    privacy). The stable key already identifies the component.
    """
    return f"{type(exc).__name__}: diagnosis component failed"


def _default_diagnosis_rules() -> tuple[DiagnosisRule, ...]:
    """The accepted v0.2 rules in the frozen dependency order (§18.1).

    D003 runs last because it may consume a previously produced D004
    candidate through ``prior_candidates``; the resolver — not this
    order — determines final roles and presentation order.
    """
    return (
        D001ToolkitResolutionConflict(),
        D002TorchToolkitMismatch(),
        D004RuntimeLibraryConflict(),
        D003PyTorchCudaUnavailable(),
    )


@dataclass
class DiagnosisResult:
    """Everything the engine concluded about one snapshot.

    ``issues``/``summary``/``check_errors`` keep their v0.1 semantics:
    the summary is always derived from ``issues`` alone, and exit-code
    behavior reads the issue layer only. ``diagnoses`` carries the
    finalized v0.2 root-cause layer (every entry role-resolved by the
    ``DiagnosisResolver``), and ``diagnosis_errors`` records bounded
    per-component diagnosis-pipeline failures (keyed by the stable
    names "facts"/"D001"/"D002"/"D004"/"D003"/"resolver"; each value
    is only the exception type name plus a stable generic message —
    never raw exception text, which can carry private paths and
    environment values). The two layers are distinct: neither feeds
    the other's summary or errors.
    """

    issues: list[Issue] = field(default_factory=list)
    diagnoses: list[Diagnosis] = field(default_factory=list)
    summary: Summary = field(default_factory=Summary)
    check_errors: dict[str, str] = field(default_factory=dict)
    diagnosis_errors: dict[str, str] = field(default_factory=dict)


class DiagnosisEngine:
    """Runs every registered check; one crashing check never stops the rest.

    Since v0.2 the engine also orchestrates the diagnosis pipeline
    additively after the check phase: one ``DiagnosisFactBuilder`` build
    over the snapshot and the final ordered issues, the accepted rules in
    the frozen dependency order with prior-candidate propagation, then
    one ``DiagnosisResolver`` call whose output lands verbatim in
    ``DiagnosisResult.diagnoses``. Every diagnosis stage is separately
    guarded (architecture §23): a failing component records a bounded
    ``diagnosis_errors`` entry and never disturbs the issue results.
    """

    def __init__(
        self,
        checks: list[Check] | None = None,
        compatibility: CompatibilityData | None = None,
        *,
        diagnosis_rules: Sequence[DiagnosisRule] | None = None,
        fact_builder: DiagnosisFactBuilder | None = None,
        resolver: DiagnosisResolver | None = None,
    ) -> None:
        self.checks = checks if checks is not None else default_checks()
        self._compatibility = compatibility
        # Optional test seams (backwards compatible, keyword-only): the
        # defaults are exactly the accepted production components.
        self._diagnosis_rules: tuple[DiagnosisRule, ...] = (
            tuple(diagnosis_rules)
            if diagnosis_rules is not None
            else _default_diagnosis_rules()
        )
        self._fact_builder = fact_builder if fact_builder is not None else DiagnosisFactBuilder()
        self._resolver = resolver if resolver is not None else DiagnosisResolver()

    def run(self, snapshot: EnvironmentSnapshot) -> DiagnosisResult:
        result = DiagnosisResult()
        compatibility = self._compatibility or self._load_compat(result)
        ctx = DiagnosticContext(snapshot=snapshot, compatibility=compatibility)

        issues: list[Issue] = []
        for check in self.checks:
            try:
                issues.extend(check.run(ctx))
            except Exception as exc:
                # Keyed by class name: two checks may share an issue code
                # (e.g. both TORCH005 aspects) and must not overwrite
                # each other's error reports.
                result.check_errors[type(check).__name__] = f"{type(exc).__name__}: {exc}"
        issues.sort(key=lambda issue: (_SEVERITY_ORDER[issue.severity], issue.code))
        result.issues = issues
        result.summary = Summary.from_issues(issues)
        self._run_diagnosis(snapshot, issues, compatibility, result)
        return result

    def _run_diagnosis(
        self,
        snapshot: EnvironmentSnapshot,
        issues: Sequence[Issue],
        compatibility: CompatibilityData | None,
        result: DiagnosisResult,
    ) -> None:
        """The additive diagnosis pipeline (architecture §21/§23).

        facts → D001 → D002 → D004 → D003 → resolver, each stage guarded
        separately. A fact-builder failure leaves no normalized evidence
        to evaluate, so the rules and resolver are skipped entirely. A
        rule failure records its code and leaves that code absent from
        the prior-candidate map — downstream rules must read absence as
        "upstream unavailable", never negative evidence (§23.1). A
        resolver failure attaches nothing: partially role-less
        candidates never become diagnoses.
        """
        try:
            facts = self._fact_builder.build(snapshot, issues, compatibility)
        except Exception as exc:
            result.diagnosis_errors["facts"] = _bounded_diagnosis_error(exc)
            return
        candidates: list[DiagnosisCandidate] = []
        prior_candidates: dict[str, DiagnosisCandidate] = {}
        for rule in self._diagnosis_rules:
            try:
                candidate = rule.evaluate(facts, issues, prior_candidates)
            except Exception as exc:
                # Keyed by the rule's stable code, never its class name:
                # the frozen component names are D001..D004 (§23). The
                # value is bounded — raw exception text never enters.
                result.diagnosis_errors[rule.code] = _bounded_diagnosis_error(exc)
                continue
            if candidate is not None:
                # A None return is a normal "not evaluated": no
                # candidate, no error — distinct from an exception.
                prior_candidates[rule.code] = candidate
                candidates.append(candidate)
        try:
            # Exactly one resolver call per successful pipeline run; its
            # output order is final — the engine never re-sorts.
            result.diagnoses = self._resolver.resolve(candidates)
        except Exception as exc:
            result.diagnosis_errors["resolver"] = _bounded_diagnosis_error(exc)

    @staticmethod
    def _load_compat(result: DiagnosisResult) -> CompatibilityData | None:
        try:
            return load_compatibility_data()
        except CompatibilityDataError as exc:
            # Unreadable bundled data must not disable diagnosis entirely.
            result.check_errors["compatibility-data"] = str(exc)
            return None
