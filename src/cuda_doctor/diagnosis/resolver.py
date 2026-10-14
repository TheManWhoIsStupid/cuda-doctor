"""Diagnosis candidate resolver (Phase 10).

Converts independently produced ``DiagnosisCandidate`` objects into the
finalized ``Diagnosis`` list with deterministic roles and ordering
(architecture §6.4/§24). The resolver is pure policy over its inputs: it
never runs rules, collects facts, reads host state or raw issues, calls
the engine, executes remediation, or renders reports.
Its one owned transformation is ``role=None -> final DiagnosisRole``
(plan §29); every rule-produced payload field survives verbatim, and
the input candidate objects are never mutated.

PRIMARY selection follows the frozen global ranking (truth tables §12):

1. causal-parent status — a candidate whose code another candidate
   explicitly names in ``causal_parent_codes`` (truth tables §12.1);
2. observed-active-failure explanation — the candidate hint only
   (§12.2), never severity, issue codes, or prose;
3. confidence HIGH > MEDIUM > LOW (§12.3);
4. severity CRITICAL > ERROR > WARNING > INFO (§12.4);
5. explicit SAFE remediation — at least one SAFE step carrying a
   command, the only structured actionability marker (§12.5,
   architecture §24.2);
6. the stable code order D001 < D002 < D003 < D004 (§12.6).

At most one PRIMARY exists, and none is forced when no candidate is
eligible: a LOW-confidence candidate is ineligible for PRIMARY unless it
explains the active failure (semantics §10.4 with the truth-table §20.8
exception, which the 12.2-before-12.3 ordering produces naturally).
Every remaining candidate receives SECONDARY (an independent established
diagnosis, or a valid consequence in the winner's causal chain —
semantics §12.2), CONTRIBUTING (a condition the PRIMARY itself names as
its causal parent — semantics §12.3), or POSSIBLE (a LOW-confidence
finding — semantics §12.4). The finalized list is ordered by role rank,
then the same frozen ranking key, so any permutation of a candidate set
yields an identical result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, replace

from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    DiagnosisCandidate,
    DiagnosisRelationship,
    DiagnosisRole,
    RemediationSafety,
)

__all__ = ["DiagnosisResolver"]

# Frozen resolver-local orderings (truth tables §12.3/§12.4/§12.6):
# explicit ranks — never enum identity, numeric value, or lexicographic
# enum-string comparison (plan §16/§17).
_CONFIDENCE_RANK = {
    Confidence.HIGH: 0,
    Confidence.MEDIUM: 1,
    Confidence.LOW: 2,
}
_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.ERROR: 1,
    Severity.WARNING: 2,
    Severity.INFO: 3,
}
# The four v0.2.0 diagnosis codes in frozen code order; any other code
# (synthetic/pathological input only) sorts after them, lexically, so
# the order stays total and deterministic.
_CODE_RANK = {"D001": 0, "D002": 1, "D003": 2, "D004": 3}
_UNKNOWN_CODE_RANK = len(_CODE_RANK)
# Presentation order of the finalized roles. Truth tables §26 group
# SECONDARY/CONTRIBUTING without an internal order; this stable
# sub-order matches the DiagnosisRole declaration and keeps POSSIBLE
# findings last ("possible causes" close the report).
_ROLE_RANK = {
    DiagnosisRole.PRIMARY: 0,
    DiagnosisRole.SECONDARY: 1,
    DiagnosisRole.CONTRIBUTING: 2,
    DiagnosisRole.POSSIBLE: 3,
}


class DiagnosisResolver:
    """Assign final roles and a deterministic order to rule candidates.

    ``resolve`` consumes already-produced candidates and returns
    finalized diagnoses: every output carries a concrete
    ``DiagnosisRole``, every rule-produced field is preserved exactly,
    and the input candidates (and their carried diagnoses) are left
    untouched — finalization builds new ``Diagnosis`` objects via
    ``dataclasses.replace``. Duplicate-code candidates are both kept
    (never silently discarded) and ordered deterministically; a causal
    parent code absent from the candidate set is simply not a parent
    here (never fabricated, never a crash); a causal cycle cannot make
    the resolver recurse because parentage is a per-candidate boolean,
    not a graph walk.
    """

    def resolve(self, candidates: Sequence[DiagnosisCandidate]) -> list[Diagnosis]:
        """Finalize ``candidates`` into role-assigned diagnoses."""
        if not candidates:
            # Empty input: no diagnoses, no synthetic PRIMARY (plan §11).
            return []
        keyed = [
            (candidate, _ranking_key(candidate, candidates)) for candidate in candidates
        ]
        winner: DiagnosisCandidate | None = None
        if any(_primary_eligible(candidate) for candidate, _ in keyed):
            # Exactly one PRIMARY when any candidate is eligible; none
            # otherwise — a PRIMARY is never forced (plan §5).
            winner = min(
                (item for item in keyed if _primary_eligible(item[0])),
                key=lambda item: item[1],
            )[0]
        finalized: list[tuple[DiagnosisRole, tuple, Diagnosis]] = []
        for candidate, key in keyed:
            role = (
                DiagnosisRole.PRIMARY
                if candidate is winner
                else _runner_up_role(candidate, winner)
            )
            finalized.append(
                (
                    role,
                    key,
                    replace(
                        candidate.diagnosis,
                        # Role is the resolver-owned mutation (plan §29);
                        # a malformed pre-set role is deterministically
                        # overwritten, never honored and never raised on.
                        role=role,
                        relationships=_final_relationships(
                            candidate.diagnosis.relationships
                        ),
                    ),
                )
            )
        finalized.sort(key=lambda item: (_ROLE_RANK[item[0]], item[1]))
        return [diagnosis for _, _, diagnosis in finalized]


def _ranking_key(
    candidate: DiagnosisCandidate, candidates: Sequence[DiagnosisCandidate]
) -> tuple:
    """The frozen §12.1-§12.6 comparison key, smallest first.

    The two boolean dimensions invert (``not``) so True ranks first;
    the rank maps order the frozen enumerations. Everything through
    §12.6 is the frozen production ranking and is preserved exactly.

    The trailing code, title, and candidate fingerprint are the
    duplicate-code fallback only: two same-code candidates that tie on
    every frozen dimension are separated by title, then by a
    whole-candidate fingerprint, so the key is a genuine total order
    and no outcome ever depends on input order (plan §24/§25:
    duplicates are retained and deterministically degraded, never
    decided silently by position). These trailing fields sit strictly
    after the frozen dimensions and never decide anything for real
    v0.2.0 inputs, which have unique codes; the fingerprint is state
    serialization, not a ranking criterion, so no free-form text
    influences the frozen ranking itself (plan §18.4/§18.10).
    """
    diagnosis = candidate.diagnosis
    return (
        not _is_named_causal_parent(candidate, candidates),  # §12.1
        not candidate.explains_active_failure,  # §12.2
        _CONFIDENCE_RANK[diagnosis.confidence],  # §12.3
        _SEVERITY_RANK[diagnosis.severity],  # §12.4
        not _has_explicit_safe_remediation(diagnosis),  # §12.5
        _CODE_RANK.get(diagnosis.code, _UNKNOWN_CODE_RANK),  # §12.6
        diagnosis.code,
        diagnosis.title,
        _candidate_fingerprint(candidate),
    )


def _candidate_fingerprint(candidate: DiagnosisCandidate) -> str:
    """A deterministic rendering of the complete candidate state.

    Only reachable as the final duplicate-code tie-break, after code
    and title have also tied. The fingerprint must cover the whole
    ``DiagnosisCandidate`` envelope — the carried hints
    (``explains_active_failure``, ``risk_only``,
    ``causal_parent_codes``) as well as the diagnosis — because the
    hints feed §12.1 and the loser-role matrix: two candidates with
    identical diagnoses but different parent hints would otherwise
    still tie, letting input order pick the winner and change another
    candidate's final role. The resolver-owned ``diagnosis.role`` is
    neutralized first so a malformed pre-set role cannot influence
    ordering; candidates differing in any other field serialize
    differently because the dataclass field order is fixed.
    """
    normalized = replace(
        candidate, diagnosis=replace(candidate.diagnosis, role=None)
    )
    return repr(asdict(normalized))


def _primary_eligible(candidate: DiagnosisCandidate) -> bool:
    """LOW confidence alone never claims PRIMARY (semantics §10.4).

    The frozen exception — a LOW-confidence diagnosis that explains the
    observed active failure, such as D003 UNKNOWN (architecture §24.3,
    truth tables §20.8) — is eligible because the §12.2 criterion
    outranks confidence, not because confidence was upgraded.
    """
    return candidate.diagnosis.confidence is not Confidence.LOW or (
        candidate.explains_active_failure
    )


def _is_named_causal_parent(
    candidate: DiagnosisCandidate, candidates: Sequence[DiagnosisCandidate]
) -> bool:
    """True iff another candidate explicitly names this candidate's code.

    Parentage comes only from ``causal_parent_codes`` (plan §7) — never
    from titles, summaries, issue codes, code numerics, or mere D004
    coexistence. A self-referencing candidate never becomes its own
    parent (plan §24), and a code named only by a candidate absent from
    this set grants nothing (plan §8).
    """
    code = candidate.diagnosis.code
    return any(
        other is not candidate and code in other.causal_parent_codes
        for other in candidates
    )


def _has_explicit_safe_remediation(diagnosis: Diagnosis) -> bool:
    """The §12.5 tie-break: an explicit structured SAFE action exists.

    Exactly one boolean: some remediation step is SAFE and carries a
    command. Step counts, prose length, or lexical content never rank
    (plan §18); every v0.2.0 rule candidate uses ``command=None``, so
    this criterion ties in practice and falls through to code order.
    """
    return any(
        step.safety is RemediationSafety.SAFE and step.command is not None
        for step in diagnosis.remediation
    )


def _runner_up_role(
    candidate: DiagnosisCandidate, winner: DiagnosisCandidate | None
) -> DiagnosisRole:
    """The frozen role matrix for every non-PRIMARY candidate.

    - the PRIMARY itself names this candidate's code as a causal parent
      → CONTRIBUTING: a condition contributing to the primary that the
      global rules did not select as dominant (semantics §12.3);
    - LOW confidence → POSSIBLE: a plausible explanation without
      sufficient evidence for a stronger classification (semantics
      §10.4/§12.4);
    - otherwise SECONDARY: an independent established diagnosis or a
      valid consequence in the winner's causal chain (semantics §12.2 —
      the D003 symptom diagnosis beside a PRIMARY D004, truth tables
      §20.5/§20.8).
    """
    if (
        winner is not None
        and candidate.diagnosis.code in winner.causal_parent_codes
    ):
        return DiagnosisRole.CONTRIBUTING
    if candidate.diagnosis.confidence is Confidence.LOW:
        return DiagnosisRole.POSSIBLE
    return DiagnosisRole.SECONDARY


def _final_relationships(
    relationships: Sequence[DiagnosisRelationship],
) -> list[DiagnosisRelationship]:
    """Rule-carried relationships, verbatim except exact duplicates.

    The resolver invents no relationships and reverses no direction
    (plan §9): the frozen D003 "explains" → D004 link passes through
    untouched. Only a repeated identical (kind, target) pair collapses
    — the same link must not appear twice in one finalized diagnosis.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[DiagnosisRelationship] = []
    for relationship in relationships:
        marker = (relationship.kind, relationship.target_code)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(relationship)
    return unique
