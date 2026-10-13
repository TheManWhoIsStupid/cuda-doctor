"""Tests for the D004 CUDA runtime library conflict rule (Revision 3).

The rule is tested over synthetic ``DiagnosisFacts`` (truth tables §19,
amendment §19.8a rows A-S): collector and normalization behavior is
pinned by the Phase-3/Phase-5 suites; here only the exact-load-name
Class-A/Class-B decision over already-normalized facts matters. Frozen
row IDs appear in test names/comments (rows A-S, N05-N13 plus the
§19.8-§19.13 scenario families) so the coverage stays auditable against
the frozen tables. Amendment §21 classification: every old test that
constructed only the deleted family-level identity was rewritten to the
split ``load_name`` / ``load_name_family_version`` /
``target_family_version`` surface, the old family-only canonical
positive (row-A shape) is now a pinned NEGATIVE, and
``test_winner_decided_inside_the_model_emits`` flips to negative by
frozen mandate — these are semantic flips required by Revision 3, not
weakenings.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from tests.factories import base_snapshot

from cuda_doctor.collectors.runtime_libraries import (
    ORIGIN_CONDA_PREFIX,
    ORIGIN_LD_LIBRARY_PATH,
    ORIGIN_LDCONFIG,
    ORIGIN_PYTHON_PACKAGE,
    ORIGIN_TOOLKIT,
    VERSION_SOURCE_FILENAME,
    VERSION_SOURCE_SYMLINK_TARGET,
)
from cuda_doctor.compatibility import load_compatibility_data
from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.core.models import (
    EnvironmentSnapshot,
    RuntimeLibraryCandidate,
    RuntimeLibraryInventory,
)
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.facts import (
    DiagnosisFactBuilder,
    DiagnosisFacts,
    IdentityReliability,
    PlatformApplicability,
    ReferenceToolkit,
    RuntimeLibraryCandidateFact,
    RuntimeLibraryFacts,
    ToolkitSelectorFact,
    Trinary,
)
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    EvidenceItem,
    ImpactItem,
    RemediationSafety,
)
from cuda_doctor.diagnosis.rules import (
    D001ToolkitResolutionConflict,
    D002TorchToolkitMismatch,
    D004RuntimeLibraryConflict,
    DiagnosisRule,
)
from cuda_doctor.diagnosis.rules import base as rules_base
from cuda_doctor.diagnosis.rules import d004 as d004_module

COMPATIBILITY = load_compatibility_data()

RULE = D004RuntimeLibraryConflict()
D001 = D001ToolkitResolutionConflict()
D002 = D002TorchToolkitMismatch()


def lib(
    family: str,
    version: str | None,
    *,
    origin: str = ORIGIN_LD_LIBRARY_PATH,
    order: int | None = None,
    reliability: IdentityReliability = IdentityReliability.DIRECT,
    active: Trinary = Trinary.UNKNOWN,
    target: str | None = None,
    name: str | None = None,
) -> RuntimeLibraryCandidateFact:
    """Synthetic normalized candidate; only the D004 surface is populated.

    Revision-3 split identity: ``version`` is the family the observed
    name CLAIMS (``load_name_family_version``); ``target`` is the
    canonical target family (``target_family_version``). ``version=None``
    means a bare name with no claim (ineligible for both classes).
    ``name`` overrides the derived exact name for multi-component tails
    — e.g. an observed ``11.0`` tail whose first numeric component
    claims family 11 — and the exact string is never normalized.
    """
    load_name = name or (f"{family}.so.{version}" if version is not None else f"{family}.so")
    return RuntimeLibraryCandidateFact(
        family=family,
        load_name=load_name,
        load_name_family_version=version,
        target_family_version=target,
        reliability=reliability,
        origin=origin,
        active_toolkit=active,
        modeled_search_order=order,
    )


def runtime_facts(
    candidates: tuple[RuntimeLibraryCandidateFact, ...] = (),
    torch_family: str | None = "12",
    cudnn_family: str | None = None,
    scan_incomplete: bool = False,
) -> RuntimeLibraryFacts:
    """A complete, successful scan unless stated otherwise."""
    return RuntimeLibraryFacts(
        candidates=candidates,
        torch_runtime_family=torch_family,
        package_local_cudnn_family=cudnn_family,
        scan_incomplete=scan_incomplete,
    )


def base_facts(platform: Platform = Platform.LINUX) -> DiagnosisFacts:
    snapshot = base_snapshot()
    if platform is not Platform.LINUX:
        snapshot.system = replace(snapshot.system, platform=platform)
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


def facts_with(
    runtime: RuntimeLibraryFacts | None = None,
    *,
    selectors: tuple[ToolkitSelectorFact, ...] = (),
    reference: ReferenceToolkit | None = None,
    applicability: PlatformApplicability | None = None,
    platform: Platform = Platform.LINUX,
) -> DiagnosisFacts:
    base = base_facts(platform)
    return replace(
        base,
        runtime_libraries=runtime if runtime is not None else runtime_facts(),
        toolkit_selectors=selectors,
        reference_toolkit=reference,
        applicability=applicability if applicability is not None else base.applicability,
    )


def ref(
    source: str = "NVCC",
    version: str | None = "11.8",
    reliability: IdentityReliability = IdentityReliability.DIRECT,
) -> ReferenceToolkit:
    """Synthetic reference toolkit (the authoritative Phase-5 result)."""
    return ReferenceToolkit(
        source=source,
        canonical_root=None,
        toolkit_version=version,
        reliability=reliability,
    )


def sel(
    name: str,
    version: str | None,
    *,
    canonical_root: str | None = None,
) -> ToolkitSelectorFact:
    """A minimal active/valid selector fact for D001 coexistence rows."""
    explicit = name in ("CUDA_HOME", "CUDA_PATH", "CUDACXX")
    effective = name in ("NVCC", "PATH_CUDA_BIN")
    return ToolkitSelectorFact(
        name=name,
        explicit=explicit,
        effective=effective,
        active=True,
        valid=True,
        canonical_root=canonical_root,
        toolkit_version=version,
        reliability=IdentityReliability.DIRECT,
    )


def conflict_facts() -> DiagnosisFacts:
    """The canonical Revision-3 D004 environment (row C): torch CUDA 12
    baseline; within the modeled LD_LIBRARY_PATH ordering the exact
    observed name libcudart.so.12 first resolves to family 11 (a
    misleading symlink) and later to family 12 — same-name shadowing."""
    return facts_with(
        runtime_facts(
            candidates=(
                lib("libcudart", "12", order=0, target="11"),
                lib("libcudart", "12", order=1, target="12"),
            )
        )
    )


def class_b_facts() -> DiagnosisFacts:
    """Row F: the misleading alias is corroborated only by an
    independently observed current-Python provider of the same exact
    name (no later LDP provider exists)."""
    return facts_with(
        runtime_facts(
            candidates=(
                lib("libcudart", "12", order=0, target="11"),
                lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
            )
        )
    )


def legacy_shape_facts() -> DiagnosisFacts:
    """The old NOT-ACCEPTED family-level canonical shape — now row A:
    different exact names never shadow one another merely because their
    claimed families differ."""
    return facts_with(
        runtime_facts(
            candidates=(
                lib("libcudart", "11", order=0, target="11"),
                lib("libcudart", "12", order=1, target="12"),
            )
        )
    )


def conflict_diagnosis() -> Diagnosis:
    candidate = RULE.evaluate(conflict_facts(), [])
    assert candidate is not None, "expected a D004 candidate"
    return candidate.diagnosis


def raw(
    family: str,
    version: str | None,
    *,
    origin: str,
    order: int | None = None,
    path: str | None = None,
    canonical_path: str | None = None,
    version_source: str | None = VERSION_SOURCE_FILENAME,
) -> RuntimeLibraryCandidate:
    """A raw Phase-3 inventory candidate for builder-based tests.

    ``canonical_path`` lets a synthetic candidate simulate what the
    strict collector records for a resolvable symlink target or a
    regular file (canonical = itself); synthetic fixtures never touch
    the real filesystem. A bare name is expressed with ``version=None``
    and an explicit bare ``path``.
    """
    suffix = f".so.{version}" if version is not None else ".so"
    return RuntimeLibraryCandidate(
        family=family,
        path=path or f"/opt/inventory/{family}{suffix}",
        canonical_path=canonical_path,
        version=version,
        version_source=version_source,
        origin=origin,
        search_order=order,
    )


def snapshot_with_inventory(
    *candidates: RuntimeLibraryCandidate,
    scan_errors: dict[str, str] | None = None,
    platform: Platform | None = None,
    cuda_version: str | None = None,
) -> EnvironmentSnapshot:
    snapshot = base_snapshot()
    snapshot.runtime_libraries = RuntimeLibraryInventory(
        candidates=list(candidates), scan_errors=scan_errors or {}
    )
    if platform is not None:
        snapshot.system = replace(snapshot.system, platform=platform)
    if cuda_version is not None:
        snapshot.pytorch = replace(snapshot.pytorch, cuda_version=cuda_version)
    return snapshot


def built_facts(snapshot: EnvironmentSnapshot) -> DiagnosisFacts:
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


class TestRuleContract:
    def test_code_is_d004(self):
        assert D004RuntimeLibraryConflict.code == "D004"
        assert D004RuntimeLibraryConflict().code == "D004"

    def test_satisfies_the_frozen_rule_protocol(self):
        assert isinstance(RULE, DiagnosisRule)

    def test_issue_and_prior_candidate_context_never_changes_the_result(self):
        # D004 is independent of D001/D002 (truth tables §20.6) and is not
        # yet consumed by D003 (Phase 9): both parameters must stay inert.
        facts = conflict_facts()
        issues = [
            Issue(code="CUDA004", severity=Severity.INFO, title="multiple installs"),
            Issue(code="TORCH005", severity=Severity.ERROR, title="torch aspect"),
        ]
        d001_candidate = D001.evaluate(
            facts_with(
                runtime_facts(),
                selectors=(
                    sel("CUDA_HOME", "11.8", canonical_root="/opt/a"),
                    sel("NVCC", "12.4", canonical_root="/opt/b"),
                ),
            ),
            [],
        )
        d002_candidate = D002.evaluate(
            facts_with(runtime_facts(), reference=ref(version="11.8")),
            [],
        )
        assert d001_candidate is not None
        assert d002_candidate is not None
        assert RULE.evaluate(facts, issues) == RULE.evaluate(facts, [])
        assert RULE.evaluate(facts, [], {"D001": d001_candidate, "D002": d002_candidate}) == (
            RULE.evaluate(facts, [])
        )


class TestPlatformGate:
    def test_linux_is_evaluated(self):
        facts = conflict_facts()
        assert facts.applicability.d004 is True
        assert RULE.evaluate(facts, []) is not None

    @pytest.mark.parametrize(
        "platform", [Platform.WINDOWS, Platform.MACOS, Platform.OTHER]
    )
    def test_not_evaluated_elsewhere(self, platform):
        # Truth tables §19.1 / semantics §23.2: Linux-only; other platforms
        # are not evaluated — never guessed and never "conflict-free".
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            ),
            platform=platform,
        )
        assert facts.applicability.d004 is False
        assert RULE.evaluate(facts, []) is None

    def test_applicability_gate_trusted_over_platform(self):
        # The rule consumes the normalized applicability fact only.
        base = base_facts(Platform.LINUX)
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            ),
            applicability=replace(base.applicability, d004=False),
        )
        assert RULE.evaluate(facts, []) is None


class TestTruthTableRows:
    """Every frozen Revision-3 row A-S (amendment §19.8a), pinned
    explicitly against the rewritten rule."""

    def test_row_a_different_names_never_shadow(self):
        # Different exact names (.so.11 vs .so.12) never form one
        # same-name group, whatever their claimed/target families are.
        assert RULE.evaluate(legacy_shape_facts(), []) is None

    def test_row_b_cuda_11_style_different_names(self):
        # baseline 11; .so.10.2 -> 10 at entry 0 and .so.11.0 -> 11 at
        # entry 1 are different lookup identities — no shadowing.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "10", order=0, target="10", name="libcudart.so.10.2"),
                    lib("libcudart", "11", order=1, target="11", name="libcudart.so.11.0"),
                ),
                torch_family="11",
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_c_same_name_class_a(self):
        candidate = RULE.evaluate(conflict_facts(), [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM
        assert candidate.diagnosis.severity is Severity.WARNING

    def test_row_d_cuda_11_style_class_a(self):
        # baseline 11; the exact name libcudart.so.11.0 resolves to
        # family 10 first and to family 11 later — same-name shadowing
        # with the exact multi-component tail preserved.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0, target="10", name="libcudart.so.11.0"),
                    lib("libcudart", "11", order=1, target="11", name="libcudart.so.11.0"),
                ),
                torch_family="11",
            )
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_row_e_bare_development_aliases(self):
        # libcudart.so -> 11 and libcudart.so -> 12: bare names are
        # ineligible lookup identities for both classes.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", None, order=0, target="11"),
                    lib("libcudart", None, order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_f_class_b_independently_corroborated(self):
        candidate = RULE.evaluate(class_b_facts(), [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM
        assert candidate.risk_only is True

    def test_row_g_uncorroborated_misleading_alias(self):
        # Anti-circularity: the alias cannot establish its own relevance
        # by claiming family 12 in its filename alone.
        facts = facts_with(
            runtime_facts(candidates=(lib("libcudart", "12", order=0, target="11"),))
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_h_earliest_same_name_target_unknown(self):
        # Do not skip the earliest provider: its unknown target family
        # makes the whole exact-name path UNKNOWN even though a later
        # same-name baseline-compatible provider exists.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0),
                    lib("libcudart", "12", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_i_later_target_unknown_without_corroboration(self):
        # Class A is not established by a later same-name provider whose
        # target family is unknown, and nothing corroborates Class B.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_i_class_b_may_still_emit_via_corroboration(self):
        # The same row-I shape with an independent same-name package
        # provider: the later unknown LDP provider is irrelevant to the
        # Class-B relation, which fires on its own evidence.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is not None

    def test_row_j_no_manufactured_cross_origin_winner(self):
        # The package-local compatible provider carries a DIFFERENT
        # exact name, so it cannot corroborate; deciding the winner
        # would need RPATH/RUNPATH knowledge — resolution UNKNOWN.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib(
                        "libcudart",
                        "12",
                        origin=ORIGIN_PYTHON_PACKAGE,
                        target="12",
                        name="libcudart.so.12.1",
                    ),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_k_ldconfig_same_name_conflict_alone(self):
        # A same-name conflict visible only through ldconfig carries no
        # modeled ordering and no precedence (§19.12).
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", origin=ORIGIN_LDCONFIG, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_LDCONFIG, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_l_active_toolkit_alone_creates_no_precedence(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize("torch_family", [None, "11", "13"])
    def test_row_m_cudnn_class_a_is_torch_invariant(self, torch_family):
        # cuDNN uses its own package-local baseline (§19.10); the torch
        # runtime family must not change the outcome.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                torch_family=torch_family,
                cudnn_family="9",
            )
        )
        baseline = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudnn", "9", order=0, target="8"),
                        lib("libcudnn", "9", order=1, target="9"),
                    ),
                    cudnn_family="9",
                )
            ),
            [],
        )
        assert baseline is not None
        assert RULE.evaluate(facts, []) == baseline

    def test_row_n_libcublas_class_a_via_observed_package_baseline(self):
        # The baseline is the uniquely reliable environment-observed
        # package-local libcublas family (torch 12 + package family 12);
        # no CUDA-major == libcublas-family numeric assumption exists.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_row_o_scan_incomplete_does_not_invalidate_class_a(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                ),
                scan_incomplete=True,
            )
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_row_o_scan_incomplete_does_not_invalidate_class_b(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                ),
                scan_incomplete=True,
            )
        )
        assert RULE.evaluate(facts, []) is not None

    def test_row_p_exact_duplicates_are_inert(self):
        baseline = RULE.evaluate(conflict_facts(), [])
        duplicated = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=1, target="12"),
                        lib("libcudart", "12", order=1, target="12"),
                    )
                )
            ),
            [],
        )
        assert duplicated is not None
        assert duplicated == baseline

    def test_row_q_irrelevant_same_name_alias(self):
        # .so.99 -> 11 and .so.99 -> 12 under baseline 12: the observed
        # name claims family 99, not the baseline — never relevant.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "99", order=0, target="11"),
                    lib("libcudart", "99", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_r_irrelevant_corroborated_class_b_shape(self):
        # Even a perfectly corroborated .so.99 alias stays irrelevant.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "99", order=0, target="11"),
                    lib("libcudart", "99", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_s_exact_name_distinction(self):
        # libcudart.so.11.0 and libcudart.so.11 are different lookup
        # identities under baseline 11 — no same-name group forms, so a
        # .11-provider cannot be the later compatible provider for .11.0.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0, target="10", name="libcudart.so.11.0"),
                    lib("libcudart", "11", order=1, target="11"),
                ),
                torch_family="11",
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_row_s_positive_needs_the_same_exact_tail(self):
        # The same environment with the exact tail repeated: Class A.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0, target="10", name="libcudart.so.11.0"),
                    lib("libcudart", "11", order=1, target="11", name="libcudart.so.11.0"),
                ),
                torch_family="11",
            )
        )
        assert RULE.evaluate(facts, []) is not None


class TestLibcudartBaseline:
    def test_19_8_example_same_name_shadowing_emits(self):
        # Truth tables §19.8 (rewritten): torch CUDA 12 build; within the
        # modeled LD_LIBRARY_PATH lookup for the observed exact name
        # libcudart.so.12, the family-11 target precedes the family-12
        # target — Class A (row C).
        diagnosis = conflict_diagnosis()
        assert diagnosis.confidence is Confidence.MEDIUM
        assert diagnosis.severity is Severity.WARNING

    def test_expected_family_preferred_emits_nothing(self):
        # D004-N01/N02-shaped: the earliest same-name target already is
        # the expected family; divergent later copies stay shadowed.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="12"),
                    lib("libcudart", "12", order=1, target="11"),
                    lib("libcudart", "12", order=2, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_multiple_copies_same_family_across_origins_no_conflict(self):
        # D004-N02: same compatible family everywhere — multiplicity in
        # ldconfig / toolkit / conda / package origins is inventory only.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", origin=ORIGIN_LDCONFIG, target="12"),
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE, target="12"),
                    lib("libcudart", "12", origin=ORIGIN_CONDA_PREFIX, target="12"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                    lib("libcudart", "12", order=0, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_unknown_torch_runtime_family_is_unknown(self):
        # §11A: no torch runtime family means no libcudart baseline — never
        # one inferred from filenames or toolkit versions.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                ),
                torch_family=None,
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_preferred_candidate_unknown_target_is_unknown(self):
        # Row H: the earliest same-name provider's target family is
        # unknown; skipping ahead would fabricate precedence.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0),
                    lib("libcudart", "12", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_reliability_never_gates_the_conflict_predicate(self):
        # R2 plan §4: reliability is not conflict evidence. Even a
        # synthetic UNKNOWN-reliability row-C pair (builder-impossible
        # shape) emits the identical candidate — the split identity
        # fields alone decide, and only baseline derivation reads
        # reliability.
        baseline = RULE.evaluate(conflict_facts(), [])
        synthetic = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib(
                            "libcudart",
                            "12",
                            order=0,
                            target="11",
                            reliability=IdentityReliability.UNKNOWN,
                        ),
                        lib(
                            "libcudart",
                            "12",
                            order=1,
                            target="12",
                            reliability=IdentityReliability.UNKNOWN,
                        ),
                    )
                )
            ),
            [],
        )
        assert synthetic is not None
        assert synthetic == baseline

    def test_incidental_candidates_never_imply_conflict(self):
        # Toolkit (incidental and active), conda, ldconfig, package copies
        # with disagreeing families but no modeled LD_LIBRARY_PATH
        # precedence at all.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib(
                        "libcudart",
                        "11",
                        origin=ORIGIN_TOOLKIT,
                        active=Trinary.FALSE,
                        target="11",
                    ),
                    lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE, target="11"),
                    lib("libcudart", "9", origin=ORIGIN_CONDA_PREFIX, target="9"),
                    lib("libcudart", "10", origin=ORIGIN_LDCONFIG, target="10"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_conflict_without_any_expected_form_is_not_medium(self):
        # A lone misleading alias with neither a later same-name LDP
        # provider nor a same-name package provider proves nothing
        # (row G).
        facts = facts_with(
            runtime_facts(candidates=(lib("libcudart", "12", order=0, target="11"),))
        )
        assert RULE.evaluate(facts, []) is None


class TestActiveToolkit:
    """The pre-Phase-6 active_toolkit distinction (plan §6): toolkit-origin
    candidates carry association state, but no toolkit existence — active,
    incidental, or unknown — ever becomes loader precedence."""

    def test_active_toolkit_true_alone_never_emits(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_active_toolkit_false_ignored_as_incidental(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib(
                        "libcudart",
                        "11",
                        origin=ORIGIN_TOOLKIT,
                        active=Trinary.FALSE,
                        target="11",
                    ),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_active_toolkit_unknown_stays_conservative(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib(
                        "libcudart",
                        "11",
                        origin=ORIGIN_TOOLKIT,
                        active=Trinary.UNKNOWN,
                        target="11",
                    ),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_active_and_incidental_coexist_without_collapsing_roles(self):
        # Adding active/incidental same-name toolkit copies (which would
        # look like compatible providers if origins leaked) to a real
        # modeled conflict changes neither the decision nor the evidence.
        standalone = RULE.evaluate(conflict_facts(), [])
        with_toolkits = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib(
                            "libcudart",
                            "12",
                            origin=ORIGIN_TOOLKIT,
                            active=Trinary.TRUE,
                            target="12",
                        ),
                        lib(
                            "libcudart",
                            "12",
                            origin=ORIGIN_TOOLKIT,
                            active=Trinary.FALSE,
                            target="12",
                        ),
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=1, target="12"),
                    )
                )
            ),
            [],
        )
        assert with_toolkits == standalone

    def test_active_toolkit_cannot_corroborate_class_b(self):
        # §19.11: only the frozen python_package same-name corroboration
        # crosses origins — an active toolkit provider never does.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestLdpOrdering:
    def test_earlier_conflicting_entry_wins_within_the_model(self):
        assert RULE.evaluate(conflict_facts(), []) is not None

    def test_reversed_ordering_emits_nothing(self):
        # N07: the earliest same-name target is the expected family; the
        # divergent later alias stays shadowed.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="12"),
                    lib("libcudart", "12", order=1, target="11"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_missing_search_order_invents_no_precedence(self):
        # Plan §8: an LDP candidate whose entry order was never established
        # is unordered — it is neither the earliest provider nor the later
        # same-name compatible one.
        unordered_conflict = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=None, target="12"),
                )
            )
        )
        unordered_expected = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=None, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(unordered_conflict, []) is None
        assert RULE.evaluate(unordered_expected, []) is None

    def test_non_ld_origins_never_receive_fake_order(self):
        # A stray modeled_search_order on a toolkit candidate (impossible
        # through the builder) must not act as modeled precedence.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, order=5, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_unknown_earliest_entry_blocks_even_with_later_conflict(self):
        # Row H: the earliest same-name provider's target family is
        # unknown, so no later entry can be promoted.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0),
                    lib("libcudart", "12", order=1, target="11"),
                    lib("libcudart", "12", order=2, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_two_distinct_candidates_at_the_earliest_entry_are_unknown(self):
        # §19.5: two distinct same-name candidates share the earliest
        # modeled position; choosing between them needs per-library
        # loader semantics the model does not cover.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=0, target="10"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_duplicate_candidates_neither_block_nor_duplicate(self):
        # Row P: exact duplicates collapse; the emission and its evidence
        # equal the non-duplicated scenario.
        baseline = RULE.evaluate(conflict_facts(), [])
        duplicated = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=1, target="12"),
                    )
                )
            ),
            [],
        )
        assert duplicated is not None
        assert duplicated == baseline

    def test_n07_divergent_later_same_name_alias_stays_shadowed(self):
        # The healthy majority shape: earliest target == baseline, a
        # divergent same-name copy later — shadowed, not a conflict.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="12"),
                    lib("libcudart", "12", order=1, target="11"),
                    lib("libcudart", "12", order=2, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_n08_normal_minor_version_symlink_is_healthy(self):
        # libcudart.so.12 -> libcudart.so.12.4.127 is the normal minor
        # symlink: the target family equals the claimed family.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="12"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestPythonPackageInteraction:
    def test_package_local_matching_candidate_alone_emits_nothing(self):
        facts = facts_with(
            runtime_facts(
                candidates=(lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),)
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_19_11_package_local_expected_vs_ldp_conflict_is_unknown(self):
        # Truth tables §19.11 / plan §24: the only way to pick the winner
        # is ELF RPATH/RUNPATH knowledge — resolution UNKNOWN, no MEDIUM.
        # (The package copy here has no exact same-name form.)
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target=None),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_old_family_level_winner_shape_flips_to_negative(self):
        # Amendment §21 mandated flip: the old
        # "winner decided inside the model" shape (different names
        # .so.11@0 / .so.12@1 plus a package-local family-12 copy)
        # emitted under the NOT-ACCEPTED family-level rule; under
        # Revision 3 the exact names differ, so nothing shadows anything
        # (row A) and the package copy corroborates nothing.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_package_local_conflict_alone_is_not_a_winner_claim(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="11"),
                    lib("libcudart", "12", order=0, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestClassBCorroboration:
    """The frozen Class-B relation (amendment §12/§14/§15): an
    independently observed current-Python same-name provider with a
    baseline-compatible target, and nothing else, corroborates."""

    def test_anti_circularity_uncorroborated_alias_never_emits(self):
        # Row G stated for Class B: the suspicious LDP candidate can
        # never establish its own relevance; without an independent
        # same-name package provider there is no Class B.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "11", origin=ORIGIN_PYTHON_PACKAGE, target="11"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_different_exact_package_name_does_not_corroborate(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib(
                        "libcudart",
                        "12",
                        origin=ORIGIN_PYTHON_PACKAGE,
                        target="12",
                        name="libcudart.so.12.1",
                    ),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_bare_package_alias_does_not_corroborate(self):
        # A bare package name (no versioned claim) is not the same
        # versioned lookup identity.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", None, origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_package_candidate_with_conflicting_target_does_not_corroborate(self):
        # Same exact name, but the package copy itself resolves to the
        # conflicting family — no baseline-compatible form observed.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="11"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize(
        "origin", [ORIGIN_LDCONFIG, ORIGIN_CONDA_PREFIX, ORIGIN_TOOLKIT]
    )
    def test_only_python_package_origin_corroborates(self, origin):
        # The frozen cross-origin allowance is current-Python
        # python_package only (§19.11/§19.12).
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=origin, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_class_b_never_claims_a_cross_origin_winner(self):
        # Row F emits — but only as a conditional static risk; nothing in
        # the environmental claims (summary, evidence, impact) says the
        # package-local copy wins, loads, or is preferred across origins.
        # The frozen §28 remediation prose ("an earlier entry wins within
        # the modeled search order") is scoped guidance about the modeled
        # ordering, not a cross-origin loader claim, so it is not scanned.
        candidate = RULE.evaluate(class_b_facts(), [])
        assert candidate is not None
        claimed = " ".join(
            [candidate.diagnosis.summary]
            + [item.value for item in candidate.diagnosis.evidence]
            + [f"{item.scope} {item.status}" for item in candidate.diagnosis.impact]
        )
        for forbidden in ("wins", "winner", "will load", "is loaded", "definitely loads"):
            assert forbidden not in claimed, forbidden
        assert "independently observed current-Python" in candidate.diagnosis.summary

    def test_class_a_preferred_over_class_b_for_the_same_conflict(self):
        # Plan §16: when the same family + exact name satisfies both
        # classes, Class A is the representative — one diagnosis, no
        # duplicated evidence.
        both = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        class_a_only = conflict_facts()
        first = RULE.evaluate(both, [])
        second = RULE.evaluate(class_a_only, [])
        assert first is not None and second is not None
        assert first == second  # Class A evidence shape, not Class B


class TestConda:
    def test_conda_candidate_existence_is_not_a_global_winner(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_CONDA_PREFIX, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_conda_candidate_leaves_modeled_conflict_unchanged(self):
        standalone = RULE.evaluate(conflict_facts(), [])
        with_conda = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "11", origin=ORIGIN_CONDA_PREFIX, target="11"),
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=1, target="12"),
                    )
                )
            ),
            [],
        )
        assert with_conda == standalone

    def test_conda_cannot_corroborate_class_b(self):
        # §19.11: conda stays inventory-only; only the frozen python_package
        # same-name corroboration crosses origins.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_CONDA_PREFIX, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestLdconfig:
    def test_ldconfig_candidate_is_inventory_only_never_winner(self):
        # Named regression (plan §25): an incompatible library visible only
        # through ldconfig, with the expected copy elsewhere, is inventory
        # evidence — not a modeled winner, not a MEDIUM D004 (§19.12).
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_LDCONFIG, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_ldconfig_candidate_leaves_modeled_conflict_unchanged(self):
        standalone = RULE.evaluate(conflict_facts(), [])
        with_ldconfig = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "9", origin=ORIGIN_LDCONFIG, target="9"),
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=1, target="12"),
                    )
                )
            ),
            [],
        )
        assert with_ldconfig == standalone

    def test_ldconfig_cannot_corroborate_class_b(self):
        # Row K for Class B: ldconfig presence must not alter an otherwise
        # identical decision and can never replace the package-local
        # corroboration.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_LDCONFIG, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_ldconfig_never_receives_a_modeled_order(self):
        # A stray search_order on an ldconfig candidate (impossible
        # through the builder) is not precedence.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", origin=ORIGIN_LDCONFIG, order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_ldconfig_only_old_library_with_complete_scan_emits_nothing(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "10", origin=ORIGIN_LDCONFIG, target="10"),
                    lib("libcudnn", "7", origin=ORIGIN_LDCONFIG, target="7"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestLibcublasBaseline:
    """Plan §32 matrix, pinned for the environment-observed baseline."""

    def _class_a(self, package_family: str) -> RuntimeLibraryFacts:
        return runtime_facts(
            candidates=(
                lib("libcublas", "12", order=0, target="11"),
                lib("libcublas", "12", order=1, target="12"),
                lib("libcublas", package_family, origin=ORIGIN_PYTHON_PACKAGE),
            )
        )

    def test_matrix_1_known_torch_and_single_package_family_establish_baseline(self):
        # §19.9: torch family 12 known and exactly one reliable
        # package-local libcublas family 12 observed — baseline 12, and
        # the same-name shadowing emits.
        candidate = RULE.evaluate(facts_with(self._class_a("12")), [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_matrix_2_unknown_torch_family_is_unknown(self):
        facts = facts_with(replace(self._class_a("12"), torch_runtime_family=None))
        assert RULE.evaluate(facts, []) is None

    def test_matrix_3_zero_package_local_families_is_unknown(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_matrix_4_multiple_disagreeing_package_families_are_unknown(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib("libcublas", "11", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_matrix_5_bare_package_alias_with_reliable_target_contributes(self):
        # A bare package alias (DERIVED_STRONG target identity, the shape
        # the strict builder records for libcublas.so -> libcublas.so.12)
        # contributes its target family to the baseline.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib(
                        "libcublas",
                        None,
                        origin=ORIGIN_PYTHON_PACKAGE,
                        reliability=IdentityReliability.DERIVED_STRONG,
                        target="12",
                    ),
                )
            )
        )
        assert RULE.evaluate(facts, []) is not None

    def test_matrix_6_no_cuda_major_assumption_observed_pair_rules(self):
        # Plan §11B: CUDA major == libcublas family is never assumed. This
        # environment observably pairs torch family 12 with a package-local
        # libcublas family 9 — that observed pair is the baseline.
        conflicting = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcublas", "9", order=0, target="12"),
                        lib("libcublas", "9", order=1, target="9"),
                        lib("libcublas", "9", origin=ORIGIN_PYTHON_PACKAGE),
                    )
                )
            ),
            [],
        )
        expected_preferred = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcublas", "9", order=0, target="9"),
                        lib("libcublas", "9", order=1, target="12"),
                        lib("libcublas", "9", origin=ORIGIN_PYTHON_PACKAGE),
                    )
                )
            ),
            [],
        )
        assert conflicting is not None
        assert expected_preferred is None

    def test_matrix_6_wrong_family_names_are_irrelevant_under_observed_baseline(self):
        # With the observed baseline 9, exact names claiming family 12
        # are irrelevant lookup identities (rows Q/R semantics) even
        # though torch runs family 12.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib("libcublas", "9", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_unreliable_package_local_identity_leaves_baseline_unknown(self):
        # Matrix-1 negative: the only package copy's identity is not
        # reliable enough for baseline derivation.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib(
                        "libcublas",
                        "12",
                        origin=ORIGIN_PYTHON_PACKAGE,
                        reliability=IdentityReliability.UNKNOWN,
                    ),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_matching_mapped_family_emits_nothing(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="12"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestLibcudnn:
    """Plan §33 matrix plus the §19.10 cuDNN rows."""

    def _cudnn_class_a(self) -> RuntimeLibraryFacts:
        return runtime_facts(
            candidates=(
                lib("libcudnn", "9", order=0, target="8"),
                lib("libcudnn", "9", order=1, target="9"),
            ),
            cudnn_family="9",
        )

    def test_matrix_1_known_package_local_baseline_conflict_is_medium(self):
        # §19.10: expected environment-local cuDNN family 9; the modeled
        # lookup for the exact name libcudnn.so.9 resolves to family 8
        # first and to family 9 later.
        candidate = RULE.evaluate(facts_with(self._cudnn_class_a()), [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_matrix_2_unknown_baseline_is_unknown(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family=None,
            )
        )
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize("torch_family", [None, "11", "13"])
    def test_matrix_3_torch_family_changes_never_alter_the_cudnn_decision(
        self, torch_family
    ):
        # Plan §26 invariant, restated for exact names: holding the cuDNN
        # baseline and candidates fixed, torch_runtime_family is inert.
        baseline = RULE.evaluate(facts_with(self._cudnn_class_a()), [])
        assert baseline is not None
        variant = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudnn", "9", order=0, target="8"),
                        lib("libcudnn", "9", order=1, target="9"),
                    ),
                    torch_family=torch_family,
                    cudnn_family="9",
                )
            ),
            [],
        )
        assert variant == baseline

    def test_matrix_4_bare_alias_cannot_establish_lookup_relevance(self):
        # Row E for cuDNN: bare development aliases are ineligible.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", None, order=0, target="8"),
                    lib("libcudnn", None, order=1, target="9"),
                ),
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_matrix_5_versioned_name_needs_the_baseline_family(self):
        # Exact names claiming family 8 are irrelevant under baseline 9,
        # however their targets diverge (rows Q/R semantics).
        wrong_family = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "8", order=0, target="9"),
                    lib("libcudnn", "8", order=1, target="8"),
                ),
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(wrong_family, []) is None
        right_family = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", origin=ORIGIN_PYTHON_PACKAGE, target="9"),
                ),
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(right_family, []) is not None

    def test_matching_candidates_emit_nothing(self):
        # D004-N03: multiple cuDNN copies, all consistent with baseline.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="9"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_multiple_package_local_baselines_are_unknown_synthetic(self):
        # The builder collapses this to package_local_cudnn_family=None
        # (pinned by the Phase-5 suite); the rule sees UNKNOWN baseline.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family=None,
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_multiple_package_local_baselines_are_unknown_builder(self):
        snapshot = snapshot_with_inventory(
            raw("libcudnn", "8", origin=ORIGIN_PYTHON_PACKAGE),
            raw("libcudnn", "9", origin=ORIGIN_PYTHON_PACKAGE),
            raw("libcudnn", "9", origin=ORIGIN_LD_LIBRARY_PATH, order=0),
            raw("libcudnn", "9", origin=ORIGIN_LD_LIBRARY_PATH, order=1),
        )
        facts = built_facts(snapshot)
        assert facts.runtime_libraries.package_local_cudnn_family is None
        assert RULE.evaluate(facts, []) is None

    def test_torch_cuda_family_is_never_a_cudnn_baseline(self):
        # Semantics §23.8 / plan §26: the conflicting cuDNN target family
        # equals the torch runtime family here — if the rule wrongly
        # compared cuDNN against torch it would treat the earliest
        # target as expected and stay silent.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="12"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                torch_family="12",
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(facts, []) is not None


class TestScanIncomplete:
    def test_successful_complete_empty_inventory_emits_nothing(self):
        facts = facts_with(runtime_facts())
        assert facts.runtime_libraries.scan_incomplete is False
        assert RULE.evaluate(facts, []) is None

    def test_incomplete_scan_with_no_conflict_emits_nothing(self):
        # §12: a failed scan is not an empty healthy inventory — and it is
        # equally never conjured into a conflict.
        facts = facts_with(runtime_facts(scan_incomplete=True))
        assert facts.runtime_libraries.scan_incomplete is True
        assert RULE.evaluate(facts, []) is None

    def test_observed_conflict_still_emits_under_incomplete_scan(self):
        # An already-established Class A stays MEDIUM; unscanned roots
        # cannot change LDP-internal precedence (rows A-S apply to what
        # is observed; scan_incomplete never caps it).
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                ),
                scan_incomplete=True,
            )
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_absence_under_incomplete_scan_is_not_healthy_proof(self):
        # Only the expected family is observed, under an incomplete scan:
        # no candidate is emitted, and nothing about the environment is
        # declared healthy (a None result is silence, not an all-clear).
        facts = facts_with(
            runtime_facts(
                candidates=(lib("libcudart", "12", order=0, target="12"),),
                scan_incomplete=True,
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestAggregation:
    def two_family_facts(self) -> DiagnosisFacts:
        return facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family="9",
            )
        )

    def test_multiple_families_aggregate_into_one_candidate(self):
        # Plan §14: one global D004 per environment, never one per pair.
        candidate = RULE.evaluate(self.two_family_facts(), [])
        assert candidate is not None
        keys = [item.key for item in candidate.diagnosis.evidence]
        assert keys == [
            "libcudart_conflicting",
            "libcudart_expected",
            "libcudnn_conflicting",
            "libcudnn_expected",
            "static_relation",
        ]

    def test_summary_names_each_conflicting_family(self):
        diagnosis = RULE.evaluate(self.two_family_facts(), []).diagnosis
        assert "libcudart" in diagnosis.summary
        assert "libcudnn" in diagnosis.summary

    def test_three_families_still_one_candidate(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family="9",
            )
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert len(candidate.diagnosis.evidence) == 7  # 3 families x 2 + relation
        families = {
            item.key.rsplit("_", 1)[0]
            for item in candidate.diagnosis.evidence
            if item.key.endswith("_conflicting")
        }
        assert families == {"libcudart", "libcublas", "libcudnn"}
        sources = {item.key: item.source for item in candidate.diagnosis.evidence}
        # Only the libcudart baseline item is torch-derived (§11A); the
        # libcublas/libcudnn expected items stay runtime-library facts.
        assert sources["libcudart_expected"] == "torch"
        assert sources["libcublas_expected"] == "runtime_libraries"
        assert sources["libcudnn_expected"] == "runtime_libraries"

    def test_duplicate_evidence_never_accumulates(self):
        baseline = RULE.evaluate(conflict_facts(), [])
        duplicated = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcudart", "12", order=1, target="12"),
                        lib("libcudart", "12", order=1, target="12"),
                    )
                )
            ),
            [],
        )
        assert duplicated == baseline
        assert len(duplicated.diagnosis.evidence) == len(baseline.diagnosis.evidence)

    def _representative_name(self, facts: DiagnosisFacts) -> str:
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        conflicting = next(
            item.value
            for item in candidate.diagnosis.evidence
            if item.key == "libcudart_conflicting"
        )
        return conflicting

    def test_class_a_is_the_representative_over_class_b(self):
        # Plan §17 selection rule 1: libcudart.so.12 is Class-B-only
        # here (no later LDP provider); libcudart.so.12.0 is Class A —
        # the Class-A finding represents the family.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                    lib("libcudart", "12", order=1, target="11", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=2, target="12", name="libcudart.so.12.0"),
                )
            )
        )
        conflicting = self._representative_name(facts)
        assert "load name libcudart.so.12.0 (" in conflicting
        assert "load name libcudart.so.12 (" not in conflicting

    def test_lower_modeled_order_wins_between_two_class_a_names(self):
        # Plan §17 selection rule 2: libcudart.so.12.0 conflicts at entry
        # 0; libcudart.so.12 conflicts at entry 1 — both Class A.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=3, target="12", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=1, target="11"),
                    lib("libcudart", "12", order=2, target="12"),
                )
            )
        )
        conflicting = self._representative_name(facts)
        assert "load name libcudart.so.12.0 (" in conflicting
        assert "load name libcudart.so.12 (" not in conflicting

    def test_lexical_name_breaks_order_ties(self):
        # Plan §17 selection rule 3: both Class A with the conflict at
        # entry 0; the lexically smaller exact name represents.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=2, target="12", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=2, target="12"),
                )
            )
        )
        conflicting = self._representative_name(facts)
        assert "load name libcudart.so.12 (" in conflicting
        assert "load name libcudart.so.12.0 (" not in conflicting

    def test_selection_is_presentation_only_and_order_independent(self):
        # Input ordering never changes the representative (or existence).
        ordered = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=2, target="12", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                )
            )
        )
        shuffled = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=2, target="12", name="libcudart.so.12.0"),
                    lib("libcudart", "12", order=0, target="11", name="libcudart.so.12.0"),
                )
            )
        )
        assert RULE.evaluate(ordered, []) == RULE.evaluate(shuffled, [])


class TestUnknownConservative:
    @pytest.mark.parametrize(
        "runtime",
        [
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                ),
                torch_family=None,
            ),
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                ),
                torch_family=None,
            ),
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family=None,
            ),
        ],
        ids=["cudart-baseline", "cublas-baseline", "cudnn-baseline"],
    )
    def test_missing_baseline_is_not_a_conflict(self, runtime):
        assert RULE.evaluate(facts_with(runtime), []) is None

    def test_partial_facts_degrade_to_silence(self):
        # Baseline present but nothing modeled at all.
        facts = facts_with(
            runtime_facts(
                candidates=(lib("libcudart", "11", origin=ORIGIN_LDCONFIG, target="11"),),
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestStaticConfidenceCeiling:
    @pytest.mark.parametrize(
        "runtime",
        [
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            ),
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0, target="11"),
                    lib("libcublas", "12", order=1, target="12"),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            ),
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family="9",
            ),
        ],
        ids=["libcudart", "libcublas", "libcudnn"],
    )
    def test_static_inventory_alone_never_reaches_high(self, runtime):
        # Plan §23: pin the contract directly — the strongest possible
        # static evidence (all-direct identities, clear same-name
        # shadowing, expected provider modeled later) still yields
        # MEDIUM, never HIGH.
        candidate = RULE.evaluate(facts_with(runtime), [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM
        assert candidate.diagnosis.confidence is not Confidence.HIGH

    def test_low_confidence_multiplicity_is_not_emitted_at_all(self):
        # D004-N04: multiple conflicting copies with no resolvable
        # precedence stay informational at the fact/issue level — the rule
        # emits no LOW D004 candidate.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_LDCONFIG, target="11"),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                    lib("libcudart", "11", origin=ORIGIN_CONDA_PREFIX, target="11"),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestRpathNonInference:
    def test_cross_origin_emission_stays_conditional_never_a_winner(self):
        # Row J: deciding which origin the loader would prefer needs
        # RPATH/RUNPATH knowledge. Revision 3 still permits the frozen
        # Class-B conditional static risk for this shape — the assertion
        # is that the output never upgrades it into a winner claim.
        candidate = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "12", order=0, target="11"),
                        lib("libcublas", "12", order=0, target="11"),
                        lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                        lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
                    )
                )
            ),
            [],
        )
        assert candidate is not None
        claimed = " ".join(
            [candidate.diagnosis.summary]
            + [item.value for item in candidate.diagnosis.evidence]
            + [f"{item.scope} {item.status}" for item in candidate.diagnosis.impact]
        )
        # The frozen §28 remediation prose describes the modeled ordering
        # only; the environmental claims are what must stay conditional.
        for forbidden in ("wins", "winner", "will load", "definitely loads", "RPATH", "RUNPATH"):
            assert forbidden not in claimed, forbidden

    @pytest.mark.parametrize("module", [rules_base, d004_module], ids=["base", "d004"])
    def test_imports_no_host_probes_or_collector_surfaces(self, module):
        # Structural purity (plan §24/§27/§34.11): resolving loader
        # ambiguity with host inspection would require importing one of
        # these. D004 is pure over DiagnosisFacts (plan §35).
        tree = ast.parse(Path(module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert imported.isdisjoint(
            {"os", "subprocess", "shutil", "site", "time", "random", "torch"}
        )
        for name in imported:
            assert not name.startswith("cuda_doctor.collectors"), name
            assert name != "cuda_doctor.core.models", name

    def test_source_names_no_collector_types_or_loader_probes(self):
        # Plan §22/§34.11: the rule must not even reference the
        # pre-normalization surface, the fact builder, or any forbidden
        # loader-inspection tooling by name.
        source = Path(d004_module.__file__).read_text()
        for forbidden in (
            "EnvironmentSnapshot",
            "RuntimeLibraryInventory",
            "DiagnosisFactBuilder",
            "ldconfig -p",
            "readelf",
            "objdump",
            "ldd",
            "LD_DEBUG",
            "/proc/",
            "dlopen",
            "ctypes",
            "DT_NEEDED",
            "DT_SONAME",
            "RPATH",
            "RUNPATH",
            "SONAME",
        ):
            assert forbidden not in source, forbidden


class TestPurityAndDeterminism:
    def test_same_facts_produce_equal_candidates(self):
        assert RULE.evaluate(conflict_facts(), []) == RULE.evaluate(conflict_facts(), [])

    def test_candidate_input_order_changes_nothing(self):
        ordered = conflict_facts()
        shuffled = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=1, target="12"),
                    lib("libcudart", "12", order=0, target="11"),
                )
            )
        )
        assert RULE.evaluate(ordered, []) == RULE.evaluate(shuffled, [])

    def test_host_environment_changes_nothing(self, monkeypatch):
        # The rule must never consult os.environ or other host state.
        facts = conflict_facts()
        before = RULE.evaluate(facts, [])
        monkeypatch.setenv("LD_LIBRARY_PATH", "/host/legacy/cuda-99/lib64")
        monkeypatch.setenv("CUDA_HOME", "/host/cuda-99.0")
        assert RULE.evaluate(facts, []) == before


class TestD001D002Interaction:
    def test_20_6_d001_and_d004_fire_independently(self):
        # Truth tables §20.6: active selector disagreement plus an
        # independent LD_LIBRARY_PATH conflict — both valid, no causal
        # edge, global ranking is a later presentation concern.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            ),
            selectors=(
                sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
                sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            ),
        )
        d001 = D001.evaluate(facts, [])
        d004 = RULE.evaluate(facts, [])
        assert d001 is not None
        assert d004 is not None
        assert d004.causal_parent_codes == ()

    def test_d002_and_d004_fire_independently(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                )
            ),
            reference=ref(version="11.8"),
        )
        assert D002.evaluate(facts, []) is not None
        standalone = RULE.evaluate(conflict_facts(), [])
        assert RULE.evaluate(facts, []) == standalone


class TestImportFailure:
    def test_import_failure_does_not_block_independent_evidence(self):
        # §19.13: D004 may emit on its own evidence — here the cuDNN
        # Class-A conflict, whose baseline never depends on torch — even
        # when PyTorch cannot be imported at all.
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch,
            installed=False,
            cuda_version=None,
            is_cuda_build=None,
            cuda_available=None,
        )
        snapshot.runtime_libraries = RuntimeLibraryInventory(
            candidates=[
                raw(
                    "libcudnn",
                    "9",
                    origin=ORIGIN_LD_LIBRARY_PATH,
                    order=0,
                    canonical_path="/opt/inventory/libcudnn.so.8.0.5",
                ),
                raw(
                    "libcudnn",
                    "9",
                    origin=ORIGIN_LD_LIBRARY_PATH,
                    order=1,
                    canonical_path="/opt/inventory/libcudnn.so.9.1",
                ),
                raw("libcudnn", "9", origin=ORIGIN_PYTHON_PACKAGE),
            ]
        )
        facts = built_facts(snapshot)
        assert facts.torch.importable is False
        assert facts.runtime_libraries.package_local_cudnn_family == "9"
        assert RULE.evaluate(facts, []) is not None

    def test_import_failure_alone_is_not_d004_evidence(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, installed=False, cuda_version=None)
        facts = built_facts(snapshot)
        assert facts.torch.importable is False
        assert RULE.evaluate(facts, []) is None


class TestCandidateOutputContent:
    def test_frozen_title_and_code(self):
        diagnosis = conflict_diagnosis()
        assert diagnosis.code == "D004"
        assert diagnosis.title == "CUDA runtime library conflict"

    def test_summary_uses_modeled_lookup_wording_only(self):
        # Semantics §23.7 / plan §25: MEDIUM conclusions are phrased in
        # terms of the modeled same-name lookup — never as proof of which
        # object will load, and never as a cross-origin winner claim.
        summary = conflict_diagnosis().summary
        assert "modeled LD_LIBRARY_PATH" in summary
        assert "not observed" in summary
        for forbidden in ("will load", "is loaded", "definitely loads", "wins", "winner"):
            assert forbidden not in summary, forbidden

    def test_summary_correct_for_class_b(self):
        # The Class-B clause must describe the corroborated-alias
        # relation, not same-name shadowing.
        summary = RULE.evaluate(class_b_facts(), []).diagnosis.summary
        assert "independently observed current-Python compatible provider" in summary
        assert "shadowed" not in summary
        assert "not observed" in summary

    def test_evidence_bounded_and_ordered_class_a(self):
        # Amendment §20 frozen evidence formats for Class A.
        assert conflict_diagnosis().evidence == [
            EvidenceItem(
                key="libcudart_conflicting",
                value=(
                    "observed versioned load name libcudart.so.12 (claims family 12) "
                    "at LD_LIBRARY_PATH entry 0 resolves to family 11 (derived "
                    "strongly from its symlink target)"
                ),
                source="runtime_libraries",
            ),
            EvidenceItem(
                key="libcudart_expected",
                value="family 12 provider of libcudart.so.12 shadowed at a later entry",
                source="torch",
            ),
            EvidenceItem(
                key="static_relation",
                value=(
                    "same-name shadowing / misleading alias on an observed "
                    "versioned load name within the modeled LD_LIBRARY_PATH ordering"
                ),
                source="compatibility",
            ),
        ]

    def test_evidence_bounded_and_ordered_class_b(self):
        # Amendment §20 frozen evidence formats for Class B.
        diagnosis = RULE.evaluate(class_b_facts(), []).diagnosis
        assert diagnosis.evidence == [
            EvidenceItem(
                key="libcudart_conflicting",
                value=(
                    "observed versioned load name libcudart.so.12 (claims family 12) "
                    "at LD_LIBRARY_PATH entry 0 resolves to family 11 (derived "
                    "strongly from its symlink target)"
                ),
                source="runtime_libraries",
            ),
            EvidenceItem(
                key="libcudart_expected",
                value="family 12 package-local provider of libcudart.so.12",
                source="torch",
            ),
            EvidenceItem(
                key="static_relation",
                value=(
                    "same-name shadowing / misleading alias on an observed "
                    "versioned load name within the modeled LD_LIBRARY_PATH ordering"
                ),
                source="compatibility",
            ),
        ]

    def test_evidence_contains_no_path_like_content(self):
        # The frozen §20 static_relation value contains one "/" as a
        # prose separator ("shadowing / misleading alias"); every other
        # slash anywhere in evidence would be path-like content.
        for item in conflict_diagnosis().evidence:
            if item.key == "static_relation":
                assert "/" not in item.value.replace("shadowing / misleading", "")
            else:
                assert "/" not in item.value
            assert "$" not in item.value

    def test_evidence_never_presents_reliability(self):
        # Plan §26: reliability is not conflict evidence.
        for item in conflict_diagnosis().evidence:
            assert "reliability" not in item.value.lower()
            assert "(direct)" not in item.value
            assert "derived strongly from" in item.value or item.key != "libcudart_conflicting"

    def test_frozen_impact(self):
        # Plan §27: alias resolution at risk; PyTorch initialization may
        # be affected conditionally; the loaded object is not observed.
        assert conflict_diagnosis().impact == [
            ImpactItem(
                scope="CUDA runtime library resolution",
                status=(
                    "alias resolution at risk under the modeled or "
                    "conditional LD_LIBRARY_PATH semantics"
                ),
            ),
            ImpactItem(
                scope="PyTorch CUDA initialization",
                status="may be affected if the modeled or conditional resolution applies",
            ),
            ImpactItem(
                scope="actually loaded library",
                status="not observed (static analysis only)",
            ),
        ]

    def test_remediation_safe_guidance_without_commands(self):
        # §19.15 (frozen; amendment OQ-3 — remediation untouched by R2):
        # remove/reorder the conflicting entry, start a clean shell,
        # re-run cuda-doctor — all SAFE, nothing executed.
        steps = conflict_diagnosis().remediation
        assert 2 <= len(steps) <= 5
        for step in steps:
            assert step.safety is RemediationSafety.SAFE
            assert step.command is None
            assert step.why
        rendered = " ".join(f"{step.action} {step.why}" for step in steps)
        assert "LD_LIBRARY_PATH" in rendered
        assert "clean shell" in rendered
        assert "cuda-doctor" in rendered

    def test_remediation_invents_no_target_and_no_destructive_action(self):
        rendered = str(asdict(conflict_diagnosis()))
        for forbidden in (
            "rm -rf",
            "rm ",
            "delete",
            "unlink",
            "overwrite",
            "uninstall",
            "apt ",
            "ln -s",
            "/usr/",
            "/opt/",
            "/home/",
            "11.8",
            "12.4",
            "cuda-11",
            "cuda-12",
        ):
            assert forbidden not in rendered, forbidden

    def test_frozen_verification_commands(self):
        # §19.16 minimum (plan §29), display-only; Linux-only rule so no
        # platform keying and no loaded-object inspection.
        assert conflict_diagnosis().verification == [
            "echo $LD_LIBRARY_PATH",
            "cuda-doctor",
            "python -c \"import torch; print(torch.cuda.is_available())\"",
        ]

    def test_diagnosis_stays_pre_resolver(self):
        diagnosis = conflict_diagnosis()
        assert diagnosis.role is None
        assert diagnosis.cause_class is None
        assert diagnosis.relationships == []
        assert diagnosis.exclusions == []

    def test_candidate_is_risk_only_without_causal_parents(self):
        # Static analysis alone: no observed failure is explained here —
        # the D003 causal chain is Phase 9/10 wiring (plan §16/§37).
        candidate = RULE.evaluate(conflict_facts(), [])
        assert candidate.risk_only is True
        assert candidate.explains_active_failure is False
        assert candidate.causal_parent_codes == ()

    def test_no_legacy_issue_linkage_or_suppression(self):
        # Plan §21: D004 owns no legacy suppression mapping — the carrier
        # stays empty and no issue code is related, whatever fired.
        for codes in (frozenset(), frozenset({"CUDA004", "TORCH005"})):
            facts = replace(conflict_facts(), issue_codes=codes)
            candidate = RULE.evaluate(facts, [])
            assert candidate is not None
            assert candidate.diagnosis.related_issues == []
            assert candidate.diagnosis.presentation_suppressed_issue_codes == []

    def test_confidence_and_severity_pinned(self):
        diagnosis = conflict_diagnosis()
        assert diagnosis.confidence is Confidence.MEDIUM
        assert diagnosis.severity is Severity.WARNING


class TestMisattribution:
    """Amendment §21 mandatory regression: the conflict is attributed to
    the divergent alias, never to the real versioned file beside it."""

    def _scenario(self) -> tuple[RuntimeLibraryCandidateFact, ...]:
        # Entry 0: a real family-11 file under its own name (healthy).
        # Entry 1: a misleading family-12 alias resolving to family 9.
        # Package: an independent family-12 provider of the same alias.
        return (
            lib("libcudart", "11", order=0, target="11"),
            lib("libcudart", "12", order=1, target="9"),
            lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE, target="12"),
        )

    def test_conflict_attributed_to_the_divergent_alias(self):
        candidate = RULE.evaluate(facts_with(runtime_facts(self._scenario())), [])
        assert candidate is not None
        conflicting = [
            item.value
            for item in candidate.diagnosis.evidence
            if item.key == "libcudart_conflicting"
        ]
        assert conflicting == [
            "observed versioned load name libcudart.so.12 (claims family 12) "
            "at LD_LIBRARY_PATH entry 1 resolves to family 9 (derived "
            "strongly from its symlink target)"
        ]
        expected = [
            item.value
            for item in candidate.diagnosis.evidence
            if item.key == "libcudart_expected"
        ]
        # Entry 1 is never cited as expected family-12 content; the
        # package-local provider is.
        assert expected == ["family 12 package-local provider of libcudart.so.12"]
        rendered = str(asdict(candidate.diagnosis))
        assert "family 11 at" not in rendered  # the real file is never the conflict

    def test_without_corroboration_the_same_scenario_stays_unknown(self):
        candidates = self._scenario()[:2]
        facts = facts_with(runtime_facts(candidates))
        assert RULE.evaluate(facts, []) is None


class TestEvidencePrivacy:
    def test_secret_inventory_path_never_reaches_the_candidate(self):
        # Plan §18: a secret-looking raw candidate path in the Phase-3
        # inventory must never appear in the D004 candidate. The fact
        # layer carries no paths at all, so it is structurally impossible.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                path="/home/secretuser/legacy/libcudart.so.12",
                canonical_path="/home/secretuser/legacy/libcudart.so.11.8.89",
                version_source=VERSION_SOURCE_SYMLINK_TARGET,
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                path="/home/secretuser/expected/libcudart.so.12",
                canonical_path="/home/secretuser/expected/libcudart.so.12",
            ),
        )
        facts = built_facts(snapshot)
        assert "secretuser" not in str(asdict(facts.runtime_libraries))
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert "secretuser" not in str(asdict(candidate.diagnosis))
        assert "secretuser" not in str(candidate.diagnosis.to_dict())
        assert "secretuser" not in str(asdict(candidate))
        assert "/home" not in str(asdict(candidate.diagnosis))


class TestBuilderEndToEnd:
    def test_row_c_through_the_fact_builder(self):
        # Real symlink shape: entry 0's libcudart.so.12 is a symlink to
        # the family-11 file; entry 1's is a regular family-12 file.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                path="/legacy/cuda-11/lib64/libcudart.so.12",
                canonical_path="/legacy/cuda-11/lib64/libcudart.so.11.8.89",
                version_source=VERSION_SOURCE_SYMLINK_TARGET,
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                path="/expected/cuda-12/lib64/libcudart.so.12",
                canonical_path="/expected/cuda-12/lib64/libcudart.so.12",
            ),
        )
        facts = built_facts(snapshot)
        assert facts.runtime_libraries.torch_runtime_family == "12"
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_row_d_exact_tail_retained_through_the_fact_builder(self):
        # CUDA-11-style: the exact observed name libcudart.so.11.0 (a
        # symlink to the family-10 file) never normalizes to .so.11.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "11.0",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                path="/legacy/libcudart.so.11.0",
                canonical_path="/legacy/libcudart.so.10.2",
            ),
            raw(
                "libcudart",
                "11.0",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                path="/expected/libcudart.so.11.0",
                canonical_path="/expected/libcudart.so.11.0",
            ),
            cuda_version="11.8",
        )
        facts = built_facts(snapshot)
        assert facts.runtime_libraries.torch_runtime_family == "11"
        names = [c.load_name for c in facts.runtime_libraries.candidates]
        assert "libcudart.so.11.0" in names
        assert RULE.evaluate(facts, []) is not None

    def test_bare_alias_target_identity_creates_no_lookup_relevance(self):
        # R2 flip of the old symlink-derived positive (amendment §21):
        # a bare observed name (load_name_family_version None,
        # DERIVED_STRONG target identity) is not an eligible versioned
        # lookup name — rows E. The fact-field assertions the old test
        # made are retained.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                None,
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                path="/opt/inventory/libcudart.so",
                canonical_path="/opt/inventory/libcudart.so.11.8",
                version_source=VERSION_SOURCE_SYMLINK_TARGET,
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                canonical_path="/opt/inventory/libcudart.so.12",
            ),
        )
        facts = built_facts(snapshot)
        preferred = facts.runtime_libraries.candidates[0]
        assert preferred.load_name == "libcudart.so"
        assert preferred.load_name_family_version is None
        assert preferred.target_family_version == "11"
        assert preferred.reliability is IdentityReliability.DERIVED_STRONG
        assert RULE.evaluate(facts, []) is None

    def test_scan_errors_mark_incomplete_without_blocking_the_conflict(self):
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                canonical_path="/opt/inventory/libcudart.so.11.8.89",
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                canonical_path="/opt/inventory/libcudart.so.12",
            ),
            scan_errors={"/root/unreadable": "permission denied"},
        )
        facts = built_facts(snapshot)
        assert facts.runtime_libraries.scan_incomplete is True
        assert RULE.evaluate(facts, []) is not None


class TestNoEngineIntegration:
    def test_engine_still_emits_no_diagnoses(self):
        # Plan §28/§38: the rule fires on the built facts, but the engine
        # neither runs it nor populates DiagnosisResult.diagnoses.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                canonical_path="/opt/inventory/libcudart.so.11.8.89",
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                canonical_path="/opt/inventory/libcudart.so.12",
            ),
        )
        result = DiagnosisEngine().run(snapshot)
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}
        assert RULE.evaluate(built_facts(snapshot), []) is not None


class TestStructuralContract:
    """R2 plan §34 binding structural tests over the rewritten rule.

    Items 7 (rows Q/R/G), 10 (imports) and 11 (source names) are pinned
    behaviorally in TestTruthTableRows / TestRpathNonInference; the
    remaining items are pinned here by AST/source inspection.
    """

    def _source(self) -> str:
        return Path(d004_module.__file__).read_text()

    def _tree(self) -> ast.Module:
        return ast.parse(self._source())

    def test_legacy_bridge_is_gone(self):
        # §34.1: the transitional R1 bridge no longer exists.
        assert not hasattr(d004_module, "_legacy_family_version")

    def test_no_transitional_r1_or_legacy_bridge_in_production_source(self):
        # §34.2: no production occurrence survives anywhere under src.
        src_root = Path(d004_module.__file__).parents[3]
        assert src_root.name == "src"
        for path in sorted(src_root.rglob("*.py")):
            text = path.read_text()
            assert "_legacy_family_version" not in text, path
            assert "TRANSITIONAL R1" not in text, path

    def test_no_lookup_name_is_constructed_from_family_ingredients(self):
        # §34.3: no string constant in the rule contains the
        # shared-object separator, so no exact name can be assembled from
        # family and version ingredients (also amendment §21 test 1 — no
        # synthesized libcudart.so.<major>).
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert ".so" not in node.value, node.value

    def test_class_a_groups_on_exact_load_name_equality(self):
        # §34.4: same-name grouping is string equality over the observed
        # exact name — never family arithmetic or prefix matching (row S
        # and the .11.0-vs-.11 behavioral pins).
        assert "candidate.load_name == name" in self._source()

    def test_relevance_gate_is_shared_by_both_classes(self):
        # §34.5/§34.6: one relevance guard — the observed name's claimed
        # family must equal the baseline — admits both Class A and
        # Class B (behavioral pins: rows Q and R).
        assert "load_name_family_version != baseline" in self._source()

    def test_class_b_requires_exact_same_name_package_corroboration(self):
        # §34.7: the corroboration predicate is exact-name equality plus
        # a baseline-compatible package-local target.
        assert "package_candidate.load_name == name" in self._source()

    def test_conflict_target_identity_comes_from_target_family_version(self):
        # §34.8: every finding's conflicting family is the earliest
        # provider's target_family_version (behavioral pins: row D and
        # the misattribution scenario; never a load-name fallback).
        assert "conflicting_target_family=preferred.target_family_version" in self._source()

    def test_reliability_read_only_in_baseline_identity(self):
        # §34.9: candidate.reliability may be read ONLY inside
        # _baseline_identity (package-local family-baseline derivation).
        offenders = [
            function.name
            for function in ast.walk(self._tree())
            if isinstance(function, ast.FunctionDef)
            and function.name != "_baseline_identity"
            and any(
                isinstance(node, ast.Attribute) and node.attr == "reliability"
                for node in ast.walk(function)
            )
        ]
        assert offenders == []

    def test_evaluate_returns_at_most_one_candidate(self):
        # §34.12: one D004 candidate maximum — the return type is a
        # single optional candidate, and multi-family facts yield one.
        multi = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0, target="11"),
                    lib("libcudart", "12", order=1, target="12"),
                    lib("libcudnn", "9", order=0, target="8"),
                    lib("libcudnn", "9", order=1, target="9"),
                ),
                cudnn_family="9",
            )
        )
        result = RULE.evaluate(multi, [])
        assert result is None or isinstance(result, type(RULE.evaluate(conflict_facts(), [])))
        single = RULE.evaluate(conflict_facts(), [])
        assert single is not None
        assert isinstance(single.diagnosis, Diagnosis)
