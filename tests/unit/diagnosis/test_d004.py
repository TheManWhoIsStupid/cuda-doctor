"""Tests for the D004 CUDA runtime library conflict rule.

The rule is tested over synthetic ``DiagnosisFacts`` (truth tables §19):
collector and normalization behavior is pinned by the Phase-3/Phase-5
suites; here only the diagnosis decision over already-normalized facts
matters. Frozen truth-table row IDs appear in test names/comments
(D004-N01 ... D004-N04 plus the §19.8-§19.13 scenario families) so the
coverage stays auditable against the frozen tables.
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
) -> RuntimeLibraryCandidateFact:
    """Synthetic normalized candidate; only the D004 surface is populated.

    R1 shape migration: the old synthetic ``version`` concept is now the
    split identity the transitional rule reads — a versioned exact load
    name claiming that family (``load_name_family_version``), optionally
    with a canonical-target family (``target``). ``version=None`` means
    no name-derived identity at all: bare name, ``load_name_family_version``
    ``None``.
    """
    load_name = f"{family}.so.{version}" if version is not None else f"{family}.so"
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
    """The canonical D004 environment (truth tables §19.8): torch CUDA 12
    build, LD_LIBRARY_PATH preferring a CUDA 11 libcudart with the expected
    CUDA 12 copy later in the same modeled ordering."""
    return facts_with(
        runtime_facts(
            candidates=(
                lib("libcudart", "11", order=0),
                lib("libcudart", "12", order=1),
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

    R1: ``canonical_path`` lets a synthetic candidate simulate what the
    strict collector records for a resolvable symlink target (synthetic
    fixtures never touch the real filesystem); a bare name is expressed
    with ``version=None`` and an explicit bare ``path``.
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
                )
            ),
            applicability=replace(base.applicability, d004=False),
        )
        assert RULE.evaluate(facts, []) is None


class TestLibcudartBaseline:
    def test_19_8_example_conflicting_earlier_entry_emits(self):
        # Truth tables §19.8: torch.version.cuda 12.4, LD_LIBRARY_PATH [0]
        # CUDA 11 libcudart, [1] CUDA 12 libcudart -> conflicting copy is
        # preferred among the modeled candidates.
        diagnosis = conflict_diagnosis()
        assert diagnosis.confidence is Confidence.MEDIUM
        assert diagnosis.severity is Severity.WARNING

    def test_expected_family_preferred_emits_nothing(self):
        # D004-N01/N02-shaped: the modeled order prefers the expected
        # family; however many copies exist, there is no conflict.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0),
                    lib("libcudart", "11", order=1),
                    lib("libcudart", "12", order=2),
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
                    lib("libcudart", "12", origin=ORIGIN_LDCONFIG),
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE),
                    lib("libcudart", "12", origin=ORIGIN_CONDA_PREFIX),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcudart", "12", order=0),
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
                ),
                torch_family=None,
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_preferred_candidate_unknown_version_is_unknown(self):
        # §13/§19.5: the preferred candidate's family is not reliably
        # identifiable; skipping it would fabricate precedence.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", None, order=0),
                    lib("libcudart", "12", order=1),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize(
        "reliability",
        [IdentityReliability.UNKNOWN, IdentityReliability.DERIVED_WEAK],
        ids=["unknown", "derived-weak"],
    )
    def test_unreliable_preferred_identity_is_not_a_conflict(self, reliability):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0, reliability=reliability),
                    lib("libcudart", "12", order=1),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_incidental_candidates_never_imply_conflict(self):
        # Toolkit (incidental and active), conda, ldconfig, package copies
        # with disagreeing families but no modeled precedence.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.FALSE),
                    lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE),
                    lib("libcudart", "9", origin=ORIGIN_CONDA_PREFIX),
                    lib("libcudart", "10", origin=ORIGIN_LDCONFIG),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_conflict_without_any_expected_copy_is_not_medium(self):
        # §19.8 requires the expected copy within the modeled ordering; a
        # lone conflicting LDP entry proves no winner.
        facts = facts_with(
            runtime_facts(candidates=(lib("libcudart", "11", order=0),))
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
                    lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_active_toolkit_false_ignored_as_incidental(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.FALSE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_active_toolkit_unknown_stays_conservative(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.UNKNOWN),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_active_and_incidental_coexist_without_collapsing_roles(self):
        # Adding active/incidental toolkit copies to a real modeled conflict
        # changes neither the decision nor the bounded evidence.
        standalone = RULE.evaluate(conflict_facts(), [])
        with_toolkits = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE),
                        lib("libcudart", "11", origin=ORIGIN_TOOLKIT, active=Trinary.FALSE),
                        lib("libcudart", "11", order=0),
                        lib("libcudart", "12", order=1),
                    )
                )
            ),
            [],
        )
        assert with_toolkits == standalone

    def test_active_toolkit_expected_with_ldp_conflict_stays_unknown(self):
        # Cross-origin winner (active toolkit vs LDP) needs unmodeled loader
        # semantics — §19.11 applies, no MEDIUM D004.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, active=Trinary.TRUE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestLdpOrdering:
    def test_earlier_conflicting_entry_wins_within_the_model(self):
        assert RULE.evaluate(conflict_facts(), []) is not None

    def test_reversed_ordering_emits_nothing(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "12", order=0),
                    lib("libcudart", "11", order=1),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_missing_search_order_invents_no_precedence(self):
        # Plan §8: an LDP candidate whose entry order was never established
        # is unordered — it is neither the preferred candidate nor the
        # expected-later one.
        unordered_conflict = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=None),
                    lib("libcudart", "12", order=1),
                )
            )
        )
        unordered_expected = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=None),
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", origin=ORIGIN_TOOLKIT, order=5),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_unknown_earliest_entry_blocks_even_with_later_conflict(self):
        # §19.5: the preferred entry's identity is unknown, so no later
        # entry can be promoted to "preferred among modeled candidates".
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", None, order=0),
                    lib("libcudart", "11", order=1),
                    lib("libcudart", "12", order=2),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_two_distinct_candidates_at_the_earliest_entry_are_unknown(self):
        # §19.5: distinguishing two files within one directory needs
        # per-library loader semantics the model does not cover.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=0),
                    lib("libcudart", "12", order=1),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_duplicate_candidates_neither_block_nor_duplicate(self):
        # Plan §22 K: exact duplicates collapse; the emission and its
        # evidence equal the non-duplicated scenario.
        baseline = RULE.evaluate(conflict_facts(), [])
        duplicated = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "11", order=0),
                        lib("libcudart", "11", order=0),
                        lib("libcudart", "12", order=1),
                        lib("libcudart", "12", order=1),
                    )
                )
            ),
            [],
        )
        assert duplicated is not None
        assert duplicated == baseline


class TestPythonPackageInteraction:
    def test_package_local_matching_candidate_alone_emits_nothing(self):
        facts = facts_with(
            runtime_facts(
                candidates=(lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),)
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_19_11_package_local_expected_vs_ldp_conflict_is_unknown(self):
        # Truth tables §19.11 / plan §24: the only way to pick the winner
        # is ELF RPATH/RUNPATH knowledge — resolution UNKNOWN, no MEDIUM.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_winner_decided_inside_the_model_emits(self):
        # The same environment with the expected copy also modeled later:
        # no host binary inspection is needed or used.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is not None

    def test_package_local_conflict_alone_is_not_a_winner_claim(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcudart", "12", order=0),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestConda:
    def test_conda_candidate_existence_is_not_a_global_winner(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_CONDA_PREFIX),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
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
                        lib("libcudart", "11", origin=ORIGIN_CONDA_PREFIX),
                        lib("libcudart", "11", order=0),
                        lib("libcudart", "12", order=1),
                    )
                )
            ),
            [],
        )
        assert with_conda == standalone

    def test_conda_expected_with_ldp_conflict_stays_unknown(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", origin=ORIGIN_CONDA_PREFIX),
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
                    lib("libcudart", "11", origin=ORIGIN_LDCONFIG),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
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
                        lib("libcudart", "9", origin=ORIGIN_LDCONFIG),
                        lib("libcudart", "11", order=0),
                        lib("libcudart", "12", order=1),
                    )
                )
            ),
            [],
        )
        assert with_ldconfig == standalone

    def test_ldconfig_never_receives_a_modeled_order(self):
        # A stray search_order on an ldconfig candidate is not precedence.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", origin=ORIGIN_LDCONFIG, order=0),
                    lib("libcudart", "12", order=1),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_ldconfig_only_old_library_with_complete_scan_emits_nothing(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "10", origin=ORIGIN_LDCONFIG),
                    lib("libcudnn", "7", origin=ORIGIN_LDCONFIG),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestLibcublasMapping:
    def _mapped(self, package_family: str) -> RuntimeLibraryFacts:
        return runtime_facts(
            candidates=(
                lib("libcublas", "11", order=0),
                lib("libcublas", "12", order=1),
                lib("libcublas", package_family, origin=ORIGIN_PYTHON_PACKAGE),
            )
        )

    def test_reliable_mapping_conflicting_family_is_medium(self):
        # §19.9: the mapping is reliable (the environment's own torch-family
        # 12 build with its package-local libcublas family 12) and the
        # modeled ordering prefers the incompatible family 11.
        diagnosis = RULE.evaluate(facts_with(self._mapped("12")), [])
        assert diagnosis is not None
        assert diagnosis.diagnosis.confidence is Confidence.MEDIUM

    def test_matching_mapped_family_emits_nothing(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "12", order=0),
                    lib("libcublas", "12", order=1),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_unmappable_family_is_unknown_even_with_clear_disagreement(self):
        # No package-local pair -> no reliable CUDA-to-SONAME mapping ->
        # compatibility UNKNOWN, no D004 conflict claim (§19.9).
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "11", order=0),
                    lib("libcublas", "12", order=1),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_multiple_disagreeing_package_local_families_are_unknown(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "11", order=0),
                    lib("libcublas", "12", order=1),
                    lib("libcublas", "11", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_unknown_torch_family_leaves_libcublas_unmapped(self):
        facts = facts_with(
            replace(self._mapped("12"), torch_runtime_family=None)
        )
        assert RULE.evaluate(facts, []) is None

    def test_unreliable_package_local_identity_leaves_mapping_unknown(self):
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcublas", "11", order=0),
                    lib("libcublas", "12", order=1),
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

    def test_mapping_follows_the_observed_pair_not_a_numeric_assumption(self):
        # Plan §11B: CUDA major == libcublas SONAME major is never assumed.
        # This environment observably pairs torch family 12 with a
        # package-local libcublas family 9 — that observed pair is the
        # baseline, so family 12 is the conflict and family 9 the expected.
        conflicting = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcublas", "12", order=0),
                        lib("libcublas", "9", order=1),
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
                        lib("libcublas", "9", order=0),
                        lib("libcublas", "12", order=1),
                        lib("libcublas", "9", origin=ORIGIN_PYTHON_PACKAGE),
                    )
                )
            ),
            [],
        )
        assert conflicting is not None
        assert expected_preferred is None


class TestLibcudnn:
    def test_single_package_local_baseline_conflict_is_medium(self):
        # §19.10: expected environment-local cuDNN family 9; the modeled
        # ordering prefers incompatible family 8 with the expected family 9
        # later in the same ordering.
        diagnosis = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudnn", "8", order=0),
                        lib("libcudnn", "9", order=1),
                    ),
                    cudnn_family="9",
                )
            ),
            [],
        )
        assert diagnosis is not None
        assert diagnosis.diagnosis.confidence is Confidence.MEDIUM

    def test_matching_candidates_emit_nothing(self):
        # D004-N03: multiple cuDNN copies, all consistent with baseline.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "9", order=0),
                    lib("libcudnn", "9", order=1),
                ),
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_zero_package_local_baselines_is_unknown(self):
        # §19.10 negative: two cuDNN copies exist but the expected baseline
        # cannot be derived — no conflict diagnosis, compatibility UNKNOWN.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "8", order=0),
                    lib("libcudnn", "9", order=1),
                ),
                cudnn_family=None,
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_multiple_package_local_baselines_are_unknown_synthetic(self):
        # The builder collapses this to package_local_cudnn_family=None
        # (pinned by the Phase-5 suite); the rule sees UNKNOWN baseline.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "8", order=0),
                    lib("libcudnn", "9", order=1),
                ),
                cudnn_family=None,
            )
        )
        assert RULE.evaluate(facts, []) is None

    def test_multiple_package_local_baselines_are_unknown_builder(self):
        snapshot = snapshot_with_inventory(
            raw("libcudnn", "8", origin=ORIGIN_PYTHON_PACKAGE),
            raw("libcudnn", "9", origin=ORIGIN_PYTHON_PACKAGE),
            raw("libcudnn", "8", origin=ORIGIN_LD_LIBRARY_PATH, order=0),
            raw("libcudnn", "9", origin=ORIGIN_LD_LIBRARY_PATH, order=1),
        )
        facts = built_facts(snapshot)
        assert facts.runtime_libraries.package_local_cudnn_family is None
        assert RULE.evaluate(facts, []) is None

    def test_torch_cuda_family_is_never_a_cudnn_baseline(self):
        # Semantics §23.8 / plan §26: the conflicting cuDNN family "12"
        # equals the torch runtime family here — if the rule wrongly
        # compared cuDNN against torch.version.cuda it would see a match
        # and stay silent instead of diagnosing the family-8-vs-9 conflict.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudnn", "12", order=0),
                    lib("libcudnn", "9", order=1),
                ),
                cudnn_family="9",
            )
        )
        assert RULE.evaluate(facts, []) is not None

    @pytest.mark.parametrize("torch_family", [None, "11", "13"])
    def test_cudnn_decision_independent_of_torch_runtime_family(
        self, torch_family
    ):
        # Plan §26 invariant: holding the cuDNN baseline and candidates
        # fixed, changing torch_runtime_family changes nothing.
        common = {
            "runtime": runtime_facts(
                candidates=(
                    lib("libcudnn", "8", order=0),
                    lib("libcudnn", "9", order=1),
                ),
                torch_family=torch_family,
                cudnn_family="9",
            )
        }
        baseline = RULE.evaluate(facts_with(**common), [])  # type: ignore[arg-type]
        assert baseline is not None
        for other in (None, "11", "13"):
            variant = RULE.evaluate(
                facts_with(
                    runtime_facts(
                        candidates=(
                            lib("libcudnn", "8", order=0),
                            lib("libcudnn", "9", order=1),
                        ),
                        torch_family=other,
                        cudnn_family="9",
                    )
                ),
                [],
            )
            assert variant == baseline


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
        # An already-modeled conflict is established inside the observed
        # ordering; unscanned roots cannot change LDP-internal precedence,
        # so the frozen MEDIUM stands (no invented confidence cap).
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
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
                candidates=(lib("libcudart", "12", order=0),),
                scan_incomplete=True,
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestAggregation:
    def two_family_facts(self) -> DiagnosisFacts:
        return facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
                    lib("libcudnn", "8", order=0),
                    lib("libcudnn", "9", order=1),
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
                    lib("libcublas", "11", order=0),
                    lib("libcublas", "12", order=1),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcudnn", "8", order=0),
                    lib("libcudnn", "9", order=1),
                ),
                cudnn_family="9",
            )
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        families = {
            item.key.rsplit("_", 1)[0]
            for item in candidate.diagnosis.evidence
            if item.key.endswith("_conflicting")
        }
        assert families == {"libcudart", "libcublas", "libcudnn"}

    def test_duplicate_evidence_never_accumulates(self):
        baseline = RULE.evaluate(conflict_facts(), [])
        duplicated = RULE.evaluate(
            facts_with(
                runtime_facts(
                    candidates=(
                        lib("libcudart", "11", order=0),
                        lib("libcudart", "12", order=1),
                        lib("libcudart", "12", order=1),
                    )
                )
            ),
            [],
        )
        assert duplicated == baseline
        assert len(duplicated.diagnosis.evidence) == len(baseline.diagnosis.evidence)


class TestUnknownConservative:
    def test_known_versus_unknown_is_never_mismatch(self):
        # §13: an unknown-identity later candidate cannot serve as the
        # expected copy, and an unknown preferred candidate blocks emission.
        unknown_expected = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1, reliability=IdentityReliability.UNKNOWN),
                )
            )
        )
        assert RULE.evaluate(unknown_expected, []) is None

    @pytest.mark.parametrize(
        ("family", "cudnn"),
        [("libcudart", None), ("libcublas", None), ("libcudnn", None)],
        ids=["cudart-baseline", "cublas-baseline", "cudnn-baseline"],
    )
    def test_missing_baseline_is_not_a_conflict(self, family, cudnn):
        candidates = (
            lib(family, "11", order=0),
            lib(family, "12", order=1),
        )
        facts = facts_with(
            runtime_facts(candidates=candidates, torch_family=None, cudnn_family=cudnn)
        )
        assert RULE.evaluate(facts, []) is None

    def test_partial_facts_degrade_to_silence(self):
        # Baseline present but nothing modeled at all.
        facts = facts_with(
            runtime_facts(
                candidates=(lib("libcudart", "11", origin=ORIGIN_LDCONFIG),),
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
                )
            ),
            runtime_facts(
                candidates=(
                    lib("libcublas", "11", order=0),
                    lib("libcublas", "12", order=1),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            ),
            runtime_facts(
                candidates=(
                    lib("libcudnn", "8", order=0),
                    lib("libcudnn", "9", order=1),
                ),
                cudnn_family="9",
            ),
        ],
        ids=["libcudart", "libcublas", "libcudnn"],
    )
    def test_static_inventory_alone_never_reaches_high(self, runtime):
        # Plan §23: pin the contract directly — the strongest possible
        # static evidence (all-direct identities, clear major conflict,
        # expected copy modeled later) still yields MEDIUM, never HIGH.
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
                    lib("libcudart", "11", origin=ORIGIN_LDCONFIG),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcudart", "11", origin=ORIGIN_CONDA_PREFIX),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None


class TestRpathNonInference:
    def test_rpath_dependent_scenario_stays_unknown(self):
        # Plan §24: the only way to choose between the LDP conflicting copy
        # and the package-local expected copy is ELF RPATH/RUNPATH
        # knowledge — Phase 8 must not upgrade it into a confident
        # diagnosis.
        facts = facts_with(
            runtime_facts(
                candidates=(
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", origin=ORIGIN_PYTHON_PACKAGE),
                    lib("libcublas", "11", order=0),
                    lib("libcublas", "12", origin=ORIGIN_PYTHON_PACKAGE),
                )
            )
        )
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize("module", [rules_base, d004_module], ids=["base", "d004"])
    def test_imports_no_host_probes_or_collector_surfaces(self, module):
        # Structural purity (plan §24/§27): resolving loader ambiguity with
        # host inspection would require importing one of these.
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
        # The rule must not even reference the pre-normalization surface or
        # any forbidden loader-inspection tooling by name.
        source = Path(d004_module.__file__).read_text()
        for forbidden in (
            "EnvironmentSnapshot",
            "RuntimeLibraryInventory",
            "ldconfig -p",
            "readelf",
            "objdump",
            "LD_DEBUG",
            "/proc/",
            "dlopen",
            "ctypes",
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
                    lib("libcudart", "12", order=1),
                    lib("libcudart", "11", order=0),
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
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
                    lib("libcudart", "11", order=0),
                    lib("libcudart", "12", order=1),
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
        # conflict, whose baseline never depends on torch — even when
        # PyTorch cannot be imported at all.
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
                raw("libcudnn", "8", origin=ORIGIN_LD_LIBRARY_PATH, order=0),
                raw("libcudnn", "9", origin=ORIGIN_LD_LIBRARY_PATH, order=1),
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

    def test_summary_uses_modeled_order_wording_only(self):
        # Semantics §23.7: MEDIUM conclusions are phrased in terms of the
        # modeled candidates — never as proof of which object will load.
        summary = conflict_diagnosis().summary
        assert "modeled LD_LIBRARY_PATH search order" in summary
        assert "not observed" in summary
        for forbidden in ("will load", "is loaded", "definitely loads"):
            assert forbidden not in summary, forbidden

    def test_evidence_bounded_and_ordered(self):
        assert conflict_diagnosis().evidence == [
            EvidenceItem(
                key="libcudart_conflicting",
                value="family 11 at LD_LIBRARY_PATH entry 0 (direct)",
                source="runtime_libraries",
            ),
            EvidenceItem(key="libcudart_expected", value="family 12", source="torch"),
            EvidenceItem(
                key="static_relation",
                value="conflicting candidate preferred within the modeled search order",
                source="compatibility",
            ),
        ]

    def test_evidence_contains_no_path_like_content(self):
        for item in conflict_diagnosis().evidence:
            assert "/" not in item.value
            assert "$" not in item.value

    def test_frozen_impact(self):
        assert conflict_diagnosis().impact == [
            ImpactItem(
                scope="CUDA runtime library resolution",
                status="conflicting family preferred within the modeled search order",
            ),
            ImpactItem(
                scope="PyTorch CUDA initialization",
                status="at risk if the modeled resolution applies",
            ),
            ImpactItem(
                scope="actually loaded library",
                status="not observed (static analysis only)",
            ),
        ]

    def test_remediation_safe_guidance_without_commands(self):
        # §19.15: remove/reorder the conflicting entry, start a clean
        # shell, re-run cuda-doctor — all SAFE, nothing executed.
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
        # §19.16 minimum, display-only; Linux-only rule so no platform
        # keying and no loaded-object inspection.
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
        # the D003 causal chain is Phase 9/10 wiring (plan §16).
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


class TestEvidencePrivacy:
    def test_secret_inventory_path_never_reaches_the_candidate(self):
        # Plan §18: a secret-looking raw candidate path in the Phase-3
        # inventory must never appear in the D004 candidate. The fact
        # layer carries no paths at all, so it is structurally impossible.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "11",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                path="/home/secretuser/legacy/libcudart.so.11",
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                path="/home/secretuser/expected/libcudart.so.12",
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
    def test_19_8_example_through_the_fact_builder(self):
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "11",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
                path="/legacy/cuda-11/lib64/libcudart.so.11",
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
                path="/expected/cuda-12/lib64/libcudart.so.12",
            ),
        )
        facts = built_facts(snapshot)
        assert facts.runtime_libraries.torch_runtime_family == "12"
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.MEDIUM

    def test_symlink_derived_identity_is_still_reliable(self):
        # R1 shape migration: a bare observed name carries no family claim
        # (``load_name_family_version`` None); the identity comes from the
        # reliably resolved canonical target (DERIVED_STRONG), which the
        # transitional rule still consumes through its filename-first /
        # target-second compatibility mapping.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "11",
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
            ),
        )
        facts = built_facts(snapshot)
        preferred = facts.runtime_libraries.candidates[0]
        assert preferred.load_name == "libcudart.so"
        assert preferred.load_name_family_version is None
        assert preferred.target_family_version == "11"
        assert preferred.reliability is IdentityReliability.DERIVED_STRONG
        assert RULE.evaluate(facts, []) is not None

    def test_scan_errors_mark_incomplete_without_blocking_the_conflict(self):
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "11",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
            ),
            scan_errors={"/root/unreadable": "permission denied"},
        )
        facts = built_facts(snapshot)
        assert facts.runtime_libraries.scan_incomplete is True
        assert RULE.evaluate(facts, []) is not None


class TestNoEngineIntegration:
    def test_engine_still_emits_no_diagnoses(self):
        # Plan §28: the rule fires on the built facts, but the engine
        # neither runs it nor populates DiagnosisResult.diagnoses.
        snapshot = snapshot_with_inventory(
            raw(
                "libcudart",
                "11",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=0,
            ),
            raw(
                "libcudart",
                "12",
                origin=ORIGIN_LD_LIBRARY_PATH,
                order=1,
            ),
        )
        result = DiagnosisEngine().run(snapshot)
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}
        assert RULE.evaluate(built_facts(snapshot), []) is not None
