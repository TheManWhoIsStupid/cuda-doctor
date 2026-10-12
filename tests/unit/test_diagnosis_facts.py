"""Tests for the v0.2 normalized diagnosis fact layer.

Synthetic snapshots only: the fact builder must be a pure normalizer of
already-collected data. Truth-table row IDs are referenced in comments
where a case pins a frozen row (D001-N14, D002-N08, D003-N08, ...).
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path

import pytest
from tests.factories import base_snapshot

from cuda_doctor.collectors.cuda import (
    VERSION_SOURCE_DERIVED_STRONG,
    VERSION_SOURCE_DERIVED_WEAK,
    VERSION_SOURCE_DIRECT,
)
from cuda_doctor.collectors.runtime_libraries import (
    ORIGIN_LD_LIBRARY_PATH,
    ORIGIN_LDCONFIG,
    ORIGIN_PYTHON_PACKAGE,
    ORIGIN_TOOLKIT,
    VERSION_SOURCE_FILENAME,
    VERSION_SOURCE_SYMLINK_TARGET,
)
from cuda_doctor.compatibility import interpret_torch_cuda, load_compatibility_data
from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.core.models import (
    CUDASelectorObservation,
    DriverInfo,
    EnvironmentSnapshot,
    NvidiaSmiInfo,
    RuntimeLibraryCandidate,
    RuntimeLibraryInventory,
)
from cuda_doctor.diagnosis import facts as facts_module
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.facts import (
    CUDAVisibleDevicesState,
    DiagnosisFactBuilder,
    DiagnosisFacts,
    IdentityReliability,
    ModeledLibraryCandidate,
    RuntimeLibraryFacts,
    ToolkitCompatibility,
    ToolkitSelectorFact,
    Trinary,
    normalize_torch_toolkit_relation,
)
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.utils.versions import CudaVersion, parse_cuda_version

COMPATIBILITY = load_compatibility_data()

SELECTOR_FIELD_NAMES = {
    "name",
    "explicit",
    "effective",
    "active",
    "valid",
    "canonical_root",
    "toolkit_version",
    "reliability",
}

FACTS_FIELD_NAMES = {
    "platform",
    "toolkit_selectors",
    "torch",
    "gpu",
    "driver",
    "visibility",
    "runtime_libraries",
    "issue_codes",
    "reference_toolkit",
    "applicability",
}


def build_facts(
    snapshot: EnvironmentSnapshot, issues: list[Issue] | None = None
) -> DiagnosisFacts:
    return DiagnosisFactBuilder().build(snapshot, issues or [], COMPATIBILITY)


def snapshot_with_selectors(
    *observations: CUDASelectorObservation, platform: Platform | None = None
) -> EnvironmentSnapshot:
    snapshot = base_snapshot()
    snapshot.cuda = replace(snapshot.cuda, selector_observations=list(observations))
    if platform is not None:
        snapshot.system = replace(snapshot.system, platform=platform)
    return snapshot


def selector_fact(facts: DiagnosisFacts, name: str) -> ToolkitSelectorFact:
    matches = [fact for fact in facts.toolkit_selectors if fact.name == name]
    assert len(matches) == 1, f"expected exactly one {name} selector fact"
    return matches[0]


def observation(name: str, **overrides) -> CUDASelectorObservation:
    params: dict = {"name": name}
    params.update(overrides)
    return CUDASelectorObservation(**params)


def inventory(*candidates: RuntimeLibraryCandidate, **overrides) -> RuntimeLibraryInventory:
    return RuntimeLibraryInventory(candidates=list(candidates), **overrides)


def library(
    family: str, path: str, *, version: str | None, source: str | None = None, **overrides
) -> RuntimeLibraryCandidate:
    params: dict = {
        "family": family,
        "path": path,
        "version": version,
        "version_source": source,
    }
    params.update(overrides)
    return RuntimeLibraryCandidate(**params)


class TestTrinary:
    def test_members_and_values(self):
        assert [(member.name, member.value) for member in Trinary] == [
            ("TRUE", "true"),
            ("FALSE", "false"),
            ("UNKNOWN", "unknown"),
        ]

    def test_unknown_is_a_real_state(self):
        # UNKNOWN must never collapse onto FALSE (or onto None-as-false):
        # it is a third value with its own identity.
        assert Trinary.UNKNOWN is not Trinary.FALSE
        assert Trinary.UNKNOWN != Trinary.FALSE
        assert Trinary.UNKNOWN.value == "unknown"

    def test_dedicated_representation(self):
        # A dedicated enum, not bool | None: the tri-state predicates use
        # it directly (architecture §12.5).
        facts = build_facts(base_snapshot())
        assert facts.gpu.visible is Trinary.TRUE
        assert type(facts.gpu.visible) is Trinary


class TestIdentityReliability:
    def test_members_and_values(self):
        assert [(member.name, member.value) for member in IdentityReliability] == [
            ("DIRECT", "direct"),
            ("DERIVED_STRONG", "derived_strong"),
            ("DERIVED_WEAK", "derived_weak"),
            ("UNKNOWN", "unknown"),
        ]

    def test_no_numeric_scores(self):
        assert all(isinstance(member.value, str) for member in IdentityReliability)

    def test_maps_collector_version_sources_verbatim(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-a",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/opt/cuda-b",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_STRONG,
            ),
            observation(
                "usr_local_cuda",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
        )
        facts = build_facts(snapshot)
        assert selector_fact(facts, "CUDA_HOME").reliability is IdentityReliability.DIRECT
        assert selector_fact(facts, "NVCC").reliability is IdentityReliability.DERIVED_STRONG
        assert (
            selector_fact(facts, "USR_LOCAL_CUDA").reliability
            is IdentityReliability.DERIVED_WEAK
        )

    def test_missing_version_is_unknown_even_with_source(self):
        # Defensive: without a version there is no identity, whatever the
        # collector's version_source string claims.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda",
                toolkit_version=None,
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        assert selector_fact(build_facts(snapshot), "CUDA_HOME").reliability is (
            IdentityReliability.UNKNOWN
        )

    @pytest.mark.parametrize("source", [None, "", "guess"])
    def test_unrecognized_source_is_unknown(self, source):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda",
                toolkit_version="12.4",
                valid=True,
                version_source=source,
            )
        )
        assert (
            selector_fact(build_facts(snapshot), "CUDA_HOME").reliability
            is IdentityReliability.UNKNOWN
        )


class TestSelectorNormalization:
    def test_field_set_exactly_architecture_shape(self):
        assert {field.name for field in fields(ToolkitSelectorFact)} == SELECTOR_FIELD_NAMES

    def test_names_normalized_to_truth_table_vocabulary(self):
        snapshot = snapshot_with_selectors(
            observation("cuda_home", canonical_root="/a", valid=True),
            observation("cuda_path", canonical_root="/b", valid=True),
            observation("cudacxx", canonical_path="/c/nvcc", valid=True),
            observation("nvcc", canonical_root="/d", valid=True),
            observation("path_cuda_bin", canonical_root="/e", valid=True),
            observation("usr_local_cuda", canonical_root="/f"),
        )
        facts = build_facts(snapshot)
        assert [fact.name for fact in facts.toolkit_selectors] == [
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDACXX",
            "NVCC",
            "PATH_CUDA_BIN",
            "USR_LOCAL_CUDA",
        ]

    def test_selector_classes(self):
        snapshot = snapshot_with_selectors(
            observation("cuda_home", canonical_root="/a", valid=True),
            observation("cuda_path", canonical_root="/b", valid=True),
            observation("cudacxx", canonical_path="/c/nvcc", valid=True),
            observation("nvcc", canonical_root="/d", valid=True),
            observation("path_cuda_bin", canonical_root="/e", valid=True),
            observation("usr_local_cuda", canonical_root="/f"),
        )
        facts = build_facts(snapshot)
        by_name = {fact.name: fact for fact in facts.toolkit_selectors}
        assert by_name["CUDA_HOME"].explicit and not by_name["CUDA_HOME"].effective
        assert by_name["CUDA_PATH"].explicit and not by_name["CUDA_PATH"].effective
        assert by_name["CUDACXX"].explicit and not by_name["CUDACXX"].effective
        assert by_name["NVCC"].effective and not by_name["NVCC"].explicit
        assert by_name["PATH_CUDA_BIN"].effective and not by_name["PATH_CUDA_BIN"].explicit
        assert not by_name["USR_LOCAL_CUDA"].explicit
        assert not by_name["USR_LOCAL_CUDA"].effective

    def test_validity_passed_through_tri_state(self):
        snapshot = snapshot_with_selectors(
            observation("cuda_home", valid=None, canonical_root="/a"),
            observation("nvcc", valid=False, canonical_root="/b"),
        )
        facts = build_facts(snapshot)
        assert selector_fact(facts, "CUDA_HOME").valid is None
        assert selector_fact(facts, "NVCC").valid is False

    def test_version_normalized_to_major_minor(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/a",
                toolkit_version="12.4.131",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        assert selector_fact(build_facts(snapshot), "CUDA_HOME").toolkit_version == "12.4"

    def test_unparseable_version_becomes_none(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/a",
                toolkit_version="not-a-version",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        fact = selector_fact(build_facts(snapshot), "CUDA_HOME")
        assert fact.toolkit_version is None
        assert fact.reliability is IdentityReliability.UNKNOWN

    def test_unknown_collector_name_stays_incidental(self):
        snapshot = snapshot_with_selectors(
            observation("custom_widget", canonical_root="/x", valid=True)
        )
        fact = selector_fact(build_facts(snapshot), "custom_widget")
        assert fact.explicit is False
        assert fact.effective is False
        assert fact.active is False


class TestSelectorActivity:
    def test_observed_explicit_selectors_active_on_linux(self):
        snapshot = snapshot_with_selectors(
            observation("cuda_home", canonical_root="/a", valid=True),
            observation("cudacxx", canonical_path="/c/nvcc", valid=True),
            observation("nvcc", canonical_root="/d", valid=True),
            observation("path_cuda_bin", canonical_root="/e", valid=True),
        )
        facts = build_facts(snapshot)
        for name in ("CUDA_HOME", "CUDACXX", "NVCC", "PATH_CUDA_BIN"):
            assert selector_fact(facts, name).active is True, name

    def test_linux_cuda_path_observed_but_never_active(self):
        # Truth tables §16.11: on Linux CUDA_PATH is inventory information
        # only — the active-selector contract never derives activity from
        # a merely observed value.
        snapshot = snapshot_with_selectors(
            observation("cuda_path", canonical_root="/opt/cuda-12.6", valid=True)
        )
        fact = selector_fact(build_facts(snapshot), "CUDA_PATH")
        assert fact.active is False

    def test_windows_cuda_path_is_active(self):
        snapshot = snapshot_with_selectors(
            observation("cuda_path", canonical_root="/opt/cuda-12.6", valid=True),
            platform=Platform.WINDOWS,
        )
        assert selector_fact(build_facts(snapshot), "CUDA_PATH").active is True


class TestUsrLocalCudaActivity:
    def test_present_but_untied_is_inactive(self):
        # D001-N14: /usr/local/cuda existing (even as a usable toolkit)
        # never makes it an active selector by itself.
        snapshot = snapshot_with_selectors(
            observation(
                "usr_local_cuda",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            )
        )
        assert selector_fact(build_facts(snapshot), "USR_LOCAL_CUDA").active is False

    def test_dangling_symlink_target_untied_is_inactive(self):
        # Stale/dangling symlink: collector records the target naming, but
        # nothing resolved through it — still not an active selector.
        snapshot = snapshot_with_selectors(
            observation(
                "usr_local_cuda",
                canonical_root="/usr/local/cuda-13.0",
                toolkit_version="13.0",
                exists=False,
                valid=None,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            )
        )
        assert selector_fact(build_facts(snapshot), "USR_LOCAL_CUDA").active is False

    def test_tied_through_active_cuda_home(self):
        # D001-P07: CUDA_HOME=/usr/local/cuda — the symlink resolution ties
        # the incidental observation to the active selector.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "usr_local_cuda",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
        )
        assert selector_fact(build_facts(snapshot), "USR_LOCAL_CUDA").active is True

    def test_tied_through_resolved_nvcc(self):
        # D001-P08: nvcc resolved via /usr/local/cuda/bin/nvcc — the
        # canonical root of the winning resolution ties the symlink.
        snapshot = snapshot_with_selectors(
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "usr_local_cuda",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
        )
        assert selector_fact(build_facts(snapshot), "USR_LOCAL_CUDA").active is True

    def test_equivalent_canonical_paths_count_as_same_target(self):
        # Different raw spellings, same canonical root: the tie is on
        # canonical identity, not on string equality of raw values.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/usr/local/cuda/",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "usr_local_cuda",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
        )
        assert selector_fact(build_facts(snapshot), "USR_LOCAL_CUDA").active is True

    def test_inactive_selector_does_not_tie(self):
        # A Linux CUDA_PATH pointing at /usr/local/cuda is not active, so
        # it must not activate the incidental observation either.
        snapshot = snapshot_with_selectors(
            observation("cuda_path", canonical_root="/usr/local/cuda", valid=True),
            observation("usr_local_cuda", canonical_root="/usr/local/cuda"),
        )
        assert selector_fact(build_facts(snapshot), "USR_LOCAL_CUDA").active is False

    def test_no_root_identity_means_no_tie(self):
        snapshot = snapshot_with_selectors(
            observation("cuda_home", canonical_root=None, valid=True),
            observation("usr_local_cuda", canonical_root=None),
        )
        assert selector_fact(build_facts(snapshot), "USR_LOCAL_CUDA").active is False


class TestToolkitIdentitySemantics:
    def test_same_reliable_version_at_different_roots(self):
        # Semantics §6: the same reliable version at two different roots is
        # not a disagreement. The facts carry both identities; versions
        # agree at the normalized level.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = build_facts(snapshot)
        roots = {fact.canonical_root for fact in facts.toolkit_selectors}
        versions = {fact.toolkit_version for fact in facts.toolkit_selectors}
        assert roots == {"/opt/cuda-12.4", "/usr/local/cuda-12.4"}
        assert versions == {"12.4"}  # path inequality alone implies nothing

    def test_path_inequality_keeps_both_identities(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "cuda_path",
                canonical_root="/opt/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            platform=Platform.WINDOWS,
        )
        facts = build_facts(snapshot)
        assert len(facts.toolkit_selectors) == 2
        assert {fact.toolkit_version for fact in facts.toolkit_selectors} == {"12.4", "12.6"}

    def test_equal_canonical_root_single_identity(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = build_facts(snapshot)
        assert {fact.canonical_root for fact in facts.toolkit_selectors} == {"/opt/cuda-12.4"}

    def test_invalid_selector_is_not_an_identity(self):
        # An unresolvable selector carries no usable identity even though
        # it is observed and (on Windows) nominally explicit.
        snapshot = snapshot_with_selectors(
            observation(
                "cudacxx",
                raw_value="/no/such/nvcc",
                exists=False,
                valid=False,
                version_source=None,
            )
        )
        fact = selector_fact(build_facts(snapshot), "CUDACXX")
        assert fact.valid is False
        assert fact.toolkit_version is None
        assert fact.canonical_root is None
        assert fact.reliability is IdentityReliability.UNKNOWN

    def test_nvidia_smi_cuda_version_is_never_a_selector(self):
        # The driver-reported "CUDA Version" (13.0 here) describes the
        # driver's UMD generation; it must never appear as a toolkit
        # selector identity or a reference toolkit.
        snapshot = base_snapshot()  # driver reports CUDA 13.0, no selectors
        facts = build_facts(snapshot)
        assert facts.toolkit_selectors == ()
        assert facts.reference_toolkit is None
        versions = [fact.toolkit_version for fact in facts.toolkit_selectors]
        assert "13.0" not in versions
        assert facts.driver.version == "580.126.09"


class TestWindowsSelectorFolding:
    def test_folded_observation_is_one_logical_identity(self):
        # Windows collector folds CUDA_HOME-or-CUDA_PATH into cuda_home; a
        # folded pair sharing root and version must not become two
        # conflicting identities.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
                canonical_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "cuda_path",
                raw_value="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
                canonical_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            platform=Platform.WINDOWS,
        )
        facts = build_facts(snapshot)
        home = selector_fact(facts, "CUDA_HOME")
        path = selector_fact(facts, "CUDA_PATH")
        assert home.active and path.active
        assert home.canonical_root == path.canonical_root
        assert home.toolkit_version == path.toolkit_version
        assert home.reliability == path.reliability

    def test_distinct_values_remain_distinguishable(self):
        # The fold must not erase genuinely distinct CUDA_HOME/CUDA_PATH
        # values: explicit distinct configurations stay two identities.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="C:/toolkits/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "cuda_path",
                canonical_root="C:/toolkits/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            platform=Platform.WINDOWS,
        )
        facts = build_facts(snapshot)
        roots = {fact.canonical_root for fact in facts.toolkit_selectors}
        assert roots == {"C:/toolkits/cuda-12.4", "C:/toolkits/cuda-12.6"}


class TestReferenceToolkit:
    def test_none_without_usable_selectors(self):
        assert build_facts(base_snapshot()).reference_toolkit is None

    def test_metadata_only_home_is_not_a_reference(self):
        # D002-N08: a root whose version comes only from path naming
        # (derived_weak) is not a usable development toolkit.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            )
        )
        assert build_facts(snapshot).reference_toolkit is None

    def test_usable_cuda_home_is_reference(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.source == "CUDA_HOME"
        assert reference.canonical_root == "/opt/cuda-12.4"
        assert reference.toolkit_version == "12.4"
        assert reference.reliability is IdentityReliability.DIRECT

    def test_cudacxx_outranks_nvcc(self):
        # Truth tables §17.8: explicit compiler selection outranks the
        # PATH-resolved nvcc.
        snapshot = snapshot_with_selectors(
            observation(
                "cudacxx",
                canonical_path="/opt/cuda-12.4/bin/nvcc",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.source == "CUDACXX"
        assert reference.toolkit_version == "12.4"

    def test_nvcc_used_when_cudacxx_absent(self):
        snapshot = snapshot_with_selectors(
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.source == "NVCC"

    def test_invalid_cudacxx_falls_through_without_fake_reference(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cudacxx",
                raw_value="/no/such/nvcc",
                exists=False,
                valid=False,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.source == "NVCC"

    def test_valid_cudacxx_with_unknown_version_still_blocks_fallback(self):
        # A wrapper-style CUDACXX: valid executable, DIRECT version, but
        # no <root>/bin/nvcc layout — root stays None while the version
        # identity exists, and the explicit compiler still outranks NVCC.
        snapshot = snapshot_with_selectors(
            observation(
                "cudacxx",
                canonical_path="/usr/local/bin/nvcc-wrapper",
                canonical_root=None,
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.source == "CUDACXX"
        assert reference.canonical_root is None
        assert reference.toolkit_version == "12.4"
        assert reference.reliability is IdentityReliability.DIRECT

    def test_valid_but_underivable_cudacxx_carries_unknown_identity(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cudacxx",
                canonical_path="/opt/bin/clang-cuda",
                canonical_root=None,
                toolkit_version=None,
                valid=True,
                version_source=None,
            )
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.source == "CUDACXX"
        assert reference.toolkit_version is None
        assert reference.reliability is IdentityReliability.UNKNOWN

    def test_cuda_home_preferred_over_cuda_path(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="C:/toolkits/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_STRONG,
            ),
            observation(
                "cuda_path",
                canonical_root="C:/toolkits/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            platform=Platform.WINDOWS,
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.source == "CUDA_HOME"

    def test_derived_strong_home_counts_as_usable(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_STRONG,
            )
        )
        reference = build_facts(snapshot).reference_toolkit
        assert reference is not None
        assert reference.reliability is IdentityReliability.DERIVED_STRONG


class TestToolkitCompatibilityAdapter:
    def test_members_and_values(self):
        assert [(member.name, member.value) for member in ToolkitCompatibility] == [
            ("EXACT", "exact"),
            ("MINOR_COMPATIBLE_INFORMATIONAL", "minor_compatible_informational"),
            ("MAJOR_INCOMPATIBLE", "major_incompatible"),
            ("UNKNOWN", "unknown"),
        ]

    def test_no_mismatch_warning_state(self):
        assert not hasattr(ToolkitCompatibility, "MISMATCH_WARNING")
        assert len(list(ToolkitCompatibility)) == 4

    @pytest.mark.parametrize("torch_v,toolkit_v", [(None, "12.4"), ("12.4", None), (None, None)])
    def test_unknown_when_either_side_missing(self, torch_v, toolkit_v):
        torch_cuda = parse_cuda_version(torch_v) if torch_v else None
        toolkit = parse_cuda_version(toolkit_v) if toolkit_v else None
        assert normalize_torch_toolkit_relation(torch_cuda, toolkit) is (
            ToolkitCompatibility.UNKNOWN
        )

    def test_exact_match(self):
        relation = normalize_torch_toolkit_relation(CudaVersion(12, 4), CudaVersion(12, 4))
        assert relation is ToolkitCompatibility.EXACT

    @pytest.mark.parametrize(
        "torch_v,toolkit_v",
        [("12.6", "12.4"), ("12.4", "12.6"), ("12.4", "12.0")],
    )
    def test_same_major_is_minor_compatible_informational(self, torch_v, toolkit_v):
        relation = normalize_torch_toolkit_relation(
            parse_cuda_version(torch_v), parse_cuda_version(toolkit_v)
        )
        assert relation is ToolkitCompatibility.MINOR_COMPATIBLE_INFORMATIONAL

    @pytest.mark.parametrize(
        "torch_v,toolkit_v",
        [("11.8", "12.4"), ("12.4", "11.8"), ("13.0", "12.4")],
    )
    def test_different_major_is_major_incompatible(self, torch_v, toolkit_v):
        relation = normalize_torch_toolkit_relation(
            parse_cuda_version(torch_v), parse_cuda_version(toolkit_v)
        )
        assert relation is ToolkitCompatibility.MAJOR_INCOMPATIBLE

    @pytest.mark.parametrize(
        "torch_v,toolkit_v",
        [
            ("12.4", "12.4"),
            ("12.6", "12.4"),
            ("12.4", "12.6"),
            ("11.8", "12.4"),
            ("12.4", "11.8"),
            ("13.0", "12.4"),
        ],
    )
    def test_mirror_of_interpret_torch_cuda_branching(self, torch_v, toolkit_v):
        # The adapter must not fork the helper's semantics: the mapped
        # state is fully determined by the helper's structured relation
        # plus its own major-version distinction.
        torch_cuda = parse_cuda_version(torch_v)
        toolkit = parse_cuda_version(toolkit_v)
        interpretation = interpret_torch_cuda(torch_cuda, toolkit)
        if interpretation.relation == "match":
            expected = ToolkitCompatibility.EXACT
        elif interpretation.relation == "unknown":
            expected = ToolkitCompatibility.UNKNOWN
        elif torch_cuda and toolkit and torch_cuda.major == toolkit.major:
            expected = ToolkitCompatibility.MINOR_COMPATIBLE_INFORMATIONAL
        else:
            expected = ToolkitCompatibility.MAJOR_INCOMPATIBLE
        assert normalize_torch_toolkit_relation(torch_cuda, toolkit) is expected


class TestTorchFacts:
    def test_healthy_environment(self):
        facts = build_facts(base_snapshot())
        assert facts.torch.importable is True
        assert facts.torch.cuda_build is True
        assert facts.torch.cuda_version == "12.4"
        assert facts.torch.cuda_available is True
        assert facts.torch.cudnn_version == "90100"

    def test_import_failure_is_not_importable(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch, installed=True, import_error="RuntimeError: simulated"
        )
        assert build_facts(snapshot).torch.importable is False

    def test_not_installed_is_not_importable(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch,
            installed=False,
            version=None,
            cuda_version=None,
            is_cuda_build=None,
            cuda_available=None,
        )
        torch_facts = build_facts(snapshot).torch
        assert torch_facts.importable is False
        assert torch_facts.cuda_build is None

    def test_cpu_build_distinguishable_from_unavailable_cuda(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch,
            is_cuda_build=False,
            cuda_version=None,
            cuda_available=False,
            device_count=0,
            devices=[],
        )
        torch_facts = build_facts(snapshot).torch
        assert torch_facts.cuda_build is False  # CPU-only build

    def test_cuda_available_none_is_preserved(self):
        # The frozen snapshot contract: None means the probe produced no
        # reliable boolean. It must stay None — never become False.
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=None)
        assert build_facts(snapshot).torch.cuda_available is None

    def test_cuda_available_false_is_preserved(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False)
        assert build_facts(snapshot).torch.cuda_available is False


class TestGpuVisibility:
    def test_visible_with_gpus(self):
        assert build_facts(base_snapshot()).gpu.visible is Trinary.TRUE

    def test_not_visible_with_clean_empty_probe(self):
        snapshot = base_snapshot()
        snapshot.gpus = []
        assert build_facts(snapshot).gpu.visible is Trinary.FALSE

    def test_unknown_when_nvidia_smi_missing(self):
        snapshot = base_snapshot()
        snapshot.nvidia_smi = None
        assert build_facts(snapshot).gpu.visible is Trinary.UNKNOWN

    def test_unknown_when_nvidia_smi_unavailable(self):
        snapshot = base_snapshot()
        snapshot.nvidia_smi = NvidiaSmiInfo(available=False, executed=False)
        assert build_facts(snapshot).gpu.visible is Trinary.UNKNOWN

    def test_unknown_when_nvidia_smi_not_executed(self):
        snapshot = base_snapshot()
        snapshot.nvidia_smi = NvidiaSmiInfo(available=True, executed=False)
        assert build_facts(snapshot).gpu.visible is Trinary.UNKNOWN

    def test_unknown_when_probe_errored_even_with_no_gpus(self):
        # A failed probe must not fabricate a negative fact (§24).
        snapshot = base_snapshot()
        snapshot.gpus = []
        snapshot.nvidia_smi = NvidiaSmiInfo(
            available=True, executed=True, error="exit-9"
        )
        assert build_facts(snapshot).gpu.visible is Trinary.UNKNOWN

    def test_independent_of_torch_availability(self):
        # Non-circularity (§18.5): torch.cuda.is_available() == False with
        # devices empty while the independent nvidia-smi path sees a GPU —
        # visibility facts come from the independent path only.
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        assert build_facts(snapshot).gpu.visible is Trinary.TRUE


class TestDriverFacts:
    def test_usable_with_version(self):
        driver = build_facts(base_snapshot()).driver
        assert driver.version == "580.126.09"
        assert driver.usable is Trinary.TRUE

    def test_not_usable_when_probe_ran_and_failed(self):
        snapshot = base_snapshot()
        snapshot.driver = DriverInfo(version=None)
        snapshot.nvidia_smi = NvidiaSmiInfo(
            available=True, executed=True, error="executable-not-found"
        )
        assert build_facts(snapshot).driver.usable is Trinary.FALSE

    def test_unknown_usable_without_any_observation(self):
        snapshot = base_snapshot()
        snapshot.driver = None
        snapshot.nvidia_smi = None
        driver = build_facts(snapshot).driver
        assert driver.version is None
        assert driver.usable is Trinary.UNKNOWN

    def test_compatible_true_within_family_minimum(self):
        # torch 12.4 on Linux requires >= 525.60.13; 580.126.09 qualifies.
        assert build_facts(base_snapshot()).driver.compatible_with_torch_runtime is (
            Trinary.TRUE
        )

    def test_compatible_false_below_family_minimum(self):
        snapshot = base_snapshot()
        snapshot.driver = DriverInfo("450.51.06", "11.0")
        assert build_facts(snapshot).driver.compatible_with_torch_runtime is Trinary.FALSE

    def test_compatible_unknown_without_torch_cuda(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_version=None, is_cuda_build=False)
        assert build_facts(snapshot).driver.compatible_with_torch_runtime is (
            Trinary.UNKNOWN
        )

    def test_compatible_unknown_for_unmodeled_family(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_version="14.0")
        assert build_facts(snapshot).driver.compatible_with_torch_runtime is (
            Trinary.UNKNOWN
        )

    def test_compatible_unknown_without_driver_version(self):
        snapshot = base_snapshot()
        snapshot.driver = DriverInfo(version=None)
        snapshot.nvidia_smi = None
        assert build_facts(snapshot).driver.compatible_with_torch_runtime is (
            Trinary.UNKNOWN
        )

    def test_compatible_unknown_without_compatibility_data(self):
        snapshot = base_snapshot()
        facts = DiagnosisFactBuilder().build(snapshot, [], None)
        assert facts.driver.compatible_with_torch_runtime is Trinary.UNKNOWN

    def test_compatible_unknown_on_macos(self):
        # The driver family table covers linux/windows only; other
        # platforms are UNKNOWN, never guessed.
        snapshot = base_snapshot()
        snapshot.system = replace(snapshot.system, platform=Platform.MACOS)
        assert build_facts(snapshot).driver.compatible_with_torch_runtime is (
            Trinary.UNKNOWN
        )


class TestVisibilityFacts:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, CUDAVisibleDevicesState.UNSET),
            ("", CUDAVisibleDevicesState.MASK_ALL),
            ("-1", CUDAVisibleDevicesState.MASK_ALL),
            ("0", CUDAVisibleDevicesState.RESTRICTED_NONEMPTY),
            ("1", CUDAVisibleDevicesState.RESTRICTED_NONEMPTY),
            ("0,1", CUDAVisibleDevicesState.RESTRICTED_NONEMPTY),
            ("3", CUDAVisibleDevicesState.RESTRICTED_NONEMPTY),  # D003-N08
            ("10,11", CUDAVisibleDevicesState.RESTRICTED_NONEMPTY),
        ],
    )
    def test_frozen_cvd_table(self, raw, expected):
        snapshot = base_snapshot()
        variables = dict(snapshot.environment.variables)
        if raw is None:
            variables.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            variables["CUDA_VISIBLE_DEVICES"] = raw
        snapshot.environment = replace(snapshot.environment, variables=variables)
        assert build_facts(snapshot).visibility.cuda_visible_devices is expected

    @pytest.mark.parametrize(
        "raw",
        ["all", "none", "void", "0, 1", " 0", "0 ", "abc", "0,abc", "-1,0", "GPU-abc", "0-1"],
    )
    def test_outside_modeled_grammar_is_unparseable(self, raw):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            variables={**snapshot.environment.variables, "CUDA_VISIBLE_DEVICES": raw},
        )
        assert build_facts(snapshot).visibility.cuda_visible_devices is (
            CUDAVisibleDevicesState.UNPARSEABLE
        )

    @pytest.mark.parametrize(
        "raw",
        ["", "-1", "0", "0,1", "3", "abc", "all", "none", "0, 1", "GPU-abc"],
    )
    def test_all_visible_or_default_is_never_produced(self, raw):
        # Phase-0 verification: the CUDA grammar documents no set-value
        # token meaning "all visible" — only unset does. The enum member
        # is reserved vocabulary, never emitted by the normalizer.
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            variables={**snapshot.environment.variables, "CUDA_VISIBLE_DEVICES": raw},
        )
        state = build_facts(snapshot).visibility.cuda_visible_devices
        assert state is not CUDAVisibleDevicesState.ALL_VISIBLE_OR_DEFAULT

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, Trinary.FALSE),
            ("", Trinary.TRUE),
            ("-1", Trinary.TRUE),
            ("0", Trinary.FALSE),
            ("0,1", Trinary.FALSE),
            ("3", Trinary.FALSE),
            ("abc", Trinary.UNKNOWN),
            ("0,abc", Trinary.UNKNOWN),
        ],
    )
    def test_explicit_all_gpu_mask_predicate(self, raw, expected):
        snapshot = base_snapshot()
        variables = dict(snapshot.environment.variables)
        if raw is None:
            variables.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            variables["CUDA_VISIBLE_DEVICES"] = raw
        snapshot.environment = replace(snapshot.environment, variables=variables)
        assert build_facts(snapshot).visibility.explicit_all_gpu_mask is expected

    def test_raw_value_not_carried_into_facts(self):
        snapshot = base_snapshot()
        secret = "GPU-abcdef0123456789"
        snapshot.environment = replace(
            snapshot.environment,
            variables={**snapshot.environment.variables, "CUDA_VISIBLE_DEVICES": secret},
        )
        assert secret not in str(asdict(build_facts(snapshot).visibility))


class TestRuntimeLibraryFacts:
    def ldp_inventory(self) -> RuntimeLibraryInventory:
        return inventory(
            library(
                "libcudart",
                "/opt/other/lib/libcudart.so.12",
                version="12",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_order=1,
            ),
            library(
                "libcudart",
                "/opt/first/lib/libcudart.so.11",
                version="11",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_order=0,
            ),
            library(
                "libcublas",
                "/opt/first/lib/libcublas.so.12",
                version="12.0.1",
                source=VERSION_SOURCE_SYMLINK_TARGET,
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_order=0,
            ),
            library(
                "libcudnn",
                "/opt/first/lib/libcudnn.so.9",
                version=None,
                source=None,
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_order=0,
            ),
            # Non-LDP origins never enter the modeled ordering.
            library(
                "libcudart",
                "/usr/local/cuda-12.4/lib64/libcudart.so",
                version="12",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_TOOLKIT,
                search_group="toolkit",
            ),
            library(
                "libcudnn",
                "/home/tester/venv/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9",
                version="9.1",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_PYTHON_PACKAGE,
                search_group="python-package",
            ),
            library(
                "libcublas",
                "/usr/lib/x86_64-linux-gnu/libcublas.so.12",
                version="12",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_LDCONFIG,
                search_group="ldconfig",
            ),
        )

    def snapshot_with_inventory(self, inv: RuntimeLibraryInventory) -> EnvironmentSnapshot:
        snapshot = base_snapshot()
        snapshot.runtime_libraries = inv
        return snapshot

    def test_modeled_candidates_only_ldp_with_order(self):
        facts = build_facts(self.snapshot_with_inventory(self.ldp_inventory()))
        modeled = facts.runtime_libraries.modeled_candidates
        assert [(c.family, c.family_version, c.search_order) for c in modeled] == [
            ("libcublas", "12", 0),
            ("libcudart", "11", 0),
            ("libcudnn", None, 0),
            ("libcudart", "12", 1),
        ]
        assert all(isinstance(c, ModeledLibraryCandidate) for c in modeled)

    def test_modeled_candidate_reliability(self):
        facts = build_facts(self.snapshot_with_inventory(self.ldp_inventory()))
        by_signature = {
            (c.family, c.search_order): c for c in facts.runtime_libraries.modeled_candidates
        }
        assert by_signature[("libcudart", 0)].reliability is IdentityReliability.DIRECT
        assert by_signature[("libcublas", 0)].reliability is (
            IdentityReliability.DERIVED_STRONG
        )
        assert by_signature[("libcudnn", 0)].reliability is IdentityReliability.UNKNOWN

    def test_ldp_candidate_without_order_is_not_modeled(self):
        inv = inventory(
            library(
                "libcudart",
                "/opt/lib/libcudart.so.12",
                version="12",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_order=None,
            )
        )
        runtime = build_facts(self.snapshot_with_inventory(inv)).runtime_libraries
        assert runtime.modeled_candidates == ()

    def test_family_counts_cover_all_origins(self):
        facts = build_facts(self.snapshot_with_inventory(self.ldp_inventory()))
        assert facts.runtime_libraries.family_counts == (
            ("libcublas", 2),
            ("libcudart", 3),
            ("libcudnn", 2),
        )

    def test_observed_origins_and_package_local_families(self):
        runtime = build_facts(
            self.snapshot_with_inventory(self.ldp_inventory())
        ).runtime_libraries
        assert runtime.observed_origins == frozenset(
            {ORIGIN_LD_LIBRARY_PATH, ORIGIN_TOOLKIT, ORIGIN_PYTHON_PACKAGE, ORIGIN_LDCONFIG}
        )
        assert runtime.package_local_families == frozenset({"libcudnn"})

    def test_torch_runtime_family(self):
        runtime = build_facts(
            self.snapshot_with_inventory(self.ldp_inventory())
        ).runtime_libraries
        assert runtime.torch_runtime_family == "12"

    def test_torch_runtime_family_none_without_torch_cuda(self):
        snapshot = self.snapshot_with_inventory(self.ldp_inventory())
        snapshot.pytorch = replace(snapshot.pytorch, cuda_version=None)
        runtime = build_facts(snapshot).runtime_libraries
        assert runtime.torch_runtime_family is None

    def test_package_local_cudnn_baseline_single_family(self):
        runtime = build_facts(
            self.snapshot_with_inventory(self.ldp_inventory())
        ).runtime_libraries
        assert runtime.package_local_cudnn_family == "9"

    def test_package_local_cudnn_baseline_ambiguous_is_none(self):
        inv = inventory(
            library(
                "libcudnn",
                "/site-a/nvidia/cudnn/lib/libcudnn.so.9",
                version="9.1",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_PYTHON_PACKAGE,
            ),
            library(
                "libcudnn",
                "/site-b/nvidia/cudnn/lib/libcudnn.so.8",
                version="8.9",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_PYTHON_PACKAGE,
            ),
        )
        runtime = build_facts(self.snapshot_with_inventory(inv)).runtime_libraries
        assert runtime.package_local_cudnn_family is None

    def test_package_local_cudnn_baseline_absent_is_none(self):
        inv = inventory(
            library(
                "libcudart",
                "/site/nvidia/cuda_runtime/lib/libcudart.so.12",
                version="12",
                source=VERSION_SOURCE_FILENAME,
                origin=ORIGIN_PYTHON_PACKAGE,
            )
        )
        runtime = build_facts(self.snapshot_with_inventory(inv)).runtime_libraries
        assert runtime.package_local_cudnn_family is None

    def test_scan_incomplete_with_errors(self):
        inv = RuntimeLibraryInventory(
            candidates=[], scan_errors={"ldconfig": "PermissionError: denied"}
        )
        runtime = build_facts(self.snapshot_with_inventory(inv)).runtime_libraries
        assert runtime.scan_incomplete is True

    def test_scan_complete_without_errors(self):
        runtime = build_facts(
            self.snapshot_with_inventory(self.ldp_inventory())
        ).runtime_libraries
        assert runtime.scan_incomplete is False

    def test_missing_inventory_yields_empty_facts(self):
        runtime = build_facts(base_snapshot()).runtime_libraries
        assert runtime == RuntimeLibraryFacts(
            modeled_candidates=(),
            family_counts=(),
            observed_origins=frozenset(),
            package_local_families=frozenset(),
            torch_runtime_family="12",
            package_local_cudnn_family=None,
            scan_incomplete=False,
        )

    def test_no_paths_or_scan_errors_carried(self):
        # Privacy (§28): facts carry aggregates only — no library paths,
        # no raw scan-error text.
        inv = RuntimeLibraryInventory(
            candidates=[
                library(
                    "libcudart",
                    "/home/secretuser/venv/nvidia/cuda_runtime/lib/libcudart.so.12",
                    version="12",
                    source=VERSION_SOURCE_FILENAME,
                    origin=ORIGIN_LD_LIBRARY_PATH,
                    search_order=0,
                )
            ],
            scan_errors={"ldconfig": "PermissionError: /home/secretuser denied"},
        )
        snapshot = self.snapshot_with_inventory(inv)
        rendered = str(asdict(build_facts(snapshot).runtime_libraries))
        assert "secretuser" not in rendered
        assert "PermissionError" not in rendered
        assert "libcudart.so" not in rendered


class TestIssueCodeProjection:
    def test_empty_without_issues(self):
        assert build_facts(base_snapshot()).issue_codes == frozenset()

    def test_projection_of_issue_codes(self):
        issues = [
            Issue(code="TORCH002", severity=Severity.ERROR, title="a"),
            Issue(code="CUDA001", severity=Severity.WARNING, title="b"),
        ]
        assert build_facts(base_snapshot(), issues).issue_codes == frozenset(
            {"TORCH002", "CUDA001"}
        )

    def test_duplicates_collapse(self):
        issues = [
            Issue(code="TORCH002", severity=Severity.ERROR, title="a"),
            Issue(code="TORCH002", severity=Severity.ERROR, title="a"),
        ]
        assert build_facts(base_snapshot(), issues).issue_codes == frozenset({"TORCH002"})

    def test_engine_issues_projected(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        result = DiagnosisEngine().run(snapshot)
        facts = DiagnosisFactBuilder().build(snapshot, result.issues, COMPATIBILITY)
        assert facts.issue_codes == frozenset({"TORCH002"})


class TestPlatformApplicability:
    @pytest.mark.parametrize(
        "platform,expected",
        [
            (Platform.LINUX, (True, True, True, True)),
            (Platform.WINDOWS, (True, True, True, False)),
            (Platform.MACOS, (False, False, False, False)),
            (Platform.OTHER, (False, False, False, False)),
        ],
    )
    def test_frozen_platform_matrix(self, platform, expected):
        snapshot = base_snapshot()
        snapshot.system = replace(snapshot.system, platform=platform)
        applicability = build_facts(snapshot).applicability
        assert (applicability.d001, applicability.d002, applicability.d003, applicability.d004) == (
            expected
        )


class TestPurityAndDeterminism:
    def test_same_inputs_produce_equal_facts(self):
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        first = DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)
        second = DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)
        assert first == second

    def test_host_environment_changes_nothing(self, monkeypatch):
        # The builder must never consult os.environ or other host state:
        # mutating the process environment between builds changes nothing.
        snapshot = base_snapshot()
        before = build_facts(snapshot)
        monkeypatch.setenv("CUDA_HOME", "/host/cuda-99.0")
        monkeypatch.setenv("CUDA_PATH", "/host/cuda-98.0")
        monkeypatch.setenv("CUDACXX", "/host/bin/nvcc")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        monkeypatch.setenv("PATH", "/host/bin")
        after = build_facts(snapshot)
        assert after == before

    def test_facts_module_imports_no_host_probes(self):
        # Purity is structural: any host probe (environment, filesystem,
        # subprocess, torch, clock, randomness) requires an import.
        tree = ast.parse(Path(facts_module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert imported.isdisjoint(
            {"os", "subprocess", "shutil", "site", "time", "random", "torch"}
        )

    def test_all_fact_models_frozen(self):
        facts = build_facts(base_snapshot())
        selector = ToolkitSelectorFact(
            name="CUDA_HOME",
            explicit=True,
            effective=False,
            active=True,
            valid=True,
            canonical_root="/opt/cuda-12.4",
            toolkit_version="12.4",
            reliability=IdentityReliability.DIRECT,
        )
        reference = facts_module.ReferenceToolkit(
            source="NVCC",
            canonical_root=None,
            toolkit_version=None,
            reliability=IdentityReliability.UNKNOWN,
        )
        modeled = ModeledLibraryCandidate(
            family="libcudart",
            family_version=None,
            reliability=IdentityReliability.UNKNOWN,
            search_order=0,
        )
        with pytest.raises(FrozenInstanceError):
            facts.platform = Platform.WINDOWS
        with pytest.raises(FrozenInstanceError):
            selector.active = False
        with pytest.raises(FrozenInstanceError):
            facts.torch.importable = False
        with pytest.raises(FrozenInstanceError):
            facts.gpu.visible = Trinary.FALSE
        with pytest.raises(FrozenInstanceError):
            facts.driver.usable = Trinary.FALSE
        with pytest.raises(FrozenInstanceError):
            facts.visibility.cuda_visible_devices = CUDAVisibleDevicesState.UNSET
        with pytest.raises(FrozenInstanceError):
            facts.runtime_libraries.scan_incomplete = True
        with pytest.raises(FrozenInstanceError):
            facts.applicability.d004 = True
        with pytest.raises(FrozenInstanceError):
            facts.reference_toolkit = None
        with pytest.raises(FrozenInstanceError):
            reference.source = "CUDA_HOME"
        with pytest.raises(FrozenInstanceError):
            modeled.family = "libcublas"

    def test_diagnosis_facts_field_set_exactly_frozen_shape(self):
        assert {f.name for f in fields(facts_module.DiagnosisFacts)} == FACTS_FIELD_NAMES

    def test_collections_are_immutable_types(self):
        facts = build_facts(base_snapshot())
        assert isinstance(facts.toolkit_selectors, tuple)
        assert isinstance(facts.issue_codes, frozenset)
        assert isinstance(facts.runtime_libraries.modeled_candidates, tuple)
        assert isinstance(facts.runtime_libraries.observed_origins, frozenset)
        assert isinstance(facts.runtime_libraries.package_local_families, frozenset)

    def test_malformed_selector_input_degrades_to_empty(self):
        # Architecture §20: one malformed fact degrades to its
        # UNKNOWN-shaped fallback instead of failing the whole build.
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, selector_observations=None)
        facts = build_facts(snapshot)
        assert facts.toolkit_selectors == ()
        assert facts.torch.importable is True  # other facts unaffected

    def test_malformed_inventory_degrades_to_empty(self):
        snapshot = base_snapshot()
        snapshot.runtime_libraries = RuntimeLibraryInventory(candidates=None)
        runtime = build_facts(snapshot).runtime_libraries
        assert runtime.modeled_candidates == ()
        assert runtime.family_counts == ()
        assert runtime.scan_incomplete is False


class TestNormalizeOnlyBoundary:
    def test_module_defines_no_diagnosis_emission_surface(self):
        # §3/§33: normalization only. No Diagnosis construction, no
        # severity/confidence/role assignment, no rule emission.
        for forbidden in ("Diagnosis(", "Severity(", "Confidence.", "DiagnosisRole."):
            assert forbidden not in Path(facts_module.__file__).read_text(), forbidden

    def test_facts_carry_no_policy_fields(self):
        facts = build_facts(base_snapshot())
        rendered = str(asdict(facts))
        for policy_word in ("should_emit", "conflict", "severity", "confidence"):
            assert policy_word not in rendered, policy_word
