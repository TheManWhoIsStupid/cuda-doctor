"""Normalized diagnosis facts (v0.2): snapshot in, immutable facts out.

This layer normalizes an already-collected :class:`EnvironmentSnapshot`
into the frozen fact view consumed by the D001-D004 rules (later phases),
so no rule reinterprets raw snapshot fields independently. It normalizes
only — it never diagnoses: nothing here decides whether an issue exists,
emits a ``Diagnosis``, or assigns severity, confidence, or role.

Guarantees (frozen architecture §12/§20):

- pure and deterministic: no host environment, filesystem, subprocess,
  torch import, network, clock, or randomness is ever consulted;
- tri-state predicates carry a dedicated representation and UNKNOWN is
  never collapsed into FALSE;
- toolkit identity separates installation identity from any conflict
  judgment: path inequality alone is never disagreement, and an unknown
  identity is never a mismatch;
- ``nvidia-smi``'s reported "CUDA Version" is never a toolkit selector;
- the D002 compatibility adapter reuses ``interpret_torch_cuda`` instead
  of forking its semantics, and driver compatibility reuses
  ``compatibility.driver`` instead of duplicating its tables.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeVar

from cuda_doctor.collectors.cuda import (
    VERSION_SOURCE_DERIVED_STRONG,
    VERSION_SOURCE_DERIVED_WEAK,
    VERSION_SOURCE_DIRECT,
)
from cuda_doctor.collectors.runtime_libraries import (
    ORIGIN_LD_LIBRARY_PATH,
    ORIGIN_PYTHON_PACKAGE,
    VERSION_SOURCE_FILENAME,
    VERSION_SOURCE_SYMLINK_TARGET,
)
from cuda_doctor.compatibility import CompatibilityData, interpret_torch_cuda
from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import (
    CUDASelectorObservation,
    EnvironmentSnapshot,
    RuntimeLibraryInventory,
)
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.utils.versions import (
    CudaVersion,
    parse_cuda_version,
    parse_driver_version,
)

__all__ = [
    "CUDAVisibleDevicesState",
    "DiagnosisFactBuilder",
    "DiagnosisFacts",
    "DriverDiagnosisFacts",
    "GPUDiagnosisFacts",
    "IdentityReliability",
    "ModeledLibraryCandidate",
    "PlatformApplicability",
    "ReferenceToolkit",
    "RuntimeLibraryFacts",
    "ToolkitCompatibility",
    "ToolkitSelectorFact",
    "TorchDiagnosisFacts",
    "Trinary",
    "VisibilityFacts",
    "normalize_torch_toolkit_relation",
]


# Normalized selector names (truth tables §5 vocabulary). The collector's
# lowercase observation names map one-to-one onto these; any unexpected
# observation name is preserved verbatim and classified incidentally.
CUDA_HOME = "CUDA_HOME"
CUDA_PATH = "CUDA_PATH"
CUDACXX = "CUDACXX"
NVCC = "NVCC"
PATH_CUDA_BIN = "PATH_CUDA_BIN"
USR_LOCAL_CUDA = "USR_LOCAL_CUDA"

_SELECTOR_NAMES = {
    "cuda_home": CUDA_HOME,
    "cuda_path": CUDA_PATH,
    "cudacxx": CUDACXX,
    "nvcc": NVCC,
    "path_cuda_bin": PATH_CUDA_BIN,
    "usr_local_cuda": USR_LOCAL_CUDA,
}

# Explicit selectors come from environment configuration; effective
# selectors are values actually chosen by executable/PATH resolution
# (semantics §20.3). ``usr_local_cuda`` is incidental by default.
_EXPLICIT_SELECTORS = frozenset({CUDA_HOME, CUDA_PATH, CUDACXX})
_EFFECTIVE_SELECTORS = frozenset({NVCC, PATH_CUDA_BIN})


class Trinary(str, Enum):
    """Dedicated three-valued predicate (architecture §12.5).

    Used where a fact is logically boolean but collection may fail.
    ``UNKNOWN`` is a real state — it must never be collapsed onto
    ``FALSE`` or confused with ``bool | None`` where ``None`` could be
    mistaken for a negative observation.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class IdentityReliability(str, Enum):
    """How reliably a toolkit identity's version was established.

    Ordering (truth tables §7): direct > derived_strong > derived_weak >
    unknown. An ``UNKNOWN`` identity cannot establish disagreement.
    """

    DIRECT = "direct"
    DERIVED_STRONG = "derived_strong"
    DERIVED_WEAK = "derived_weak"
    UNKNOWN = "unknown"


class CUDAVisibleDevicesState(str, Enum):
    """Normalized ``CUDA_VISIBLE_DEVICES`` state (truth tables §18.10).

    Only the two Phase-0-verified mask-all tokens are classified as
    ``MASK_ALL``: the empty string and ``-1``. No other special token is
    guessed. ``ALL_VISIBLE_OR_DEFAULT`` is part of the frozen vocabulary
    but has no verified producing value in v0.2.0 (the authoritative CUDA
    grammar documents no set-value form that means "all visible"), so the
    normalizer never emits it; a nonempty value is either a plain
    non-negative integer index list (``RESTRICTED_NONEMPTY`` — even when
    the indices select no existing GPU, a deliberate conservatism) or
    outside the modeled grammar (``UNPARSEABLE``).
    """

    UNSET = "unset"
    ALL_VISIBLE_OR_DEFAULT = "all_visible_or_default"
    RESTRICTED_NONEMPTY = "restricted_nonempty"
    MASK_ALL = "mask_all"
    UNPARSEABLE = "unparseable"


class ToolkitCompatibility(str, Enum):
    """Normalized torch/local-toolkit relation (truth tables §17.5).

    Adapted one-to-one from ``interpret_torch_cuda``'s structured result;
    there is deliberately no minor-mismatch-warning state.
    """

    EXACT = "exact"
    MINOR_COMPATIBLE_INFORMATIONAL = "minor_compatible_informational"
    MAJOR_INCOMPATIBLE = "major_incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolkitSelectorFact:
    """One normalized CUDA toolkit selector observation.

    ``active`` is computed here according to the frozen selector-class
    semantics (semantics §20.2-§20.4, truth tables §16.9-§16.11) — the
    collector never sets it. ``explicit``/``effective`` distinguish the
    frozen selector classes; an incidental selector (``usr_local_cuda``)
    carries both as False and becomes active only when an active
    selector's canonical resolution ties to it. ``valid`` keeps the
    collector's tri-state shape. ``toolkit_version`` is normalized to
    ``major.minor``; identity fields alone imply no conflict judgment.
    """

    name: str
    explicit: bool
    effective: bool
    active: bool
    valid: bool | None
    canonical_root: str | None
    toolkit_version: str | None
    reliability: IdentityReliability


@dataclass(frozen=True)
class TorchDiagnosisFacts:
    """PyTorch facts as observed by collection (never re-imported here).

    ``cuda_available`` keeps the snapshot's ``bool | None`` shape: ``None``
    is the frozen "probe produced no reliable boolean" contract, distinct
    from ``False``. ``cuda_build`` distinguishes a CPU-only build
    (``False``) from a CUDA build whose availability is unknown
    (``True`` with ``cuda_available=None``).
    """

    importable: bool
    cuda_build: bool | None
    cuda_version: str | None
    cuda_available: bool | None
    cudnn_version: str | None


@dataclass(frozen=True)
class GPUDiagnosisFacts:
    """GPU visibility from the independent nvidia-smi/GPU path only.

    Never derived from PyTorch's own ``is_available``/``device_count`` or
    from ``CUDA_VISIBLE_DEVICES`` (non-circularity, truth tables §18.5).
    A failed or unavailable probe is ``UNKNOWN`` — never a fabricated
    negative fact.
    """

    visible: Trinary


@dataclass(frozen=True)
class DriverDiagnosisFacts:
    """Driver facts from the independent collection path.

    ``compatible_with_torch_runtime`` reuses the existing
    ``compatibility.driver.evaluate`` verdict (documented family minimums
    and cross-generation gaps); no second compatibility table exists
    here. The nvidia-smi "CUDA Version" is not part of these facts.
    """

    version: str | None
    usable: Trinary
    compatible_with_torch_runtime: Trinary


@dataclass(frozen=True)
class VisibilityFacts:
    """Normalized device-visibility masking state.

    Only the classification is carried — never the raw environment
    value. ``explicit_all_gpu_mask`` is the tri-state predicate over the
    verified mask-all tokens (empty string / ``-1``).
    """

    cuda_visible_devices: CUDAVisibleDevicesState
    explicit_all_gpu_mask: Trinary


@dataclass(frozen=True)
class ModeledLibraryCandidate:
    """One runtime-library candidate inside the modeled ordering.

    Only candidates from ordered ``LD_LIBRARY_PATH`` entries carry a
    ``search_order``; everything the loader model cannot establish
    (RPATH, ld.so.cache precedence, ...) stays out of the facts.
    ``family_version`` is the normalized comparable component (the
    leading dotted component of the filename-derived version, e.g. ``12``
    from ``libcudart.so.12``); ``None`` means the family is UNKNOWN.
    """

    family: str
    family_version: str | None
    reliability: IdentityReliability
    search_order: int


@dataclass(frozen=True)
class RuntimeLibraryFacts:
    """Normalized view of the bounded runtime-library inventory (D004 input).

    Inventory facts only — no loader winner, conflict, or precedence
    beyond the explicitly modeled ``LD_LIBRARY_PATH`` ordering is decided
    here. No library paths or scan-error text are carried: facts stay
    bounded and shareable. ``torch_runtime_family`` is the torch CUDA
    major family (the libcudart baseline; libcublas may use it only where
    a candidate's own mapping is reliable). cuDNN uses its own
    environment-local baseline (``package_local_cudnn_family``) and is
    never compared against ``torch.version.cuda``. A ``None`` baseline
    means UNKNOWN — not a conflict and not an all-clear.
    """

    modeled_candidates: tuple[ModeledLibraryCandidate, ...]
    family_counts: tuple[tuple[str, int], ...]
    observed_origins: frozenset[str]
    package_local_families: frozenset[str]
    torch_runtime_family: str | None
    package_local_cudnn_family: str | None
    scan_incomplete: bool


@dataclass(frozen=True)
class ReferenceToolkit:
    """The effective local development toolkit (D002 reference).

    Computed once by the fact builder under the frozen precedence
    valid CUDACXX → resolved active NVCC → valid usable CUDA_HOME/
    CUDA_PATH development toolkit → none (truth tables §17.2); D002 must
    consume this instead of reimplementing the precedence. A valid
    CUDACXX with an underivable version still blocks the fallback (an
    explicit compiler selection outranks PATH nvcc) and simply carries
    ``UNKNOWN`` reliability; its root may be ``None`` (wrapper layouts
    record no toolkit-root identity).
    """

    source: str
    canonical_root: str | None
    toolkit_version: str | None
    reliability: IdentityReliability


@dataclass(frozen=True)
class PlatformApplicability:
    """Which diagnosis families are evaluated on this platform.

    Exposure only — applicability never emits or suppresses anything by
    itself. D004 is Linux-only in v0.2.0; other platforms are "not
    evaluated", never guessed (truth tables §8, §19.1).
    """

    d001: bool
    d002: bool
    d003: bool
    d004: bool


@dataclass(frozen=True)
class DiagnosisFacts:
    """The immutable normalized fact aggregate consumed by D001-D004.

    Carries the frozen architecture §12.1 view plus the two frozen
    derived members the rules must not recompute: the D002
    ``reference_toolkit`` (§12.7) and the platform ``applicability``
    gates. ``issue_codes`` is a pure projection of the issues the engine
    already produced — it implies nothing about diagnoses.
    """

    platform: Platform
    toolkit_selectors: tuple[ToolkitSelectorFact, ...]
    torch: TorchDiagnosisFacts
    gpu: GPUDiagnosisFacts
    driver: DriverDiagnosisFacts
    visibility: VisibilityFacts
    runtime_libraries: RuntimeLibraryFacts
    issue_codes: frozenset[str]
    reference_toolkit: ReferenceToolkit | None
    applicability: PlatformApplicability


# -- pure normalizers -------------------------------------------------------


def normalize_torch_toolkit_relation(
    torch_cuda: CudaVersion | None, toolkit: CudaVersion | None
) -> ToolkitCompatibility:
    """Adapt ``interpret_torch_cuda`` to the frozen D002 states (§12.8).

    Uses the helper's structured ``relation`` plus the same major-version
    distinction its own explanation branching makes — never its prose.
    The mapping is one-to-one with truth tables §17.5 and introduces no
    new band (there is no minor-mismatch-warning state).
    """
    interpretation = interpret_torch_cuda(torch_cuda, toolkit)
    if interpretation.relation == "unknown":
        return ToolkitCompatibility.UNKNOWN
    if interpretation.relation == "match":
        return ToolkitCompatibility.EXACT
    assert torch_cuda is not None and toolkit is not None  # relation implies both
    if torch_cuda.major == toolkit.major:
        return ToolkitCompatibility.MINOR_COMPATIBLE_INFORMATIONAL
    return ToolkitCompatibility.MAJOR_INCOMPATIBLE


def _identity_reliability(
    observation: CUDASelectorObservation, normalized_version: str | None
) -> IdentityReliability:
    """Map the collector's version_source onto IdentityReliability (§13.5).

    A missing or unnormalizable version is ``UNKNOWN`` regardless of any
    source string the collector might carry: without a normalized version
    there is no identity to be reliable about.
    """
    if normalized_version is None:
        return IdentityReliability.UNKNOWN
    return {
        VERSION_SOURCE_DIRECT: IdentityReliability.DIRECT,
        VERSION_SOURCE_DERIVED_STRONG: IdentityReliability.DERIVED_STRONG,
        VERSION_SOURCE_DERIVED_WEAK: IdentityReliability.DERIVED_WEAK,
    }.get(observation.version_source or "", IdentityReliability.UNKNOWN)


def _normalized_version(raw: str | None) -> str | None:
    """Normalize an observed toolkit version to ``major.minor`` (§5)."""
    parsed = parse_cuda_version(raw) if raw else None
    return str(parsed) if parsed is not None else None


def _toolkit_selector_facts(snapshot: EnvironmentSnapshot) -> tuple[ToolkitSelectorFact, ...]:
    """Normalize the six selector observations; compute activity here.

    Explicit directory/compiler selectors are active when observed
    (except ``CUDA_PATH`` on Linux, which is recorded but never active —
    truth tables §16.11). Effective selectors (``nvcc``/``path_cuda_bin``)
    are active when observed: the collector only records the resolution
    that actually won. ``usr_local_cuda`` is active only when an active
    selector's canonical root ties to it (semantics §20.4).
    """
    platform = snapshot.system.platform
    facts: list[ToolkitSelectorFact] = []
    usr_local_root: str | None = None

    for observation in snapshot.cuda.selector_observations:
        name = _SELECTOR_NAMES.get(observation.name, observation.name)
        explicit = name in _EXPLICIT_SELECTORS
        effective = name in _EFFECTIVE_SELECTORS
        active = _selector_is_active(name, platform, explicit, effective)
        if name == USR_LOCAL_CUDA:
            usr_local_root = observation.canonical_root
            active = False  # recomputed below once other selectors are known
        toolkit_version = _normalized_version(observation.toolkit_version)
        facts.append(
            ToolkitSelectorFact(
                name=name,
                explicit=explicit,
                effective=effective,
                active=active,
                valid=observation.valid,
                canonical_root=observation.canonical_root,
                toolkit_version=toolkit_version,
                reliability=_identity_reliability(observation, toolkit_version),
            )
        )

    if usr_local_root is not None:
        # /usr/local/cuda participates only when an active selector's
        # canonical resolution actually ties to it — never merely because
        # the symlink exists (D001-N14). Equivalent canonical paths count
        # as the same target regardless of the spelling used.
        tied = any(
            fact.name != USR_LOCAL_CUDA
            and fact.active
            and fact.canonical_root == usr_local_root
            for fact in facts
        )
        facts = [
            replace(fact, active=tied) if fact.name == USR_LOCAL_CUDA else fact
            for fact in facts
        ]
    return tuple(facts)


def _selector_is_active(
    name: str, platform: Platform, explicit: bool, effective: bool
) -> bool:
    if name == CUDA_PATH:
        # Observed on both platforms, but a D001 selector on Windows only
        # (truth tables §16.11); on Linux it stays inventory information.
        return platform is Platform.WINDOWS
    return explicit or effective


def _torch_facts(snapshot: EnvironmentSnapshot) -> TorchDiagnosisFacts:
    info = snapshot.pytorch
    return TorchDiagnosisFacts(
        importable=bool(info.installed) and not info.import_error,
        cuda_build=info.is_cuda_build,
        cuda_version=info.cuda_version,
        cuda_available=info.cuda_available,
        cudnn_version=info.cudnn_version,
    )


def _gpu_facts(snapshot: EnvironmentSnapshot) -> GPUDiagnosisFacts:
    """Independent GPU visibility; a failed probe is never FALSE."""
    info = snapshot.nvidia_smi
    if info is None or not info.available or not info.executed or info.error:
        return GPUDiagnosisFacts(visible=Trinary.UNKNOWN)
    return GPUDiagnosisFacts(visible=Trinary.TRUE if snapshot.gpus else Trinary.FALSE)


def _driver_facts(
    snapshot: EnvironmentSnapshot,
    torch_facts: TorchDiagnosisFacts,
    compatibility: CompatibilityData | None,
) -> DriverDiagnosisFacts:
    driver = snapshot.driver
    info = snapshot.nvidia_smi
    version = driver.version if driver is not None else None

    if version:
        usable = Trinary.TRUE  # the driver identified itself through nvidia-smi
    elif info is not None and info.available and info.executed and info.error:
        # nvidia-smi ran and reported a failure: observed breakage, not a
        # fabricated negative (truth tables §24).
        usable = Trinary.FALSE
    else:
        usable = Trinary.UNKNOWN

    compatible = Trinary.UNKNOWN
    torch_cuda = parse_cuda_version(torch_facts.cuda_version)
    installed = parse_driver_version(version) if version else None
    if compatibility is not None and torch_cuda is not None and installed is not None:
        verdict = compatibility.driver.evaluate(torch_cuda, installed, snapshot.system.platform)
        compatible = {
            None: Trinary.UNKNOWN,
            True: Trinary.TRUE,
            False: Trinary.FALSE,
        }[verdict.compatible]
    return DriverDiagnosisFacts(
        version=version, usable=usable, compatible_with_torch_runtime=compatible
    )


def _normalize_cuda_visible_devices(raw: str | None) -> CUDAVisibleDevicesState:
    """Classify per the Phase-0-verified CUDA grammar only (§18.10)."""
    if raw is None:
        return CUDAVisibleDevicesState.UNSET
    if raw in ("", "-1"):
        return CUDAVisibleDevicesState.MASK_ALL
    tokens = raw.split(",")
    if all(token.isascii() and token.isdigit() for token in tokens):
        # Deliberate conservatism: an index list that selects no existing
        # GPU (e.g. "3" on a 2-GPU machine) stays RESTRICTED_NONEMPTY —
        # the runtime masking effect is device-count-dependent and is not
        # modeled here (D003-N08).
        return CUDAVisibleDevicesState.RESTRICTED_NONEMPTY
    return CUDAVisibleDevicesState.UNPARSEABLE


def _visibility_facts(snapshot: EnvironmentSnapshot) -> VisibilityFacts:
    raw = snapshot.environment.variables.get("CUDA_VISIBLE_DEVICES")
    state = _normalize_cuda_visible_devices(raw)
    explicit_mask = {
        CUDAVisibleDevicesState.MASK_ALL: Trinary.TRUE,
        CUDAVisibleDevicesState.UNPARSEABLE: Trinary.UNKNOWN,
    }.get(state, Trinary.FALSE)
    return VisibilityFacts(cuda_visible_devices=state, explicit_all_gpu_mask=explicit_mask)


def _library_reliability(version_source: str | None) -> IdentityReliability:
    """Filename observation is direct; a symlink target's filename is one
    derivation step; anything else is UNKNOWN."""
    return {
        VERSION_SOURCE_FILENAME: IdentityReliability.DIRECT,
        VERSION_SOURCE_SYMLINK_TARGET: IdentityReliability.DERIVED_STRONG,
    }.get(version_source or "", IdentityReliability.UNKNOWN)


def _family_version(version: str | None) -> str | None:
    if not version:
        return None
    return version.split(".")[0]


def _runtime_library_facts(
    inventory: RuntimeLibraryInventory | None, torch_facts: TorchDiagnosisFacts
) -> RuntimeLibraryFacts:
    candidates = inventory.candidates if inventory is not None else []

    modeled = sorted(
        (
            ModeledLibraryCandidate(
                family=candidate.family,
                family_version=_family_version(candidate.version),
                reliability=_library_reliability(candidate.version_source),
                search_order=candidate.search_order,
            )
            for candidate in candidates
            if candidate.origin == ORIGIN_LD_LIBRARY_PATH
            and candidate.search_order is not None
        ),
        key=lambda candidate: (
            candidate.search_order,
            candidate.family,
            candidate.family_version or "",
        ),
    )

    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.family] = counts.get(candidate.family, 0) + 1
    package_local = [c for c in candidates if c.origin == ORIGIN_PYTHON_PACKAGE]
    cudnn_families = {
        _family_version(c.version) for c in package_local if c.family == "libcudnn"
    } - {None}
    # A single distinct environment-local cuDNN family is the observable
    # baseline; zero or several disagreeing ones mean UNKNOWN (§19.10).
    package_local_cudnn = next(iter(cudnn_families)) if len(cudnn_families) == 1 else None

    torch_cuda = parse_cuda_version(torch_facts.cuda_version)
    return RuntimeLibraryFacts(
        modeled_candidates=tuple(modeled),
        family_counts=tuple(sorted(counts.items())),
        observed_origins=frozenset(c.origin for c in candidates),
        package_local_families=frozenset(c.family for c in package_local),
        torch_runtime_family=str(torch_cuda.major) if torch_cuda is not None else None,
        package_local_cudnn_family=package_local_cudnn,
        scan_incomplete=bool(inventory.scan_errors) if inventory is not None else False,
    )


def _reference_toolkit(selectors: Sequence[ToolkitSelectorFact]) -> ReferenceToolkit | None:
    """Frozen D002 precedence: CUDACXX → NVCC → usable CUDA_HOME/CUDA_PATH.

    A directory merely existing is insufficient at step three: the
    selector must be a usable development toolkit (a directly or strongly
    derived version means ``<root>/bin/nvcc`` answered the probe), which
    excludes metadata-only roots (D002-N08).
    """
    by_name = {fact.name: fact for fact in selectors}

    def reference(name: str, fact: ToolkitSelectorFact) -> ReferenceToolkit:
        return ReferenceToolkit(
            source=name,
            canonical_root=fact.canonical_root,
            toolkit_version=fact.toolkit_version,
            reliability=fact.reliability,
        )

    cudacxx = by_name.get(CUDACXX)
    if cudacxx is not None and cudacxx.valid is True:
        return reference(CUDACXX, cudacxx)
    nvcc = by_name.get(NVCC)
    if nvcc is not None and nvcc.valid is True:
        return reference(NVCC, nvcc)
    for name in (CUDA_HOME, CUDA_PATH):
        fact = by_name.get(name)
        if (
            fact is not None
            and fact.valid is True
            and fact.reliability
            in (IdentityReliability.DIRECT, IdentityReliability.DERIVED_STRONG)
        ):
            return reference(name, fact)
    return None


def _applicability(platform: Platform) -> PlatformApplicability:
    workstation = platform in (Platform.LINUX, Platform.WINDOWS)
    return PlatformApplicability(
        d001=workstation,
        d002=workstation,
        d003=workstation,
        d004=platform is Platform.LINUX,
    )


_T = TypeVar("_T")


def _safe(build: Callable[[], _T], fallback: _T) -> _T:
    """Per-fact failure isolation: prefer UNKNOWN over aborting (§20)."""
    try:
        return build()
    except Exception:
        # Normalization must never crash diagnosis: one malformed fact
        # degrades to its UNKNOWN-shaped fallback.
        return fallback


class DiagnosisFactBuilder:
    """Builds the immutable :class:`DiagnosisFacts` view of one snapshot.

    Deterministic and side-effect-free: the same snapshot, issues, and
    compatibility data always produce equal facts, and no external state
    is consulted. Unexpected failures of one derived fact degrade that
    fact to an UNKNOWN-shaped fallback instead of failing the build.
    """

    def build(
        self,
        snapshot: EnvironmentSnapshot,
        issues: Sequence[Issue],
        compatibility: CompatibilityData | None,
    ) -> DiagnosisFacts:
        torch_facts = _safe(
            lambda: _torch_facts(snapshot),
            TorchDiagnosisFacts(
                importable=False,
                cuda_build=None,
                cuda_version=None,
                cuda_available=None,
                cudnn_version=None,
            ),
        )
        empty_selectors: tuple[ToolkitSelectorFact, ...] = ()
        selectors = _safe(lambda: _toolkit_selector_facts(snapshot), empty_selectors)
        return DiagnosisFacts(
            platform=snapshot.system.platform,
            toolkit_selectors=selectors,
            torch=torch_facts,
            gpu=_safe(
                lambda: _gpu_facts(snapshot), GPUDiagnosisFacts(visible=Trinary.UNKNOWN)
            ),
            driver=_safe(
                lambda: _driver_facts(snapshot, torch_facts, compatibility),
                DriverDiagnosisFacts(
                    version=None,
                    usable=Trinary.UNKNOWN,
                    compatible_with_torch_runtime=Trinary.UNKNOWN,
                ),
            ),
            visibility=_safe(
                lambda: _visibility_facts(snapshot),
                VisibilityFacts(
                    cuda_visible_devices=CUDAVisibleDevicesState.UNPARSEABLE,
                    explicit_all_gpu_mask=Trinary.UNKNOWN,
                ),
            ),
            runtime_libraries=_safe(
                lambda: _runtime_library_facts(snapshot.runtime_libraries, torch_facts),
                RuntimeLibraryFacts(
                    modeled_candidates=(),
                    family_counts=(),
                    observed_origins=frozenset(),
                    package_local_families=frozenset(),
                    torch_runtime_family=None,
                    package_local_cudnn_family=None,
                    scan_incomplete=False,
                ),
            ),
            issue_codes=frozenset(issue.code for issue in issues),
            reference_toolkit=_safe(lambda: _reference_toolkit(selectors), None),
            applicability=_safe(
                lambda: _applicability(snapshot.system.platform),
                PlatformApplicability(d001=False, d002=False, d003=False, d004=False),
            ),
        )
