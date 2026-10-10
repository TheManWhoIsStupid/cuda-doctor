"""Bounded CUDA runtime-library inventory collector (v0.2, D004 input).

Observation only: this collector records *where* runtime-library files of
the supported families exist within a small set of approved roots. It never
decides which library would actually load, whether candidates conflict, or
whether any diagnosis applies — that is diagnosis-layer work (frozen
architecture §10.6). D004 is Linux-only, so other targets get an empty,
not-applicable inventory instead of an error (truth tables §19.1).
"""

from __future__ import annotations

import os
import re
import site
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import (
    CUDAInfo,
    EnvironmentInfo,
    RuntimeLibraryCandidate,
    RuntimeLibraryInventory,
)
from cuda_doctor.utils.commands import CommandRunner
from cuda_doctor.utils.platform import current_platform

# Frozen family scope (truth tables §19.2): no other family may enter the
# v0.2 inventory without a reviewed spec revision.
SUPPORTED_FAMILIES = ("libcudart", "libcublas", "libcudnn")

# Neutral origin vocabulary (architecture §10.4): ``toolkit``, never
# ``active_toolkit`` — which toolkit is active is derived later by the
# diagnosis fact builder, never observed here.
ORIGIN_LD_LIBRARY_PATH = "ld_library_path"
ORIGIN_TOOLKIT = "toolkit"
ORIGIN_PYTHON_PACKAGE = "python_package"
ORIGIN_CONDA_PREFIX = "conda_prefix"
ORIGIN_LDCONFIG = "ldconfig"
ORIGINS = (
    ORIGIN_LD_LIBRARY_PATH,
    ORIGIN_TOOLKIT,
    ORIGIN_PYTHON_PACKAGE,
    ORIGIN_CONDA_PREFIX,
    ORIGIN_LDCONFIG,
)

# Conservative filename-based version sources (architecture §10.3): no ELF
# parsing, no per-library subprocess execution, no invented version mappings.
VERSION_SOURCE_FILENAME = "FILENAME"
VERSION_SOURCE_SYMLINK_TARGET = "SYMLINK_TARGET"
VERSION_SOURCE_UNKNOWN = "UNKNOWN"

# Exactly one read-only cache query through the injectable CommandRunner.
LDCONFIG_COMMAND = ("ldconfig", "-p")

# ``libcudart.so`` / ``libcudart.so.12`` / ``libcudart.so.12.4.127`` — the
# version tail must be dotted digits; anything else (``libcudart.so.bak``,
# ``libcublasLt.so.12``, ``libcudart_static.a``) is not a family candidate.
_LIBRARY_FILE_RE = re.compile(
    r"^(libcudart|libcublas|libcudnn)\.so(?:\.(\d+(?:\.\d+)*))?$"
)
# ``libcudart.so.12 (libc6,x86-64) => /usr/local/cuda-12.4/lib64/libcudart.so.12``
_LDCONFIG_LINE_RE = re.compile(r"^\s*(\S+)\s*\([^)]*\)\s*=>\s*(\S+)")


class RuntimeLibraryCollector:
    """Inventories CUDA runtime-library candidates from bounded roots only."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        environment: EnvironmentInfo | None = None,
        platform: Platform | None = None,
        toolkit_roots: Sequence[str] = (),
        site_packages: Sequence[str] | None = None,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.environment = environment if environment is not None else EnvironmentInfo()
        self.platform = platform or current_platform()
        # Roots already resolved by CUDA collection (architecture §10.3):
        # the collector never rediscovers installations itself.
        self.toolkit_roots = tuple(toolkit_roots)
        # Overridable so tests never scan the real interpreter's packages.
        self.site_packages = (
            tuple(site_packages) if site_packages is not None else _default_site_packages()
        )

    def collect(self) -> RuntimeLibraryInventory:
        if self.platform is not Platform.LINUX:
            # D004 platform gate: not evaluated, not an error. POSIX library
            # semantics must never run against a Windows/macOS target.
            return RuntimeLibraryInventory()
        inventory = RuntimeLibraryInventory()
        scanned: set[tuple[str, str, int | None]] = set()

        # 1. LD_LIBRARY_PATH — the one modeled explicit ordering source
        #    (Phase-1 ordered representation; source order preserved).
        for order, entry in self.environment.ld_library_path_entries:
            self._scan_directory(
                inventory,
                entry,
                origin=ORIGIN_LD_LIBRARY_PATH,
                search_group=ORIGIN_LD_LIBRARY_PATH,
                search_order=order,
                scanned=scanned,
                explicit_root=True,
            )

        # 2. Observed toolkit library directories (origin stays neutral).
        for root in self.toolkit_roots:
            for lib_dir in _toolkit_lib_dirs(root):
                self._scan_directory(
                    inventory,
                    str(lib_dir),
                    origin=ORIGIN_TOOLKIT,
                    search_group=ORIGIN_TOOLKIT,
                    search_order=None,
                    scanned=scanned,
                    explicit_root=False,
                )

        # 3. NVIDIA package libraries of the CURRENT Python environment only.
        for lib_dir in self._python_package_lib_dirs(inventory):
            self._scan_directory(
                inventory,
                str(lib_dir),
                origin=ORIGIN_PYTHON_PACKAGE,
                search_group=ORIGIN_PYTHON_PACKAGE,
                search_order=None,
                scanned=scanned,
                explicit_root=False,
            )

        # 4. Current conda environment lib, when one is established.
        conda_prefix = self.environment.variables.get("CONDA_PREFIX")
        if conda_prefix:
            self._scan_directory(
                inventory,
                os.path.join(conda_prefix, "lib"),
                origin=ORIGIN_CONDA_PREFIX,
                search_group=ORIGIN_CONDA_PREFIX,
                search_order=None,
                scanned=scanned,
                explicit_root=False,
            )

        # 5. Single ldconfig cache query — inventory only, no search order.
        self._collect_ldconfig(inventory)
        return inventory

    # -- filesystem inventory -------------------------------------------------

    def _scan_directory(
        self,
        inventory: RuntimeLibraryInventory,
        directory: str,
        *,
        origin: str,
        search_group: str,
        search_order: int | None,
        scanned: set[tuple[str, str, int | None]],
        explicit_root: bool,
    ) -> None:
        """Record family candidates found directly inside ``directory``.

        ``explicit_root`` marks user-named roots (LD_LIBRARY_PATH entries):
        scanning them directly records a stale/inaccessible entry as scan
        evidence. Structural sub-directories (toolkit ``lib``/``lib64``,
        package ``lib``, conda ``lib``) are probed instead — absence simply
        means no candidates there.
        """
        key = (search_group, directory, search_order)
        if key in scanned:
            return
        scanned.add(key)
        if not explicit_root:
            try:
                if not Path(directory).is_dir():
                    return
            except OSError:
                return
        try:
            names = sorted(os.listdir(directory))
        except OSError as exc:
            inventory.scan_errors[directory] = f"{type(exc).__name__}: {exc}"
            return
        for name in names:
            family = _family_for_filename(name)
            if family is None:
                continue
            self._add_file_candidate(
                inventory,
                os.path.join(directory, name),
                family,
                origin=origin,
                search_group=search_group,
                search_order=search_order,
            )

    def _add_file_candidate(
        self,
        inventory: RuntimeLibraryInventory,
        path: str,
        family: str,
        *,
        origin: str,
        search_group: str,
        search_order: int | None,
    ) -> None:
        try:
            if Path(path).is_dir():
                return  # a directory named like a library is not a file
        except OSError:
            return
        canonical = _canonicalize(path)
        # Version from the observed filename first; only when that carries
        # none may the symlink target's filename contribute (architecture
        # §10.3) — never more precision than a filename actually shows.
        version = _version_from_filename(path)
        if version is not None:
            source = VERSION_SOURCE_FILENAME
        else:
            version = _version_from_filename(canonical) if canonical else None
            source = (
                VERSION_SOURCE_SYMLINK_TARGET if version is not None
                else VERSION_SOURCE_UNKNOWN
            )
        inventory.candidates.append(
            RuntimeLibraryCandidate(
                family=family,
                path=path,
                canonical_path=canonical,
                soname=None,  # reliable only from ldconfig output (§11.6)
                version=version,
                version_source=source,
                origin=origin,
                search_group=search_group,
                search_order=search_order,
            )
        )

    def _python_package_lib_dirs(self, inventory: RuntimeLibraryInventory) -> list[Path]:
        """``<site-packages>/nvidia/<package>/lib`` directories, bounded.

        The pip ``nvidia`` namespace layout is the approved current-Python
        NVIDIA package root (architecture §11). One glob level only — no
        other Python installations, no unrelated virtualenvs.
        """
        dirs: list[Path] = []
        for site_dir in self.site_packages:
            nvidia = Path(site_dir) / "nvidia"
            try:
                if not nvidia.is_dir():
                    continue
                packages = sorted(nvidia.iterdir())
            except OSError as exc:
                inventory.scan_errors[str(nvidia)] = f"{type(exc).__name__}: {exc}"
                continue
            for package in packages:
                dirs.append(package / "lib")
        return dirs

    # -- ldconfig inventory ---------------------------------------------------

    def _collect_ldconfig(self, inventory: RuntimeLibraryInventory) -> None:
        """One ``ldconfig -p`` cache query; supported families only.

        ldconfig entries are inventory only: no ``search_order`` is invented
        (architecture §10.5) — the cache carries no loader-order semantics
        this model may claim. A failed query is a bounded scan error and
        never discards candidates from other roots.
        """
        result = self.runner.run(LDCONFIG_COMMAND)
        if not result.success:
            detail = result.error or f"exit {result.return_code}"
            inventory.scan_errors[ORIGIN_LDCONFIG] = f"ldconfig -p failed: {detail}"
            return
        seen: set[tuple[str, str]] = set()
        for line in result.stdout.splitlines():
            match = _LDCONFIG_LINE_RE.match(line)
            if not match:
                continue
            soname, path = match.group(1), match.group(2)
            family = _family_for_filename(soname)
            if family is None or (soname, path) in seen:
                continue
            seen.add((soname, path))
            version = _version_from_filename(path)
            if version is None:
                version = _version_from_filename(soname)
            inventory.candidates.append(
                RuntimeLibraryCandidate(
                    family=family,
                    path=path,
                    canonical_path=_canonicalize(path),
                    soname=soname,  # the cache prints the recorded SONAME
                    version=version,
                    version_source=(
                        VERSION_SOURCE_FILENAME if version is not None
                        else VERSION_SOURCE_UNKNOWN
                    ),
                    origin=ORIGIN_LDCONFIG,
                    search_group=ORIGIN_LDCONFIG,
                    search_order=None,
                )
            )


def toolkit_library_roots(cuda: CUDAInfo | None) -> tuple[str, ...]:
    """Toolkit roots already resolved during CUDA collection (§10.3).

    Canonical roots of the resolved selector observations plus the
    discovered installations list. The collector receives these roots — it
    never decides which toolkit is *active*; that is fact-layer work.
    """
    if cuda is None:
        return ()
    roots: list[str] = []

    def _add(root: str | None) -> None:
        if root:
            normalized = os.path.normpath(root)
            if normalized not in roots:
                roots.append(normalized)

    for observation in cuda.selector_observations:
        _add(observation.canonical_root)
    for installation in cuda.installations:
        _add(installation.path)
    return tuple(roots)


def _toolkit_lib_dirs(root: str) -> list[Path]:
    """Bounded toolkit library sub-directories: ``lib``, ``lib64``, and
    ``targets/<arch>/lib`` (one glob level, architecture §10.3)."""
    base = Path(root)
    dirs = [base / "lib", base / "lib64"]
    targets = base / "targets"
    try:
        if targets.is_dir():
            dirs.extend(arch / "lib" for arch in sorted(targets.iterdir()) if arch.is_dir())
    except OSError:
        pass  # an unreadable targets/ tree adds no candidates
    return dirs


def _default_site_packages() -> tuple[str, ...]:
    """Site-packages roots of the *current* interpreter only (§11).

    Besides ``site.getsitepackages()`` this includes the interpreter's
    ACTIVE per-user site (``pip install --user``) — part of the current
    environment, not a different one — but only when the interpreter
    reports user-site packages enabled (``site.ENABLE_USER_SITE is
    True``). Never other users' sites, other installations, or generic
    ``sys.path``/PYTHONPATH discovery. Roots are deduplicated with the
    system order first; any failure of either source degrades to fewer
    roots instead of breaking runtime-library collection.
    """
    roots: list[str] = []
    with suppress(Exception):  # pragma: no cover - unusual interpreter setups
        roots.extend(site.getsitepackages())
    if site.ENABLE_USER_SITE is True:
        with suppress(Exception):  # an unreachable user-site adds no root
            roots.append(site.getusersitepackages())
    return tuple(dict.fromkeys(roots))


def _family_for_filename(name: str) -> str | None:
    """The supported family of an exact library filename, else ``None``."""
    match = _LIBRARY_FILE_RE.match(name)
    return match.group(1) if match else None


def _version_from_filename(path: str | None) -> str | None:
    """The dotted-digits version tail of a family filename, else ``None``.

    Never more precise than the filename: ``libcudart.so.12`` yields "12",
    a bare ``libcudart.so`` yields ``None`` (UNKNOWN).
    """
    if not path:
        return None
    match = _LIBRARY_FILE_RE.match(os.path.basename(str(path)))
    return match.group(2) if match and match.group(2) else None


def _canonicalize(path: str) -> str | None:
    """Symlink-resolved absolute form; ``None`` when resolution fails."""
    try:
        return str(Path(path).resolve(strict=False))
    except (OSError, ValueError, RuntimeError):
        return None
