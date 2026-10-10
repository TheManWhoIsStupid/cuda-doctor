"""CUDA Toolkit collector."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import CUDAInfo, CUDASelectorObservation
from cuda_doctor.utils.commands import CommandRunner
from cuda_doctor.utils.parsing import parse_nvcc_version
from cuda_doctor.utils.paths import (
    USR_LOCAL_CUDA,
    discover_cuda_installations,
    find_executable_on_path,
)
from cuda_doctor.utils.platform import current_platform
from cuda_doctor.utils.versions import cuda_version_from_path

# Frozen architecture §9.5 version-source vocabulary. The collector emits
# only these values; how they map to identity reliability is decided later
# by the diagnosis fact layer.
VERSION_SOURCE_DIRECT = "DIRECT"
VERSION_SOURCE_DERIVED_STRONG = "DERIVED_STRONG"
VERSION_SOURCE_DERIVED_WEAK = "DERIVED_WEAK"
VERSION_SOURCE_UNKNOWN = "UNKNOWN"


class CUDACollector:
    """Collects nvcc, CUDA_HOME/CUDA_PATH, and on-disk toolkit installations."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        env: Mapping[str, str] | None = None,
        roots: Sequence[str] = (),
        platform: Platform | None = None,
        usr_local_cuda_path: str = USR_LOCAL_CUDA,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.env: dict[str, str] = dict(env if env is not None else os.environ)
        self.roots = tuple(roots)
        self.platform = platform or current_platform()
        # Overridable like ``roots`` so tests never depend on the host's
        # real /usr/local/cuda (read-only observation either way).
        self.usr_local_cuda_path = usr_local_cuda_path
        self._version_probes: dict[str, str | None] = {}

    def collect(self) -> CUDAInfo:
        info = CUDAInfo()
        lookup = self._env_lookup()
        path_entries = tuple(lookup.get("PATH", "").split(self._pathsep()))
        nvcc_binary = self._nvcc_binary_name()
        # PATH resolution runs against the injected environment only (frozen
        # architecture §9.5 binding rule 1) — never the host's PATH.
        nvcc_resolved = find_executable_on_path(nvcc_binary, path_entries)
        if nvcc_resolved:
            info.nvcc_found = True
            info.nvcc_path = nvcc_resolved
            version = self._probe_toolkit_version(nvcc_resolved)
            if version:
                info.toolkit_version = version
        self._collect_env_vars(info, lookup)
        self._collect_home_facts(info)
        info.installations = discover_cuda_installations(
            self.platform, env=self.env, roots=self.roots
        )
        info.selector_observations = self._collect_selector_observations(
            lookup, path_entries, nvcc_binary, nvcc_resolved
        )
        return info

    # -- selector observations (frozen architecture §9.5) --------------------

    def _collect_selector_observations(
        self,
        lookup: Mapping[str, str],
        path_entries: tuple[str, ...],
        nvcc_binary: str,
        nvcc_resolved: str | None,
    ) -> list[CUDASelectorObservation]:
        """Observe the CUDA toolkit selectors — facts only, never policy.

        Whether a selector is active, conflicting or diagnosis-relevant is
        decided by the diagnosis fact layer from these observations.
        """
        observations: list[CUDASelectorObservation] = []

        # cuda_home: Windows keeps the v0.1.x folding (CUDA_HOME or CUDA_PATH).
        if self.platform is Platform.WINDOWS:
            cuda_home_raw = lookup.get("CUDA_HOME") or lookup.get("CUDA_PATH")
        else:
            cuda_home_raw = lookup.get("CUDA_HOME")
        if cuda_home_raw:
            observations.append(self._directory_observation("cuda_home", cuda_home_raw))
        # cuda_path is recorded on both platforms; on Linux it is simply not
        # treated as an active D001 selector by the fact layer.
        cuda_path_raw = lookup.get("CUDA_PATH")
        if cuda_path_raw:
            observations.append(self._directory_observation("cuda_path", cuda_path_raw))

        cudacxx_raw = lookup.get("CUDACXX")
        if cudacxx_raw:
            observations.append(self._cudacxx_observation(cudacxx_raw, path_entries))

        if nvcc_resolved:
            nvcc_observation = self._nvcc_observation(nvcc_binary, nvcc_resolved)
            observations.append(nvcc_observation)
            winning_entry = self._winning_path_entry(nvcc_binary, nvcc_resolved, path_entries)
            if winning_entry is not None:
                observations.append(
                    self._path_cuda_bin_observation(winning_entry, nvcc_observation)
                )

        if self.platform is not Platform.WINDOWS:
            observations.append(self._usr_local_cuda_observation())

        return observations

    def _directory_observation(self, name: str, raw: str) -> CUDASelectorObservation:
        """Observe a directory selector (cuda_home / cuda_path).

        Version derivation order per §9.5: ``<home>/bin/nvcc --version`` when
        present, else canonical-root naming. (Toolkit metadata would sit
        between them, but v0.2 has no toolkit version file to read, so that
        step is intentionally unreachable — never guessed.)
        """
        exists, valid, canonical_root = self._directory_facts(raw)
        version: str | None = None
        source = VERSION_SOURCE_UNKNOWN
        if valid:
            home_nvcc = Path(raw) / "bin" / self._nvcc_binary_name()
            if self._is_executable_file(home_nvcc):
                version = self._probe_toolkit_version(str(home_nvcc))
                if version:
                    source = VERSION_SOURCE_DIRECT
        if version is None and canonical_root:
            derived = cuda_version_from_path(canonical_root)
            if derived:
                version, source = str(derived), VERSION_SOURCE_DERIVED_WEAK
        return CUDASelectorObservation(
            name=name,
            raw_value=raw,
            canonical_root=canonical_root,
            toolkit_version=version,
            exists=exists,
            valid=valid,
            version_source=source,
        )

    def _cudacxx_observation(
        self, raw: str, path_entries: tuple[str, ...]
    ) -> CUDASelectorObservation:
        """Observe the CUDACXX selector.

        Absolute/path-like values are used as-is; a bare name is resolved
        against the injected PATH. Per §9.5 the version comes only from the
        executable ``--version`` probe — never from ``cuda-12.4``-style
        naming, which is reserved for directory selectors.
        """
        if "/" in raw or "\\" in raw:
            resolved: str | None = raw
        else:
            resolved = find_executable_on_path(raw, path_entries)
        exists = False
        valid = False
        canonical_path: str | None = None
        canonical_root: str | None = None
        version: str | None = None
        source = VERSION_SOURCE_UNKNOWN
        if resolved:
            path = Path(resolved)
            try:
                exists = path.is_file()
            except OSError:
                exists = False
            valid = self._is_executable_file(path)
            if exists:
                canonical_path = self._canonicalize(resolved)
                if canonical_path:
                    canonical_root = str(Path(canonical_path).parent.parent)
            if valid:
                # Probe only real executables; an unparseable or failing
                # probe yields UNKNOWN — never a root-naming fallback.
                version = self._probe_toolkit_version(resolved)
                if version:
                    source = VERSION_SOURCE_DIRECT
        return CUDASelectorObservation(
            name="cudacxx",
            raw_value=raw,
            resolved_path=resolved,
            canonical_path=canonical_path,
            canonical_root=canonical_root,
            toolkit_version=version,
            exists=exists,
            valid=valid,
            version_source=source,
        )

    def _nvcc_observation(self, binary_name: str, resolved: str) -> CUDASelectorObservation:
        """Observe the PATH-resolved nvcc; canonical root is parent.parent."""
        canonical_path = self._canonicalize(resolved)
        canonical_root = str(Path(canonical_path).parent.parent) if canonical_path else None
        executable = self._is_executable_file(Path(resolved))
        version = self._probe_toolkit_version(resolved)
        return CUDASelectorObservation(
            name="nvcc",
            raw_value=binary_name,
            resolved_path=resolved,
            canonical_path=canonical_path,
            canonical_root=canonical_root,
            toolkit_version=version,
            exists=executable,
            valid=executable,
            version_source=VERSION_SOURCE_DIRECT if version else VERSION_SOURCE_UNKNOWN,
        )

    def _path_cuda_bin_observation(
        self, winning_entry: str, nvcc_observation: CUDASelectorObservation
    ) -> CUDASelectorObservation:
        """Observe the single PATH entry that supplied the resolved nvcc.

        Only the winning entry is observed (frozen truth tables §16.10);
        other CUDA-looking PATH entries are incidental and never recorded
        here. Version facts are inherited from the resolved nvcc.
        """
        try:
            entry_exists = Path(winning_entry).is_dir()
        except OSError:
            entry_exists = False
        return CUDASelectorObservation(
            name="path_cuda_bin",
            raw_value=winning_entry,
            resolved_path=nvcc_observation.resolved_path,
            canonical_path=nvcc_observation.canonical_path,
            canonical_root=nvcc_observation.canonical_root,
            toolkit_version=nvcc_observation.toolkit_version,
            exists=entry_exists,
            valid=entry_exists and nvcc_observation.valid is True,
            version_source=nvcc_observation.version_source,
        )

    def _usr_local_cuda_observation(self) -> CUDASelectorObservation:
        """Observe /usr/local/cuda as inventory only.

        Existence, symlink target and canonical root are recorded; whether
        the symlink participates in active resolution is computed later by
        the fact builder (frozen truth tables §20.4). A symlink target is
        informative even when stale/dangling (§16.9), so it is resolved
        regardless of target existence.
        """
        raw = self.usr_local_cuda_path
        path = Path(raw)
        try:
            exists: bool | None = path.exists()
            is_symlink = path.is_symlink()
            is_dir: bool | None = path.is_dir()
        except OSError:
            exists = None
            is_symlink = False
            is_dir = None
        canonical_root: str | None = None
        if is_symlink or exists:
            canonical_root = self._canonicalize(raw)
        version: str | None = None
        source = VERSION_SOURCE_UNKNOWN
        if canonical_root:
            derived = cuda_version_from_path(canonical_root)
            if derived:
                version, source = str(derived), VERSION_SOURCE_DERIVED_WEAK
        valid: bool | None = None
        if exists is not None and is_dir is not None:
            valid = exists and is_dir
        return CUDASelectorObservation(
            name="usr_local_cuda",
            raw_value=raw,
            canonical_root=canonical_root,
            toolkit_version=version,
            exists=exists,
            valid=valid,
            version_source=source,
        )

    # -- shared observation helpers -------------------------------------------

    def _directory_facts(self, raw: str) -> tuple[bool | None, bool | None, str | None]:
        """(exists, is-a-directory, canonical root) for a directory selector.

        The canonical root is only recorded for an existing directory: a
        missing or non-directory selector has no toolkit-root identity, and
        resolving nonexistent paths would be host-cwd-dependent noise.
        """
        path = Path(raw)
        try:
            exists = path.exists()
            is_dir = path.is_dir()
        except OSError:
            # Unstattable paths stay UNKNOWN rather than "absent".
            return None, None, None
        if not exists or not is_dir:
            return exists, is_dir if exists else False, None
        return True, True, self._canonicalize(raw)

    def _is_executable_file(self, path: Path) -> bool:
        try:
            return path.is_file() and os.access(path, os.X_OK)
        except OSError:
            return False

    def _canonicalize(self, path_str: str) -> str | None:
        """Symlink-resolved absolute form; None when resolution fails."""
        try:
            return str(Path(path_str).resolve(strict=False))
        except (OSError, ValueError, RuntimeError):
            return None

    def _winning_path_entry(
        self, binary_name: str, resolved: str, path_entries: tuple[str, ...]
    ) -> str | None:
        """The first PATH entry whose ``join(entry, binary)`` is the resolved path."""
        target = os.path.normpath(resolved)
        for entry in path_entries:
            if not entry:
                continue
            if os.path.normpath(os.path.join(entry, binary_name)) == target:
                return entry
        return None

    def _probe_toolkit_version(self, executable: str) -> str | None:
        """Read-only ``--version`` probe, memoized per canonical executable.

        One nvcc binary is often reachable through several selectors (PATH
        entry, CUDA_HOME/bin, symlink aliases); each distinct binary is
        probed at most once and every observation agrees.
        """
        key = self._canonicalize(executable) or executable
        if key not in self._version_probes:
            result = self.runner.run((executable, "--version"))
            parsed = parse_nvcc_version(result.stdout) if result.success else None
            self._version_probes[key] = parsed or None
        return self._version_probes[key]

    # -- v0.1.x field collection (unchanged semantics) -------------------------

    def _env_lookup(self) -> Mapping[str, str]:
        # Case-insensitive lookups are correct on Windows only (see
        # EnvironmentCollector for the Linux rationale).
        if self.platform is Platform.WINDOWS:
            return {key.upper(): value for key, value in self.env.items()}
        return self.env

    def _nvcc_binary_name(self) -> str:
        return "nvcc.exe" if self.platform is Platform.WINDOWS else "nvcc"

    def _pathsep(self) -> str:
        """PATH separator for the *target* platform, never the host's."""
        return ";" if self.platform is Platform.WINDOWS else ":"

    def _collect_env_vars(self, info: CUDAInfo, lookup: Mapping[str, str]) -> None:
        info.windows_cuda_path_vars = {
            key: value
            for key, value in lookup.items()
            if key.startswith("CUDA_PATH_V") and value
        }
        if self.platform is Platform.WINDOWS:
            info.cuda_path = lookup.get("CUDA_PATH") or None
            info.cuda_home = lookup.get("CUDA_HOME") or info.cuda_path
        else:
            info.cuda_home = lookup.get("CUDA_HOME") or None
            info.cuda_path = lookup.get("CUDA_PATH") or None

    def _collect_home_facts(self, info: CUDAInfo) -> None:
        home = info.cuda_home
        if not home:
            return
        home_path = Path(home)
        try:
            info.cuda_home_exists = home_path.is_dir()
            nvcc_binary = "nvcc.exe" if self.platform is Platform.WINDOWS else "nvcc"
            info.cuda_home_has_nvcc = (home_path / "bin" / nvcc_binary).is_file()
        except OSError:
            # Inaccessible paths are simply reported as non-existent.
            info.cuda_home_exists = False
            info.cuda_home_has_nvcc = False
