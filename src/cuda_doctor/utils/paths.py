"""Filesystem and PATH helpers for CUDA discovery."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import CUDAInstallation
from cuda_doctor.utils.versions import cuda_version_from_path, parse_cuda_version

LINUX_CUDA_ROOT_DIRS = ("/usr/local", "/opt")
WINDOWS_TOOLKIT_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
USR_LOCAL_CUDA = "/usr/local/cuda"


def find_executable(name: str) -> str | None:
    """Locate an executable on PATH (shutil.which wrapper; None when absent)."""
    return shutil.which(name)


def find_executable_on_path(name: str, path_entries: Sequence[str]) -> str | None:
    """Locate ``name`` in already-split PATH entries of an injected environment.

    Unlike :func:`find_executable` this never consults the host process
    environment, so collector tests simulating any target platform stay
    deterministic regardless of the machine running them. The first entry
    holding an executable ``name`` wins, matching PATH-resolution semantics.
    """
    for entry in path_entries:
        if not entry:
            continue
        candidate = os.path.join(entry, name)
        try:
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                return candidate
        except OSError:
            continue
    return None


def is_cuda_related(path_entry: str) -> bool:
    """True when a PATH / LD_LIBRARY_PATH entry looks CUDA-related."""
    if not path_entry:
        return False
    return "cuda" in path_entry.replace("\\", "/").lower()


def nvcc_within(cuda_home: str, nvcc_path: str) -> bool:
    """True when ``nvcc_path`` is the nvcc binary inside ``cuda_home``."""
    if not cuda_home or not nvcc_path:
        return False
    try:
        nvcc = Path(nvcc_path).resolve()
        home = Path(cuda_home).resolve()
    except OSError:
        return False
    return nvcc.parent.parent == home and nvcc.name.lower().startswith("nvcc")


def discover_cuda_installations(
    platform: Platform,
    env: Mapping[str, str] | None = None,
    roots: Sequence[str] = (),
) -> list[CUDAInstallation]:
    """Find standard CUDA Toolkit installations.

    On Linux: ``/usr/local/cuda*`` and ``/opt/cuda*``.
    On Windows: ``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*``
    plus any ``CUDA_PATH_V*`` environment variables.

    ``roots`` overrides the standard search roots (used by tests).
    """
    env = env or {}
    found: dict[str, CUDAInstallation] = {}

    def _add(path_str: str) -> None:
        path_str = os.path.normpath(path_str)
        if path_str in found:
            return
        version = cuda_version_from_path(path_str)
        found[path_str] = CUDAInstallation(
            path=path_str, version=str(version) if version else None
        )

    if platform is Platform.WINDOWS:
        search_roots = tuple(roots) or (WINDOWS_TOOLKIT_ROOT,)
        for root in search_roots:
            root_path = Path(root)
            if root_path.is_dir():
                for entry in sorted(root_path.glob("v*")):
                    if entry.is_dir():
                        _add(str(entry))
        for key, value in sorted(env.items()):
            if key.upper().startswith("CUDA_PATH_V") and value:
                _add(value)
    else:
        search_roots = tuple(roots) or LINUX_CUDA_ROOT_DIRS
        for root in search_roots:
            root_path = Path(root)
            if not root_path.is_dir():
                continue
            for entry in sorted(root_path.glob("cuda*")):
                if not entry.is_dir():
                    continue
                version = cuda_version_from_path(entry.name)
                if version is None and entry.name.lower() != "cuda":
                    continue  # e.g. /usr/local/cudnn
                _add(str(entry))

    def _sort_key(install: CUDAInstallation) -> tuple[tuple[int, int], str]:
        # ``install.version`` is a bare "major.minor" string (see _add above),
        # so parse it directly — the path-oriented regex would never match it
        # and every install would sort as (0, 0), i.e. by path text.
        version = parse_cuda_version(install.version) or (0, 0)
        return (version, install.path)

    return sorted(found.values(), key=_sort_key, reverse=True)
