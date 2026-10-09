"""Normalized environment models.

Collectors produce these dataclasses; checks and reporters consume them.
Unavailable information is represented with ``None`` / empty collections —
never with exceptions. A missing tool is a diagnostic fact, not an error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from cuda_doctor.core.enums import Platform
from cuda_doctor.utils.redact import redact_value

__all__ = [
    "CUDAInfo",
    "CUDAInstallation",
    "CompilerInfo",
    "DriverInfo",
    "EnvironmentInfo",
    "EnvironmentSnapshot",
    "GPUInfo",
    "NvidiaSmiInfo",
    "PyTorchInfo",
    "PythonInfo",
    "SystemInfo",
    "ToolInfo",
    "TorchDevice",
    "snapshot_to_dict",
]


@dataclass
class SystemInfo:
    """Operating system and Python runtime facts."""

    os_name: str
    os_version: str
    kernel_version: str
    architecture: str
    platform: Platform
    python_version: str
    python_executable: str


@dataclass
class GPUInfo:
    """One NVIDIA GPU as reported by nvidia-smi."""

    index: int
    name: str
    uuid: str | None = None
    memory_total_mb: int | None = None
    compute_capability: str | None = None


@dataclass
class DriverInfo:
    """NVIDIA driver facts.

    ``cuda_version`` is the CUDA version reported by nvidia-smi — the CUDA
    UMD generation the driver was validated with, not a hard limit on the
    CUDA applications it can run (see CUDA minor-version compatibility).
    """

    version: str | None = None
    cuda_version: str | None = None
    source: str = "nvidia-smi"


@dataclass
class CUDAInstallation:
    """A CUDA Toolkit directory discovered on disk."""

    path: str
    version: str | None = None  # derived from the directory name when possible


@dataclass
class CUDAInfo:
    """CUDA Toolkit facts."""

    nvcc_found: bool = False
    nvcc_path: str | None = None
    toolkit_version: str | None = None  # from `nvcc --version`
    cuda_home: str | None = None  # CUDA_HOME (or CUDA_PATH on Windows)
    cuda_path: str | None = None  # CUDA_PATH, when distinct from cuda_home
    cuda_home_exists: bool | None = None
    cuda_home_has_nvcc: bool | None = None
    installations: list[CUDAInstallation] = field(default_factory=list)
    windows_cuda_path_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class PythonInfo:
    """Python interpreter facts."""

    version: str
    executable: str
    in_virtual_env: bool = False
    virtual_env_path: str | None = None
    pip_version: str | None = None


@dataclass
class TorchDevice:
    """One GPU as enumerated through PyTorch."""

    index: int
    name: str | None = None
    compute_capability: str | None = None


@dataclass
class PyTorchInfo:
    """PyTorch facts; PyTorch itself is an optional dependency.

    Distinctions that matter downstream:
    - ``installed=False``            -> PyTorch not installed;
    - ``is_cuda_build=False``        -> CPU-only build;
    - ``is_cuda_build=True`` but ``cuda_available=False`` -> GPU build that
      cannot initialize CUDA (driver/runtime problem);
    - ``cuda_available=True``        -> CUDA working normally.
    """

    installed: bool = False
    version: str | None = None
    cuda_version: str | None = None  # torch.version.cuda; None on CPU builds
    is_cuda_build: bool | None = None
    cuda_available: bool | None = None
    device_count: int | None = None
    cudnn_version: str | None = None
    devices: list[TorchDevice] = field(default_factory=list)
    import_error: str | None = None  # non-ImportError failures carry a message


@dataclass
class ToolInfo:
    """A generic external tool (compiler, CMake, Ninja, ...)."""

    name: str
    found: bool = False
    path: str | None = None
    version: str | None = None


@dataclass
class CompilerInfo:
    """Host compiler toolchain facts."""

    compilers: list[ToolInfo] = field(default_factory=list)

    @property
    def any_found(self) -> bool:
        return any(tool.found for tool in self.compilers)


@dataclass
class EnvironmentInfo:
    """CUDA-relevant environment variable facts.

    PATH and LD_LIBRARY_PATH are stored as analyzed entries; reporters never
    dump them verbatim, only the CUDA-relevant subset.
    """

    variables: dict[str, str] = field(default_factory=dict)  # e.g. CUDA_HOME
    path_entries: list[tuple[int, str]] = field(default_factory=list)
    cuda_path_entries: list[tuple[int, str]] = field(default_factory=list)
    ld_library_path: str | None = None
    cuda_ld_library_entries: list[str] = field(default_factory=list)


@dataclass
class NvidiaSmiInfo:
    """Outcome of invoking nvidia-smi, kept for diagnostics/evidence."""

    available: bool = False  # executable could be located
    executed: bool = False  # at least one invocation returned
    error: str | None = None  # "executable-not-found" | "timeout" | ...
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None


@dataclass
class EnvironmentSnapshot:
    """The complete normalized view of the local environment."""

    system: SystemInfo
    gpus: list[GPUInfo] = field(default_factory=list)
    driver: DriverInfo | None = None
    nvidia_smi: NvidiaSmiInfo | None = None
    cuda: CUDAInfo = field(default_factory=CUDAInfo)
    python: PythonInfo | None = None
    pytorch: PyTorchInfo = field(default_factory=PyTorchInfo)
    compiler: CompilerInfo = field(default_factory=CompilerInfo)
    cmake: ToolInfo | None = None
    ninja: ToolInfo | None = None
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    collection_errors: dict[str, str] = field(default_factory=dict)
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def snapshot_to_dict(
    snapshot: EnvironmentSnapshot,
    *,
    redact: bool = True,
    home: str | None = None,
) -> dict[str, Any]:
    """Convert a snapshot into plain, JSON-ready data.

    With ``redact=True`` (default) all embedded strings are redacted using
    ``home`` (or the current home directory). With ``redact=False`` values are
    only normalized (enums to values, tuples to lists).

    The full ``PATH`` and raw ``LD_LIBRARY_PATH`` are never included: reports
    must stay shareable, so only the CUDA-relevant subsets analyzed in
    ``cuda_path_entries`` / ``cuda_ld_library_entries`` are emitted.
    """
    data = asdict(snapshot)
    environment = data.get("environment")
    if isinstance(environment, dict):
        environment.pop("path_entries", None)
        environment.pop("ld_library_path", None)
    if redact:
        return redact_value(data, home=home)
    return redact_value(data, home="")
