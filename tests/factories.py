"""Snapshot factories shared across check/reporter tests."""

from __future__ import annotations

from cuda_doctor.compatibility import load_compatibility_data
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import (
    CompilerInfo,
    CUDAInfo,
    CUDAInstallation,
    DriverInfo,
    EnvironmentInfo,
    EnvironmentSnapshot,
    GPUInfo,
    NvidiaSmiInfo,
    PythonInfo,
    PyTorchInfo,
    SystemInfo,
    ToolInfo,
    TorchDevice,
)


def base_snapshot(**overrides) -> EnvironmentSnapshot:
    """A healthy Linux / GPU / CUDA 12.4 / torch environment.

    Any top-level snapshot field can be overridden via kwargs, e.g.
    ``base_snapshot(driver=None)``.
    """
    snapshot = EnvironmentSnapshot(
        system=SystemInfo(
            os_name="Linux",
            os_version="Ubuntu 22.04.5 LTS",
            kernel_version="5.15.0",
            architecture="x86_64",
            platform=Platform.LINUX,
            python_version="3.11.15",
            python_executable="/home/tester/venv/bin/python",
        ),
        gpus=[GPUInfo(0, "NVIDIA H20-3e", "GPU-abc", 143771, "9.0")],
        driver=DriverInfo("580.126.09", "13.0"),
        nvidia_smi=NvidiaSmiInfo(available=True, executed=True),
        cuda=CUDAInfo(
            nvcc_found=True,
            nvcc_path="/usr/local/cuda-12.4/bin/nvcc",
            toolkit_version="12.4",
            cuda_home="/usr/local/cuda-12.4",
            cuda_home_exists=True,
            cuda_home_has_nvcc=True,
            installations=[CUDAInstallation("/usr/local/cuda-12.4", "12.4")],
        ),
        python=PythonInfo("3.11.15", "/home/tester/venv/bin/python", True),
        pytorch=PyTorchInfo(
            installed=True,
            version="2.6.0+cu124",
            cuda_version="12.4",
            is_cuda_build=True,
            cuda_available=True,
            device_count=1,
            cudnn_version="90100",
            devices=[TorchDevice(0, "NVIDIA H20-3e", "9.0")],
        ),
        compiler=CompilerInfo(
            compilers=[
                ToolInfo("gcc", True, "/usr/bin/gcc", "11.4.0"),
                ToolInfo("g++", True, "/usr/bin/g++", "11.4.0"),
            ]
        ),
        cmake=ToolInfo("cmake", True, "/usr/bin/cmake", "3.28.3"),
        ninja=ToolInfo("ninja", True, "/usr/bin/ninja", "1.11.1"),
        environment=EnvironmentInfo(
            variables={"CUDA_HOME": "/usr/local/cuda-12.4"},
            path_entries=[(0, "/usr/bin")],
            # Deliberately empty: ENV001 judges entry existence on the real
            # filesystem, so the shared "healthy" snapshot must stay neutral.
            cuda_path_entries=[],
        ),
    )
    for key, value in overrides.items():
        setattr(snapshot, key, value)
    return snapshot


def context_for(snapshot: EnvironmentSnapshot) -> DiagnosticContext:
    """Diagnostic context with the real bundled compatibility data."""
    return DiagnosticContext(snapshot=snapshot, compatibility=load_compatibility_data())
