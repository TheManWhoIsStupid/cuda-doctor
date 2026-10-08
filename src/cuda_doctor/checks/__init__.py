"""Diagnostic checks: the complete rule registry."""

from __future__ import annotations

from cuda_doctor.checks.base import Check
from cuda_doctor.checks.compiler_checks import CompilerCudaCompatibility, NoHostCompiler
from cuda_doctor.checks.cuda_checks import (
    CudaHomeInvalid,
    CudaHomeNotSet,
    MultipleCudaBinsInPath,
    MultipleToolkits,
    NvccDiffersFromCudaHome,
    NvccNotFound,
)
from cuda_doctor.checks.driver_checks import DriverRuntimeCompatibility, DriverUndetermined
from cuda_doctor.checks.environment_checks import (
    ConflictingLdLibraryPaths,
    DuplicateCudaPaths,
    InvalidCudaPaths,
    OlderCudaShadowsNewer,
)
from cuda_doctor.checks.gpu_checks import GpuSmiUnavailable, NoNvidiaGpu, NvidiaSmiFailed
from cuda_doctor.checks.pytorch_checks import (
    PyTorchCannotEnumerate,
    PyTorchCpuOnlyBuild,
    PyTorchCudaUnavailable,
    PyTorchImportFailed,
    PyTorchNewerThanDriver,
    PyTorchNotInstalled,
    PyTorchRuntimeDiffers,
)

__all__ = ["ALL_CHECKS", "Check", "default_checks"]

ALL_CHECKS: list[type[Check]] = [
    # GPU
    GpuSmiUnavailable,
    NoNvidiaGpu,
    NvidiaSmiFailed,
    # Driver
    DriverUndetermined,
    DriverRuntimeCompatibility,
    # CUDA
    NvccNotFound,
    CudaHomeNotSet,
    CudaHomeInvalid,
    MultipleToolkits,
    MultipleCudaBinsInPath,
    NvccDiffersFromCudaHome,
    # PyTorch
    PyTorchNotInstalled,
    PyTorchImportFailed,
    PyTorchCudaUnavailable,
    PyTorchCpuOnlyBuild,
    PyTorchRuntimeDiffers,
    PyTorchNewerThanDriver,
    PyTorchCannotEnumerate,
    # Compiler
    NoHostCompiler,
    CompilerCudaCompatibility,
    # Environment
    InvalidCudaPaths,
    DuplicateCudaPaths,
    ConflictingLdLibraryPaths,
    OlderCudaShadowsNewer,
]


def default_checks() -> list[Check]:
    """Fresh instances of every registered check, in stable order."""
    return [check_class() for check_class in ALL_CHECKS]
