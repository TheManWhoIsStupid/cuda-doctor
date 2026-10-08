"""PyTorch checks.

The torch/CUDA relationship is subtle, and these checks encode the product's
most important nuance: a local-toolkit/runtime version difference is normally
harmless (TORCH004 stays INFO), while a runtime *newer than the driver
supports* is a real error (TORCH006).
"""

from __future__ import annotations

from cuda_doctor.checks.base import Check
from cuda_doctor.compatibility import interpret_torch_cuda
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.recommendations import recommendations_for
from cuda_doctor.utils.versions import (
    compare_versions,
    parse_cuda_version,
)


class PyTorchNotInstalled(Check):
    """TORCH001: no torch importable (informational, not an error)."""

    code = "TORCH001"
    category = "pytorch"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        torch_info = ctx.snapshot.pytorch
        if torch_info.installed or torch_info.import_error:
            return []
        return [
            self.issue(
                severity=Severity.INFO,
                title="PyTorch not installed",
                description=(
                    "No PyTorch installation was detected in this Python "
                    "environment. If that is unexpected, install it."
                ),
                evidence=["import torch raised ImportError"],
                recommendations=recommendations_for(self.code),
            )
        ]


class PyTorchImportFailed(Check):
    """TORCH005 (import aspect): torch exists but cannot be imported."""

    code = "TORCH005"
    category = "pytorch"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        torch_info = ctx.snapshot.pytorch
        if not torch_info.import_error:
            return []
        recommendations = recommendations_for(self.code)
        known = None
        if ctx.compatibility is not None:
            known = ctx.compatibility.known_issues.match("torch_import", torch_info.import_error)
        if known is not None:
            recommendations = [f"{known.title}: {known.advice}", *recommendations]
        title = "PyTorch cannot be imported"
        if known is not None:
            title = f"PyTorch cannot be imported — {known.title}"
        return [
            self.issue(
                severity=Severity.ERROR,
                title=title,
                description=(
                    "Importing torch raised an exception. The environment is broken; "
                    "the error text usually identifies the missing library."
                ),
                evidence=[torch_info.import_error],
                recommendations=recommendations,
            )
        ]


class PyTorchCudaUnavailable(Check):
    """TORCH002: CUDA build but torch.cuda.is_available() is False."""

    code = "TORCH002"
    category = "pytorch"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        torch_info = ctx.snapshot.pytorch
        if not torch_info.installed or not torch_info.is_cuda_build:
            return []
        if torch_info.cuda_available is not False:
            return []
        evidence = [
            f"torch {torch_info.version} (built for CUDA {torch_info.cuda_version})",
            "torch.cuda.is_available() -> False",
        ]
        driver = ctx.snapshot.driver
        if driver is not None and driver.cuda_version:
            evidence.append(f"driver supports at most CUDA {driver.cuda_version}")
        return [
            self.issue(
                severity=Severity.ERROR,
                title="torch.cuda.is_available() is False",
                description=(
                    "This PyTorch build supports CUDA, but CUDA could not be "
                    "initialized. Typical causes: no NVIDIA driver, a driver too old "
                    "for the build's CUDA runtime, or a broken driver installation."
                ),
                evidence=evidence,
                recommendations=recommendations_for(self.code),
            )
        ]


class PyTorchCpuOnlyBuild(Check):
    """TORCH003: CPU-only wheel installed."""

    code = "TORCH003"
    category = "pytorch"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        torch_info = ctx.snapshot.pytorch
        if not torch_info.installed or torch_info.is_cuda_build is not False:
            return []
        return [
            self.issue(
                severity=Severity.WARNING,
                title="CPU-only PyTorch build detected",
                description=(
                    f"torch {torch_info.version or '(unknown version)'} has no CUDA "
                    "support (torch.version.cuda is None). This is the classic "
                    "'accidentally installed the CPU wheel' situation; the GPU cannot "
                    "be used from PyTorch at all."
                ),
                evidence=[
                    f"torch.version.cuda = {torch_info.cuda_version}",
                    "CPU-only build",
                ],
                recommendations=recommendations_for(self.code),
            )
        ]


class PyTorchRuntimeDiffers(Check):
    """TORCH004: torch's bundled CUDA differs from the local toolkit.

    Deliberately INFO (never ERROR): official PyTorch wheels bundle their own
    CUDA runtime, so this difference is normally valid.
    """

    code = "TORCH004"
    category = "pytorch"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        snapshot = ctx.snapshot
        if not snapshot.pytorch.installed:
            return []
        torch_cuda = parse_cuda_version(snapshot.pytorch.cuda_version)
        toolkit = parse_cuda_version(snapshot.cuda.toolkit_version)
        if torch_cuda is None or toolkit is None or torch_cuda == toolkit:
            return []
        interpretation = interpret_torch_cuda(torch_cuda, toolkit)
        return [
            self.issue(
                severity=Severity.INFO,
                title=(
                    "PyTorch CUDA runtime "
                    f"({torch_cuda}) differs from local toolkit ({toolkit})"
                ),
                description=interpretation.explanation,
                evidence=[
                    f"torch.version.cuda = {torch_cuda}",
                    f"local toolkit (nvcc) = {toolkit}",
                ],
                recommendations=recommendations_for(self.code),
            )
        ]


class PyTorchNewerThanDriver(Check):
    """TORCH006: torch's CUDA runtime exceeds what the driver supports."""

    code = "TORCH006"
    category = "pytorch"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        snapshot = ctx.snapshot
        torch_cuda = parse_cuda_version(snapshot.pytorch.cuda_version)
        driver = snapshot.driver
        if driver is None or not driver.cuda_version or torch_cuda is None:
            return []
        driver_max = parse_cuda_version(driver.cuda_version)
        if driver_max is None or compare_versions(torch_cuda, driver_max) <= 0:
            return []
        return [
            self.issue(
                severity=Severity.ERROR,
                title="PyTorch CUDA runtime newer than the driver supports",
                description=(
                    f"PyTorch was built for CUDA {torch_cuda}, but the installed "
                    f"driver ({driver.version}) supports at most CUDA {driver_max}. "
                    "torch.cuda will not work until the driver is updated (or a "
                    "PyTorch build for an older CUDA is installed)."
                ),
                evidence=[
                    f"torch.version.cuda = {torch_cuda}",
                    f"driver {driver.version} (max CUDA {driver_max})",
                ],
                recommendations=recommendations_for(self.code),
            )
        ]


class PyTorchCannotEnumerate(Check):
    """TORCH005 (enumeration aspect): CUDA 'available' but devices unusable."""

    code = "TORCH005"
    category = "pytorch"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        torch_info = ctx.snapshot.pytorch
        if not torch_info.installed or torch_info.import_error:
            return []
        if torch_info.cuda_available is not True:
            return []  # TORCH002 owns the unavailable case
        if torch_info.device_count is None or torch_info.device_count <= 0:
            return [
                self.issue(
                    severity=Severity.ERROR,
                    title="PyTorch cannot enumerate GPU devices",
                    description=(
                        "torch.cuda.is_available() is True but no devices could be "
                        "enumerated — an inconsistent state that usually indicates a "
                        "broken driver/runtime setup."
                    ),
                    evidence=[
                        f"torch.cuda.device_count() = {torch_info.device_count}",
                    ],
                    recommendations=recommendations_for(self.code),
                )
            ]
        return []
