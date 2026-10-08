"""GPU presence checks."""

from __future__ import annotations

from cuda_doctor.checks.base import Check
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.recommendations import recommendations_for


class GpuSmiUnavailable(Check):
    """GPU001: nvidia-smi could not be located."""

    code = "GPU001"
    category = "gpu"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        smi = ctx.snapshot.nvidia_smi
        if smi is None or smi.available:
            return []
        # On macOS no NVIDIA driver is expected — informational, not an error.
        severity = (
            Severity.INFO if ctx.platform is Platform.MACOS else Severity.ERROR
        )
        return [
            self.issue(
                severity=severity,
                title="nvidia-smi unavailable",
                description=(
                    "The nvidia-smi tool was not found on PATH. It ships with the "
                    "NVIDIA driver, so usually this means no NVIDIA driver is installed."
                ),
                evidence=[f"lookup error: {smi.error or 'not found'}"],
                recommendations=recommendations_for(self.code),
            )
        ]


class NoNvidiaGpu(Check):
    """GPU002: nvidia-smi works but reports no GPUs."""

    code = "GPU002"
    category = "gpu"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        snapshot = ctx.snapshot
        smi = snapshot.nvidia_smi
        if smi is None or not smi.available or smi.error is not None:
            return []  # GPU001/GPU003 own the unavailable/failed cases
        if snapshot.gpus:
            return []
        severity = (
            Severity.INFO if ctx.platform is Platform.MACOS else Severity.ERROR
        )
        return [
            self.issue(
                severity=severity,
                title="No NVIDIA GPU detected",
                description=(
                    "nvidia-smi ran successfully but reported no GPUs. CUDA will not "
                    "be usable on this machine."
                ),
                evidence=["nvidia-smi query returned an empty GPU list"],
                recommendations=recommendations_for(self.code),
            )
        ]


class NvidiaSmiFailed(Check):
    """GPU003: nvidia-smi was found but its execution failed."""

    code = "GPU003"
    category = "gpu"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        smi = ctx.snapshot.nvidia_smi
        if smi is None or not smi.available or smi.error is None:
            return []
        evidence = [f"execution error: {smi.error}"]
        if smi.stderr_excerpt:
            evidence.append(smi.stderr_excerpt)
        return [
            self.issue(
                severity=Severity.ERROR,
                title="nvidia-smi execution failed",
                description=(
                    "nvidia-smi exists but failed to produce GPU information, most "
                    "commonly because the NVIDIA driver is broken or is mid-update."
                ),
                evidence=evidence,
                recommendations=recommendations_for(self.code),
            )
        ]
