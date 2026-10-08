"""NVIDIA driver checks."""

from __future__ import annotations

from cuda_doctor.checks.base import Check
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.recommendations import recommendations_for
from cuda_doctor.utils.versions import (
    compare_versions,
    parse_cuda_version,
    parse_driver_version,
)


class DriverUndetermined(Check):
    """DRV001: the driver version could not be determined."""

    code = "DRV001"
    category = "driver"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        snapshot = ctx.snapshot
        if not snapshot.gpus:
            return []  # no GPU -> nothing to verify
        driver = snapshot.driver
        if driver is not None and driver.version:
            return []
        evidence = ["nvidia-smi did not report a usable driver version"]
        smi = snapshot.nvidia_smi
        if smi is not None:
            if smi.stdout_excerpt:
                evidence.append(smi.stdout_excerpt)
            if smi.stderr_excerpt:
                evidence.append(smi.stderr_excerpt)
        return [
            self.issue(
                severity=Severity.WARNING,
                title="Unable to determine NVIDIA driver version",
                description=(
                    "GPUs are present but the driver version is unknown, so "
                    "driver/CUDA compatibility cannot be verified."
                ),
                evidence=evidence,
                recommendations=recommendations_for(self.code),
            )
        ]


class DriverRuntimeCompatibility(Check):
    """DRV002: driver vs. CUDA toolkit compatibility looks problematic.

    Primary, deterministic signal: the maximum CUDA version the driver reports
    (nvidia-smi header) versus the installed toolkit. Fallback: the bundled
    minimum-driver table, worded as a potential issue.
    """

    code = "DRV002"
    category = "driver"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        snapshot = ctx.snapshot
        toolkit = parse_cuda_version(snapshot.cuda.toolkit_version)
        if toolkit is None:
            return []
        driver = snapshot.driver
        if driver is None or not driver.version:
            return []  # DRV001 owns the missing-driver case

        driver_max = parse_cuda_version(driver.cuda_version) if driver.cuda_version else None
        if driver_max is not None and compare_versions(toolkit, driver_max) > 0:
            return [
                self.issue(
                    severity=Severity.ERROR,
                    title="CUDA Toolkit newer than the installed driver supports",
                    description=(
                        f"nvcc reports CUDA {toolkit}, but the driver "
                        f"({driver.version}) supports at most CUDA {driver_max}. "
                        "Binaries built with this toolkit will fail to run until the "
                        "driver is updated."
                    ),
                    evidence=[
                        f"toolkit (nvcc): CUDA {toolkit}",
                        f"driver: {driver.version} (max CUDA {driver_max})",
                    ],
                    recommendations=recommendations_for(self.code),
                )
            ]

        # Fallback: bundled minimum-driver table (uncertain -> cautious wording).
        if driver_max is None and ctx.compatibility is not None:
            installed = parse_driver_version(driver.version)
            if installed is None:
                return []
            verdict = ctx.compatibility.driver.evaluate(toolkit, installed, ctx.platform)
            if verdict.compatible is False:
                return [
                    self.issue(
                        severity=Severity.WARNING,
                        title="Potential driver/runtime compatibility issue",
                        description=verdict.message,
                        evidence=[
                            f"toolkit (nvcc): CUDA {toolkit}",
                            f"driver: {driver.version}",
                        ],
                        recommendations=recommendations_for(self.code),
                    )
                ]
        return []
