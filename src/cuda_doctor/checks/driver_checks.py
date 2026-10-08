"""NVIDIA driver checks."""

from __future__ import annotations

from cuda_doctor.checks.base import Check
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.recommendations import recommendations_for
from cuda_doctor.utils.versions import parse_cuda_version, parse_driver_version


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
    """DRV002: driver vs. CUDA toolkit compatibility.

    Uses NVIDIA's CUDA 11+ *minor-version compatibility* model instead of a
    strict ceiling: the CUDA version reported by nvidia-smi is the CUDA UMD
    version (the toolkit generation the driver was validated with), not a
    hard limit. Within a CUDA major family, a newer toolkit usually runs on
    an older driver as long as the driver meets the NVIDIA-documented family
    minimum. Cross-generation toolkits are a genuine incompatibility.
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
        installed = parse_driver_version(driver.version)
        verdict = None
        if ctx.compatibility is not None and installed is not None:
            verdict = ctx.compatibility.driver.evaluate(toolkit, installed, ctx.platform)

        # Cross-generation: the driver's CUDA family is older than the
        # toolkit's — minor-version compatibility cannot help here.
        if driver_max is not None and driver_max.major < toolkit.major:
            return [
                self.issue(
                    severity=Severity.ERROR,
                    title="CUDA Toolkit is from a newer CUDA generation than the driver",
                    description=(
                        f"nvcc reports CUDA {toolkit}, but the driver "
                        f"({driver.version}, max CUDA {driver_max}) belongs to the "
                        f"CUDA {driver_max.major}.x generation. Applications built "
                        "with this toolkit are not expected to run on this driver "
                        "unless a CUDA forward-compatibility package is installed "
                        "(datacenter GPUs only, per NVIDIA documentation)."
                    ),
                    evidence=[
                        f"toolkit (nvcc): CUDA {toolkit}",
                        f"driver: {driver.version} (CUDA UMD version {driver_max})",
                    ],
                    recommendations=recommendations_for(self.code),
                )
            ]

        # Driver below the NVIDIA-documented minimum for the toolkit's family.
        if verdict is not None and verdict.compatible is False:
            return [
                self.issue(
                    severity=Severity.ERROR,
                    title="Driver is below the documented minimum for this CUDA generation",
                    description=(
                        f"{verdict.message} Applications built with CUDA {toolkit} "
                        "may fail to run or initialize CUDA until the driver is "
                        "updated (unless a CUDA forward-compatibility package is "
                        "installed, datacenter GPUs only)."
                    ),
                    evidence=[
                        f"toolkit (nvcc): CUDA {toolkit}",
                        f"driver: {driver.version}",
                        f"documented minimum for CUDA {toolkit.major}.x: "
                        f"{'.'.join(str(p) for p in verdict.minimum_driver or ())}",
                    ],
                    recommendations=recommendations_for(self.code),
                )
            ]

        # Same family, toolkit minor above the driver's UMD version, driver
        # meets the family minimum -> minor-version compatibility applies.
        if (
            driver_max is not None
            and driver_max.major == toolkit.major
            and toolkit.minor > driver_max.minor
            and verdict is not None
            and verdict.compatible is True
        ):
            return [
                self.issue(
                    severity=Severity.INFO,
                    title=(
                        "Toolkit newer than the driver's CUDA version "
                        "(minor-version compatibility)"
                    ),
                    description=(
                        f"The local toolkit is CUDA {toolkit}, while the driver "
                        f"({driver.version}) reports CUDA {driver_max} — the CUDA UMD "
                        "version the driver was validated with. This is not a hard "
                        "ceiling: CUDA minor-version compatibility allows "
                        f"applications built with CUDA {toolkit} to run on drivers "
                        f"from the same CUDA {toolkit.major}.x generation "
                        "(the installed driver meets the documented family "
                        "minimum). Newer driver-dependent features and PTX produced "
                        "by the newer toolkit may still require a driver update."
                    ),
                    evidence=[
                        f"toolkit (nvcc): CUDA {toolkit}",
                        f"driver: {driver.version} (CUDA UMD version {driver_max})",
                    ],
                    recommendations=recommendations_for(self.code),
                )
            ]

        # Everything else: toolkit within the driver's validated CUDA version,
        # or not enough information — no verdict.
        return []
