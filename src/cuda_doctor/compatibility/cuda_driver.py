"""NVIDIA driver <-> CUDA compatibility rules.

Data-driven from ``cuda_driver_compatibility.json``. The model follows
NVIDIA's *CUDA minor-version compatibility* documentation (CUDA 11+):

- compatibility is primarily determined by the CUDA **major family**;
- an application built with a newer CUDA minor release runs on a driver
  from the same major family, provided the driver meets the documented
  family minimum (with caveats: newer driver-dependent features and newer
  PTX may still fail);
- the driver shipped with a toolkit release is a *different concept* from
  the minor-compatibility minimum and is deliberately not stored here.

Lookups are exact per major family: a CUDA family that is absent from the
data (e.g. a future CUDA 14) is UNKNOWN — never inferred from an older
family.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cuda_doctor.core.enums import Platform
from cuda_doctor.utils.versions import (
    CudaVersion,
    compare_versions,
    parse_driver_version,
)

_PLATFORM_KEYS = {
    "linux": Platform.LINUX,
    "windows": Platform.WINDOWS,
}


@dataclass(frozen=True)
class DriverVerdict:
    """Outcome of a driver/toolkit comparison."""

    compatible: bool | None  # None = cannot determine (UNKNOWN)
    minimum_driver: tuple[int, ...] | None
    message: str


class DriverCompatibility:
    """Documented minimum driver per CUDA major family, per platform."""

    def __init__(
        self, family_minimums: Mapping[int, Mapping[Platform, tuple[int, ...]]]
    ) -> None:
        self._families: dict[int, dict[Platform, tuple[int, ...]]] = {
            major: dict(entries) for major, entries in family_minimums.items()
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> DriverCompatibility:
        section = data.get("minor_version_compatibility", {})
        families: dict[int, dict[Platform, tuple[int, ...]]] = {}
        if isinstance(section, Mapping):
            for family_text, platform_section in section.items():
                if not family_text.isdigit():
                    continue
                entries = platform_section if isinstance(platform_section, Mapping) else {}
                minimums: dict[Platform, tuple[int, ...]] = {}
                for key, platform in _PLATFORM_KEYS.items():
                    raw = entries.get(key)
                    if isinstance(raw, str):
                        minimum = parse_driver_version(raw.lstrip(">="))
                        if minimum:
                            minimums[platform] = minimum
                if minimums:
                    families[int(family_text)] = minimums
        return cls(families)

    def family_minimum(
        self, cuda: CudaVersion, platform: Platform
    ) -> tuple[int, ...] | None:
        """Documented family minimum, or None when the family is unknown.

        Exact major-family lookup only: a newer family never inherits an
        older family's requirements, and an unknown minor within a known
        family is covered by the family rule (that is what NVIDIA
        documents).
        """
        family = self._families.get(cuda.major)
        if family is None:
            return None
        return family.get(platform)

    def evaluate(
        self, cuda: CudaVersion, driver: tuple[int, ...], platform: Platform
    ) -> DriverVerdict:
        minimum = self.family_minimum(cuda, platform)
        if minimum is None:
            return DriverVerdict(
                None,
                None,
                f"No NVIDIA-documented driver minimum for CUDA {cuda.major}.x on this "
                "platform; compatibility is unknown.",
            )
        ok = compare_versions(driver, minimum) >= 0
        minimum_text = ".".join(str(part) for part in minimum)
        driver_text = ".".join(str(part) for part in driver)
        if ok:
            message = (
                f"Driver {driver_text} meets the documented minimum ({minimum_text}) "
                f"for CUDA {cuda.major}.x minor-version compatibility."
            )
        else:
            message = (
                f"CUDA {cuda.major}.x requires driver {minimum_text} or newer for "
                "minor-version compatibility "
                f"(installed: {driver_text})."
            )
        return DriverVerdict(ok, minimum, message)
