"""NVIDIA driver <-> CUDA toolkit compatibility rules.

Data-driven from ``cuda_driver_compatibility.json``; never hardcoded at call
sites so the table can be updated without touching logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cuda_doctor.compatibility._util import nearest_lower
from cuda_doctor.core.enums import Platform
from cuda_doctor.utils.versions import (
    CudaVersion,
    compare_versions,
    parse_driver_version,
    parse_version,
)


@dataclass(frozen=True)
class DriverVerdict:
    """Outcome of a driver/toolkit comparison."""

    compatible: bool | None  # None = cannot determine
    minimum_driver: tuple[int, ...] | None
    message: str


class DriverCompatibility:
    """Minimum driver version per CUDA version, per platform."""

    def __init__(
        self, min_drivers: Mapping[Platform, Mapping[CudaVersion, tuple[int, ...]]]
    ) -> None:
        self._table: dict[Platform, dict[CudaVersion, tuple[int, ...]]] = {
            platform: dict(entries) for platform, entries in min_drivers.items()
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Mapping[str, str]]) -> DriverCompatibility:
        platform_keys = {
            "linux": Platform.LINUX,
            "windows": Platform.WINDOWS,
        }
        table: dict[Platform, dict[CudaVersion, tuple[int, ...]]] = {}
        for key, platform in platform_keys.items():
            section = data.get(key, {})
            entries: dict[CudaVersion, tuple[int, ...]] = {}
            for cuda_text, driver_text in section.items():
                parts = parse_version(cuda_text)
                if not parts or len(parts) < 2:
                    continue
                minimum = parse_driver_version(str(driver_text).lstrip(">="))
                if minimum:
                    entries[CudaVersion(parts[0], parts[1])] = minimum
            table[platform] = entries
        return cls(table)

    def minimum_driver(
        self, cuda: CudaVersion, platform: Platform
    ) -> tuple[int, ...] | None:
        table = self._table.get(platform, {})
        return nearest_lower(table, cuda)

    def evaluate(
        self, cuda: CudaVersion, driver: tuple[int, ...], platform: Platform
    ) -> DriverVerdict:
        minimum = self.minimum_driver(cuda, platform)
        if minimum is None:
            return DriverVerdict(
                None, None, "No driver compatibility data for this CUDA version."
            )
        ok = compare_versions(driver, minimum) >= 0
        minimum_text = ".".join(str(part) for part in minimum)
        if ok:
            message = f"Driver meets the minimum ({minimum_text}) for CUDA {cuda}."
        else:
            message = (
                f"CUDA {cuda} requires driver {minimum_text} or newer "
                f"(installed: {'.'.join(str(p) for p in driver)})."
            )
        return DriverVerdict(ok, minimum, message)
