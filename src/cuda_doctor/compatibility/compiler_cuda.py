"""Host compiler <-> CUDA toolkit compatibility rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cuda_doctor.compatibility._util import nearest_lower
from cuda_doctor.utils.versions import CudaVersion, parse_version


@dataclass(frozen=True)
class CompilerVerdict:
    """Outcome of a compiler/toolkit comparison."""

    compatible: bool | None  # None = cannot determine
    message: str


class CompilerCompatibility:
    """Coarse, data-driven host compiler support per CUDA version."""

    def __init__(
        self,
        max_gcc_major: Mapping[CudaVersion, int],
        min_vs_major: Mapping[CudaVersion, int],
        max_vs_major: Mapping[CudaVersion, int],
        vs_years: Mapping[int, int],
    ) -> None:
        self._max_gcc = dict(max_gcc_major)
        self._min_vs = dict(min_vs_major)
        self._max_vs = dict(max_vs_major)
        self._vs_years = dict(vs_years)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> CompilerCompatibility:
        linux_raw = data.get("linux", {})
        windows_raw = data.get("windows", {})
        linux = linux_raw if isinstance(linux_raw, Mapping) else {}
        windows = windows_raw if isinstance(windows_raw, Mapping) else {}

        def _int_table(raw: object) -> dict[CudaVersion, int]:
            table: dict[CudaVersion, int] = {}
            if isinstance(raw, Mapping):
                for cuda_text, value in raw.items():
                    parts = parse_version(str(cuda_text))
                    if parts and len(parts) >= 2 and isinstance(value, int):
                        table[CudaVersion(parts[0], parts[1])] = value
            return table

        vs_years_raw = windows.get("installation_version_to_vs_year", {})
        vs_years: dict[int, int] = {}
        if isinstance(vs_years_raw, Mapping):
            for key, value in vs_years_raw.items():
                parts = parse_version(str(key))
                if parts:
                    vs_years[parts[0]] = int(value)
        return cls(
            max_gcc_major=_int_table(linux.get("max_supported_gcc_major", {})),
            min_vs_major=_int_table(windows.get("min_vs_major", {})),
            max_vs_major=_int_table(windows.get("max_vs_major", {})),
            vs_years=vs_years,
        )

    def max_supported_gcc_major(self, cuda: CudaVersion) -> int | None:
        return nearest_lower(self._max_gcc, cuda)

    def vs_range(self, cuda: CudaVersion) -> tuple[int | None, int | None]:
        return (
            nearest_lower(self._min_vs, cuda),
            nearest_lower(self._max_vs, cuda),
        )

    def vs_year(self, installation_major: int) -> int | None:
        return self._vs_years.get(installation_major)

    def evaluate_gcc(
        self, gcc_version: tuple[int, ...], cuda: CudaVersion
    ) -> CompilerVerdict:
        """Compare a gcc major against the supported maximum.

        The table is coarse, so a negative verdict is always phrased as a
        *potential* incompatibility, never a certainty.
        """
        maximum = self.max_supported_gcc_major(cuda)
        if maximum is None:
            return CompilerVerdict(None, "No compiler compatibility data for this CUDA version.")
        gcc_major = gcc_version[0] if gcc_version else 0
        if gcc_major <= maximum:
            return CompilerVerdict(
                True, f"gcc {gcc_major}.x is within the supported range for CUDA {cuda}."
            )
        return CompilerVerdict(
            False,
            f"gcc {gcc_major}.x is newer than the maximum supported by CUDA {cuda} "
            f"(approximately gcc {maximum}.x, per the CUDA installation guide). "
            "This is a potential compatibility issue: nvcc may reject the host compiler.",
        )

    def evaluate_visual_studio(
        self, installation_version: tuple[int, ...], cuda: CudaVersion
    ) -> CompilerVerdict:
        vs_min, vs_max = self.vs_range(cuda)
        if not installation_version or vs_min is None or vs_max is None:
            return CompilerVerdict(None, "No Visual Studio compatibility data.")
        major = installation_version[0]
        year_min = self.vs_year(vs_min)
        year_max = self.vs_year(vs_max)
        if vs_min <= major <= vs_max:
            return CompilerVerdict(True, "Visual Studio version is within the supported range.")
        years = f"VS {year_min}-VS {year_max}" if year_min and year_max else "the documented range"
        return CompilerVerdict(
            False,
            f"Visual Studio {self.vs_year(major) or major} is outside the range supported by "
            f"CUDA {cuda} (approximately {years}). This is a potential compatibility issue; "
            "install the Visual Studio version listed in the CUDA installation guide, "
            "or use a CUDA toolkit that supports your Visual Studio.",
        )
