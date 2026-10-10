"""CUDA-relevant environment variable collector.

PATH and LD_LIBRARY_PATH are analyzed into entries; only the CUDA-relevant
subset is ever surfaced in reports (never a full PATH dump).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import EnvironmentInfo
from cuda_doctor.utils.paths import is_cuda_related
from cuda_doctor.utils.platform import current_platform

SCALAR_VARIABLES = ("CUDA_HOME", "CUDA_PATH")

# v0.2 diagnosis inputs, captured raw. Unlike SCALAR_VARIABLES these are
# captured on *presence*, not truthiness: ``CUDA_VISIBLE_DEVICES=""`` masks
# all GPUs, so an empty value is a meaningful observation, not an unset one.
PRESENCE_VARIABLES = ("CUDACXX", "CUDA_VISIBLE_DEVICES", "CONDA_PREFIX")


class EnvironmentCollector:
    """Collects CUDA-relevant environment variable facts."""

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        platform: Platform | None = None,
    ) -> None:
        self.env: dict[str, str] = dict(env if env is not None else os.environ)
        self.platform = platform or current_platform()

    def collect(self) -> EnvironmentInfo:
        info = EnvironmentInfo()
        # Environment names are case-insensitive only on Windows; folding them
        # everywhere else would conflate distinct Linux variables (e.g. a
        # lowercase ``cuda_home`` is NOT CUDA_HOME on Linux).
        if self.platform is Platform.WINDOWS:
            lookup = {key.upper(): value for key, value in self.env.items()}
        else:
            lookup = self.env

        for name in SCALAR_VARIABLES:
            if lookup.get(name):
                info.variables[name] = lookup[name]
        for name in PRESENCE_VARIABLES:
            if name in lookup:
                info.variables[name] = lookup[name]
        for key, value in sorted(lookup.items()):
            if key.startswith("CUDA_PATH_V") and value:
                info.variables[key] = value

        for index, entry in enumerate(lookup.get("PATH", "").split(self._pathsep())):
            if not entry:
                continue
            info.path_entries.append((index, entry))
            if is_cuda_related(entry):
                info.cuda_path_entries.append((index, entry))

        if self.platform is not Platform.WINDOWS:
            ld = lookup.get("LD_LIBRARY_PATH")
            if ld is not None:
                info.ld_library_path = ld
                info.cuda_ld_library_entries = [
                    entry for entry in ld.split(":") if is_cuda_related(entry)
                ]
                # Ordered raw entries for later v0.2 diagnosis phases. Like
                # the PATH loop above, empty entries are skipped but keep
                # their original position. Never serialized in reports.
                info.ld_library_path_entries = [
                    (index, entry)
                    for index, entry in enumerate(ld.split(":"))
                    if entry
                ]
        return info

    def _pathsep(self) -> str:
        """PATH separator for the *target* platform, never the host's.

        The collector supports an injected target platform (tests simulate
        Windows targets on POSIX hosts and vice versa), so ``os.pathsep`` —
        which describes the machine running Python — would be wrong.
        """
        return ";" if self.platform is Platform.WINDOWS else ":"
