"""Environment variable checks (CUDA-relevant subset only)."""

from __future__ import annotations

import os

from cuda_doctor.checks.base import Check
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.recommendations import recommendations_for
from cuda_doctor.utils.versions import CudaVersion, cuda_version_from_path


class InvalidCudaPaths(Check):
    """ENV001: CUDA-related PATH entries that do not exist on disk."""

    code = "ENV001"
    category = "environment"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        missing: list[str] = []
        for _, entry in ctx.snapshot.environment.cuda_path_entries:
            if not os.path.isdir(entry):
                missing.append(entry)
        for value in ctx.snapshot.environment.cuda_ld_library_entries:
            if not os.path.isdir(value):
                missing.append(value)
        if not missing:
            return []
        return [
            self.issue(
                severity=Severity.WARNING,
                title="CUDA paths point to non-existent directories",
                description=(
                    "These CUDA-related entries do not exist. They are usually "
                    "leftovers from a removed toolkit and can shadow or confuse "
                    "builds."
                ),
                evidence=missing,
                recommendations=recommendations_for(self.code),
            )
        ]


class DuplicateCudaPaths(Check):
    """ENV002: the same CUDA path appears more than once in PATH."""

    code = "ENV002"
    category = "environment"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        seen: dict[str, int] = {}
        for _, entry in ctx.snapshot.environment.cuda_path_entries:
            key = os.path.normpath(entry)
            seen[key] = seen.get(key, 0) + 1
        duplicates = [entry for entry, count in seen.items() if count > 1]
        if not duplicates:
            return []
        return [
            self.issue(
                severity=Severity.INFO,
                title="Duplicated CUDA paths in PATH",
                description="The same CUDA directory appears multiple times in PATH.",
                evidence=[f"{entry} (x{seen[entry]})" for entry in duplicates],
                recommendations=recommendations_for(self.code),
            )
        ]


class ConflictingLdLibraryPaths(Check):
    """ENV003: multiple CUDA versions in LD_LIBRARY_PATH."""

    code = "ENV003"
    category = "environment"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        entries = ctx.snapshot.environment.cuda_ld_library_entries
        versions = {
            version
            for version in (cuda_version_from_path(entry) for entry in entries)
            if version
        }
        if len(versions) < 2:
            return []
        return [
            self.issue(
                severity=Severity.WARNING,
                title="Conflicting CUDA versions in LD_LIBRARY_PATH",
                description=(
                    f"LD_LIBRARY_PATH references {len(versions)} CUDA versions "
                    f"({', '.join(str(v) for v in sorted(versions))}). The dynamic "
                    "linker loads the first match, which can silently mix runtime "
                    "libraries across CUDA versions."
                ),
                evidence=entries,
                recommendations=recommendations_for(self.code),
            )
        ]


class OlderCudaShadowsNewer(Check):
    """ENV004: an older CUDA bin appears in PATH before a newer one."""

    code = "ENV004"
    category = "environment"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        positioned: list[tuple[int, CudaVersion, str]] = []
        for index, entry in ctx.snapshot.environment.cuda_path_entries:
            version = cuda_version_from_path(entry)
            if version:
                positioned.append((index, version, entry))
        if len(positioned) < 2:
            return []
        positioned.sort()
        _, newest, _ = max(positioned, key=lambda item: item[1])
        newest_position, _, _ = min(
            (item for item in positioned if item[1] == newest),
            key=lambda item: item[0],
        )
        older_before = [
            entry
            for index, version, entry in positioned
            if version < newest and index < newest_position
        ]
        if not older_before:
            return []
        return [
            self.issue(
                severity=Severity.INFO,
                title="Older CUDA appears earlier in PATH than newer CUDA",
                description=(
                    f"A bin directory of CUDA older than {newest} is listed before "
                    "the newer toolkit, so commands like nvcc resolve to the older "
                    "version. Reorder PATH if that is unintended."
                ),
                evidence=older_before,
                recommendations=recommendations_for(self.code),
            )
        ]
