"""CUDA Toolkit checks."""

from __future__ import annotations

from cuda_doctor.checks.base import Check
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.recommendations import recommendations_for
from cuda_doctor.utils.paths import nvcc_within
from cuda_doctor.utils.versions import cuda_version_from_path


class NvccNotFound(Check):
    """CUDA001: no nvcc on PATH."""

    code = "CUDA001"
    category = "cuda"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        if ctx.snapshot.cuda.nvcc_found:
            return []
        return [
            self.issue(
                severity=Severity.WARNING,
                title="CUDA Toolkit (nvcc) not found",
                description=(
                    "No nvcc executable was found on PATH. A local toolkit is only "
                    "needed to compile CUDA code (custom kernels, extensions); running "
                    "prebuilt PyTorch wheels does not require it."
                ),
                evidence=["nvcc not found on PATH"],
                recommendations=recommendations_for(self.code),
            )
        ]


class CudaHomeNotSet(Check):
    """CUDA002: neither CUDA_HOME nor CUDA_PATH is set."""

    code = "CUDA002"
    category = "cuda"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        cuda = ctx.snapshot.cuda
        if cuda.cuda_home or cuda.cuda_path:
            return []
        return [
            self.issue(
                severity=Severity.INFO,
                title="CUDA_HOME / CUDA_PATH not set",
                description=(
                    "No CUDA_HOME (Linux) or CUDA_PATH (Windows) variable is set. "
                    "This is optional — builds that need it usually configure it "
                    "themselves — but some tools look for it."
                ),
                evidence=["CUDA_HOME and CUDA_PATH are unset"],
                recommendations=recommendations_for(self.code),
            )
        ]


class CudaHomeInvalid(Check):
    """CUDA003: CUDA_HOME points to a non-existent or non-toolkit directory."""

    code = "CUDA003"
    category = "cuda"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        cuda = ctx.snapshot.cuda
        if not cuda.cuda_home or cuda.cuda_home_exists is None:
            return []
        if cuda.cuda_home_exists and cuda.cuda_home_has_nvcc:
            return []
        if cuda.cuda_home_exists and cuda.cuda_home_has_nvcc is False:
            problem = "does not look like a CUDA Toolkit (no bin/nvcc inside)"
        elif not cuda.cuda_home_exists:
            problem = "does not exist"
        else:
            return []
        return [
            self.issue(
                severity=Severity.WARNING,
                title="CUDA_HOME points to an invalid location",
                description=(
                    f"CUDA_HOME/CUDA_PATH is set to {cuda.cuda_home}, which {problem}."
                ),
                evidence=[
                    f"CUDA_HOME={cuda.cuda_home}",
                    f"directory exists: {cuda.cuda_home_exists}",
                    f"contains bin/nvcc: {cuda.cuda_home_has_nvcc}",
                ],
                recommendations=recommendations_for(self.code),
            )
        ]


class MultipleToolkits(Check):
    """CUDA004: several CUDA Toolkit installations on disk (informational)."""

    code = "CUDA004"
    category = "cuda"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        versioned = [
            install
            for install in ctx.snapshot.cuda.installations
            if install.version
        ]
        if len(versioned) < 2:
            return []
        listing = [
            f"CUDA {install.version}: {install.path}" for install in versioned
        ]
        return [
            self.issue(
                severity=Severity.INFO,
                title="Multiple CUDA Toolkit installations detected",
                description=(
                    "Several CUDA Toolkit versions are installed. That is common and "
                    "not a problem by itself — just be deliberate about which one "
                    "PATH and CUDA_HOME select."
                ),
                evidence=listing,
                recommendations=recommendations_for(self.code),
            )
        ]


class MultipleCudaBinsInPath(Check):
    """CUDA005: more than one CUDA version's bin directory in PATH."""

    code = "CUDA005"
    category = "cuda"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        entries = ctx.snapshot.environment.cuda_path_entries
        versions = {
            version
            for version in (
                cuda_version_from_path(entry) for _, entry in entries
            )
            if version
        }
        if len(versions) < 2:
            return []
        listing = [
            f"PATH[{index}]: {entry}" for index, entry in entries
        ]
        return [
            self.issue(
                severity=Severity.WARNING,
                title="Multiple CUDA bin directories in PATH",
                description=(
                    f"PATH contains bin directories for {len(versions)} CUDA "
                    f"versions ({', '.join(str(v) for v in sorted(versions))}). The "
                    "earliest entry wins, which may not be the one you intend."
                ),
                evidence=listing,
                recommendations=recommendations_for(self.code),
            )
        ]


class NvccDiffersFromCudaHome(Check):
    """CUDA006: the nvcc on PATH belongs to a different toolkit than CUDA_HOME."""

    code = "CUDA006"
    category = "cuda"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        cuda = ctx.snapshot.cuda
        if not (cuda.cuda_home and cuda.nvcc_path):
            return []
        if nvcc_within(cuda.cuda_home, cuda.nvcc_path):
            return []
        return [
            self.issue(
                severity=Severity.WARNING,
                title="nvcc does not belong to CUDA_HOME",
                description=(
                    f"nvcc resolves to {cuda.nvcc_path}, but CUDA_HOME is set to "
                    f"{cuda.cuda_home}. Tools that trust CUDA_HOME (e.g. extension "
                    "builders) may compile against a different toolkit than the one "
                    "you get on the command line."
                ),
                evidence=[
                    f"nvcc: {cuda.nvcc_path} (nvcc --version: {cuda.toolkit_version or 'unknown'})",
                    f"CUDA_HOME: {cuda.cuda_home}",
                ],
                recommendations=recommendations_for(self.code),
            )
        ]
