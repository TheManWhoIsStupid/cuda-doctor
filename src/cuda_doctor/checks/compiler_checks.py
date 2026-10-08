"""Host compiler checks."""

from __future__ import annotations

from cuda_doctor.checks.base import Check
from cuda_doctor.compatibility import CompilerCompatibility
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.recommendations import recommendations_for
from cuda_doctor.utils.versions import CudaVersion, parse_cuda_version, parse_version


class NoHostCompiler(Check):
    """CMP001: no usable host compiler."""

    code = "CMP001"
    category = "compiler"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        if ctx.snapshot.compiler.any_found:
            return []
        if ctx.platform is Platform.WINDOWS:
            detail = (
                "Neither Visual Studio (vswhere) nor cl.exe was detected. cl is only "
                "on PATH inside a Visual Studio developer prompt."
            )
        elif ctx.platform is Platform.MACOS:
            detail = "No C/C++ compiler detected (macOS: install Xcode command line tools)."
        else:
            detail = "None of gcc/g++/clang/clang++ was found on PATH."
        return [
            self.issue(
                severity=Severity.WARNING,
                title="No host compiler detected",
                description=(
                    f"{detail} A host compiler is required to build CUDA code "
                    "(nvcc compiles host code through it)."
                ),
                evidence=["no compiler executable found"],
                recommendations=recommendations_for(self.code),
            )
        ]


class CompilerCudaCompatibility(Check):
    """CMP002: host compiler potentially unsupported by the CUDA toolkit.

    The compatibility table is coarse, so findings are always worded as
    *potential* issues, never certainties.
    """

    code = "CMP002"
    category = "compiler"

    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        compatibility = ctx.compatibility
        if compatibility is None:
            return []
        cuda = parse_cuda_version(ctx.snapshot.cuda.toolkit_version)
        if cuda is None:
            return []
        if ctx.platform is Platform.WINDOWS:
            return self._check_windows(ctx, cuda, compatibility.compiler)
        return self._check_linux(ctx, cuda, compatibility.compiler)

    def _check_linux(
        self,
        ctx: DiagnosticContext,
        cuda: CudaVersion,
        compatibility: CompilerCompatibility,
    ) -> list[Issue]:
        issues: list[Issue] = []
        for tool in ctx.snapshot.compiler.compilers:
            if tool.name != "gcc" or not tool.found or not tool.version:
                continue
            gcc = parse_version(tool.version)
            if gcc is None:
                continue
            verdict = compatibility.evaluate_gcc(gcc, cuda)
            if verdict.compatible is False:
                issues.append(
                    self.issue(
                        severity=Severity.WARNING,
                        title="Host gcc potentially too new for this CUDA Toolkit",
                        description=verdict.message,
                        evidence=[
                            f"gcc {tool.version} at {tool.path}",
                            f"CUDA Toolkit {cuda}",
                        ],
                        recommendations=recommendations_for(self.code),
                    )
                )
        return issues

    def _check_windows(
        self,
        ctx: DiagnosticContext,
        cuda: CudaVersion,
        compatibility: CompilerCompatibility,
    ) -> list[Issue]:
        issues: list[Issue] = []
        for tool in ctx.snapshot.compiler.compilers:
            if "vswhere" not in tool.name or not tool.version:
                continue
            version = parse_version(tool.version)
            if version is None:
                continue
            verdict = compatibility.evaluate_visual_studio(version, cuda)
            if verdict.compatible is False:
                issues.append(
                    self.issue(
                        severity=Severity.WARNING,
                        title="Visual Studio version potentially unsupported by CUDA",
                        description=verdict.message,
                        evidence=[
                            f"Visual Studio installationVersion {tool.version}",
                            f"CUDA Toolkit {cuda}",
                        ],
                        recommendations=recommendations_for(self.code),
                    )
                )
        return issues
