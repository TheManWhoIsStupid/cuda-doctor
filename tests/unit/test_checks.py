"""Rule-by-rule tests for every diagnostic check."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from tests.factories import base_snapshot, context_for

from cuda_doctor.checks import ALL_CHECKS, Check
from cuda_doctor.checks.compiler_checks import CompilerCudaCompatibility, NoHostCompiler
from cuda_doctor.checks.cuda_checks import (
    CudaHomeInvalid,
    CudaHomeNotSet,
    MultipleCudaBinsInPath,
    MultipleToolkits,
    NvccDiffersFromCudaHome,
    NvccNotFound,
)
from cuda_doctor.checks.driver_checks import DriverRuntimeCompatibility, DriverUndetermined
from cuda_doctor.checks.environment_checks import (
    ConflictingLdLibraryPaths,
    DuplicateCudaPaths,
    InvalidCudaPaths,
    OlderCudaShadowsNewer,
)
from cuda_doctor.checks.gpu_checks import GpuSmiUnavailable, NoNvidiaGpu, NvidiaSmiFailed
from cuda_doctor.checks.pytorch_checks import (
    PyTorchCannotEnumerate,
    PyTorchCpuOnlyBuild,
    PyTorchCudaUnavailable,
    PyTorchImportFailed,
    PyTorchNewerThanDriver,
    PyTorchNotInstalled,
    PyTorchRuntimeDiffers,
)
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.core.models import (
    CUDAInstallation,
    NvidiaSmiInfo,
    ToolInfo,
)
from cuda_doctor.diagnosis.recommendations import DEFAULT_RECOMMENDATIONS


def run_check(check: type[Check], snapshot):
    """Run one check class against a snapshot with real compatibility data."""
    return check().run(context_for(snapshot))


def no_cuda(snapshot):
    """A CUDAInfo representing 'no toolkit anywhere'."""
    return replace(
        snapshot.cuda,
        nvcc_found=False,
        nvcc_path=None,
        toolkit_version=None,
        installations=[],
    )


def no_torch(snapshot):
    return replace(snapshot.pytorch, installed=False, import_error=None)


class TestRegistry:
    def test_all_codes_well_formed(self):
        for check in ALL_CHECKS:
            assert re.fullmatch(r"[A-Z]{3,5}\d{3}", check.code), check.code
            assert check.category, check.code

    def test_recommendations_cover_every_code(self):
        codes = {check.code for check in ALL_CHECKS}
        assert codes <= set(DEFAULT_RECOMMENDATIONS), codes - set(
            DEFAULT_RECOMMENDATIONS
        )

    def test_registry_count(self):
        # 3 GPU + 2 driver + 6 CUDA + 7 torch + 2 compiler + 4 environment
        assert len(ALL_CHECKS) == 24


class TestGpuChecks:
    def test_gpu001_fires_when_not_found(self):
        snapshot = base_snapshot(
            nvidia_smi=NvidiaSmiInfo(available=False, executed=False, error="not found")
        )
        issues = run_check(GpuSmiUnavailable, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR
        assert "nvidia-smi unavailable" in issues[0].title

    def test_gpu001_is_info_on_macos(self):
        snapshot = base_snapshot(
            nvidia_smi=NvidiaSmiInfo(available=False, executed=False, error="not found"),
        )
        snapshot.system = replace(snapshot.system, platform=Platform.MACOS)
        issues = run_check(GpuSmiUnavailable, snapshot)
        assert issues[0].severity is Severity.INFO

    def test_gpu001_silent_when_available(self):
        assert run_check(GpuSmiUnavailable, base_snapshot()) == []

    def test_gpu002_fires_on_empty_gpu_list(self):
        snapshot = base_snapshot(gpus=[], driver=None)
        issues = run_check(NoNvidiaGpu, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR

    def test_gpu002_silent_when_smi_failed(self):
        # GPU003 owns the failed-execution case.
        snapshot = base_snapshot(
            gpus=[],
            driver=None,
            nvidia_smi=NvidiaSmiInfo(available=True, executed=True, error="exit-9"),
        )
        assert run_check(NoNvidiaGpu, snapshot) == []

    def test_gpu002_silent_with_gpus(self):
        assert run_check(NoNvidiaGpu, base_snapshot()) == []

    def test_gpu003_fires_with_stderr_evidence(self):
        snapshot = base_snapshot(
            nvidia_smi=NvidiaSmiInfo(
                available=True, executed=True, error="exit-9", stderr_excerpt="ERR: driver crashed"
            )
        )
        issues = run_check(NvidiaSmiFailed, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR
        assert any("ERR: driver crashed" in line for line in issues[0].evidence)

    def test_gpu003_silent_without_error(self):
        assert run_check(NvidiaSmiFailed, base_snapshot()) == []


class TestDriverChecks:
    def test_drv001_fires_when_version_unknown(self):
        snapshot = base_snapshot(driver=None)
        issues = run_check(DriverUndetermined, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING

    def test_drv001_silent_without_gpus(self):
        assert run_check(DriverUndetermined, base_snapshot(gpus=[], driver=None)) == []

    def test_drv002_error_when_toolkit_exceeds_driver_max(self):
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, toolkit_version="13.0")
        snapshot.driver = replace(snapshot.driver, version="550.54.14", cuda_version="12.4")
        issues = run_check(DriverRuntimeCompatibility, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR
        assert "at most CUDA 12.4" in issues[0].description

    def test_drv002_silent_at_boundary(self):
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, toolkit_version="13.0")
        snapshot.driver = replace(snapshot.driver, version="580.126.09", cuda_version="13.0")
        assert run_check(DriverRuntimeCompatibility, snapshot) == []

    def test_drv002_warning_via_fallback_table(self):
        # Driver max CUDA unknown -> fall back to the minimum-driver table.
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, toolkit_version="12.4")
        snapshot.driver = replace(snapshot.driver, version="450.36.06", cuda_version=None)
        issues = run_check(DriverRuntimeCompatibility, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert "Potential" in issues[0].title

    def test_drv002_silent_without_driver(self):
        snapshot = base_snapshot(driver=None)
        assert run_check(DriverRuntimeCompatibility, snapshot) == []


class TestCudaChecks:
    def test_cuda001_fires_without_nvcc(self):
        snapshot = base_snapshot()
        snapshot.cuda = no_cuda(snapshot)
        issues = run_check(NvccNotFound, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert "does not require" in issues[0].description  # reassurance present

    def test_cuda002_info_when_home_unset(self):
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, cuda_home=None, cuda_path=None)
        issues = run_check(CudaHomeNotSet, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.INFO

    def test_cuda003_home_does_not_exist(self):
        snapshot = base_snapshot()
        snapshot.cuda = replace(
            snapshot.cuda, cuda_home="/opt/cuda-gone", cuda_home_exists=False
        )
        issues = run_check(CudaHomeInvalid, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert "does not exist" in issues[0].description

    def test_cuda003_home_without_nvcc(self):
        snapshot = base_snapshot()
        snapshot.cuda = replace(
            snapshot.cuda,
            cuda_home="/opt/not-a-toolkit",
            cuda_home_exists=True,
            cuda_home_has_nvcc=False,
        )
        issues = run_check(CudaHomeInvalid, snapshot)
        assert len(issues) == 1
        assert "no bin/nvcc" in issues[0].description

    def test_cuda003_silent_when_valid(self):
        assert run_check(CudaHomeInvalid, base_snapshot()) == []

    def test_cuda004_info_for_multiple_toolkits(self):
        snapshot = base_snapshot()
        snapshot.cuda = replace(
            snapshot.cuda,
            installations=[
                CUDAInstallation("/usr/local/cuda-12.6", "12.6"),
                CUDAInstallation("/usr/local/cuda-12.4", "12.4"),
            ],
        )
        issues = run_check(MultipleToolkits, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.INFO
        assert any("12.6" in line for line in issues[0].evidence)

    def test_cuda005_warns_on_two_versions_in_path(self):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            cuda_path_entries=[
                (0, "/usr/local/cuda-11.8/bin"),
                (1, "/usr/local/cuda-12.4/bin"),
            ],
        )
        issues = run_check(MultipleCudaBinsInPath, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING

    def test_cuda006_warns_when_nvcc_differs_from_home(self):
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, nvcc_path="/usr/local/cuda-12.6/bin/nvcc")
        issues = run_check(NvccDiffersFromCudaHome, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING

    def test_cuda006_silent_when_consistent(self):
        assert run_check(NvccDiffersFromCudaHome, base_snapshot()) == []


class TestPyTorchChecks:
    def test_torch001_info_when_not_installed(self):
        snapshot = base_snapshot()
        snapshot.pytorch = no_torch(snapshot)
        issues = run_check(PyTorchNotInstalled, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.INFO

    def test_torch005_error_with_known_issue_match(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch,
            installed=True,
            import_error="RuntimeError: libcuda.so.1: cannot open shared object file",
        )
        issues = run_check(PyTorchImportFailed, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR
        assert "NVIDIA driver library" in issues[0].title  # known-issue title
        assert any("libcuda" in r for r in issues[0].recommendations)

    def test_torch005_error_without_match(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch, installed=True, import_error="ValueError: nonsense"
        )
        issues = run_check(PyTorchImportFailed, snapshot)
        assert issues[0].title == "PyTorch cannot be imported"

    def test_torch002_error_when_cuda_unavailable(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, cuda_available=False, devices=[])
        issues = run_check(PyTorchCudaUnavailable, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR
        assert any("driver supports" in e for e in issues[0].evidence)

    def test_torch002_silent_for_cpu_build(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch, is_cuda_build=False, cuda_version=None, cuda_available=False
        )
        assert run_check(PyTorchCudaUnavailable, snapshot) == []

    def test_torch003_warns_on_cpu_build(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(
            snapshot.pytorch,
            is_cuda_build=False,
            cuda_version=None,
            cuda_available=False,
            device_count=0,
            devices=[],
        )
        issues = run_check(PyTorchCpuOnlyBuild, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert "CPU-only" in issues[0].title

    def test_torch004_is_info_never_error(self):
        # The product's core nuance: local toolkit != torch runtime is FINE.
        snapshot = base_snapshot()
        snapshot.cuda = replace(snapshot.cuda, toolkit_version="12.6")
        issues = run_check(PyTorchRuntimeDiffers, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.INFO
        assert "usually fine" in issues[0].description.lower()

    def test_torch004_silent_when_equal(self):
        assert run_check(PyTorchRuntimeDiffers, base_snapshot()) == []

    def test_torch006_error_when_torch_exceeds_driver(self):
        snapshot = base_snapshot()
        snapshot.driver = replace(snapshot.driver, version="535.104.05", cuda_version="12.2")
        issues = run_check(PyTorchNewerThanDriver, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR

    def test_torch006_silent_within_driver_support(self):
        assert run_check(PyTorchNewerThanDriver, base_snapshot()) == []

    def test_torch005_enumeration_error(self):
        snapshot = base_snapshot()
        snapshot.pytorch = replace(snapshot.pytorch, device_count=0, devices=[])
        issues = run_check(PyTorchCannotEnumerate, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR
        assert "enumerate" in issues[0].title


class TestCompilerChecks:
    def test_cmp001_warns_without_compiler(self):
        snapshot = base_snapshot()
        snapshot.compiler = replace(snapshot.compiler, compilers=[])
        issues = run_check(NoHostCompiler, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING

    def test_cmp001_windows_hint(self):
        snapshot = base_snapshot()
        snapshot.system = replace(snapshot.system, platform=Platform.WINDOWS)
        snapshot.compiler = replace(snapshot.compiler, compilers=[])
        issues = run_check(NoHostCompiler, snapshot)
        assert "Visual Studio" in issues[0].description

    def test_cmp002_warns_on_too_new_gcc(self):
        snapshot = base_snapshot()
        snapshot.compiler = replace(
            snapshot.compiler,
            compilers=[ToolInfo("gcc", True, "/usr/bin/gcc", "14.2.0")],
        )
        issues = run_check(CompilerCudaCompatibility, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert "potentially" in issues[0].title.lower()

    def test_cmp002_silent_on_supported_gcc(self):
        assert run_check(CompilerCudaCompatibility, base_snapshot()) == []

    def test_cmp002_windows_vs_too_old(self):
        snapshot = base_snapshot()
        snapshot.system = replace(snapshot.system, platform=Platform.WINDOWS)
        snapshot.compiler = replace(
            snapshot.compiler,
            compilers=[ToolInfo("vswhere", True, r"C:\Program Files\vswhere.exe", "15.0")],
        )
        issues = run_check(CompilerCudaCompatibility, snapshot)
        assert len(issues) == 1
        assert "Visual Studio" in issues[0].title

    def test_cmp002_silent_without_compatibility_data(self):
        snapshot = base_snapshot()
        snapshot.compiler = replace(
            snapshot.compiler,
            compilers=[ToolInfo("gcc", True, "/usr/bin/gcc", "14.2.0")],
        )
        ctx = DiagnosticContext(snapshot=snapshot, compatibility=None)
        assert CompilerCudaCompatibility().run(ctx) == []


class TestEnvironmentChecks:
    def test_env001_warns_on_missing_directory(self, tmp_path):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            cuda_path_entries=[(0, str(tmp_path / "cuda-gone" / "bin"))],
        )
        issues = run_check(InvalidCudaPaths, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING

    def test_env001_silent_when_entries_exist(self, tmp_path):
        (tmp_path / "bin").mkdir()
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment, cuda_path_entries=[(0, str(tmp_path / "bin"))]
        )
        assert run_check(InvalidCudaPaths, snapshot) == []

    def test_env002_info_on_duplicates(self):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            cuda_path_entries=[
                (0, "/usr/local/cuda-12.4/bin"),
                (1, "/usr/local/cuda-12.4/bin/"),
            ],
        )
        issues = run_check(DuplicateCudaPaths, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.INFO

    def test_env003_warns_on_two_ld_versions(self):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            cuda_ld_library_entries=[
                "/usr/local/cuda-11.8/lib64",
                "/usr/local/cuda-12.4/lib64",
            ],
        )
        issues = run_check(ConflictingLdLibraryPaths, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING

    def test_env004_info_when_older_shadows_newer(self):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            cuda_path_entries=[
                (0, "/usr/local/cuda-11.8/bin"),
                (1, "/usr/local/cuda-12.4/bin"),
            ],
        )
        issues = run_check(OlderCudaShadowsNewer, snapshot)
        assert len(issues) == 1
        assert issues[0].severity is Severity.INFO

    def test_env004_silent_when_newer_first(self):
        snapshot = base_snapshot()
        snapshot.environment = replace(
            snapshot.environment,
            cuda_path_entries=[
                (0, "/usr/local/cuda-12.4/bin"),
                (1, "/usr/local/cuda-11.8/bin"),
            ],
        )
        assert run_check(OlderCudaShadowsNewer, snapshot) == []


@pytest.mark.parametrize("check", ALL_CHECKS)
def test_healthy_snapshot_triggers_no_check(check):
    """The shared healthy snapshot must produce zero issues, on any machine."""
    assert check().run(context_for(base_snapshot())) == []
