"""Tests for environment collectors (all external I/O faked or injected)."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
import types

import pytest
from tests.conftest import FakeRunner, failed, load_fixture, ok

from cuda_doctor.collectors.cmake import CMakeCollector
from cuda_doctor.collectors.compiler import CompilerCollector
from cuda_doctor.collectors.cuda import CUDACollector
from cuda_doctor.collectors.environment import EnvironmentCollector
from cuda_doctor.collectors.ninja import NinjaCollector
from cuda_doctor.collectors.nvidia_smi import (
    QUERY_GPU_ARGS,
    QUERY_GPU_MINIMAL_ARGS,
    XML_ARGS,
    NvidiaSmiClient,
)
from cuda_doctor.collectors.python_env import PythonEnvCollector
from cuda_doctor.collectors.pytorch import PyTorchCollector
from cuda_doctor.collectors.system import SystemCollector
from cuda_doctor.core.enums import Platform
from cuda_doctor.utils.commands import ERROR_NOT_FOUND, CommandResult


class TestSystemCollector:
    def test_populates_fields(self):
        info = SystemCollector().collect()
        assert info.os_name
        assert info.python_version
        assert info.kernel_version
        assert info.platform in list(Platform)

    def test_platform_injection(self):
        # The behavioral classification must follow the injected target, not
        # the host OS (host-descriptive fields like os_name stay host-read).
        info = SystemCollector(platform=Platform.MACOS).collect()
        assert info.platform is Platform.MACOS

    def test_platform_defaults_to_host(self):
        from cuda_doctor.utils.platform import current_platform

        assert SystemCollector().collect().platform is current_platform()

    def test_no_personal_data_in_model(self):
        info = SystemCollector().collect()
        # Hostname/username are intentionally not part of the model.
        assert not hasattr(info, "hostname")
        assert not hasattr(info, "username")


class TestNvidiaSmiClient:
    @pytest.fixture()
    def fake_smi_on_path(self, monkeypatch):
        """Hermeticity: CI machines have no nvidia-smi to locate.

        The client resolves the executable before consulting the injected
        runner, so pretend it exists regardless of the host machine.
        """
        monkeypatch.setattr(
            "cuda_doctor.collectors.nvidia_smi.find_executable",
            lambda name: f"/fake/bin/{name}",
        )

    def test_happy_path(self, fixtures_dir, fake_smi_on_path):
        runner = FakeRunner(
            {
                QUERY_GPU_ARGS: ok(load_fixture(fixtures_dir, "nvidia_smi/query_csv_single.txt")),
                XML_ARGS: ok(load_fixture(fixtures_dir, "nvidia_smi/xml_h20.xml")),
            }
        )
        result = NvidiaSmiClient(runner).query()
        assert len(result.gpus) == 1
        assert result.gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert result.driver is not None
        assert result.driver.version == "580.126.09"
        assert result.driver.cuda_version == "13.0"
        assert result.info.available and result.info.executed

    def test_executable_not_found(self):
        result = NvidiaSmiClient(FakeRunner(), executable="nvidia-smi-missing-xyz").query()
        assert result.gpus == []
        assert result.driver is None
        assert result.info.error == ERROR_NOT_FOUND
        assert result.info.available is False

    def test_driver_failure_preserved_as_evidence(self, fixtures_dir, fake_smi_on_path):
        stderr = load_fixture(fixtures_dir, "nvidia_smi/stderr_failed.txt")
        runner = FakeRunner(default=failed(stderr, return_code=6))
        result = NvidiaSmiClient(runner).query()
        assert result.gpus == []
        assert result.info.executed is True
        assert result.info.stderr_excerpt and "NVIDIA-SMI has failed" in result.info.stderr_excerpt

    def test_banner_fallback_when_xml_unparseable(self, fixtures_dir, fake_smi_on_path):
        runner = FakeRunner(
            {
                QUERY_GPU_ARGS: ok(load_fixture(fixtures_dir, "nvidia_smi/query_csv_single.txt")),
                XML_ARGS: ok("<broken"),
                (): ok(load_fixture(fixtures_dir, "nvidia_smi/plain_banner.txt")),
            }
        )
        result = NvidiaSmiClient(runner).query()
        assert result.driver is not None
        assert result.driver.version == "580.126.09"
        assert result.driver.source == "nvidia-smi"

    def test_timeout(self, fake_smi_on_path):
        timeout_result = CommandResult(("<fake>",), None, "", "", "timeout")
        result = NvidiaSmiClient(FakeRunner(default=timeout_result)).query()
        assert result.gpus == []
        assert result.info.error == "timeout"

    def test_old_driver_compute_cap_retry(self, fake_smi_on_path):
        old_driver_error = failed('Field "compute_cap" is not a valid field to be queried')
        retry_output = "0, NVIDIA Tesla V100-SXM2-32GB, GPU-abc, 32510 MiB"
        runner = FakeRunner(
            {
                QUERY_GPU_ARGS: old_driver_error,
                QUERY_GPU_MINIMAL_ARGS: ok(retry_output),
            }
        )
        result = NvidiaSmiClient(runner).query()
        assert len(result.gpus) == 1
        assert result.gpus[0].compute_capability is None


class TestCUDACollector:
    def test_full_toolkit(self, tmp_path, fixtures_dir, monkeypatch):
        home = tmp_path / "cuda-12.4"
        (home / "bin" / "nvcc").parent.mkdir(parents=True)
        (home / "bin" / "nvcc").touch()
        other_root = tmp_path / "toolkits"
        (other_root / "cuda-11.8").mkdir(parents=True)
        monkeypatch.setattr(
            "cuda_doctor.collectors.cuda.find_executable",
            lambda name: str(home / "bin" / "nvcc") if name == "nvcc" else None,
        )
        runner = FakeRunner(
            {("--version",): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={"CUDA_HOME": str(home), "PATH": "/usr/bin"},
            roots=(str(other_root),),
            platform=Platform.LINUX,
        ).collect()
        assert info.nvcc_found is True
        assert info.toolkit_version == "12.4"
        assert info.cuda_home == str(home)
        assert info.cuda_home_exists is True
        assert info.cuda_home_has_nvcc is True
        assert [install.version for install in info.installations] == ["11.8"]

    def test_nvcc_missing(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.collectors.cuda.find_executable", lambda name: None)
        info = CUDACollector(
            FakeRunner(), env={"PATH": "/usr/bin"}, platform=Platform.LINUX
        ).collect()
        assert info.nvcc_found is False
        assert info.toolkit_version is None
        assert info.cuda_home is None

    def test_cuda_home_points_nowhere(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.collectors.cuda.find_executable", lambda name: None)
        info = CUDACollector(
            FakeRunner(),
            env={"CUDA_HOME": "/definitely/not/here", "PATH": "/usr/bin"},
            platform=Platform.LINUX,
        ).collect()
        assert info.cuda_home_exists is False
        assert info.cuda_home_has_nvcc is False

    def test_windows_env_vars(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.collectors.cuda.find_executable", lambda name: None)
        info = CUDACollector(
            FakeRunner(),
            env={
                "CUDA_PATH": r"C:\CUDA\v12.4",
                "CUDA_PATH_V11_8": r"C:\CUDA\v11.8",
                "PATH": r"C:\Windows",
            },
            platform=Platform.WINDOWS,
        ).collect()
        # On Windows CUDA_PATH acts as the effective home when CUDA_HOME is unset.
        assert info.cuda_home == r"C:\CUDA\v12.4"
        assert info.cuda_path == r"C:\CUDA\v12.4"
        assert info.windows_cuda_path_vars == {"CUDA_PATH_V11_8": r"C:\CUDA\v11.8"}


def _fake_torch(cuda: str | None, available: bool = True, devices: int = 1):
    def import_torch(name):
        assert name == "torch"
        return types.SimpleNamespace(
            __version__="2.4.0+cu121",
            version=types.SimpleNamespace(cuda=cuda),
            cuda=types.SimpleNamespace(
                is_available=lambda: available,
                device_count=lambda: devices if available else 0,
                get_device_name=lambda i: f"NVIDIA Fake GPU {i}",
                get_device_capability=lambda i: (9, 0),
            ),
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(version=lambda: 90100)
            ),
        )

    return import_torch


class TestPyTorchCollector:
    def test_not_installed(self):
        def raises(name):
            # What a genuinely absent torch raises on a real interpreter.
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        info = PyTorchCollector(raises).collect()
        assert info.installed is False
        assert info.import_error is None

    def test_broken_submodule_is_not_reported_as_absent(self):
        def raises(name):
            # torch/ exists but e.g. torch._C is missing/corrupted.
            raise ModuleNotFoundError("No module named 'torch._C'", name="torch._C")

        info = PyTorchCollector(raises).collect()
        assert info.installed is True
        assert info.import_error and "torch._C" in info.import_error

    def test_import_failure_captured(self):
        def raises(name):
            raise RuntimeError("libcuda.so.1: cannot open shared object file")

        info = PyTorchCollector(raises).collect()
        # The package is present; only the import itself failed.
        assert info.installed is True
        assert info.import_error and "libcuda" in info.import_error

    def test_full_cuda_torch(self):
        info = PyTorchCollector(_fake_torch("12.1")).collect()
        assert info.installed is True
        assert info.version == "2.4.0+cu121"
        assert info.cuda_version == "12.1"
        assert info.is_cuda_build is True
        assert info.cuda_available is True
        assert info.device_count == 1
        assert info.cudnn_version == "90100"
        assert info.devices[0].compute_capability == "9.0"

    def test_cpu_only_build(self):
        info = PyTorchCollector(_fake_torch(None, available=False)).collect()
        assert info.is_cuda_build is False
        assert info.cuda_available is False
        assert info.devices == []

    def test_cuda_build_unavailable(self):
        info = PyTorchCollector(_fake_torch("12.1", available=False)).collect()
        assert info.is_cuda_build is True
        assert info.cuda_available is False
        assert info.device_count is None


class TestCompilerCollectorLinux:
    @pytest.fixture(autouse=True)
    def fake_executables(self, monkeypatch):
        monkeypatch.setattr(
            "cuda_doctor.collectors.find_executable",
            lambda name: f"/usr/bin/{name}" if name in ("gcc", "g++") else None,
        )

    def test_probe_results(self):
        runner = FakeRunner({("--version",): ok("gcc (Ubuntu 11.4.0-1ubuntu1) 11.4.0")})
        info = CompilerCollector(
            runner, env={"PATH": "/usr/bin"}, platform=Platform.LINUX
        ).collect()
        by_name = {tool.name: tool for tool in info.compilers}
        assert by_name["gcc"].found and by_name["gcc"].version == "11.4.0"
        assert by_name["g++"].found is True
        assert by_name["clang"].found is False
        assert by_name["clang++"].found is False
        assert info.any_found is True


class TestBuildTools:
    def test_cmake(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.collectors.find_executable", lambda name: "/usr/bin/cmake")
        info = CMakeCollector(FakeRunner({("--version",): ok("cmake version 3.28.3")})).collect()
        assert info.found is True
        assert info.version == "3.28.3"
        assert info.path == "/usr/bin/cmake"

    def test_ninja(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.collectors.find_executable", lambda name: None)
        info = NinjaCollector(FakeRunner()).collect()
        assert info.found is False

    def test_ninja_version(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.collectors.find_executable", lambda name: "/usr/bin/ninja")
        info = NinjaCollector(FakeRunner({("--version",): ok("1.11.1")})).collect()
        assert info.version == "1.11.1"


class TestEnvironmentCollector:
    def test_linux(self):
        info = EnvironmentCollector(
            env={
                "PATH": "/usr/local/cuda-12.4/bin:/usr/bin:/opt/cuda/lib",
                "CUDA_HOME": "/usr/local/cuda-12.4",
                "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:/usr/lib/x86_64",
                "IRRELEVANT_SECRET": "should-not-appear",
            },
            platform=Platform.LINUX,
        ).collect()
        assert info.variables == {"CUDA_HOME": "/usr/local/cuda-12.4"}
        assert info.cuda_path_entries == [(0, "/usr/local/cuda-12.4/bin"), (2, "/opt/cuda/lib")]
        assert info.cuda_ld_library_entries == ["/usr/local/cuda-12.4/lib64"]
        assert info.ld_library_path == "/usr/local/cuda-12.4/lib64:/usr/lib/x86_64"
        # Only relevant variables are ever collected.
        assert all("SECRET" not in key for key in info.variables)

    def test_windows_ignores_ld_library_path(self):
        info = EnvironmentCollector(
            env={"PATH": r"C:\CUDA\v12.4\bin;C:\Windows", "CUDA_PATH": r"C:\CUDA\v12.4"},
            platform=Platform.WINDOWS,
        ).collect()
        assert info.variables == {"CUDA_PATH": r"C:\CUDA\v12.4"}
        assert info.cuda_path_entries == [(0, r"C:\CUDA\v12.4\bin")]
        assert info.ld_library_path is None

    def test_linux_env_names_are_case_sensitive(self):
        # A lowercase "cuda_home" is a *different* variable on Linux and must
        # not be conflated with CUDA_HOME (or "path" with PATH).
        info = EnvironmentCollector(
            env={"cuda_home": "/opt/cuda-12.4", "path": "/opt/cuda-12.4/bin:/usr/bin"},
            platform=Platform.LINUX,
        ).collect()
        assert info.variables == {}
        assert info.cuda_path_entries == []

    def test_windows_env_names_fold_case(self):
        info = EnvironmentCollector(
            env={"cuda_path": r"C:\CUDA\v12.4", "path": r"C:\CUDA\v12.4\bin;C:\Windows"},
            platform=Platform.WINDOWS,
        ).collect()
        assert info.variables == {"CUDA_PATH": r"C:\CUDA\v12.4"}
        assert info.cuda_path_entries == [(0, r"C:\CUDA\v12.4\bin")]

    def test_pathsep_follows_target_platform_not_host(self):
        # The separator must come from the injected target platform, never
        # from os.pathsep (each target is simulated on any host).
        windows_path = r"C:\CUDA\v12.4\bin;C:\Windows"
        linux_path = "/usr/local/cuda-12.4/bin:/usr/bin"

        windows_target = EnvironmentCollector(
            env={"PATH": windows_path}, platform=Platform.WINDOWS
        ).collect()
        assert windows_target.cuda_path_entries == [(0, r"C:\CUDA\v12.4\bin")]

        linux_target = EnvironmentCollector(
            env={"PATH": linux_path}, platform=Platform.LINUX
        ).collect()
        assert linux_target.cuda_path_entries == [(0, "/usr/local/cuda-12.4/bin")]

        # A Linux-style PATH seen by a Windows target stays one unsplit
        # entry (';' never occurs) — proving ';' was the separator used.
        cross = EnvironmentCollector(
            env={"PATH": linux_path}, platform=Platform.WINDOWS
        ).collect()
        assert cross.path_entries == [(0, linux_path)]

        # A Windows-style PATH seen by a Linux target splits on ':' into
        # drive-less fragments — proving ':' was the separator used.
        cross = EnvironmentCollector(
            env={"PATH": windows_path}, platform=Platform.LINUX
        ).collect()
        assert [entry for _, entry in cross.path_entries] == [
            "C",
            r"\CUDA\v12.4\bin;C",
            r"\Windows",
        ]


class TestPythonEnvCollector:
    def test_system_python_outside_venv(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        info = PythonEnvCollector(env={}).collect()
        assert info.in_virtual_env is False
        assert info.virtual_env_path is None
        assert info.version == platform.python_version()
        assert info.executable

    def test_venv_detected_via_prefix(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/opt/venv")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        info = PythonEnvCollector(env={}).collect()
        assert info.in_virtual_env is True
        assert info.virtual_env_path == "/opt/venv"

    def test_venv_detected_via_virtual_env_variable(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        info = PythonEnvCollector(env={"VIRTUAL_ENV": "/opt/venv"}).collect()
        assert info.in_virtual_env is True

    def test_pip_version_reported(self, monkeypatch):
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "24.0")
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        info = PythonEnvCollector(env={}).collect()
        assert info.pip_version == "24.0"

    def test_pip_version_absent_is_not_an_error(self, monkeypatch):
        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", missing)
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        info = PythonEnvCollector(env={}).collect()
        assert info.pip_version is None
