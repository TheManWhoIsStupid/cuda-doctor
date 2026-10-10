"""Tests for environment collectors (all external I/O faked or injected)."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import tempfile
import types
from pathlib import Path

import pytest
from tests.conftest import FakeRunner, failed, load_fixture, ok

from cuda_doctor.collectors.cmake import CMakeCollector
from cuda_doctor.collectors.compiler import CompilerCollector
from cuda_doctor.collectors.cuda import (
    VERSION_SOURCE_DERIVED_STRONG,
    VERSION_SOURCE_DERIVED_WEAK,
    VERSION_SOURCE_DIRECT,
    VERSION_SOURCE_UNKNOWN,
    CUDACollector,
)
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
            "cuda_doctor.collectors.cuda.find_executable_on_path",
            lambda name, entries: str(home / "bin" / "nvcc") if name == "nvcc" else None,
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
        monkeypatch.setattr(
            "cuda_doctor.collectors.cuda.find_executable_on_path", lambda name, entries: None
        )
        info = CUDACollector(
            FakeRunner(), env={"PATH": "/usr/bin"}, platform=Platform.LINUX
        ).collect()
        assert info.nvcc_found is False
        assert info.toolkit_version is None
        assert info.cuda_home is None

    def test_cuda_home_points_nowhere(self, monkeypatch):
        monkeypatch.setattr(
            "cuda_doctor.collectors.cuda.find_executable_on_path", lambda name, entries: None
        )
        info = CUDACollector(
            FakeRunner(),
            env={"CUDA_HOME": "/definitely/not/here", "PATH": "/usr/bin"},
            platform=Platform.LINUX,
        ).collect()
        assert info.cuda_home_exists is False
        assert info.cuda_home_has_nvcc is False

    def test_windows_env_vars(self, monkeypatch):
        monkeypatch.setattr(
            "cuda_doctor.collectors.cuda.find_executable_on_path", lambda name, entries: None
        )
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


def _make_toolkit(root: Path, name: str = "cuda-12.4", *, nvcc: bool = True) -> Path:
    """A real on-disk toolkit directory; bin/nvcc is a genuine executable."""
    home = root / name
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if nvcc:
        nvcc_bin = bin_dir / "nvcc"
        nvcc_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        nvcc_bin.chmod(0o755)
    return home


def _by_name(info) -> dict:
    return {obs.name: obs for obs in info.selector_observations}


def _version_probes(runner: FakeRunner) -> list[tuple[str, ...]]:
    return [call for call in runner.calls if call[1:] == ("--version",)]


def _can_symlink() -> bool:
    if not hasattr(os, "symlink"):
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.touch()
            os.symlink(target, Path(tmp) / "link")
        return True
    except OSError:
        return False


requires_symlinks = pytest.mark.skipif(not _can_symlink(), reason="symlinks unavailable")


class TestNVCCSelectorObservation:
    def test_agrees_with_legacy_fields(self, tmp_path, fixtures_dir):
        toolkit = _make_toolkit(tmp_path)
        nvcc_bin = toolkit / "bin" / "nvcc"
        runner = FakeRunner(
            {(str(nvcc_bin), "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={"PATH": str(toolkit / "bin")},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["nvcc"]
        assert obs.raw_value == "nvcc"
        assert obs.resolved_path == info.nvcc_path == str(nvcc_bin)
        assert obs.canonical_path == str(nvcc_bin.resolve())
        assert obs.canonical_root == str(toolkit.resolve())
        assert obs.toolkit_version == info.toolkit_version == "12.4"
        assert obs.version_source == VERSION_SOURCE_DIRECT
        assert obs.exists is True and obs.valid is True

    def test_probe_failure_yields_unknown_version(self, tmp_path):
        toolkit = _make_toolkit(tmp_path)
        info = CUDACollector(
            FakeRunner(),  # every probe fails
            env={"PATH": str(toolkit / "bin")},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["nvcc"]
        assert info.nvcc_found is True
        assert obs.toolkit_version is None and info.toolkit_version is None
        assert obs.version_source == VERSION_SOURCE_UNKNOWN

    def test_resolution_uses_injected_path_not_host(self, tmp_path):
        # §10.3: PATH resolution depends only on the injected environment —
        # a host machine with nvcc installed anywhere must not leak in.
        toolkit = _make_toolkit(tmp_path)
        found = CUDACollector(
            FakeRunner(),
            env={"PATH": str(toolkit / "bin")},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        assert found.nvcc_found is True

        missing = CUDACollector(
            FakeRunner(),
            env={"PATH": "/nonexistent-cuda-doctor-xyz"},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        assert missing.nvcc_found is False
        names = _by_name(missing)
        assert "nvcc" not in names and "path_cuda_bin" not in names


class TestPathCudaBinObservation:
    def test_records_winning_entry_only(self, tmp_path, fixtures_dir):
        winning = _make_toolkit(tmp_path, "cuda-12.4")
        shadowed = _make_toolkit(tmp_path, "cuda-11.8")
        winning_nvcc = winning / "bin" / "nvcc"
        runner = FakeRunner(
            {(str(winning_nvcc), "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={
                "PATH": f"{winning / 'bin'}:{shadowed / 'bin'}",
            },
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        observations = info.selector_observations
        path_bins = [obs for obs in observations if obs.name == "path_cuda_bin"]
        assert len(path_bins) == 1  # exactly one, never one per CUDA-looking entry
        obs = path_bins[0]
        assert obs.raw_value == str(winning / "bin")
        assert obs.resolved_path == str(winning_nvcc)
        assert obs.canonical_root == str(winning.resolve())
        # The non-winning CUDA-looking entry is incidental (§16.10) and
        # appears in no observation.
        assert all(str(shadowed / "bin") != obs.raw_value for obs in observations)
        # Version facts are inherited from the resolved nvcc.
        assert obs.toolkit_version == "12.4"
        assert obs.version_source == VERSION_SOURCE_DIRECT


class TestCUDACXXObservation:
    def test_absolute_path_probed_directly(self, tmp_path, fixtures_dir):
        toolkit = _make_toolkit(tmp_path)
        cudacxx = toolkit / "bin" / "nvcc"
        runner = FakeRunner(
            {(str(cudacxx), "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={"PATH": "/nonexistent", "CUDACXX": str(cudacxx)},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["cudacxx"]
        assert obs.raw_value == str(cudacxx)
        assert obs.resolved_path == str(cudacxx)  # path-like values are used as-is
        assert obs.canonical_path == str(cudacxx.resolve())
        assert obs.canonical_root == str(toolkit.resolve())
        assert obs.exists is True and obs.valid is True
        assert obs.toolkit_version == "12.4"
        assert obs.version_source == VERSION_SOURCE_DIRECT

    def test_nonexistent_path_is_invalid_without_probing(self, tmp_path):
        runner = FakeRunner()
        info = CUDACollector(
            runner,
            env={"PATH": "/nonexistent", "CUDACXX": "/definitely/not/here/nvcc"},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["cudacxx"]
        assert obs.exists is False and obs.valid is False
        assert obs.canonical_root is None
        assert obs.toolkit_version is None
        assert obs.version_source == VERSION_SOURCE_UNKNOWN
        assert _version_probes(runner) == []

    def test_version_never_derived_from_path_naming(self, tmp_path, fixtures_dir):
        # §9.5 assigns CUDACXX only the executable-probe source: a toolkit
        # named cuda-11.8 whose probe output is unparseable must stay UNKNOWN,
        # never "11.8" guessed from the directory name.
        toolkit = _make_toolkit(tmp_path, "cuda-11.8")
        cudacxx = toolkit / "bin" / "nvcc"
        runner = FakeRunner(
            {(str(cudacxx), "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_malformed.txt"))}
        )
        info = CUDACollector(
            runner,
            env={"PATH": "/nonexistent", "CUDACXX": str(cudacxx)},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["cudacxx"]
        assert obs.toolkit_version is None
        assert obs.version_source == VERSION_SOURCE_UNKNOWN

    def test_bare_name_resolved_on_injected_path(self, tmp_path, fixtures_dir):
        toolkit = _make_toolkit(tmp_path)
        nvcc_bin = toolkit / "bin" / "nvcc"
        runner = FakeRunner(
            {(str(nvcc_bin), "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={"PATH": f"/nonexistent:{toolkit / 'bin'}", "CUDACXX": "nvcc"},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["cudacxx"]
        assert obs.resolved_path == str(nvcc_bin)
        assert obs.canonical_root == str(toolkit.resolve())
        assert obs.toolkit_version == "12.4"

    @requires_symlinks
    def test_symlinked_cudacxx_resolves_to_real_binary(self, tmp_path, fixtures_dir):
        toolkit = _make_toolkit(tmp_path)
        real_nvcc = toolkit / "bin" / "nvcc"
        shims = tmp_path / "shims"
        shims.mkdir()
        os.symlink(real_nvcc, shims / "nvcc")
        shim_nvcc = str(shims / "nvcc")
        runner = FakeRunner(
            {(shim_nvcc, "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={"PATH": "/nonexistent", "CUDACXX": str(shims / "nvcc")},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["cudacxx"]
        assert obs.resolved_path == str(shims / "nvcc")
        assert obs.canonical_path == str(real_nvcc.resolve())
        assert obs.canonical_root == str(toolkit.resolve())

    def test_empty_cudacxx_is_no_selector(self, tmp_path):
        info = CUDACollector(
            FakeRunner(),
            env={"PATH": "/nonexistent", "CUDACXX": ""},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        assert "cudacxx" not in _by_name(info)


class TestCUDAHomeObservation:
    def _collect(self, tmp_path, env_extra: dict, runner=None):
        return CUDACollector(
            runner or FakeRunner(),
            env={"PATH": "/nonexistent", **env_extra},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()

    def test_direct_version_from_home_nvcc_probed_once(self, tmp_path, fixtures_dir):
        toolkit = _make_toolkit(tmp_path)
        nvcc_bin = toolkit / "bin" / "nvcc"
        runner = FakeRunner(
            {
                (str(nvcc_bin), "--version"): ok(
                    load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt")
                )
            }
        )
        # The same binary is reachable both through PATH and CUDA_HOME/bin;
        # it must be probed exactly once (§33 performance budget).
        info = CUDACollector(
            runner,
            env={"PATH": str(toolkit / "bin"), "CUDA_HOME": str(toolkit)},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        assert len(_version_probes(runner)) == 1
        nvcc_obs = _by_name(info)["nvcc"]
        home_obs = _by_name(info)["cuda_home"]
        assert home_obs.raw_value == str(toolkit)
        assert home_obs.exists is True and home_obs.valid is True
        assert home_obs.canonical_root == str(toolkit.resolve())
        assert home_obs.toolkit_version == "12.4" == nvcc_obs.toolkit_version
        assert home_obs.version_source == VERSION_SOURCE_DIRECT
        # Legacy fields stay consistent with the observation.
        assert info.cuda_home_exists is True and info.cuda_home_has_nvcc is True

    def test_version_falls_back_to_root_naming(self, tmp_path):
        # No bin/nvcc inside; only the canonical root name carries a version.
        toolkit = _make_toolkit(tmp_path, "cuda-11.8", nvcc=False)
        info = self._collect(tmp_path, {"CUDA_HOME": str(toolkit)})
        obs = _by_name(info)["cuda_home"]
        assert obs.toolkit_version == "11.8"
        assert obs.version_source == VERSION_SOURCE_DERIVED_WEAK

    def test_unversioned_root_stays_unknown(self, tmp_path):
        # Conda-style roots carry no version in their naming (§20.6):
        # never guessed.
        toolkit = _make_toolkit(tmp_path, "env-torch", nvcc=False)
        info = self._collect(tmp_path, {"CUDA_HOME": str(toolkit)})
        obs = _by_name(info)["cuda_home"]
        assert obs.toolkit_version is None
        assert obs.version_source == VERSION_SOURCE_UNKNOWN
        assert obs.exists is True and obs.valid is True

    def test_nonexistent_home_is_invalid(self, tmp_path):
        info = self._collect(tmp_path, {"CUDA_HOME": "/definitely/not/here"})
        obs = _by_name(info)["cuda_home"]
        assert obs.exists is False and obs.valid is False
        assert obs.canonical_root is None
        assert obs.toolkit_version is None
        assert obs.version_source == VERSION_SOURCE_UNKNOWN
        assert info.cuda_home_exists is False  # legacy behavior unchanged

    @requires_symlinks
    def test_alias_and_nvcc_normalize_to_same_root(self, tmp_path, fixtures_dir):
        real = _make_toolkit(tmp_path, "cuda-12.4")
        alias = tmp_path / "cuda"
        os.symlink(real, alias)
        runner = FakeRunner(
            {(str(real / "bin" / "nvcc"), "--version"): ok(
                load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt")
            )}
        )
        info = CUDACollector(
            runner,
            env={"PATH": str(real / "bin"), "CUDA_HOME": str(alias)},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        by_name = _by_name(info)
        # Raw strings differ, canonical identities do not (path-string
        # inequality is not toolkit-identity inequality).
        assert by_name["cuda_home"].raw_value == str(alias)
        assert by_name["nvcc"].resolved_path == str(real / "bin" / "nvcc")
        assert by_name["cuda_home"].canonical_root == by_name["nvcc"].canonical_root
        # Alias and real binary are the same file: still one probe.
        assert len(_version_probes(runner)) == 1


class TestCudaPathObservationLinux:
    def test_recorded_on_linux(self, tmp_path):
        # CUDA_PATH is collected on Linux too; treating it as an inactive
        # D001 selector is the fact layer's decision, not the collector's.
        info = CUDACollector(
            FakeRunner(),
            env={"PATH": "/nonexistent", "CUDA_PATH": "/opt/cuda-12.4"},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        obs = _by_name(info)["cuda_path"]
        assert obs.raw_value == "/opt/cuda-12.4"
        assert info.cuda_path == "/opt/cuda-12.4"  # legacy field unchanged


class TestUsrLocalCudaObservation:
    def _usr_root(self, tmp_path: Path) -> Path:
        usr_root = tmp_path / "usr-local"
        usr_root.mkdir(exist_ok=True)
        return usr_root

    def _collect(self, tmp_path, usr_root: Path):
        return CUDACollector(
            FakeRunner(),
            env={"PATH": "/nonexistent"},
            platform=Platform.LINUX,
            usr_local_cuda_path=str(usr_root / "cuda"),
        ).collect()

    def test_absent(self, tmp_path):
        info = self._collect(tmp_path, self._usr_root(tmp_path))
        obs = _by_name(info)["usr_local_cuda"]
        assert obs.exists is False and obs.valid is False
        assert obs.canonical_root is None
        assert obs.toolkit_version is None
        assert obs.version_source == VERSION_SOURCE_UNKNOWN

    def test_plain_directory_without_version(self, tmp_path):
        usr_root = self._usr_root(tmp_path)
        (usr_root / "cuda").mkdir()
        info = self._collect(tmp_path, usr_root)
        obs = _by_name(info)["usr_local_cuda"]
        assert obs.exists is True and obs.valid is True
        assert obs.canonical_root == str((usr_root / "cuda").resolve())
        assert obs.toolkit_version is None
        assert obs.version_source == VERSION_SOURCE_UNKNOWN

    @requires_symlinks
    def test_symlink_target_naming_derives_version(self, tmp_path):
        usr_root = self._usr_root(tmp_path)
        (usr_root / "cuda-12.4").mkdir()
        os.symlink(usr_root / "cuda-12.4", usr_root / "cuda")
        info = self._collect(tmp_path, usr_root)
        obs = _by_name(info)["usr_local_cuda"]
        assert obs.exists is True and obs.valid is True
        assert obs.canonical_root == str((usr_root / "cuda-12.4").resolve())
        assert obs.toolkit_version == "12.4"
        assert obs.version_source == VERSION_SOURCE_DERIVED_WEAK

    @requires_symlinks
    def test_stale_symlink_records_dangling_target(self, tmp_path):
        # §16.9: a stale symlink is inventory; its target naming stays
        # observable even though the target directory is gone.
        usr_root = self._usr_root(tmp_path)
        os.symlink(usr_root / "cuda-11.8", usr_root / "cuda")  # target absent
        info = self._collect(tmp_path, usr_root)
        obs = _by_name(info)["usr_local_cuda"]
        assert obs.exists is False and obs.valid is False
        assert obs.canonical_root == str(usr_root / "cuda-11.8")
        assert obs.toolkit_version == "11.8"
        assert obs.version_source == VERSION_SOURCE_DERIVED_WEAK


class TestSelectorObservationPlatformGating:
    def test_windows_target_has_no_usr_local_cuda(self, tmp_path):
        usr_root = tmp_path / "usr-local"
        usr_root.mkdir()
        (usr_root / "cuda").mkdir()  # exists, but Windows never consults it
        info = CUDACollector(
            FakeRunner(),
            env={"PATH": "C:\\Windows"},
            platform=Platform.WINDOWS,
            usr_local_cuda_path=str(usr_root / "cuda"),
        ).collect()
        assert "usr_local_cuda" not in _by_name(info)

    def test_other_platforms_get_the_observation(self, tmp_path):
        for target in (Platform.LINUX, Platform.MACOS, Platform.OTHER):
            info = CUDACollector(
                FakeRunner(),
                env={"PATH": "/nonexistent"},
                platform=target,
                usr_local_cuda_path=str(tmp_path / "absent"),
            ).collect()
            assert "usr_local_cuda" in _by_name(info)


class TestSelectorVersionSourceVocabulary:
    def test_emitted_sources_stay_in_frozen_vocabulary(self, tmp_path, fixtures_dir):
        toolkit = _make_toolkit(tmp_path)
        nvcc_bin = toolkit / "bin" / "nvcc"
        # A versioned root without bin/nvcc can only derive its version
        # from naming (DERIVED_WEAK); the absent /usr/local/cuda stays UNKNOWN.
        versioned_without_nvcc = _make_toolkit(tmp_path, "cuda-11.8", nvcc=False)
        runner = FakeRunner(
            {(str(nvcc_bin), "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={
                "PATH": str(toolkit / "bin"),
                "CUDA_HOME": str(toolkit),
                "CUDA_PATH": str(versioned_without_nvcc),
                "CUDACXX": str(nvcc_bin),
            },
            platform=Platform.LINUX,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        vocabulary = {
            VERSION_SOURCE_DIRECT,
            VERSION_SOURCE_DERIVED_STRONG,
            VERSION_SOURCE_DERIVED_WEAK,
            VERSION_SOURCE_UNKNOWN,
        }
        assert info.selector_observations
        for obs in info.selector_observations:
            assert obs.version_source in vocabulary
        sources = {obs.version_source for obs in info.selector_observations}
        assert VERSION_SOURCE_DIRECT in sources
        assert VERSION_SOURCE_DERIVED_WEAK in sources
        assert VERSION_SOURCE_UNKNOWN in sources


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

    def test_v02_environment_variables_captured(self):
        info = EnvironmentCollector(
            env={
                "PATH": "/usr/bin",
                "CUDACXX": "/opt/cuda-12.4/bin/nvcc",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "CONDA_PREFIX": "/opt/conda/envs/torch",
            },
            platform=Platform.LINUX,
        ).collect()
        assert info.variables["CUDACXX"] == "/opt/cuda-12.4/bin/nvcc"
        assert info.variables["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert info.variables["CONDA_PREFIX"] == "/opt/conda/envs/torch"

    def test_cuda_visible_devices_empty_string_is_meaningful(self):
        # "" masks all GPUs (Phase-0-verified CUDA semantics), so it must be
        # preserved as an observed value — distinct from the variable being
        # unset (absent from `variables` entirely).
        info = EnvironmentCollector(
            env={"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": ""}, platform=Platform.LINUX
        ).collect()
        assert info.variables["CUDA_VISIBLE_DEVICES"] == ""

    def test_scalar_variables_keep_truthy_only_semantics(self):
        # v0.1.x behavior unchanged: an empty CUDA_HOME is NOT promoted to an
        # observation (only the new PRESENCE_VARIABLES capture empty values).
        info = EnvironmentCollector(
            env={"PATH": "/usr/bin", "CUDA_HOME": ""}, platform=Platform.LINUX
        ).collect()
        assert info.variables == {}

    def test_cuda_visible_devices_captured_on_windows_with_case_folding(self):
        info = EnvironmentCollector(
            env={"PATH": r"C:\Windows", "cuda_visible_devices": "-1"},
            platform=Platform.WINDOWS,
        ).collect()
        assert info.variables["CUDA_VISIBLE_DEVICES"] == "-1"

    def test_ld_library_path_entries_ordered_linux(self):
        info = EnvironmentCollector(
            env={"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/opt/lib"},
            platform=Platform.LINUX,
        ).collect()
        assert info.ld_library_path_entries == [
            (0, "/usr/local/cuda/lib64"),
            (1, "/opt/lib"),
        ]

    def test_ld_library_path_entries_preserve_index_across_empty_segments(self):
        info = EnvironmentCollector(
            env={"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/a::/b:"},
            platform=Platform.LINUX,
        ).collect()
        # Empty segments are skipped exactly like PATH entries, while the
        # original positions (indexes) are kept.
        assert info.ld_library_path_entries == [(0, "/a"), (2, "/b")]

    def test_ld_library_path_entries_empty_when_unset(self):
        info = EnvironmentCollector(env={"PATH": "/usr/bin"}, platform=Platform.LINUX).collect()
        assert info.ld_library_path is None
        assert info.ld_library_path_entries == []

    def test_windows_target_has_no_ordered_ld_library_path(self):
        # The ordered representation is gated exactly like the raw
        # ld_library_path: a Windows target never collects it, regardless of
        # what the (simulated) environment contains or the host OS is.
        info = EnvironmentCollector(
            env={
                "PATH": r"C:\Windows",
                "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/opt/lib",
            },
            platform=Platform.WINDOWS,
        ).collect()
        assert info.ld_library_path is None
        assert info.ld_library_path_entries == []


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
