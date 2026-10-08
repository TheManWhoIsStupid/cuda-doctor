"""End-to-end pipeline tests: fake machines through collectors -> engine -> reporters."""

from __future__ import annotations

import json

import pytest
from tests.conftest import FakeRunner, load_fixture, ok
from typer.testing import CliRunner

from cuda_doctor.cli import app
from cuda_doctor.core.enums import EnvironmentStatus, Platform
from cuda_doctor.core.runner import CollectionRunner
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.reporters import JsonReporter, MarkdownReporter, ReportInputs

SMI = "/fake/bin/nvidia-smi"
NVCC = "/fake/bin/nvcc"
GCC = "/fake/bin/gcc"
GXX = "/fake/bin/g++"
CMAKE = "/fake/bin/cmake"
NINJA = "/fake/bin/ninja"

CSV_ARGS = (
    "--query-gpu=index,name,uuid,memory.total,compute_cap",
    "--format=csv,noheader",
)
XML_ARGS = ("-q", "-x")

FAKE_PATHS = {
    "nvidia-smi": SMI,
    "nvcc": NVCC,
    "gcc": GCC,
    "g++": GXX,
    "cmake": CMAKE,
    "ninja": NINJA,
}


class FakeTorchVersion:
    cuda = "12.4"


class FakeTorchCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(index):
        return "NVIDIA GeForce RTX 4090"

    @staticmethod
    def get_device_capability(index):
        return (8, 9)


class FakeCudnn:
    @staticmethod
    def version():
        return 90100


class FakeBackends:
    cudnn = FakeCudnn


class FakeTorch:
    __version__ = "2.6.0+cu124"
    version = FakeTorchVersion
    cuda = FakeTorchCuda
    backends = FakeBackends


class FakeCpuTorch:
    __version__ = "2.6.0+cpu"

    class version:
        cuda = None

    class cuda:
        @staticmethod
        def is_available():
            return False

    class backends:
        class cudnn:
            @staticmethod
            def version():
                return None


def fake_torch_import(torch_module):
    def _import(name):
        if name != "torch":
            raise ImportError(name)
        return torch_module

    return _import


def broken_torch_import(name):
    raise RuntimeError("libcuda.so.1: cannot open shared object file: No such file")


@pytest.fixture()
def patch_executables(monkeypatch):
    """Route all executable lookups through the FAKE_PATHS table."""

    def _install(paths):
        def lookup(name):
            return paths.get(name)

        monkeypatch.setattr("cuda_doctor.collectors.find_executable", lookup)
        monkeypatch.setattr("cuda_doctor.collectors.cuda.find_executable", lookup)
        monkeypatch.setattr("cuda_doctor.collectors.nvidia_smi.find_executable", lookup)

    return _install


@pytest.fixture()
def fixtures(fixtures_dir):
    class _F:
        csv = load_fixture(fixtures_dir, "nvidia_smi/query_csv_single.txt")
        xml = load_fixture(fixtures_dir, "nvidia_smi/xml_h20.xml")
        nvcc = load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt")

    return _F


def healthy_runner(fixtures, nvcc: str = NVCC) -> FakeRunner:
    return FakeRunner(
        {
            (SMI, *CSV_ARGS): ok(fixtures.csv),
            (SMI, *XML_ARGS): ok(fixtures.xml),
            (nvcc, "--version"): ok(fixtures.nvcc),
            (GCC, "--version"): ok("gcc (Ubuntu 11.4.0) 11.4.0\n"),
            (GXX, "--version"): ok("g++ (Ubuntu 11.4.0) 11.4.0\n"),
            (CMAKE, "--version"): ok("cmake version 3.28.3\n"),
            (NINJA, "--version"): ok("1.11.1\n"),
        }
    )


@pytest.fixture()
def healthy_environment(tmp_path):
    """A coherent on-disk world: CUDA_HOME exists and holds bin/nvcc."""
    toolkit = tmp_path / "toolkits" / "cuda-12.4"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
    plain = tmp_path / "plain-bin"
    plain.mkdir()
    env = {
        "PATH": f"{toolkit / 'bin'}:{plain}",
        "CUDA_HOME": str(toolkit),
    }
    return env, (str(tmp_path / "toolkits"),), toolkit


def healthy_setup(fixtures, toolkit, torch_module=None):
    """Consistent executable table where nvcc lives inside CUDA_HOME."""
    nvcc = str(toolkit / "bin" / "nvcc")
    paths = {**FAKE_PATHS, "nvcc": nvcc}
    runner = healthy_runner(fixtures, nvcc)
    return paths, runner


class TestHealthyMachine:
    def test_full_pipeline_reports_healthy(self, patch_executables, fixtures, healthy_environment):
        env, roots, toolkit = healthy_environment
        paths, runner = healthy_setup(fixtures, toolkit)
        patch_executables(paths)
        snapshot = CollectionRunner(
            platform=Platform.LINUX,
            command_runner=runner,
            env=env,
            roots=roots,
            torch_import=fake_torch_import(FakeTorch),
        ).collect()

        assert snapshot.collection_errors == {}
        assert snapshot.gpus and snapshot.gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert snapshot.driver is not None and snapshot.driver.version == "580.126.09"
        assert snapshot.cuda.toolkit_version == "12.4"
        assert snapshot.cuda.cuda_home_has_nvcc is True
        assert snapshot.pytorch.cuda_available is True

        result = DiagnosisEngine().run(snapshot)
        assert [issue.code for issue in result.issues] == []
        assert result.summary.status is EnvironmentStatus.HEALTHY

    def test_reports_render(self, patch_executables, fixtures, healthy_environment):
        env, roots, toolkit = healthy_environment
        paths, runner = healthy_setup(fixtures, toolkit)
        patch_executables(paths)
        snapshot = CollectionRunner(
            platform=Platform.LINUX,
            command_runner=runner,
            env=env,
            roots=roots,
            torch_import=fake_torch_import(FakeTorch),
        ).collect()
        result = DiagnosisEngine().run(snapshot)
        inputs = ReportInputs(
            snapshot=snapshot, result=result, generated_at="2026-10-08T00:00:00+00:00"
        )
        payload = json.loads(JsonReporter().render(inputs))
        assert payload["summary"]["status"] == "HEALTHY"
        markdown = MarkdownReporter().render(inputs)
        assert markdown.startswith("# CUDA Doctor Report")
        assert "No issues found" in markdown


class TestBrokenMachine:
    def test_missing_everything(self, patch_executables, tmp_path):
        patch_executables({})  # nothing on PATH at all
        snapshot = CollectionRunner(
            platform=Platform.LINUX,
            command_runner=FakeRunner(),
            env={"PATH": "/nonexistent", "CUDA_HOME": str(tmp_path / "gone" / "cuda")},
            roots=(str(tmp_path),),
            torch_import=broken_torch_import,
        ).collect()

        result = DiagnosisEngine().run(snapshot)
        codes = {issue.code for issue in result.issues}
        assert "GPU001" in codes  # no nvidia-smi
        assert "CUDA001" in codes  # no nvcc
        assert "CMP001" in codes  # no compiler
        assert "TORCH005" in codes  # torch import failure (libcuda)
        assert "CUDA003" in codes  # CUDA_HOME points nowhere
        assert result.summary.status is EnvironmentStatus.DEGRADED

    def test_import_failure_matches_known_issue(self, patch_executables, tmp_path):
        patch_executables({})
        snapshot = CollectionRunner(
            platform=Platform.LINUX,
            command_runner=FakeRunner(),
            env={"PATH": "/nonexistent"},
            roots=(str(tmp_path),),
            torch_import=broken_torch_import,
        ).collect()
        result = DiagnosisEngine().run(snapshot)
        torch_issue = next(i for i in result.issues if i.code == "TORCH005")
        assert "NVIDIA driver library" in torch_issue.title
        assert any("libcuda" in r for r in torch_issue.recommendations)

    def test_cli_on_broken_machine(self, patch_executables, tmp_path, monkeypatch):
        patch_executables({})
        monkeypatch.setenv("PATH", "/nonexistent")
        monkeypatch.setenv("CUDA_HOME", str(tmp_path / "gone" / "cuda"))

        def fake_collect():
            return CollectionRunner(
                platform=Platform.LINUX,
                command_runner=FakeRunner(),
                env={"PATH": "/nonexistent", "CUDA_HOME": str(tmp_path / "gone" / "cuda")},
                roots=(str(tmp_path),),
                torch_import=broken_torch_import,
            ).collect()

        monkeypatch.setattr("cuda_doctor.cli._collect", fake_collect)
        cli = CliRunner()
        result = cli.invoke(app, ["diagnose", "--format", "json"])
        assert result.exit_code == 0  # findings never change the exit code
        payload = json.loads(result.output)
        assert payload["summary"]["status"] == "DEGRADED"
        codes = {issue["code"] for issue in payload["issues"]}
        assert {"GPU001", "TORCH005"} <= codes


class TestCpuOnlyMachine:
    def test_cpu_torch_flags_warning(self, patch_executables, fixtures, healthy_environment):
        env, roots, toolkit = healthy_environment
        paths, runner = healthy_setup(fixtures, toolkit)
        patch_executables(paths)
        snapshot = CollectionRunner(
            platform=Platform.LINUX,
            command_runner=runner,
            env=env,
            roots=roots,
            torch_import=fake_torch_import(FakeCpuTorch),
        ).collect()
        result = DiagnosisEngine().run(snapshot)
        codes = [issue.code for issue in result.issues]
        assert "TORCH003" in codes  # CPU-only build
        assert result.summary.status is EnvironmentStatus.USABLE_WITH_WARNINGS


class TestMultiToolkitMachine:
    def test_multiple_installations_and_shadowing(self, patch_executables, fixtures, tmp_path):
        for version in ("11.8", "12.4"):
            toolkit_bin = tmp_path / f"cuda-{version}" / "bin"
            toolkit_bin.mkdir(parents=True)
            (toolkit_bin / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
        nvcc = str(tmp_path / "cuda-11.8" / "bin" / "nvcc")  # first on PATH = CUDA_HOME
        patch_executables({**FAKE_PATHS, "nvcc": nvcc})
        env = {
            "PATH": f"{tmp_path / 'cuda-11.8' / 'bin'}:{tmp_path / 'cuda-12.4' / 'bin'}",
            "CUDA_HOME": str(tmp_path / "cuda-11.8"),
        }
        snapshot = CollectionRunner(
            platform=Platform.LINUX,
            command_runner=healthy_runner(fixtures, nvcc),
            env=env,
            roots=(str(tmp_path),),
            torch_import=fake_torch_import(FakeTorch),
        ).collect()
        result = DiagnosisEngine().run(snapshot)
        codes = [issue.code for issue in result.issues]
        assert "CUDA004" in codes  # multiple toolkits on disk
        assert "CUDA005" in codes  # two cuda bin dirs in PATH
        assert "ENV004" in codes  # older cuda listed first
        assert "CUDA006" not in codes  # nvcc sits inside CUDA_HOME
        assert "TORCH004" not in codes  # torch 12.4 == nvcc 12.4
