"""Tests for the collection orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import FakeRunner

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.runner import CollectionRunner


def _no_torch(name: str) -> object:
    raise ModuleNotFoundError(name)


@pytest.mark.parametrize("target", list(Platform))
def test_injected_platform_is_authoritative(target):
    # Regression (Windows CI): the system collector used to read the *host*
    # OS, so a Linux-target simulation collected on a Windows host recorded
    # system.platform=WINDOWS and was diagnosed with Windows compatibility
    # minimums (528.33 instead of the Linux 525.60.13). The injected platform
    # must govern the whole snapshot on any host.
    snapshot = CollectionRunner(
        command_runner=FakeRunner(),
        env={"PATH": "/nonexistent"},
        roots=(),
        platform=target,
        torch_import=_no_torch,
    ).collect()
    assert snapshot.system.platform is target
    assert snapshot.collection_errors == {}


def test_default_platform_is_the_host():
    # Production behavior is unchanged: without an override the snapshot
    # reports the platform the tool actually runs on.
    from cuda_doctor.utils.platform import current_platform

    snapshot = CollectionRunner(
        command_runner=FakeRunner(),
        env={"PATH": "/nonexistent"},
        roots=(),
        torch_import=_no_torch,
    ).collect()
    assert snapshot.system.platform is current_platform()


def test_real_collection_never_crashes():
    snapshot = CollectionRunner().collect()
    assert snapshot.system.os_name
    assert snapshot.system.python_version
    assert snapshot.collection_errors == {}
    assert snapshot.collected_at


def test_crashing_collector_is_isolated(monkeypatch):
    class ExplodingSystem:
        def __init__(self, platform=None):
            pass  # stub matching SystemCollector's signature

        def collect(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("cuda_doctor.core.runner.SystemCollector", ExplodingSystem)
    snapshot = CollectionRunner().collect()
    assert "system" in snapshot.collection_errors
    assert "RuntimeError: boom" in snapshot.collection_errors["system"]
    # Fallback system info keeps the pipeline alive.
    assert snapshot.system.os_name == "Unknown"
    # Other collectors still ran.
    assert snapshot.cuda.nvcc_found in (True, False)


def test_injected_dependencies_flow_through(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cuda_doctor.collectors.cuda.find_executable_on_path",
        lambda name, entries: "/opt/cuda/bin/nvcc" if name == "nvcc" else None,
    )
    (tmp_path / "cuda-12.4").mkdir()
    snapshot = CollectionRunner(
        platform=Platform.LINUX,
        env={"PATH": "/usr/bin", "CUDA_HOME": str(tmp_path / "cuda-12.4")},
        roots=(str(tmp_path),),
    ).collect()
    assert snapshot.cuda.cuda_home == str(tmp_path / "cuda-12.4")
    assert [install.version for install in snapshot.cuda.installations] == ["12.4"]


class TestRuntimeLibraryIntegration:
    def _toolkit_with_libs(self, tmp_path) -> Path:
        toolkit = tmp_path / "cuda-12.4"
        lib64 = toolkit / "lib64"
        lib64.mkdir(parents=True)
        (lib64 / "libcudart.so.12").write_text("", encoding="utf-8")
        return toolkit

    def test_inventory_attached_from_toolkit_root(self, tmp_path, monkeypatch):
        # The toolkit root flows from CUDA collection (canonical root of the
        # resolved cuda_home observation) into the runtime-library collector.
        monkeypatch.setattr(
            "cuda_doctor.collectors.cuda.find_executable_on_path",
            lambda name, entries: None,
        )
        toolkit = self._toolkit_with_libs(tmp_path)
        snapshot = CollectionRunner(
            command_runner=FakeRunner(),
            env={"PATH": "/nonexistent", "CUDA_HOME": str(toolkit)},
            roots=(),
            platform=Platform.LINUX,
            torch_import=_no_torch,
        ).collect()
        assert snapshot.runtime_libraries is not None
        found = {
            (c.family, c.origin) for c in snapshot.runtime_libraries.candidates
        }
        assert ("libcudart", "toolkit") in found
        # The failed ldconfig (FakeRunner default) is a scan error, never a
        # collection error: the collector degrades, it does not crash.
        assert snapshot.collection_errors == {}
        assert "ldconfig" in snapshot.runtime_libraries.scan_errors

    def test_exploding_collector_is_isolated(self, monkeypatch):
        class ExplodingRuntimeLibraries:
            def __init__(self, *args, **kwargs):
                pass

            def collect(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "cuda_doctor.core.runner.RuntimeLibraryCollector", ExplodingRuntimeLibraries
        )
        snapshot = CollectionRunner(
            command_runner=FakeRunner(),
            env={"PATH": "/nonexistent"},
            roots=(),
            platform=Platform.LINUX,
            torch_import=_no_torch,
        ).collect()
        assert snapshot.runtime_libraries is None
        assert "runtime_libraries" in snapshot.collection_errors
        assert "RuntimeError: boom" in snapshot.collection_errors["runtime_libraries"]
        # Other collectors still ran.
        assert snapshot.cuda.nvcc_found is False

    def test_windows_target_gets_empty_not_applicable_inventory(self):
        snapshot = CollectionRunner(
            command_runner=FakeRunner(),
            env={"PATH": r"C:\Windows"},
            roots=(),
            platform=Platform.WINDOWS,
            torch_import=_no_torch,
        ).collect()
        assert snapshot.runtime_libraries is not None
        assert snapshot.runtime_libraries.candidates == []
        assert snapshot.runtime_libraries.scan_errors == {}
        assert "runtime_libraries" not in snapshot.collection_errors
