"""Tests for the collection orchestrator."""

from __future__ import annotations

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
