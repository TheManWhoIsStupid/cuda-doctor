"""Tests for the collection orchestrator."""

from __future__ import annotations

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.runner import CollectionRunner


def test_real_collection_never_crashes():
    snapshot = CollectionRunner().collect()
    assert snapshot.system.os_name
    assert snapshot.system.python_version
    assert snapshot.collection_errors == {}
    assert snapshot.collected_at


def test_crashing_collector_is_isolated(monkeypatch):
    class ExplodingSystem:
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
        "cuda_doctor.collectors.cuda.find_executable",
        lambda name: "/opt/cuda/bin/nvcc" if name == "nvcc" else None,
    )
    (tmp_path / "cuda-12.4").mkdir()
    snapshot = CollectionRunner(
        platform=Platform.LINUX,
        env={"PATH": "/usr/bin", "CUDA_HOME": str(tmp_path / "cuda-12.4")},
        roots=(str(tmp_path),),
    ).collect()
    assert snapshot.cuda.cuda_home == str(tmp_path / "cuda-12.4")
    assert [install.version for install in snapshot.cuda.installations] == ["12.4"]
