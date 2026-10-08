"""Tests for the normalized environment models."""

from __future__ import annotations

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import (
    CompilerInfo,
    CUDAInfo,
    CUDAInstallation,
    EnvironmentSnapshot,
    PythonInfo,
    PyTorchInfo,
    SystemInfo,
    ToolInfo,
    snapshot_to_dict,
)

HOME = "/home/alice"


def make_snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        system=SystemInfo(
            os_name="Linux",
            os_version="22.04",
            kernel_version="5.15.0",
            architecture="x86_64",
            platform=Platform.LINUX,
            python_version="3.11.15",
            python_executable=f"{HOME}/venv/bin/python",
        ),
        cuda=CUDAInfo(
            nvcc_found=True,
            nvcc_path=f"{HOME}/cuda/bin/nvcc",
            toolkit_version="12.4",
            cuda_home=f"{HOME}/cuda",
            installations=[CUDAInstallation(path="/usr/local/cuda-12.4", version="12.4")],
        ),
        python=PythonInfo(version="3.11.15", executable=f"{HOME}/venv/bin/python"),
        pytorch=PyTorchInfo(installed=True, version="2.4.0", cuda_version="12.1"),
        compiler=CompilerInfo(
            compilers=[ToolInfo(name="gcc", found=True, path="/usr/bin/gcc", version="11.4.0")]
        ),
        cmake=ToolInfo(name="cmake", found=True, path="/usr/bin/cmake", version="3.28.3"),
        ninja=ToolInfo(name="ninja", found=False),
    )


class TestSnapshotToDict:
    def test_redacts_paths(self):
        data = snapshot_to_dict(make_snapshot(), home=HOME)
        assert data["system"]["python_executable"] == "~/venv/bin/python"
        assert data["cuda"]["nvcc_path"] == "~/cuda/bin/nvcc"
        # Non-home paths stay intact.
        assert data["cuda"]["installations"][0]["path"] == "/usr/local/cuda-12.4"

    def test_enums_become_plain_values(self):
        data = snapshot_to_dict(make_snapshot(), home=HOME)
        assert data["system"]["platform"] == "linux"

    def test_no_redaction_normalizes_only(self):
        data = snapshot_to_dict(make_snapshot(), redact=False)
        assert data["system"]["python_executable"] == f"{HOME}/venv/bin/python"
        assert data["system"]["platform"] == "linux"

    def test_collected_at_present(self):
        data = snapshot_to_dict(make_snapshot(), home=HOME)
        assert isinstance(data["collected_at"], str)
        assert data["collected_at"]


class TestDefaults:
    def test_optional_absence_is_not_an_error(self):
        snapshot = make_snapshot()
        assert snapshot.gpus == []
        assert snapshot.driver is None
        assert snapshot.ninja is not None and snapshot.ninja.found is False
        assert snapshot.collection_errors == {}

    def test_compiler_any_found(self):
        assert make_snapshot().compiler.any_found is True
        assert CompilerInfo(compilers=[]).any_found is False
