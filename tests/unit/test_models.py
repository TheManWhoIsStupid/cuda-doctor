"""Tests for the normalized environment models."""

from __future__ import annotations

from dataclasses import replace

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import (
    CompilerInfo,
    CUDAInfo,
    CUDAInstallation,
    CUDASelectorObservation,
    EnvironmentInfo,
    EnvironmentSnapshot,
    PythonInfo,
    PyTorchInfo,
    RuntimeLibraryCandidate,
    RuntimeLibraryInventory,
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


class TestV02ObservationModelDefaults:
    """v0.2 observation models construct safely with safe defaults only."""

    def test_cuda_selector_observation(self):
        obs = CUDASelectorObservation(name="cuda_home")
        assert obs.raw_value is None
        assert obs.resolved_path is None
        assert obs.canonical_root is None
        assert obs.toolkit_version is None
        assert obs.exists is None
        assert obs.valid is None
        assert obs.version_source is None

    def test_cuda_info_selector_observations_default_empty(self):
        assert CUDAInfo().selector_observations == []

    def test_runtime_library_inventory(self):
        inventory = RuntimeLibraryInventory()
        assert inventory.candidates == []
        assert inventory.scan_errors == {}

    def test_runtime_library_candidate(self):
        candidate = RuntimeLibraryCandidate(family="libcudart", path="/usr/lib/libcudart.so")
        assert candidate.canonical_path is None
        assert candidate.soname is None
        assert candidate.version is None
        assert candidate.version_source is None
        assert candidate.origin == ""
        assert candidate.search_group == ""
        assert candidate.search_order is None

    def test_environment_info_ordered_entries_default_empty(self):
        assert EnvironmentInfo().ld_library_path_entries == []


class TestOrderedLdLibraryPathNeverSerialized:
    def _snapshot_with_ordered_entries(self) -> EnvironmentSnapshot:
        snapshot = make_snapshot()
        snapshot.environment = EnvironmentInfo(
            variables={"CUDA_HOME": "/usr/local/cuda-12.4"},
            path_entries=[(0, f"{HOME}/bin")],
            ld_library_path=f"{HOME}/lib:/usr/lib",
            ld_library_path_entries=[(0, f"{HOME}/lib"), (1, "/usr/lib")],
        )
        return snapshot

    def test_ordered_entries_absent_from_snapshot_to_dict(self):
        data = snapshot_to_dict(self._snapshot_with_ordered_entries(), home=HOME)
        env = data["environment"]
        assert "ld_library_path_entries" not in env
        assert "path_entries" not in env
        assert "ld_library_path" not in env

    def test_ordered_entries_absent_even_without_redaction(self):
        # The exclusion happens before redaction: no configuration of
        # snapshot_to_dict can emit the ordered full LD_LIBRARY_PATH list.
        data = snapshot_to_dict(self._snapshot_with_ordered_entries(), redact=False)
        assert "ld_library_path_entries" not in data["environment"]

    def test_home_bearing_entries_not_reconstructed_elsewhere(self):
        data = snapshot_to_dict(self._snapshot_with_ordered_entries(), home=HOME)
        assert f"{HOME}/lib" not in str(data)


class TestSelectorObservationSerialization:
    def test_selector_observations_serialize_redacted_like_sibling_fields(self):
        # selector_observations intentionally duplicate already-serialized
        # fields (nvcc_path et al.); they must pass through the same
        # recursive redaction as every other snapshot path.
        snapshot = make_snapshot()
        snapshot.cuda = replace(
            snapshot.cuda,
            selector_observations=[
                CUDASelectorObservation(
                    name="cudacxx",
                    raw_value=f"{HOME}/cuda/bin/nvcc",
                    resolved_path=f"{HOME}/cuda/bin/nvcc",
                    canonical_root=f"{HOME}/cuda",
                )
            ],
        )
        data = snapshot_to_dict(snapshot, home=HOME)
        obs = data["cuda"]["selector_observations"][0]
        assert obs["name"] == "cudacxx"
        assert obs["raw_value"] == "~/cuda/bin/nvcc"
        assert obs["canonical_root"] == "~/cuda"
