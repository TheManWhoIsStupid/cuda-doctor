"""Tests for CUDA path discovery helpers (uses temporary directories)."""

from __future__ import annotations

from cuda_doctor.core.enums import Platform
from cuda_doctor.utils.paths import (
    discover_cuda_installations,
    is_cuda_related,
    nvcc_within,
)


def _make_layout(tmp_path):
    (tmp_path / "cuda-12.4").mkdir()
    (tmp_path / "cuda-11.8").mkdir()
    (tmp_path / "cuda").mkdir()
    (tmp_path / "cudnn").mkdir()  # must be ignored
    (tmp_path / "unrelated").mkdir()


class TestDiscoverLinux:
    def test_layout_found_and_sorted_newest_first(self, tmp_path):
        _make_layout(tmp_path)
        installs = discover_cuda_installations(Platform.LINUX, roots=(str(tmp_path),))
        paths = [install.path for install in installs]
        assert len(installs) == 3
        assert paths[0].endswith("cuda-12.4")
        assert paths[1].endswith("cuda-11.8")
        assert any(p.endswith("/cuda") for p in paths)
        by_name = {install.path: install for install in installs}
        assert by_name[str(tmp_path / "cuda-12.4")].version == "12.4"
        assert by_name[str(tmp_path / "cuda")].version is None

    def test_missing_root(self):
        assert discover_cuda_installations(Platform.LINUX, roots=("/definitely/not/here",)) == []


class TestDiscoverWindows:
    def test_versioned_dirs_and_env_vars(self, tmp_path):
        (tmp_path / "v12.4").mkdir()
        (tmp_path / "v11.8").mkdir()
        (tmp_path / "junk").mkdir()
        other = tmp_path / "env-toolkit"
        other.mkdir()
        installs = discover_cuda_installations(
            Platform.WINDOWS,
            env={"CUDA_PATH_V11_8": str(other)},
            roots=(str(tmp_path),),
        )
        paths = [install.path for install in installs]
        assert any(p.endswith("v12.4") for p in paths)
        assert any(p.endswith("v11.8") for p in paths)
        assert not any(p.endswith("junk") for p in paths)
        assert str(other) in paths


class TestNvccWithin:
    def test_matches(self, tmp_path):
        home = tmp_path / "cuda-12.4"
        nvcc = home / "bin" / "nvcc"
        nvcc.parent.mkdir(parents=True)
        nvcc.touch()
        assert nvcc_within(str(home), str(nvcc)) is True

    def test_different_home(self, tmp_path):
        home = tmp_path / "cuda-12.4"
        home.mkdir()
        other = tmp_path / "cuda-11.8" / "bin" / "nvcc"
        other.parent.mkdir(parents=True)
        other.touch()
        assert nvcc_within(str(home), str(other)) is False

    def test_empty_inputs(self):
        assert nvcc_within("", "/x/nvcc") is False
        assert nvcc_within("/x", "") is False


class TestIsCudaRelated:
    def test_positive(self):
        assert is_cuda_related("/usr/local/cuda-12.4/bin") is True
        assert is_cuda_related(r"C:\CUDA\v12.4\lib\x64") is True
        assert is_cuda_related("/opt/cuda/include") is True

    def test_negative(self):
        assert is_cuda_related("/usr/bin") is False
        assert is_cuda_related("") is False
