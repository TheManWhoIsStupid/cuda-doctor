"""Tests for version parsing and comparison."""

from __future__ import annotations

from cuda_doctor.utils.versions import (
    CudaVersion,
    compare_versions,
    cuda_version_from_path,
    format_version,
    parse_cuda_version,
    parse_driver_version,
    parse_version,
)


class TestParseVersion:
    def test_plain(self):
        assert parse_version("12.4.131") == (12, 4, 131)

    def test_embedded_in_text(self):
        assert parse_version("gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0") == (11, 4, 0)

    def test_single_number(self):
        assert parse_version("version 3") == (3,)

    def test_two_components(self):
        assert parse_version("CUDA 12.4") == (12, 4)

    def test_none_and_empty(self):
        assert parse_version(None) is None
        assert parse_version("") is None

    def test_no_digits(self):
        assert parse_version("no version here") is None


class TestParseCudaVersion:
    def test_release_line(self):
        text = "Cuda compilation tools, release 12.4, V12.4.131"
        assert parse_cuda_version(text) == CudaVersion(12, 4)

    def test_major_only_defaults_minor_zero(self):
        assert parse_cuda_version("CUDA 11") == CudaVersion(11, 0)

    def test_invalid(self):
        assert parse_cuda_version("") is None

    def test_str(self):
        assert str(CudaVersion(12, 4)) == "12.4"


class TestParseDriverVersion:
    def test_linux_style(self):
        assert parse_driver_version("580.126.09") == (580, 126, 9)

    def test_windows_style(self):
        assert parse_driver_version("552.22") == (552, 22)


class TestCompareVersions:
    def test_padding_equality(self):
        assert compare_versions((12, 4), (12, 4, 0)) == 0

    def test_less_than(self):
        assert compare_versions((11, 8), (12, 1)) == -1

    def test_greater_than(self):
        assert compare_versions((550,), (450, 80)) == 1

    def test_minor_ordering(self):
        assert compare_versions((12, 9), (12, 10)) == -1

    def test_one_component_branch_minimum(self):
        # Branch-level family minimums such as (580,) must compare correctly
        # against full driver versions via zero padding.
        assert compare_versions((580, 65, 6), (580,)) == 1
        assert compare_versions((590, 44, 1), (580,)) == 1
        assert compare_versions((575, 57, 8), (580,)) == -1
        assert compare_versions((580,), (580, 0, 0)) == 0


class TestCudaVersionFromPath:
    def test_linux_versioned(self):
        assert cuda_version_from_path("/usr/local/cuda-12.4") == CudaVersion(12, 4)

    def test_linux_underscore(self):
        assert cuda_version_from_path("cuda_11.8") == CudaVersion(11, 8)

    def test_no_separator(self):
        assert cuda_version_from_path("cuda12.1") == CudaVersion(12, 1)

    def test_windows_style(self):
        assert cuda_version_from_path(r"C:\CUDA\v11.8") == CudaVersion(11, 8)

    def test_unversioned_default(self):
        assert cuda_version_from_path("/usr/local/cuda") is None

    def test_unrelated_name(self):
        assert cuda_version_from_path("/usr/local/cudnn") is None

    def test_empty(self):
        assert cuda_version_from_path(None) is None


class TestFormatVersion:
    def test_format(self):
        assert format_version((12, 4, 131)) == "12.4.131"

    def test_none(self):
        assert format_version(None) is None
