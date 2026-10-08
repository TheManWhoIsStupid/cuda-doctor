"""Tests for the compatibility knowledge modules."""

from __future__ import annotations

import pytest

from cuda_doctor.compatibility import (
    DriverCompatibility,
    KnownIssues,
    interpret_torch_cuda,
    load_compatibility_data,
)
from cuda_doctor.core.enums import Platform
from cuda_doctor.utils.versions import CudaVersion


@pytest.fixture(scope="module")
def compat():
    return load_compatibility_data()


class TestDriverCompatibility:
    def test_table_loads(self, compat):
        assert compat.driver.minimum_driver(CudaVersion(12, 4), Platform.LINUX) == (550, 54, 14)

    def test_exact_windows(self, compat):
        assert compat.driver.minimum_driver(CudaVersion(12, 4), Platform.WINDOWS) == (551, 61)

    def test_unknown_minor_falls_back_to_nearest_lower(self, compat):
        # 12.9 is not in the table; 12.6 requirements apply as a lower bound.
        assert compat.driver.minimum_driver(CudaVersion(12, 9), Platform.LINUX) == (560, 28, 4)

    def test_future_major_falls_back_to_newest_known(self, compat):
        assert compat.driver.minimum_driver(CudaVersion(14, 0), Platform.LINUX) == (580, 65, 6)

    def test_evaluate_ok(self, compat):
        verdict = compat.driver.evaluate(
            CudaVersion(12, 4), (550, 54, 14), Platform.LINUX
        )
        assert verdict.compatible is True

    def test_evaluate_too_old(self, compat):
        verdict = compat.driver.evaluate(
            CudaVersion(12, 4), (545, 23), Platform.LINUX
        )
        assert verdict.compatible is False
        assert "requires driver 550.54.14" in verdict.message

    def test_from_json_direct(self):
        data = {"linux": {"12.4": ">=550.54.14"}}
        compat = DriverCompatibility.from_json(data)
        assert compat.minimum_driver(CudaVersion(12, 4), Platform.LINUX) == (550, 54, 14)


class TestCompilerCompatibility:
    def test_max_gcc(self, compat):
        assert compat.compiler.max_supported_gcc_major(CudaVersion(11, 8)) == 11
        assert compat.compiler.max_supported_gcc_major(CudaVersion(12, 4)) == 13

    def test_evaluate_gcc_ok(self, compat):
        verdict = compat.compiler.evaluate_gcc((11, 4, 0), CudaVersion(12, 4))
        assert verdict.compatible is True

    def test_evaluate_gcc_too_new_is_potential_only(self, compat):
        verdict = compat.compiler.evaluate_gcc((14, 2, 0), CudaVersion(12, 4))
        assert verdict.compatible is False
        assert "potential compatibility issue" in verdict.message

    def test_vs_range(self, compat):
        assert compat.compiler.vs_range(CudaVersion(11, 4)) == (15, 16)
        assert compat.compiler.vs_range(CudaVersion(12, 4)) == (16, 17)

    def test_vs_years(self, compat):
        assert compat.compiler.vs_year(17) == 2022

    def test_evaluate_vs_out_of_range(self, compat):
        verdict = compat.compiler.evaluate_visual_studio((17,), CudaVersion(11, 4))
        assert verdict.compatible is False
        assert "potential" in verdict.message

    def test_evaluate_vs_in_range(self, compat):
        verdict = compat.compiler.evaluate_visual_studio((16, 11), CudaVersion(12, 4))
        assert verdict.compatible is True


class TestTorchCudaInterpretation:
    def test_match(self):
        result = interpret_torch_cuda(CudaVersion(12, 4), CudaVersion(12, 4))
        assert result.relation == "match"

    def test_unknown(self):
        result = interpret_torch_cuda(None, CudaVersion(12, 4))
        assert result.relation == "unknown"

    def test_minor_mismatch_explained_as_fine(self):
        result = interpret_torch_cuda(CudaVersion(12, 1), CudaVersion(12, 4))
        assert result.relation == "torch-older"
        assert "usually fine" in result.explanation
        assert "bundle" in result.explanation

    def test_major_mismatch_mentions_extensions(self):
        result = interpret_torch_cuda(CudaVersion(13, 0), CudaVersion(12, 4))
        assert result.relation == "torch-newer"
        assert "custom CUDA extensions" in result.explanation


class TestKnownIssues:
    def test_match_libcuda(self, compat):
        issue = compat.known_issues.match(
            "torch_import", "RuntimeError: libcuda.so.1: cannot open shared object file"
        )
        assert issue is not None and issue.id == "KI-TORCH-001"

    def test_match_undefined_symbol(self, compat):
        issue = compat.known_issues.match("torch_import", "ImportError: undefined symbol: _ZN...")
        assert issue is not None and issue.id == "KI-TORCH-003"

    def test_no_match(self, compat):
        assert compat.known_issues.match("torch_import", "some unrelated error") is None

    def test_no_text(self, compat):
        assert compat.known_issues.match("torch_import", None) is None

    def test_empty_data(self):
        matcher = KnownIssues.from_json({"comment": "empty"})
        assert matcher.match("torch_import", "libcuda.so") is None
