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
    def test_documented_family_minimums(self, compat):
        # NVIDIA-documented minor-version-compatibility baselines.
        assert (
            compat.driver.family_minimum(CudaVersion(11, 8), Platform.LINUX)
            == (450, 80, 2)
        )
        assert (
            compat.driver.family_minimum(CudaVersion(11, 0), Platform.WINDOWS)
            == (452, 39)
        )
        assert (
            compat.driver.family_minimum(CudaVersion(12, 6), Platform.LINUX)
            == (525, 60, 13)
        )
        assert (
            compat.driver.family_minimum(CudaVersion(12, 0), Platform.WINDOWS)
            == (528, 33)
        )
        assert (
            compat.driver.family_minimum(CudaVersion(13, 0), Platform.LINUX)
            == (580,)
        )
        assert (
            compat.driver.family_minimum(CudaVersion(13, 1), Platform.WINDOWS)
            == (580,)
        )

    def test_cuda13_family_rule_is_branch_level(self, compat):
        # NVIDIA documents the CUDA 13.x minor-compatibility range as the R580
        # branch (driver >= 580), not an exact patch release: the 580.65.06 /
        # 580.88 drivers are the ones *packaged with* the CUDA 13.0 toolkit and
        # must not be stored as the family minimum.
        linux = compat.driver.family_minimum(CudaVersion(13, 0), Platform.LINUX)
        windows = compat.driver.family_minimum(CudaVersion(13, 0), Platform.WINDOWS)
        assert linux == (580,) and len(linux) == 1
        assert windows == (580,) and len(windows) == 1

    def test_cuda13_r580_driver_satisfies_family_minimum(self, compat):
        # Any 580.x driver (e.g. the one shipped with CUDA 13.0) satisfies
        # the >= 580 branch rule via zero-padded comparison.
        verdict = compat.driver.evaluate(CudaVersion(13, 0), (580, 65, 6), Platform.LINUX)
        assert verdict.compatible is True
        assert (
            compat.driver.evaluate(CudaVersion(13, 2), (580, 88), Platform.WINDOWS)
        ).compatible is True

    def test_cuda13_newer_driver_branch_is_compatible(self, compat):
        # A 590.x driver is above the family minimum: normal backward
        # compatibility, no toolkit-driver pairing required.
        verdict = compat.driver.evaluate(CudaVersion(13, 0), (590, 44, 1), Platform.LINUX)
        assert verdict.compatible is True

    def test_cuda13_driver_below_family_branch_minimum(self, compat):
        # An R575 driver is below the documented R580-family minimum.
        verdict = compat.driver.evaluate(CudaVersion(13, 0), (575, 57, 8), Platform.LINUX)
        assert verdict.compatible is False
        assert "580" in verdict.message
        windows = compat.driver.evaluate(CudaVersion(13, 0), (575, 12), Platform.WINDOWS)
        assert windows.compatible is False

    def test_future_major_is_unknown_not_inherited(self, compat):
        # CUDA 14 must NOT reuse CUDA 13 requirements (or any older family).
        assert compat.driver.family_minimum(CudaVersion(14, 0), Platform.LINUX) is None
        verdict = compat.driver.evaluate(CudaVersion(14, 0), (580, 65, 6), Platform.LINUX)
        assert verdict.compatible is None
        assert "unknown" in verdict.message

    def test_unknown_minor_within_family_uses_family_rule(self, compat):
        # The family rule is documented across the entire major family,
        # so an unknown 12.9 still resolves to the CUDA 12.x minimum.
        assert (
            compat.driver.family_minimum(CudaVersion(12, 9), Platform.LINUX)
            == (525, 60, 13)
        )

    def test_platform_without_table_is_unknown(self, compat):
        assert compat.driver.family_minimum(CudaVersion(12, 4), Platform.MACOS) is None

    def test_evaluate_ok(self, compat):
        verdict = compat.driver.evaluate(
            CudaVersion(12, 4), (550, 54, 14), Platform.LINUX
        )
        assert verdict.compatible is True
        assert "minor-version compatibility" in verdict.message

    def test_evaluate_below_family_minimum(self, compat):
        verdict = compat.driver.evaluate(
            CudaVersion(12, 6), (515, 43, 4), Platform.LINUX
        )
        assert verdict.compatible is False
        assert "525.60.13" in verdict.message

    def test_from_json_direct(self):
        data = {
            "minor_version_compatibility": {
                "12": {"linux": ">=525.60.13", "windows": ">=528.33"}
            }
        }
        compat = DriverCompatibility.from_json(data)
        assert compat.family_minimum(CudaVersion(12, 9), Platform.LINUX) == (525, 60, 13)
        assert compat.family_minimum(CudaVersion(12, 9), Platform.WINDOWS) == (528, 33)
        assert compat.family_minimum(CudaVersion(11, 8), Platform.LINUX) is None

    def test_from_json_ignores_malformed_entries(self):
        data = {
            "minor_version_compatibility": {
                "comment-ish": {"linux": ">=1"},
                "12": {"linux": 525, "windows": ">=528.33"},
            }
        }
        compat = DriverCompatibility.from_json(data)
        assert compat.family_minimum(CudaVersion(12, 0), Platform.LINUX) is None
        assert compat.family_minimum(CudaVersion(12, 0), Platform.WINDOWS) == (528, 33)


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

    def test_future_cuda_major_is_unknown_not_inherited(self, compat):
        # CUDA 14 must not reuse CUDA 13 (or any older) compiler rules.
        assert compat.compiler.max_supported_gcc_major(CudaVersion(14, 0)) is None
        verdict = compat.compiler.evaluate_gcc((11, 4, 0), CudaVersion(14, 0))
        assert verdict.compatible is None
        assert "unknown" in verdict.message
        assert compat.compiler.vs_range(CudaVersion(14, 0)) == (None, None)
        vs = compat.compiler.evaluate_visual_studio((17, 11), CudaVersion(14, 0))
        assert vs.compatible is None

    def test_unknown_minor_is_unknown_not_inherited(self, compat):
        # Compiler support is documented per toolkit release, not per family:
        # an unknown 12.9 must not silently reuse 12.6's rules.
        assert compat.compiler.max_supported_gcc_major(CudaVersion(12, 9)) is None
        verdict = compat.compiler.evaluate_gcc((11, 4, 0), CudaVersion(12, 9))
        assert verdict.compatible is None
        assert compat.compiler.vs_range(CudaVersion(12, 9)) == (None, None)

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
