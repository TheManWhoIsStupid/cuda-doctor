"""Windows code-path tests (run anywhere: platform is injected)."""

from __future__ import annotations

import os
from pathlib import Path

from tests.conftest import FakeRunner, load_fixture, ok

from cuda_doctor.collectors.compiler import (
    VSWHERE_RELATIVE_PATH,
    CompilerCollector,
)
from cuda_doctor.collectors.cuda import CUDACollector
from cuda_doctor.core.enums import Platform
from cuda_doctor.utils.platform import current_platform


class TestCurrentPlatform:
    def test_classifies_known_platforms(self, monkeypatch):
        for value, expected in (
            ("win32", Platform.WINDOWS),
            ("linux", Platform.LINUX),
            ("darwin", Platform.MACOS),
            ("sunos5", Platform.OTHER),
        ):
            monkeypatch.setattr("sys.platform", value)
            assert current_platform() is expected

    def test_current_machine_is_classified(self):
        # Never OTHER on supported development machines.
        assert current_platform() in (Platform.LINUX, Platform.WINDOWS, Platform.MACOS)


def _fake_isfile(vswhere: Path):
    """Match a vswhere candidate regardless of POSIX/Windows separators
    (tests run the Windows code path on Linux hosts too)."""

    def isfile(path) -> bool:
        normalized = Path(str(path).replace("\\", "/"))
        return normalized == vswhere

    return isfile


class TestWindowsCompilerCollector:
    def test_vswhere_found_and_version_parsed(self, monkeypatch, tmp_path):
        program_files = tmp_path / "ProgramFiles(x86)"
        vswhere = program_files / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        vswhere.parent.mkdir(parents=True)
        vswhere.write_text("", encoding="utf-8")
        monkeypatch.setattr("os.path.isfile", _fake_isfile(vswhere))

        # The collector builds the vswhere path with os.path.join, which keeps
        # the backslash separators from VSWHERE_RELATIVE_PATH on POSIX hosts.
        candidate = os.path.join(str(program_files), VSWHERE_RELATIVE_PATH)
        runner = FakeRunner(
            {
                (candidate, "-latest", "-products", "*", "-property", "installationVersion"): ok(
                    "17.11.35327.3\n"
                ),
                ("C:\\VC\\bin\\cl.exe", "/?"): ok(
                    "Microsoft (R) C/C++ Optimizing Compiler Version 19.41 for x64\n"
                ),
            }
        )
        monkeypatch.setattr(
            "cuda_doctor.collectors.find_executable",
            lambda name: "C:\\VC\\bin\\cl.exe" if name == "cl" else None,
        )
        info = CompilerCollector(
            runner=runner,
            env={"ProgramFiles(x86)": str(program_files)},
            platform=Platform.WINDOWS,
        ).collect()
        by_name = {tool.name: tool for tool in info.compilers}
        assert by_name["Visual Studio (vswhere)"].version == "17.11.35327.3"
        assert by_name["Visual Studio (vswhere)"].found is True
        assert by_name["cl"].found is True

    def test_vswhere_missing_and_cl_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "cuda_doctor.collectors.find_executable", lambda name: None
        )
        # Point ProgramFiles(x86) at a real (empty) dir: hardcoding
        # C:\Program Files (x86) leaks the host state — Windows runners
        # have a real vswhere.exe there, so absence could not be simulated.
        info = CompilerCollector(
            runner=FakeRunner(),
            env={"ProgramFiles(x86)": str(tmp_path)},
            platform=Platform.WINDOWS,
        ).collect()
        # Only the cl probe ran; vswhere was not on disk.
        assert [tool.name for tool in info.compilers] == ["cl"]
        assert info.any_found is False

    def test_vswhere_runs_but_output_unparseable(self, monkeypatch, tmp_path):
        program_files = tmp_path / "ProgramFiles(x86)"
        vswhere = program_files / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        vswhere.parent.mkdir(parents=True)
        vswhere.write_text("", encoding="utf-8")
        monkeypatch.setattr("os.path.isfile", _fake_isfile(vswhere))
        candidate = os.path.join(str(program_files), VSWHERE_RELATIVE_PATH)
        runner = FakeRunner(
            {
                (candidate, "-latest", "-products", "*", "-property", "installationVersion"): ok(
                    ""
                )
            }
        )
        monkeypatch.setattr("cuda_doctor.collectors.find_executable", lambda name: None)
        info = CompilerCollector(
            runner=runner,
            env={"ProgramFiles(x86)": str(program_files)},
            platform=Platform.WINDOWS,
        ).collect()
        vs_tool = info.compilers[0]
        assert vs_tool.name == "Visual Studio (vswhere)"
        assert vs_tool.found is False  # no version -> not usable

    def test_linux_ignores_vswhere(self, monkeypatch):
        monkeypatch.setattr("cuda_doctor.collectors.find_executable", lambda name: None)
        info = CompilerCollector(
            runner=FakeRunner(), env={}, platform=Platform.LINUX
        ).collect()
        assert [tool.name for tool in info.compilers] == ["gcc", "g++", "clang", "clang++"]


def _make_windows_toolkit(root: Path, name: str = "cuda-12.4") -> Path:
    """A real on-disk Windows toolkit directory with bin/nvcc.exe."""
    home = root / name
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    nvcc_bin = bin_dir / "nvcc.exe"
    nvcc_bin.write_text("", encoding="utf-8")
    nvcc_bin.chmod(0o755)
    return home


class TestWindowsCUDASelectorObservations:
    def test_cuda_path_folding_is_one_logical_identity(self, tmp_path):
        # Windows keeps the v0.1.x folding: with CUDA_HOME unset, cuda_home
        # comes from CUDA_PATH. Both observations must then carry the same
        # canonical root (frozen architecture §9.5 binding rule 3) so the
        # fact layer can never count them as two disagreeing selectors.
        toolkit = _make_windows_toolkit(tmp_path)
        info = CUDACollector(
            FakeRunner(),
            env={"PATH": "C:\\Windows", "CUDA_PATH": str(toolkit)},
            platform=Platform.WINDOWS,
            usr_local_cuda_path=str(toolkit),  # exists, but Windows never consults it
        ).collect()
        by_name = {obs.name: obs for obs in info.selector_observations}
        assert "usr_local_cuda" not in by_name  # POSIX convention, not a Windows selector
        home = by_name["cuda_home"]
        cuda_path = by_name["cuda_path"]
        assert home.raw_value == cuda_path.raw_value == str(toolkit)
        assert home.canonical_root == cuda_path.canonical_root == str(toolkit.resolve())
        # Legacy folding behavior is unchanged.
        assert info.cuda_home == info.cuda_path == str(toolkit)
        assert info.cuda_home_exists is True

    def test_cuda_home_and_cuda_path_both_set_stay_distinct(self, tmp_path):
        home_toolkit = _make_windows_toolkit(tmp_path, "cuda-12.4")
        path_toolkit = _make_windows_toolkit(tmp_path, "cuda-11.8")
        info = CUDACollector(
            FakeRunner(),
            env={
                "PATH": "C:\\Windows",
                "CUDA_HOME": str(home_toolkit),
                "CUDA_PATH": str(path_toolkit),
            },
            platform=Platform.WINDOWS,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        by_name = {obs.name: obs for obs in info.selector_observations}
        assert by_name["cuda_home"].raw_value == str(home_toolkit)
        assert by_name["cuda_path"].raw_value == str(path_toolkit)
        assert by_name["cuda_home"].canonical_root != by_name["cuda_path"].canonical_root
        assert info.cuda_home == str(home_toolkit)  # no folding when CUDA_HOME is set

    def test_env_names_fold_case(self, tmp_path):
        toolkit = _make_windows_toolkit(tmp_path)
        info = CUDACollector(
            FakeRunner(),
            env={
                "path": "C:\\Windows",
                "cuda_path": str(toolkit),
                "cudacxx": "nvcc.exe",
            },
            platform=Platform.WINDOWS,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        by_name = {obs.name: obs for obs in info.selector_observations}
        assert by_name["cuda_home"].raw_value == str(toolkit)  # folded from cuda_path
        assert by_name["cuda_path"].raw_value == str(toolkit)
        # Bare-name CUDACXX resolution is case-folded too but finds nothing
        # on this PATH — recorded as an unresolvable selector, not omitted.
        assert by_name["cudacxx"].raw_value == "nvcc.exe"

    def test_semicolon_path_resolution_finds_nvcc_exe(self, tmp_path, fixtures_dir):
        # Resolution uses the target platform's binary name and separator:
        # "nvcc.exe" found in the second ";"-separated entry wins.
        empty = tmp_path / "winbin-empty"
        empty.mkdir()
        toolkit = _make_windows_toolkit(tmp_path)
        nvcc_bin = toolkit / "bin" / "nvcc.exe"
        runner = FakeRunner(
            {(str(nvcc_bin), "--version"): ok(load_fixture(fixtures_dir, "nvcc/nvcc_12_4.txt"))}
        )
        info = CUDACollector(
            runner,
            env={"PATH": f"{empty};{toolkit / 'bin'}"},
            platform=Platform.WINDOWS,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        by_name = {obs.name: obs for obs in info.selector_observations}
        assert info.nvcc_found is True
        nvcc = by_name["nvcc"]
        assert nvcc.raw_value == "nvcc.exe"
        assert nvcc.resolved_path == str(nvcc_bin)
        assert nvcc.toolkit_version == "12.4"  # nvcc.exe output parses the same
        path_bin = by_name["path_cuda_bin"]
        assert path_bin.raw_value == str(toolkit / "bin")
        assert path_bin.toolkit_version == "12.4"  # inherited from the resolved nvcc

    def test_colon_separated_path_is_not_split_on_windows(self, tmp_path):
        # A POSIX-style PATH seen by a Windows target is one single entry
        # (the separator is ";" for the target), so nothing resolves unless
        # the whole string is a directory holding nvcc.exe.
        toolkit = _make_windows_toolkit(tmp_path)
        posix_path = f"/usr/local/cuda/bin:{toolkit / 'bin'}"
        info = CUDACollector(
            FakeRunner(),
            env={"PATH": posix_path},
            platform=Platform.WINDOWS,
            usr_local_cuda_path=str(tmp_path / "absent"),
        ).collect()
        assert info.nvcc_found is False  # the unsplit entry is not a real directory
