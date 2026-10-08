"""Windows code-path tests (run anywhere: platform is injected)."""

from __future__ import annotations

import os
from pathlib import Path

from tests.conftest import FakeRunner, ok

from cuda_doctor.collectors.compiler import (
    VSWHERE_RELATIVE_PATH,
    CompilerCollector,
)
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

    def test_vswhere_missing_and_cl_absent(self, monkeypatch):
        monkeypatch.setattr(
            "cuda_doctor.collectors.find_executable", lambda name: None
        )
        info = CompilerCollector(
            runner=FakeRunner(),
            env={"ProgramFiles(x86)": r"C:\Program Files (x86)"},
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
