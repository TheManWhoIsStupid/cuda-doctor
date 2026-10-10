"""Tests for the bounded CUDA runtime-library inventory collector.

All filesystem state lives under pytest tmp_path; platform, environment,
command runner, and site-packages roots are injected so no test ever
observes the real host (§4 DLP/toolkit roots come from tmp_path, and
``site_packages`` is pinned to tmp paths or ``()`` everywhere).
"""

from __future__ import annotations

import os
import site
import tempfile
from pathlib import Path

import pytest
from tests.conftest import FakeRunner, ok

from cuda_doctor.collectors.runtime_libraries import (
    LDCONFIG_COMMAND,
    ORIGIN_CONDA_PREFIX,
    ORIGIN_LD_LIBRARY_PATH,
    ORIGIN_LDCONFIG,
    ORIGIN_PYTHON_PACKAGE,
    ORIGIN_TOOLKIT,
    ORIGINS,
    SUPPORTED_FAMILIES,
    VERSION_SOURCE_FILENAME,
    VERSION_SOURCE_SYMLINK_TARGET,
    VERSION_SOURCE_UNKNOWN,
    RuntimeLibraryCollector,
    _default_site_packages,
    toolkit_library_roots,
)
from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import (
    CUDAInfo,
    CUDAInstallation,
    CUDASelectorObservation,
    EnvironmentInfo,
)


def _can_symlink() -> bool:
    if not hasattr(os, "symlink"):
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.touch()
            os.symlink(target, Path(tmp) / "link")
        return True
    except OSError:
        return False


requires_symlinks = pytest.mark.skipif(not _can_symlink(), reason="symlinks unavailable")


def _env_info(
    ld_library_path: str | None = None, conda_prefix: str | None = None
) -> EnvironmentInfo:
    """EnvironmentInfo mirroring EnvironmentCollector's ordered split."""
    info = EnvironmentInfo()
    if ld_library_path is not None:
        info.ld_library_path = ld_library_path
        info.ld_library_path_entries = [
            (index, entry)
            for index, entry in enumerate(ld_library_path.split(":"))
            if entry
        ]
    if conda_prefix is not None:
        info.variables["CONDA_PREFIX"] = conda_prefix
    return info


def _lib(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lib = directory / name
    lib.write_text("", encoding="utf-8")
    return lib


def _collect(
    environment: EnvironmentInfo | None = None,
    runner: FakeRunner | None = None,
    platform: Platform = Platform.LINUX,
    toolkit_roots=(),
    site_packages=(),
):
    return RuntimeLibraryCollector(
        runner or FakeRunner(),
        environment=environment or EnvironmentInfo(),
        platform=platform,
        toolkit_roots=toolkit_roots,
        site_packages=site_packages,
    ).collect()


class TestPlatformGate:
    def test_non_linux_targets_get_empty_not_applicable_inventory(self, tmp_path):
        # D004 is Linux-only: a Windows/macOS/other target is "not evaluated",
        # never an error and never POSIX loader behavior on the wrong target.
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12")
        for target in (Platform.WINDOWS, Platform.MACOS, Platform.OTHER):
            runner = FakeRunner()
            inventory = _collect(
                environment=_env_info(ld_library_path=str(lib_dir)),
                runner=runner,
                platform=target,
            )
            assert inventory.candidates == []
            assert inventory.scan_errors == {}
            assert runner.calls == []  # not even ldconfig runs

    def test_linux_target_inventories(self, tmp_path):
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12")
        inventory = _collect(environment=_env_info(ld_library_path=str(lib_dir)))
        assert [c.family for c in inventory.candidates] == ["libcudart"]


class TestFamilyScope:
    def test_frozen_family_constant(self):
        assert SUPPORTED_FAMILIES == ("libcudart", "libcublas", "libcudnn")

    def test_only_supported_families_collected(self, tmp_path):
        lib_dir = tmp_path / "libs"
        for name in (
            "libcudart.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libfoo.so.1",  # unrelated family
            "libcudart_static.a",  # static archive, not a shared object
            "libcublasLt.so.12",  # Lt variant outside the frozen families
            "libcudart.so.bak",  # non-numeric suffix
            "libcudart.so.12.debug",  # debug suffix is not a version
            "readme.txt",
        ):
            _lib(lib_dir, name)
        inventory = _collect(environment=_env_info(ld_library_path=str(lib_dir)))
        found = {(c.family, c.path) for c in inventory.candidates}
        assert found == {
            (family, str(lib_dir / f"{family}{suffix}"))
            for family, suffix in (
                ("libcudart", ".so.12"),
                ("libcublas", ".so.12"),
                ("libcudnn", ".so.9"),
            )
        }


class TestLDLibraryPathInventory:
    def test_source_order_preserved_with_entry_indexes(self, tmp_path):
        # "/a::/b" — the empty segment keeps index 1 out, positions survive.
        first = tmp_path / "first"
        second = tmp_path / "second"
        _lib(first, "libcudart.so.12")
        _lib(second, "libcudart.so.11")
        inventory = _collect(
            environment=_env_info(ld_library_path=f"{first}::{second}")
        )
        by_order = {c.search_order: c for c in inventory.candidates}
        assert sorted(by_order) == [0, 2]  # the gap at index 1 is preserved
        assert by_order[0].path == str(first / "libcudart.so.12")
        assert by_order[2].path == str(second / "libcudart.so.11")
        for candidate in inventory.candidates:
            assert candidate.origin == ORIGIN_LD_LIBRARY_PATH
            assert candidate.search_group == ORIGIN_LD_LIBRARY_PATH

    def test_same_directory_at_two_positions_keeps_both(self, tmp_path):
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12")
        inventory = _collect(
            environment=_env_info(ld_library_path=f"{lib_dir}::{lib_dir}")
        )
        assert [c.search_order for c in inventory.candidates] == [0, 2]

    def test_missing_entry_is_scan_error_not_crash(self, tmp_path):
        missing = tmp_path / "missing-entry"
        present = tmp_path / "present"
        _lib(present, "libcublas.so.12")
        inventory = _collect(
            environment=_env_info(ld_library_path=f"{missing}:{present}")
        )
        assert str(missing) in inventory.scan_errors
        assert [c.path for c in inventory.candidates] == [str(present / "libcublas.so.12")]


class TestToolkitRootInventory:
    def test_lib_lib64_and_targets_layouts_scanned(self, tmp_path):
        root = tmp_path / "cuda-12.4"
        _lib(root / "lib", "libcudart.so.12")
        _lib(root / "lib64", "libcublas.so.12")
        _lib(root / "targets" / "x86_64-linux" / "lib", "libcudnn.so.9")
        inventory = _collect(toolkit_roots=(str(root),))
        paths = {c.path for c in inventory.candidates}
        assert paths == {
            str(root / "lib" / "libcudart.so.12"),
            str(root / "lib64" / "libcublas.so.12"),
            str(root / "targets" / "x86_64-linux" / "lib" / "libcudnn.so.9"),
        }
        for candidate in inventory.candidates:
            assert candidate.origin == ORIGIN_TOOLKIT
            assert candidate.search_order is None  # no invented ordering

    def test_toolkit_origin_stays_neutral(self, tmp_path):
        root = tmp_path / "cuda-12.4"
        _lib(root / "lib64", "libcudart.so.12")
        inventory = _collect(toolkit_roots=(str(root),))
        assert {c.origin for c in inventory.candidates} == {ORIGIN_TOOLKIT}
        # "toolkit" is the frozen neutral vocabulary — never active/winning.
        assert ORIGIN_TOOLKIT == "toolkit"
        assert all("active" not in origin for origin in ORIGINS)

    def test_root_without_library_dirs_is_silent(self, tmp_path):
        root = tmp_path / "cuda-12.4"
        root.mkdir()
        inventory = _collect(
            runner=FakeRunner({LDCONFIG_COMMAND[1:]: ok("")}), toolkit_roots=(str(root),)
        )
        assert inventory.candidates == []
        assert inventory.scan_errors == {}  # absence is not an error

    def test_nonexistent_root_is_silent(self, tmp_path):
        inventory = _collect(
            runner=FakeRunner({LDCONFIG_COMMAND[1:]: ok("")}),
            toolkit_roots=(str(tmp_path / "absent"),),
        )
        assert inventory.candidates == []
        assert inventory.scan_errors == {}

    def test_no_recursive_scan_beyond_library_dirs(self, tmp_path):
        # A library deeper than lib/lib64/targets/<arch>/lib is out of scope.
        root = tmp_path / "cuda"
        _lib(root / "deep" / "nested" / "lib", "libcudart.so.12")
        _lib(root / "lib" / "extra" / "sub", "libcublas.so.12")
        inventory = _collect(toolkit_roots=(str(root),))
        assert inventory.candidates == []


class TestToolkitLibraryRootsHelper:
    def test_none_cuda_yields_no_roots(self):
        assert toolkit_library_roots(None) == ()

    def test_combines_observation_roots_and_installations_deduped(self):
        cuda = CUDAInfo(
            selector_observations=[
                CUDASelectorObservation(name="cuda_home", canonical_root="/opt/cuda"),
                CUDASelectorObservation(name="nvcc", canonical_root="/opt/cuda/"),  # alias form
                CUDASelectorObservation(name="cudacxx", canonical_root=None),
            ],
            installations=[CUDAInstallation(path="/usr/local/cuda-12.4", version="12.4")],
        )
        assert toolkit_library_roots(cuda) == (
            "/opt/cuda",
            "/usr/local/cuda-12.4",
        )

    def test_empty_cuda_yields_no_roots(self):
        assert toolkit_library_roots(CUDAInfo()) == ()


class TestPythonPackageInventory:
    def test_nvidia_package_libs_collected(self, tmp_path):
        site = tmp_path / "site-packages"
        _lib(site / "nvidia" / "cudnn" / "lib", "libcudnn.so.9")
        inventory = _collect(site_packages=(str(site),))
        assert [c.path for c in inventory.candidates] == [
            str(site / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9")
        ]
        assert inventory.candidates[0].origin == ORIGIN_PYTHON_PACKAGE

    def test_only_nvidia_namespace_is_scanned(self, tmp_path):
        site = tmp_path / "site-packages"
        _lib(site / "torch" / "lib", "libcudart.so.12")  # not the nvidia namespace
        _lib(site / "nvidia" / "cudnn" / "share", "libcudnn.so.9")  # not a lib dir
        inventory = _collect(site_packages=(str(site),))
        assert inventory.candidates == []

    def test_site_without_nvidia_dir_is_silent(self, tmp_path):
        site = tmp_path / "site-packages"
        site.mkdir()
        inventory = _collect(
            runner=FakeRunner({LDCONFIG_COMMAND[1:]: ok("")}), site_packages=(str(site),)
        )
        assert inventory.candidates == []
        assert inventory.scan_errors == {}

    def test_empty_site_packages_scans_nothing(self):
        # Hermeticity contract: () means no interpreter state is consulted.
        inventory = _collect(site_packages=())
        assert inventory.candidates == []

    def test_nonexistent_site_root_is_silent(self, tmp_path):
        inventory = _collect(
            runner=FakeRunner({LDCONFIG_COMMAND[1:]: ok("")}),
            site_packages=(str(tmp_path / "absent-site"),),
        )
        assert inventory.candidates == []
        assert inventory.scan_errors == {}


class TestDefaultSiteRoots:
    """Review fix: the default root set includes the ACTIVE user site."""

    USER_SITE = "/home/alice/.local/lib/python3.11/site-packages"
    SYSTEM_SITES = ("/venv/lib/python3.11/site-packages", "/usr/lib/python3/dist-packages")

    def _pin(self, monkeypatch, *, enabled, user_site=USER_SITE, system=SYSTEM_SITES):
        monkeypatch.setattr(site, "getsitepackages", lambda: list(system))
        monkeypatch.setattr(site, "ENABLE_USER_SITE", enabled)
        if callable(user_site):
            monkeypatch.setattr(site, "getusersitepackages", user_site)
        else:
            monkeypatch.setattr(site, "getusersitepackages", lambda: user_site)

    def test_system_roots_remain_included(self, monkeypatch):
        self._pin(monkeypatch, enabled=False)
        assert _default_site_packages() == tuple(self.SYSTEM_SITES)

    def test_enabled_user_site_is_appended(self, monkeypatch):
        self._pin(monkeypatch, enabled=True)
        assert _default_site_packages() == (
            "/venv/lib/python3.11/site-packages",
            "/usr/lib/python3/dist-packages",
            self.USER_SITE,
        )

    def test_disabled_user_site_is_excluded(self, monkeypatch):
        # Both False and None mean "user-site packages not enabled" for
        # this interpreter — never scan the user site then.
        for disabled in (False, None):
            self._pin(monkeypatch, enabled=disabled)
            assert _default_site_packages() == tuple(self.SYSTEM_SITES)

    def test_duplicate_root_between_system_and_user_appears_once(self, monkeypatch):
        self._pin(
            monkeypatch,
            enabled=True,
            system=["/shared/site", "/usr/lib/python3/dist-packages"],
            user_site="/shared/site",
        )
        # Dedup preserves deterministic order: system roots first.
        assert _default_site_packages() == (
            "/shared/site",
            "/usr/lib/python3/dist-packages",
        )

    def test_user_site_failure_does_not_crash_collection(self, monkeypatch):
        def unreachable():
            raise RuntimeError("no user site base")

        self._pin(monkeypatch, enabled=True, user_site=unreachable)
        # The system roots survive; the helper never raises.
        assert _default_site_packages() == tuple(self.SYSTEM_SITES)

    def test_user_site_nvidia_libs_follow_the_same_bounded_rule(self, tmp_path):
        # A pip --user NVIDIA wheel lives under <user-site>/nvidia/<pkg>/lib
        # and is inventoried exactly like any other injected site root.
        user_site = tmp_path / ".local" / "lib" / "python3.11" / "site-packages"
        _lib(user_site / "nvidia" / "cudnn" / "lib", "libcudnn.so.9")
        inventory = _collect(site_packages=(str(user_site),))
        candidate = inventory.candidates[0]
        assert candidate.path == str(user_site / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9")
        assert candidate.origin == ORIGIN_PYTHON_PACKAGE

    def test_collector_default_flows_from_site_configuration(self, tmp_path, monkeypatch):
        # End-to-end wiring: without an explicit site_packages injection the
        # constructor pulls the enabled user site from the site module.
        user_site = tmp_path / "user-site"
        _lib(user_site / "nvidia" / "cudnn" / "lib", "libcudnn.so.9")
        self._pin(monkeypatch, enabled=True, system=[], user_site=str(user_site))
        inventory = RuntimeLibraryCollector(FakeRunner(), platform=Platform.LINUX).collect()
        assert [c.origin for c in inventory.candidates] == [ORIGIN_PYTHON_PACKAGE]

    def test_other_python_installations_are_not_scanned(self, tmp_path):
        # Only exact configured roots are consulted: a sibling installation's
        # site-packages (e.g. another pythonX.Y) is never discovered.
        user_site = tmp_path / ".local" / "lib" / "python3.11" / "site-packages"
        other_install = tmp_path / ".local" / "lib" / "python3.10" / "site-packages"
        _lib(user_site / "nvidia" / "cublas" / "lib", "libcublas.so.12")
        _lib(other_install / "nvidia" / "cudnn" / "lib", "libcudnn.so.9")
        inventory = _collect(site_packages=(str(user_site),))
        assert [c.path for c in inventory.candidates] == [
            str(user_site / "nvidia" / "cublas" / "lib" / "libcublas.so.12")
        ]


class TestCondaInventory:
    def _quiet_runner(self) -> FakeRunner:
        # ldconfig succeeds with empty output so scan silence is attributable
        # to the conda root alone.
        return FakeRunner({LDCONFIG_COMMAND[1:]: ok("")})

    def test_current_conda_lib_scanned(self, tmp_path):
        conda = tmp_path / "conda" / "envs" / "torch"
        _lib(conda / "lib", "libcudart.so.12")
        inventory = _collect(
            runner=self._quiet_runner(),
            environment=_env_info(conda_prefix=str(conda)),
        )
        assert [c.path for c in inventory.candidates] == [str(conda / "lib" / "libcudart.so.12")]
        assert inventory.candidates[0].origin == ORIGIN_CONDA_PREFIX

    def test_absent_conda_prefix_is_not_an_error(self):
        inventory = _collect(
            runner=self._quiet_runner(), environment=EnvironmentInfo()
        )  # no CONDA_PREFIX at all
        assert inventory.candidates == []
        assert inventory.scan_errors == {}

    def test_empty_conda_prefix_is_ignored(self):
        inventory = _collect(
            runner=self._quiet_runner(), environment=_env_info(conda_prefix="")
        )
        assert inventory.candidates == []
        assert inventory.scan_errors == {}


LDCONFIG_SAMPLE = """\
1976 caches found in /etc/ld.so.cache
        libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1
        libcudart.so.12 (libc6,x86-64) => /usr/local/cuda-12.4/lib64/libcudart.so.12
        libcublas.so.12 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcublas.so.12
        libcublas.so.12 (libc6,x86-64) => /opt/cuda/lib64/libcublas.so.12
        libcudnn.so.9 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudnn.so.9
        libgcc_s.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libgcc_s.so.1
"""


class TestLdconfigInventory:
    def test_supported_families_parsed_with_sonames(self):
        runner = FakeRunner({LDCONFIG_COMMAND[1:]: ok(LDCONFIG_SAMPLE)})
        inventory = _collect(runner=runner)
        by_soname = {c.soname: c for c in inventory.candidates}
        assert set(by_soname) == {"libcudart.so.12", "libcublas.so.12", "libcudnn.so.9"}
        cudart = by_soname["libcudart.so.12"]
        assert cudart.path == "/usr/local/cuda-12.4/lib64/libcudart.so.12"
        assert cudart.version == "12"
        assert cudart.version_source == VERSION_SOURCE_FILENAME
        assert cudart.origin == ORIGIN_LDCONFIG
        assert cudart.search_group == ORIGIN_LDCONFIG
        assert cudart.search_order is None  # no invented loader ordering
        # Same soname at two paths stays two distinct candidates.
        cublas = [c for c in inventory.candidates if c.soname == "libcublas.so.12"]
        assert len(cublas) == 2

    def test_exactly_one_query_through_injected_runner(self):
        runner = FakeRunner({LDCONFIG_COMMAND[1:]: ok(LDCONFIG_SAMPLE)})
        _collect(runner=runner)
        assert runner.calls == [("ldconfig", "-p")]

    def test_failure_is_scan_error_and_other_roots_survive(self, tmp_path):
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12")
        runner = FakeRunner()  # default: every command fails
        inventory = _collect(
            environment=_env_info(ld_library_path=str(lib_dir)), runner=runner
        )
        assert ORIGIN_LDCONFIG in inventory.scan_errors
        assert inventory.scan_errors[ORIGIN_LDCONFIG].startswith("ldconfig -p failed:")
        assert [c.path for c in inventory.candidates] == [str(lib_dir / "libcudart.so.12")]

    def test_header_line_ignored(self):
        # The real output starts with a human header line — never parsed as
        # a library entry.
        runner = FakeRunner({LDCONFIG_COMMAND[1:]: ok("1896 caches found in /etc/ld.so.cache\n")})
        inventory = _collect(runner=runner)
        assert inventory.candidates == []
        assert inventory.scan_errors == {}


class TestVersionSemantics:
    def test_unversioned_soname_is_unknown(self, tmp_path):
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so")
        inventory = _collect(environment=_env_info(ld_library_path=str(lib_dir)))
        candidate = inventory.candidates[0]
        assert candidate.version is None
        assert candidate.version_source == VERSION_SOURCE_UNKNOWN

    def test_major_only_version_from_filename(self, tmp_path):
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12")
        candidate = _collect(
            environment=_env_info(ld_library_path=str(lib_dir))
        ).candidates[0]
        assert candidate.version == "12"
        assert candidate.version_source == VERSION_SOURCE_FILENAME

    def test_multi_component_version_kept_verbatim(self, tmp_path):
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12.4.127")
        candidate = _collect(
            environment=_env_info(ld_library_path=str(lib_dir))
        ).candidates[0]
        assert candidate.version == "12.4.127"
        assert candidate.version_source == VERSION_SOURCE_FILENAME

    @requires_symlinks
    def test_bare_link_gets_version_from_symlink_target(self, tmp_path):
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12.4.127")
        os.symlink(lib_dir / "libcudart.so.12.4.127", lib_dir / "libcudart.so")
        candidate = _collect(
            environment=_env_info(ld_library_path=str(lib_dir))
        ).candidates[0]
        assert candidate.path == str(lib_dir / "libcudart.so")
        assert candidate.canonical_path == str(lib_dir / "libcudart.so.12.4.127")
        assert candidate.version == "12.4.127"
        assert candidate.version_source == VERSION_SOURCE_SYMLINK_TARGET

    @requires_symlinks
    def test_observed_version_is_never_upgraded_by_target(self, tmp_path):
        # libcudart.so.12 -> libcudart.so.12.4.127: the observed filename
        # says "12", so "12" it stays — no precision the filename lacks.
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12.4.127")
        os.symlink(lib_dir / "libcudart.so.12.4.127", lib_dir / "libcudart.so.12")
        candidate = _collect(
            environment=_env_info(ld_library_path=str(lib_dir))
        ).candidates[0]
        assert candidate.version == "12"
        assert candidate.version_source == VERSION_SOURCE_FILENAME

    @requires_symlinks
    def test_dangling_symlink_is_recorded_conservatively(self, tmp_path):
        lib_dir = tmp_path / "libs"
        lib_dir.mkdir()
        os.symlink(lib_dir / "libcudart.so.12.9.0", lib_dir / "libcudart.so.12")  # target absent
        candidate = _collect(
            environment=_env_info(ld_library_path=str(lib_dir))
        ).candidates[0]
        assert candidate.path == str(lib_dir / "libcudart.so.12")
        assert candidate.version == "12"  # from the observed filename
        assert candidate.version_source == VERSION_SOURCE_FILENAME


class TestCandidateModelSemantics:
    def test_origins_stay_in_frozen_vocabulary(self, tmp_path):
        site = tmp_path / "site"
        conda = tmp_path / "conda"
        lib_dir = tmp_path / "ldp"
        toolkit = tmp_path / "toolkit"
        _lib(lib_dir, "libcudart.so.12")
        _lib(toolkit / "lib64", "libcublas.so.12")
        _lib(site / "nvidia" / "cublas" / "lib", "libcublas.so.11")
        _lib(conda / "lib", "libcudnn.so.9")
        inventory = _collect(
            environment=_env_info(
                ld_library_path=str(lib_dir), conda_prefix=str(conda)
            ),
            runner=FakeRunner({LDCONFIG_COMMAND[1:]: ok(LDCONFIG_SAMPLE)}),
            toolkit_roots=(str(toolkit),),
            site_packages=(str(site),),
        )
        assert {c.origin for c in inventory.candidates} == set(ORIGINS)

    def test_filesystem_candidates_do_not_claim_sonames(self, tmp_path):
        # A SONAME is reliably known only from ldconfig output; directory
        # scanning never invents one.
        lib_dir = tmp_path / "libs"
        _lib(lib_dir, "libcudart.so.12")
        candidate = _collect(
            environment=_env_info(ld_library_path=str(lib_dir))
        ).candidates[0]
        assert candidate.soname is None

    def test_directory_named_like_a_library_is_not_a_file(self, tmp_path):
        lib_dir = tmp_path / "libs"
        (lib_dir / "libcudart.so.12").mkdir(parents=True)
        inventory = _collect(environment=_env_info(ld_library_path=str(lib_dir)))
        assert inventory.candidates == []


class TestBoundedBehavior:
    def test_library_outside_approved_roots_not_found(self, tmp_path):
        # A real family library in an unrelated directory appears in the
        # inventory only through an approved root.
        outside = tmp_path / "elsewhere"
        _lib(outside, "libcudart.so.12")
        inventory = _collect()
        assert inventory.candidates == []

    def test_no_ldconfig_query_without_linux_target(self):
        runner = FakeRunner()
        _collect(runner=runner, platform=Platform.WINDOWS)
        assert runner.calls == []
