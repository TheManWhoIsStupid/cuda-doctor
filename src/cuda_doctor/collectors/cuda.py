"""CUDA Toolkit collector."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import CUDAInfo
from cuda_doctor.utils.commands import CommandRunner
from cuda_doctor.utils.parsing import parse_nvcc_version
from cuda_doctor.utils.paths import discover_cuda_installations, find_executable
from cuda_doctor.utils.platform import current_platform


class CUDACollector:
    """Collects nvcc, CUDA_HOME/CUDA_PATH, and on-disk toolkit installations."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        env: Mapping[str, str] | None = None,
        roots: Sequence[str] = (),
        platform: Platform | None = None,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.env: dict[str, str] = dict(env if env is not None else os.environ)
        self.roots = tuple(roots)
        self.platform = platform or current_platform()

    def collect(self) -> CUDAInfo:
        info = CUDAInfo()
        nvcc_path = find_executable("nvcc")
        if nvcc_path:
            info.nvcc_found = True
            info.nvcc_path = nvcc_path
            version = self.runner.run((nvcc_path, "--version"))
            if version.success:
                info.toolkit_version = parse_nvcc_version(version.stdout)
        self._collect_env_vars(info)
        self._collect_home_facts(info)
        info.installations = discover_cuda_installations(
            self.platform, env=self.env, roots=self.roots
        )
        return info

    def _collect_env_vars(self, info: CUDAInfo) -> None:
        env = {key.upper(): value for key, value in self.env.items()}
        info.windows_cuda_path_vars = {
            key: value for key, value in env.items() if key.startswith("CUDA_PATH_V") and value
        }
        if self.platform is Platform.WINDOWS:
            info.cuda_path = env.get("CUDA_PATH") or None
            info.cuda_home = env.get("CUDA_HOME") or info.cuda_path
        else:
            info.cuda_home = env.get("CUDA_HOME") or None
            info.cuda_path = env.get("CUDA_PATH") or None

    def _collect_home_facts(self, info: CUDAInfo) -> None:
        home = info.cuda_home
        if not home:
            return
        home_path = Path(home)
        try:
            info.cuda_home_exists = home_path.is_dir()
            nvcc_binary = "nvcc.exe" if self.platform is Platform.WINDOWS else "nvcc"
            info.cuda_home_has_nvcc = (home_path / "bin" / nvcc_binary).is_file()
        except OSError:
            # Inaccessible paths are simply reported as non-existent.
            info.cuda_home_exists = False
            info.cuda_home_has_nvcc = False
