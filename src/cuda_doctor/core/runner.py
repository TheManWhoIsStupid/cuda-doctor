"""Collection orchestration.

Runs every collector with per-collector isolation: one crashing collector is
recorded in ``collection_errors`` and the rest continue.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from cuda_doctor.collectors.cmake import CMakeCollector
from cuda_doctor.collectors.compiler import CompilerCollector
from cuda_doctor.collectors.cuda import CUDACollector
from cuda_doctor.collectors.environment import EnvironmentCollector
from cuda_doctor.collectors.gpu import GPUCollector
from cuda_doctor.collectors.ninja import NinjaCollector
from cuda_doctor.collectors.python_env import PythonEnvCollector
from cuda_doctor.collectors.pytorch import PyTorchCollector
from cuda_doctor.collectors.runtime_libraries import (
    RuntimeLibraryCollector,
    toolkit_library_roots,
)
from cuda_doctor.collectors.system import SystemCollector
from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import (
    EnvironmentSnapshot,
    SystemInfo,
)
from cuda_doctor.utils.commands import CommandRunner
from cuda_doctor.utils.platform import current_platform

_T = TypeVar("_T")


class CollectionRunner:
    """Builds an :class:`EnvironmentSnapshot` from all collectors.

    All dependencies (command runner, environment mapping, CUDA search roots,
    torch import function) are injectable for testing.
    """

    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        env: Mapping[str, str] | None = None,
        roots: Sequence[str] = (),
        platform: Platform | None = None,
        torch_import: Callable[[str], object] | None = None,
    ) -> None:
        self.runner = command_runner or CommandRunner()
        self.env = env
        self.roots = tuple(roots)
        self.platform = platform or current_platform()
        self.torch_import = torch_import

    def collect(self) -> EnvironmentSnapshot:
        errors: dict[str, str] = {}

        system = self._guarded(
            errors, "system", lambda: SystemCollector(self.platform).collect()
        )
        gpu = self._guarded(errors, "gpu", lambda: GPUCollector(self.runner).collect())
        cuda = self._guarded(
            errors,
            "cuda",
            lambda: CUDACollector(self.runner, self.env, self.roots, self.platform).collect(),
        )
        python = self._guarded(errors, "python", lambda: PythonEnvCollector(self.env).collect())
        pytorch = self._guarded(
            errors, "pytorch", lambda: PyTorchCollector(self.torch_import).collect()
        )
        compiler = self._guarded(
            errors,
            "compiler",
            lambda: CompilerCollector(self.runner, self.env, self.platform).collect(),
        )
        cmake = self._guarded(errors, "cmake", lambda: CMakeCollector(self.runner).collect())
        ninja = self._guarded(errors, "ninja", lambda: NinjaCollector(self.runner).collect())
        environment = self._guarded(
            errors, "environment", lambda: EnvironmentCollector(self.env, self.platform).collect()
        )
        runtime_libraries = self._guarded(
            errors,
            "runtime_libraries",
            lambda: RuntimeLibraryCollector(
                self.runner,
                environment=environment,
                platform=self.platform,
                toolkit_roots=toolkit_library_roots(cuda),
            ).collect(),
        )

        snapshot = EnvironmentSnapshot(system=system or self._fallback_system())
        if gpu is not None:
            snapshot.gpus = gpu.gpus
            snapshot.driver = gpu.driver
            snapshot.nvidia_smi = gpu.info
        if cuda is not None:
            snapshot.cuda = cuda
        if python is not None:
            snapshot.python = python
        if pytorch is not None:
            snapshot.pytorch = pytorch
        if compiler is not None:
            snapshot.compiler = compiler
        if cmake is not None:
            snapshot.cmake = cmake
        if ninja is not None:
            snapshot.ninja = ninja
        if environment is not None:
            snapshot.environment = environment
        if runtime_libraries is not None:
            snapshot.runtime_libraries = runtime_libraries
        snapshot.collection_errors = errors
        return snapshot

    @staticmethod
    def _guarded(
        errors: dict[str, str],
        name: str,
        factory: Callable[[], _T],
    ) -> _T | None:
        try:
            return factory()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    def _fallback_system(self) -> SystemInfo:
        return SystemInfo(
            os_name="Unknown",
            os_version="unknown",
            kernel_version=platform.release() or "unknown",
            architecture=platform.machine() or "unknown",
            platform=self.platform,
            python_version=platform.python_version()
            or ".".join(map(str, sys.version_info[:3])),
            python_executable=sys.executable or "unknown",
        )
