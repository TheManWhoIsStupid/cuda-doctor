"""Python environment collector."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping

from cuda_doctor.core.models import PythonInfo


class PythonEnvCollector:
    """Collects interpreter and virtual-environment facts."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self.env: dict[str, str] = dict(env if env is not None else os.environ)

    def collect(self) -> PythonInfo:
        prefix = sys.prefix
        base = getattr(sys, "base_prefix", prefix)
        in_venv = prefix != base or bool(self.env.get("VIRTUAL_ENV"))
        return PythonInfo(
            version=platform.python_version(),
            executable=sys.executable or "unknown",
            in_virtual_env=in_venv,
            virtual_env_path=prefix if in_venv else None,
            pip_version=self._pip_version(),
        )

    @staticmethod
    def _pip_version() -> str | None:
        try:
            return importlib.metadata.version("pip")
        except importlib.metadata.PackageNotFoundError:
            return None
