"""Platform detection helpers."""

from __future__ import annotations

import sys

from cuda_doctor.core.enums import Platform


def current_platform() -> Platform:
    """Classify the current operating system for CUDA-relevant behavior."""
    if sys.platform.startswith("win"):
        return Platform.WINDOWS
    if sys.platform.startswith("linux"):
        return Platform.LINUX
    if sys.platform.startswith("darwin"):
        return Platform.MACOS
    return Platform.OTHER
