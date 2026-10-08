"""Version parsing and comparison helpers.

These parsers are intentionally forgiving: tool output varies across vendors,
releases, and locales. When nothing can be parsed, ``None`` is returned and the
caller treats the version as unknown.
"""

from __future__ import annotations

import re
from typing import NamedTuple

_DOTTED_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?")

# Matches "cuda-12.4", "cuda_11.8", "CUDA\v12.4", "cuda12.1", ...
_CUDA_DIR_RE = re.compile(r"cuda[/\\_.\s-]*v?(\d{1,2})[._-](\d{1,2})", re.IGNORECASE)


class CudaVersion(NamedTuple):
    """A CUDA version as normally written: major.minor (e.g. 12.4)."""

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """Extract the first dotted number group from free text.

    >>> parse_version("gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0")
    (11, 4, 0)
    """
    if not text:
        return None
    match = _DOTTED_VERSION_RE.search(str(text))
    if not match:
        return None
    parts = tuple(int(group) for group in match.groups() if group is not None)
    return parts or None


def parse_cuda_version(text: str | None) -> CudaVersion | None:
    """Extract a CUDA version (major.minor) from free text."""
    parts = parse_version(text)
    if not parts:
        return None
    return CudaVersion(parts[0], parts[1] if len(parts) > 1 else 0)


def parse_driver_version(text: str | None) -> tuple[int, ...] | None:
    """Extract an NVIDIA driver version such as 550.54.14 or 552.22."""
    return parse_version(text)


def compare_versions(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Compare dotted versions, padding with zeros ("12.4" == "12.4.0")."""
    size = max(len(a), len(b))
    pa = a + (0,) * (size - len(a))
    pb = b + (0,) * (size - len(b))
    if pa < pb:
        return -1
    if pa > pb:
        return 1
    return 0


def cuda_version_from_path(text: str | None) -> CudaVersion | None:
    """Extract a CUDA version from a directory name or path.

    Handles "/usr/local/cuda-12.4", "cuda_11.8", r"C:\\...\\CUDA\\v12.4".
    Returns ``None`` for unversioned names like "/usr/local/cuda" or
    unrelated names like "cudnn".
    """
    if not text:
        return None
    match = _CUDA_DIR_RE.search(str(text))
    if not match:
        return None
    return CudaVersion(int(match.group(1)), int(match.group(2)))


def format_version(parts: tuple[int, ...] | None) -> str | None:
    """Render a parsed version tuple back to a dotted string."""
    if not parts:
        return None
    return ".".join(str(part) for part in parts)
