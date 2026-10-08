"""Internal helpers shared by the compatibility modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from cuda_doctor.utils.versions import CudaVersion

_T = TypeVar("_T")


def nearest_lower(table: Mapping[CudaVersion, _T], cuda: CudaVersion) -> _T | None:
    """Value for the largest known CUDA version ``<= cuda``.

    Compatibility requirements grow monotonically with the CUDA version, so
    the nearest lower entry is a safe lower bound for unknown versions.
    """
    candidates = [key for key in table if key <= cuda]
    return table[max(candidates)] if candidates else None
