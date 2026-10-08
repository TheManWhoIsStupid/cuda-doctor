"""Exception hierarchy for cuda-doctor.

The normal absence of software (no GPU, no nvcc, no PyTorch, ...) must never
raise: it is a diagnostic fact. These exceptions are reserved for genuinely
unexpected internal failures.
"""

from __future__ import annotations


class CudaDoctorError(Exception):
    """Base class for all cuda-doctor errors."""


class CollectorError(CudaDoctorError):
    """A collector failed unexpectedly (collectors normally never raise)."""


class CompatibilityDataError(CudaDoctorError):
    """Bundled compatibility data is missing or unreadable."""


class ReportError(CudaDoctorError):
    """Rendering or writing a report failed."""
