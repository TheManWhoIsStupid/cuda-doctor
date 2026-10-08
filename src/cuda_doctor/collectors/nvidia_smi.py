"""nvidia-smi client: invocation and parsing.

Machine-readable formats are preferred: a CSV ``--query-gpu`` call for the
GPU list and ``-q -x`` XML for driver facts. The plain banner table is only
parsed as a last-resort fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cuda_doctor.core.models import DriverInfo, GPUInfo, NvidiaSmiInfo
from cuda_doctor.utils.commands import ERROR_NOT_FOUND, CommandResult, CommandRunner
from cuda_doctor.utils.parsing import (
    parse_driver_banner,
    parse_nvidia_smi_xml,
    parse_query_gpu_csv,
)
from cuda_doctor.utils.paths import find_executable

QUERY_GPU_ARGS = (
    "--query-gpu=index,name,uuid,memory.total,compute_cap",
    "--format=csv,noheader",
)
# Fallback for very old drivers that do not know the compute_cap query field.
QUERY_GPU_MINIMAL_ARGS = (
    "--query-gpu=index,name,uuid,memory.total",
    "--format=csv,noheader",
)
XML_ARGS = ("-q", "-x")

_EXCERPT_LIMIT = 400


@dataclass
class NvidiaSmiResult:
    """Everything nvidia-smi told us, normalized."""

    gpus: list[GPUInfo] = field(default_factory=list)
    driver: DriverInfo | None = None
    info: NvidiaSmiInfo = field(default_factory=NvidiaSmiInfo)


def _excerpt(text: str) -> str | None:
    return (text[:_EXCERPT_LIMIT].strip() or None) if text else None


class NvidiaSmiClient:
    """Runs nvidia-smi and turns its output into facts."""

    def __init__(self, runner: CommandRunner, executable: str = "nvidia-smi") -> None:
        self.runner = runner
        self.executable = executable

    def query(self) -> NvidiaSmiResult:
        result = NvidiaSmiResult()
        path = find_executable(self.executable)
        if path is None:
            result.info = NvidiaSmiInfo(available=False, error=ERROR_NOT_FOUND)
            return result
        result.info.available = True

        csv = self.runner.run((path, *QUERY_GPU_ARGS))
        if csv.error is None:
            result.info.executed = True
        if self._should_retry_minimal(csv):
            # Older driver: retry without the compute_cap field.
            retry = self.runner.run((path, *QUERY_GPU_MINIMAL_ARGS))
            if retry.error is None:
                csv = retry
        if csv.error is None:
            result.gpus = parse_query_gpu_csv(csv.stdout)
            result.info.stdout_excerpt = _excerpt(csv.stdout)
            if csv.return_code != 0 and not result.gpus:
                # Executed but failed (e.g. driver problem): keep as evidence.
                result.info.error = f"exit-{csv.return_code}"
                result.info.stderr_excerpt = _excerpt(csv.stderr)
        else:
            result.info.error = csv.error
            result.info.stderr_excerpt = _excerpt(csv.stderr)

        xml = self.runner.run((path, *XML_ARGS))
        driver = parse_nvidia_smi_xml(xml.stdout) if xml.stdout else None
        if driver is None and xml.error is None:
            banner = self.runner.run((path,))
            if banner.stdout:
                driver = parse_driver_banner(banner.stdout)
            if driver is None and banner.stderr:
                result.info.stderr_excerpt = result.info.stderr_excerpt or _excerpt(banner.stderr)
        if driver is not None:
            result.driver = driver
        return result

    @staticmethod
    def _should_retry_minimal(result: CommandResult) -> bool:
        """Old drivers reject the compute_cap query field (non-zero exit)."""
        if not result.stderr:
            return False
        return "not a valid field" in result.stderr.lower()
