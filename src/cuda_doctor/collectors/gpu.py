"""GPU and NVIDIA driver collector (orchestrates the nvidia-smi client)."""

from __future__ import annotations

from cuda_doctor.collectors.nvidia_smi import NvidiaSmiClient, NvidiaSmiResult
from cuda_doctor.utils.commands import CommandRunner


class GPUCollector:
    """Collects detected GPUs and driver facts via nvidia-smi.

    An unavailable or failing nvidia-smi is reported through
    ``NvidiaSmiResult.info`` — it never raises.
    """

    def __init__(self, runner: CommandRunner | None = None, executable: str = "nvidia-smi") -> None:
        self._client = NvidiaSmiClient(runner or CommandRunner(), executable)

    def collect(self) -> NvidiaSmiResult:
        return self._client.query()
