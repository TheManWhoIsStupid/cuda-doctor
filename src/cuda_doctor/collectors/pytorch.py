"""PyTorch collector (PyTorch is an optional dependency).

Every torch call can fail on a broken environment — that failure mode is
itself valuable diagnostic data, so failures are captured instead of raised.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from cuda_doctor.core.models import PyTorchInfo, TorchDevice


def _safe(call: Callable[[], Any]) -> Any:
    """Run a torch call, returning None on any failure."""
    try:
        return call()
    except Exception:
        return None


class PyTorchCollector:
    """Collects PyTorch facts via an injectable ``import_module``."""

    def __init__(self, import_module: Callable[[str], Any] | None = None) -> None:
        self._import_module = import_module or importlib.import_module

    def collect(self) -> PyTorchInfo:
        info = PyTorchInfo()
        try:
            torch = self._import_module("torch")
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                return info  # genuinely absent: a normal fact, not an error
            # e.g. "No module named 'torch._C'" — the package is installed
            # but broken; reporting "not installed" would hide the problem.
            info.installed = True
            info.import_error = f"{type(exc).__name__}: {exc}"[:300]
            return info
        except Exception as exc:
            # Any other import-time failure (missing native library, DLL
            # load failure, ...) means a broken install, not an absent one.
            info.installed = True
            info.import_error = f"{type(exc).__name__}: {exc}"[:300]
            return info

        info.installed = True
        info.version = _safe(lambda: torch.__version__)
        info.cuda_version = _safe(lambda: torch.version.cuda)
        info.is_cuda_build = info.cuda_version is not None
        info.cuda_available = _safe(lambda: torch.cuda.is_available())
        if info.cuda_available:
            count = _safe(lambda: torch.cuda.device_count())
            info.device_count = count if isinstance(count, int) else None
            cudnn = _safe(lambda: torch.backends.cudnn.version())
            info.cudnn_version = str(cudnn) if cudnn is not None else None
            for index in range(info.device_count or 0):
                device = TorchDevice(index=index)
                try:
                    device.name = torch.cuda.get_device_name(index)
                except Exception:
                    device.name = None
                try:
                    capability = torch.cuda.get_device_capability(index)
                    device.compute_capability = (
                        ".".join(str(part) for part in capability)
                        if isinstance(capability, tuple)
                        else None
                    )
                except Exception:
                    device.compute_capability = None
                info.devices.append(device)
        return info
