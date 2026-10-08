"""Central recommendation texts.

Checks may append contextual advice, but the default guidance per issue code
lives here so wording stays consistent, reviewable, and never contradicts
the tool's read-only philosophy.
"""

from __future__ import annotations

DRIVER_DOWNLOAD = (
    "Update the NVIDIA driver: https://www.nvidia.com/Download "
    "(on Linux, the driver shipped by your distribution is usually fine)."
)
CUDA_DOWNLOAD = "Install the CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
PYTORCH_INSTALL = (
    "Install a CUDA-enabled PyTorch wheel: https://pytorch.org/get-started/locally "
    "(official wheels bundle their own CUDA runtime)."
)
CUDA_TOOLKIT_DOCS = (
    "See the CUDA Toolkit Installation Guide for Linux/Windows: "
    "https://docs.nvidia.com/cuda/"
)

DEFAULT_RECOMMENDATIONS: dict[str, list[str]] = {
    "GPU001": [
        "nvidia-smi ships with the NVIDIA driver; install the driver to use CUDA.",
        DRIVER_DOWNLOAD,
    ],
    "GPU002": [
        "No NVIDIA GPU is visible to the driver. If this is unexpected, check that the "
        "GPU is installed/seated and that the system BIOS enables it.",
    ],
    "GPU003": [
        "nvidia-smi failed to talk to the driver. A driver reinstall usually fixes this; "
        "the stderr output above often names the exact cause.",
        DRIVER_DOWNLOAD,
    ],
    "DRV001": [
        "Without the driver version, compatibility cannot be verified. "
        "Run nvidia-smi manually to see what it reports.",
    ],
    "DRV002": [
        "Update the NVIDIA driver to one supporting your CUDA version, or use an older "
        "CUDA toolkit that your driver supports.",
        DRIVER_DOWNLOAD,
        CUDA_TOOLKIT_DOCS,
    ],
    "CUDA001": [
        "Install the CUDA Toolkit if you need to compile CUDA code; running prebuilt "
        "PyTorch wheels does not require a local nvcc.",
        CUDA_DOWNLOAD,
    ],
    "CUDA002": [
        "Set CUDA_HOME (Linux) / CUDA_PATH (Windows) to your toolkit directory if build "
        "tools need it; many workflows run fine without it.",
    ],
    "CUDA003": [
        "Point CUDA_HOME/CUDA_PATH at an existing CUDA Toolkit directory "
        "(one containing bin/nvcc), or unset it.",
    ],
    "CUDA004": [
        "Multiple toolkits are fine; just make sure PATH/CUDA_HOME deliberately select "
        "the one you intend to use.",
    ],
    "CUDA005": [
        "Keep only one CUDA bin directory in PATH (usually the one matching CUDA_HOME) "
        "so the intended nvcc wins.",
    ],
    "CUDA006": [
        "Align CUDA_HOME with the nvcc you actually use (export "
        "CUDA_HOME=<dir containing that nvcc>), or prepend the intended toolkit's bin "
        "directory to PATH.",
    ],
    "TORCH001": [
        "Install PyTorch if you use it: https://pytorch.org/get-started/locally"
    ],
    "TORCH002": [
        "torch.cuda.is_available() returned False despite a CUDA build — the usual causes "
        "are a missing/outdated driver or a CUDA runtime newer than the driver supports.",
        DRIVER_DOWNLOAD,
    ],
    "TORCH003": [
        "A CPU-only PyTorch build cannot use the GPU. Reinstall a CUDA wheel.",
        PYTORCH_INSTALL,
    ],
    "TORCH004": [
        "No action needed: PyTorch wheels bundle their own CUDA runtime, so this "
        "mismatch is normally harmless. Only align versions when building custom "
        "CUDA extensions.",
    ],
    "TORCH005": [
        "The error message above usually identifies the missing library; common fixes: "
        "reinstall the NVIDIA driver, reinstall PyTorch, or start from a clean virtual "
        "environment.",
    ],
    "TORCH006": [
        "Update the NVIDIA driver to a version supporting CUDA "
        "(or install a PyTorch build for an older CUDA).",
        DRIVER_DOWNLOAD,
    ],
    "CMP001": [
        "Install a host compiler: gcc/g++ (Linux) or Visual Studio Build Tools with "
        "C++ workload (Windows) — required to compile CUDA code.",
    ],
    "CMP002": [
        "Check the CUDA Installation Guide for the compilers your toolkit supports; "
        "usually a matching gcc/Visual Studio version is available in your package "
        "manager.",
    ],
    "ENV001": [
        "Remove stale CUDA entries from PATH/CUDA_HOME, or reinstall the toolkit so "
        "the paths exist again.",
    ],
    "ENV002": [
        "Duplicated PATH entries are harmless but confusing; keep one.",
    ],
    "ENV003": [
        "Multiple CUDA library directories in LD_LIBRARY_PATH can shadow each other; "
        "keep only the intended version (usually none is needed for pip-installed "
        "PyTorch).",
    ],
    "ENV004": [
        "PATH is searched in order: the earlier (older) CUDA bin will win. Reorder if "
        "that is not what you want.",
    ],
}


def recommendations_for(code: str) -> list[str]:
    """Default recommendations for an issue code (empty when unknown)."""
    return list(DEFAULT_RECOMMENDATIONS.get(code, []))
