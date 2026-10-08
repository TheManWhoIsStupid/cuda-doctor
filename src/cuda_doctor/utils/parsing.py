"""Parsers for external tool output.

Machine-readable query formats are preferred over scraping the pretty
terminal tables; the plain-banner parser exists only as a fallback.
"""

from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET

from cuda_doctor.core.models import DriverInfo, GPUInfo

_MEMORY_MB_RE = re.compile(r"(\d[\d,]*)\s*MiB", re.IGNORECASE)
_DRIVER_BANNER_RE = re.compile(
    r"Driver Version\s*:\s*([\d.]+)\s+CUDA Version\s*:\s*([\d.]+)"
)
_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)


def _clean(value: str | None) -> str | None:
    """Trim a CSV field; map nvidia-smi's "[N/A]" placeholders to None."""
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() in {"[N/A]", "N/A", "-"}:
        return None
    return text


def parse_memory_mb(text: str | None) -> int | None:
    """Parse VRAM sizes like '24576 MiB' into an integer MB count."""
    if not text:
        return None
    match = _MEMORY_MB_RE.search(text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_query_gpu_csv(stdout: str) -> list[GPUInfo]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader`` output.

    Expected field order: index,name,uuid,memory.total,compute_cap,driver_version.
    Malformed rows are skipped rather than raising.
    """
    gpus: list[GPUInfo] = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < 2:
            continue
        try:
            index = int(_clean(row[0]) or "-1")
        except ValueError:
            continue
        name = _clean(row[1]) or "Unknown GPU"
        uuid = _clean(row[2]) if len(row) > 2 else None
        memory = parse_memory_mb(row[3]) if len(row) > 3 else None
        compute_cap = _clean(row[4]) if len(row) > 4 else None
        gpus.append(
            GPUInfo(
                index=index,
                name=name,
                uuid=uuid,
                memory_total_mb=memory,
                compute_capability=compute_cap,
            )
        )
    return gpus


def parse_nvidia_smi_xml(xml_text: str) -> DriverInfo | None:
    """Extract driver and max-supported-CUDA versions from ``nvidia-smi -q -x``."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    driver_version = root.findtext(".//driver_version")
    cuda_version = root.findtext(".//cuda_version")
    if driver_version is None and cuda_version is None:
        return None
    return DriverInfo(
        version=_clean(driver_version),
        cuda_version=_clean(cuda_version),
        source="nvidia-smi -q -x",
    )


def parse_driver_banner(stdout: str) -> DriverInfo | None:
    """Fallback: parse the plain ``nvidia-smi`` banner table header."""
    match = _DRIVER_BANNER_RE.search(stdout)
    if not match:
        return None
    return DriverInfo(version=match.group(1), cuda_version=match.group(2), source="nvidia-smi")


def parse_nvcc_version(stdout: str) -> str | None:
    """Parse ``nvcc --version`` output (e.g. 'release 12.4, V12.4.131').

    nvcc always prints a ``release X.Y`` line; anything else is treated as
    unknown rather than guessing from loose digit matches.
    """
    match = _NVCC_RELEASE_RE.search(stdout)
    return match.group(1) if match else None
