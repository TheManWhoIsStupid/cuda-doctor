"""Fixture-based tests for external tool output parsers."""

from __future__ import annotations

from cuda_doctor.utils.parsing import (
    parse_driver_banner,
    parse_memory_mb,
    parse_nvcc_version,
    parse_nvidia_smi_xml,
    parse_query_gpu_csv,
)


def _load(fixtures_dir, relative):
    return (fixtures_dir / relative).read_text(encoding="utf-8")


class TestQueryGpuCsv:
    def test_real_h20_multi_gpu(self, fixtures_dir):
        gpus = parse_query_gpu_csv(_load(fixtures_dir, "nvidia_smi/query_csv_h20_multi.txt"))
        assert len(gpus) == 8
        assert gpus[0].name == "NVIDIA H20-3e"
        assert gpus[0].index == 0
        assert gpus[0].memory_total_mb == 143771
        assert gpus[0].compute_capability == "9.0"
        assert gpus[0].uuid and gpus[0].uuid.startswith("GPU-")

    def test_single_gpu(self, fixtures_dir):
        gpus = parse_query_gpu_csv(_load(fixtures_dir, "nvidia_smi/query_csv_single.txt"))
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert gpus[0].memory_total_mb == 24564

    def test_malformed_rows_skipped_not_fatal(self, fixtures_dir):
        gpus = parse_query_gpu_csv(_load(fixtures_dir, "nvidia_smi/query_csv_malformed.txt"))
        # Error banner, blank and junk rows are skipped; usable rows survive.
        names = [gpu.name for gpu in gpus]
        assert names == ["NVIDIA H100 PCIe [partial row]", "NVIDIA A100-SXM4-40GB"]
        assert gpus[1].memory_total_mb is None  # '[N/A]'
        assert gpus[1].compute_capability == "8.0"

    def test_empty_output(self):
        assert parse_query_gpu_csv("") == []


class TestNvidiaSmiXml:
    def test_real_h20_xml(self, fixtures_dir):
        driver = parse_nvidia_smi_xml(_load(fixtures_dir, "nvidia_smi/xml_h20.xml"))
        assert driver is not None
        assert driver.version == "580.126.09"
        assert driver.cuda_version == "13.0"

    def test_invalid_xml(self):
        assert parse_nvidia_smi_xml("this is not xml <") is None

    def test_missing_fields(self):
        assert parse_nvidia_smi_xml("<root></root>") is None


class TestDriverBannerFallback:
    def test_real_banner(self, fixtures_dir):
        driver = parse_driver_banner(_load(fixtures_dir, "nvidia_smi/plain_banner.txt"))
        assert driver is not None
        assert driver.version == "580.126.09"
        assert driver.cuda_version == "13.0"

    def test_no_banner(self):
        assert parse_driver_banner("something else entirely") is None


class TestNvccVersion:
    def test_real_12_4(self, fixtures_dir):
        assert parse_nvcc_version(_load(fixtures_dir, "nvcc/nvcc_12_4.txt")) == "12.4"

    def test_11_5(self, fixtures_dir):
        assert parse_nvcc_version(_load(fixtures_dir, "nvcc/nvcc_11_5.txt")) == "11.5"

    def test_malformed(self, fixtures_dir):
        assert parse_nvcc_version(_load(fixtures_dir, "nvcc/nvcc_malformed.txt")) is None

    def test_empty(self):
        assert parse_nvcc_version("") is None


class TestMemoryParsing:
    def test_mib(self):
        assert parse_memory_mb("24576 MiB") == 24576

    def test_thousands_separator(self):
        assert parse_memory_mb("1,024 MiB") == 1024

    def test_na(self):
        assert parse_memory_mb("[N/A]") is None

    def test_empty(self):
        assert parse_memory_mb("") is None
