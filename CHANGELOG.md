# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-08

### Added

- Initial MVP release.
- Environment collectors: system, GPU (nvidia-smi), NVIDIA driver, CUDA Toolkit
  (nvcc / CUDA_HOME / installations), Python, PyTorch, compilers, CMake, Ninja,
  and CUDA-relevant environment variables.
- Normalized `EnvironmentSnapshot` data model; missing software is a fact, not
  an error.
- Diagnosis engine with structured issue model and stable issue codes
  (GPU*, DRV*, CUDA*, TORCH*, CMP*, ENV*).
- Compatibility knowledge (driver ↔ CUDA, host compiler ↔ CUDA, PyTorch CUDA
  runtime interpretation) loaded from versioned JSON data files.
- Reporters: Rich terminal output, JSON, and Markdown.
- CLI (`cuda-doctor`, `cuda-doctor diagnose`, `cuda-doctor info`,
  `cuda-doctor --version`) with `--format terminal|json|markdown`, `--output`,
  and `--verbose` options.
- Read-only tool: never installs, uninstalls, or modifies anything.

[Unreleased]: https://github.com/TheManWhoIsStupid/cuda-doctor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TheManWhoIsStupid/cuda-doctor/releases/tag/v0.1.0
