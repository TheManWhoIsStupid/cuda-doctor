# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Target: v0.1.1 — correctness pass on CUDA compatibility semantics
(branch `fix/v0.1.1-correctness`, pending external review; not released).

### Changed

- Driver/CUDA compatibility now follows NVIDIA's CUDA 11+ minor-version
  compatibility model instead of treating the `nvidia-smi` CUDA version as a
  strict ceiling. The bundled driver table stores the documented per-family
  minimums (11.x: Linux 450.80.02 / Windows 452.39; 12.x: 525.60.13 / 528.33;
  13.x: 580.65.06 / 580.88) and no longer mixes them with the
  drivers-shipped-with-toolkit-releases table.
- `DRV002`: a toolkit minor above the driver's reported CUDA UMD version
  within the same CUDA generation is now at most INFO (minor-version
  compatibility applies, with PTX/feature caveats). Hard findings are
  reserved for generation gaps and drivers below the documented family
  minimum, always worded with the forward-compatibility-package caveat.
- `TORCH006`: only fires for generation gaps or documented-minimum
  violations, and never when `torch.cuda.is_available()` is `True` —
  observed runtime success overrides static version comparisons. `TORCH002`
  now folds the documented family minimum into its evidence as a likely
  cause when the driver is below it.
- Unknown/future CUDA versions are UNKNOWN: removed the nearest-lower
  fallback in driver and compiler lookups, so CUDA 14 no longer inherits
  CUDA 13 rules and an unlisted toolkit minor no longer inherits compiler
  rules.
- Environment PATH splitting uses the injected target platform's separator
  (`;` vs `:`) instead of the host's `os.pathsep`.
- The terminal report's Compatibility section no longer marks a same-family
  minor gap with the error symbol (info mark instead, matching `DRV002` INFO),
  and the torch-vs-driver row stays green whenever
  `torch.cuda.is_available()` is `True`.

### Added

- GitHub Actions CI (`.github/workflows/ci.yml`): pytest on
  Ubuntu/Windows × Python 3.10–3.13, plus a ruff + mypy job. No GPU, CUDA,
  driver, or PyTorch required.
- Regression tests for all of the above, including the five scenarios from
  the correctness review (same-family minor gaps, working-runtime
  contradiction guard, below-minimum driver as likely cause, future-major
  UNKNOWN handling for both driver and compiler rules, and cross-platform
  PATH separators).

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
- Hermetic test suite (unit + end-to-end simulated machines; no GPU required)
  with real nvidia-smi/nvcc parser fixtures captured from H20 and RTX 4090
  machines.
- Privacy-first reporting: home-directory and username redaction, no hostname
  collection, CUDA-only PATH excerpts.
- Read-only tool: never installs, uninstalls, or modifies anything.

### Fixed

Self-review hardening pass:
- JSON reports no longer embed the full `PATH` / raw `LD_LIBRARY_PATH`
  (only the CUDA-relevant subsets, as documented) and now redact
  `check_errors` texts like every other field.
- Toolkit installations sort by parsed version, not path text
  (single-digit versions such as CUDA 9.0 vs 10.0 previously misordered).
- A corrupted PyTorch install (e.g. missing `torch._C`) is reported as
  "installed but broken" instead of "not installed".
- Environment-variable names are case-insensitive on Windows only; lowercase
  Linux variables are no longer conflated with their uppercase forms.
- Unwritable `--output` paths and renderer failures exit with a friendly
  message (code 2) instead of a traceback.
- Internal check errors are keyed by check class, so two checks sharing an
  issue code can no longer overwrite each other's diagnostics.

[Unreleased]: https://github.com/TheManWhoIsStupid/cuda-doctor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TheManWhoIsStupid/cuda-doctor/releases/tag/v0.1.0
