# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-10-09

Distribution release: PyPI-ready packaging and reproducible publication
support. No diagnostic behavior changes.

### Changed

- Packaging metadata modernized to PEP 639: SPDX license expression
  (`license = "MIT"`), explicit `license-files`, and the deprecated license
  Trove classifier removed (build-system requirement raised to
  `setuptools>=77.0.3` accordingly). The project license itself is unchanged.
- README Quick Start is now PyPI-first (`python -m pip install cuda-doctor`
  on Linux/macOS, `py -m pip install cuda-doctor` on Windows) and explains
  that cuda-doctor should be installed into the environment whose PyTorch
  the user wants to diagnose — including a concise note on why pipx's
  isolation makes it a poor fit for this tool. Source/development installs
  moved to the Development section.

### Added

- Package CI (`.github/workflows/package.yml`): a distribution smoke matrix
  (Ubuntu/Windows × Python 3.10/3.13) that builds the wheel and sdist,
  validates them with `twine check`, installs the built wheel into a clean
  environment, and runs `cuda-doctor --version` / `python -m cuda_doctor
  --version` / `cuda-doctor info` from outside the checkout.
- PyPI publishing workflow (`.github/workflows/publish.yml`) using PyPI
  Trusted Publishing (`pypa/gh-action-pypi-publish`): token-free, triggered
  only by an actual GitHub Release publication, gated by the protected
  `pypi` environment, with the built wheel and sdist also attached to the
  GitHub Release by a separately permissioned job. This prepares PyPI
  distribution; the first actual upload will occur when the release
  workflow runs.

## [0.1.1] - 2026-10-09

Correctness pass on CUDA compatibility semantics.

### Changed

- Driver/CUDA compatibility now follows NVIDIA's CUDA 11+ minor-version
  compatibility model instead of treating the `nvidia-smi` CUDA version as a
  strict ceiling. The bundled driver table stores the documented per-family
  minimums (11.x: Linux 450.80.02 / Windows 452.39; 12.x: 525.60.13 / 528.33;
  13.x: the R580 branch rule, `>= 580` on both platforms — not the 580.65.06 /
  580.88 drivers packaged with the CUDA 13.0 toolkit) and no longer mixes them
  with the drivers-shipped-with-toolkit-releases table.
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
- `TORCH002`/`TORCH006` de-duplication: when CUDA is *observed* unavailable
  (`is_available()` False), `TORCH002` is the single primary diagnosis and
  carries the static driver evidence (generation gap, documented family
  minimum) as likely causes — `TORCH006` no longer adds a second ERROR for
  the same root cause. `TORCH006` now speaks only when the availability
  probe itself failed (`None`), worded conservatively to note that runtime
  behavior was not directly confirmed.
- Terminology: user-facing output no longer labels the `nvidia-smi` CUDA
  version a maximum. The terminal report says "Reported CUDA" (was
  "Max CUDA"), the Markdown report says "reported CUDA X.Y", and issue
  wording says "reported CUDA" — reflecting that CUDA minor-version
  compatibility makes the reported version a validated-with generation, not
  a hard ceiling.
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

### Fixed

- `CollectionRunner(platform=...)` now records the injected platform in
  `snapshot.system.platform` on any host OS. The system collector previously
  read the host OS, so a snapshot simulated for one platform but collected on
  another was diagnosed with the *host's* compatibility minimums (surfaced as
  the Windows CI failure of the driver-too-old integration scenario). With no
  override, production behavior is unchanged: the snapshot still reports the
  actual host platform.

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

[Unreleased]: https://github.com/TheManWhoIsStupid/cuda-doctor/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/TheManWhoIsStupid/cuda-doctor/releases/tag/v0.1.2
[0.1.1]: https://github.com/TheManWhoIsStupid/cuda-doctor/releases/tag/v0.1.1
[0.1.0]: https://github.com/TheManWhoIsStupid/cuda-doctor/releases/tag/v0.1.0
