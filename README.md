# cuda-doctor

[![CI](https://github.com/TheManWhoIsStupid/cuda-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/TheManWhoIsStupid/cuda-doctor/actions/workflows/ci.yml)

> Read-only local diagnostics for CUDA / PyTorch development environments.
>
> Detect → Analyze → Explain → Recommend. Never modifies your system.

`cuda-doctor` inspects your machine the way a colleague would before helping you
debug a CUDA problem: it looks at the GPUs, the NVIDIA driver, the CUDA toolkit(s)
on disk and on `PATH`, the host compiler, and your PyTorch install — then tells
you what looks wrong, why, and what to do about it.

It is inspired by `flutter doctor`: findings are explained in plain language,
and a completed run always exits `0` (only internal tool failures exit non-zero).

## Quick start

```bash
# Python 3.10+
pip install -e ".[dev]"     # or: pip install .        (from a checkout)

cuda-doctor                  # run the full diagnosis (terminal report)
```

```text
CUDA Doctor v0.1.0

System
  OS              Linux Ubuntu 22.04.5 LTS (kernel 5.15.0-91-generic)
  ...
NVIDIA Driver
  Version         580.126.09
  Max CUDA        13.0
...
Summary
  Status: HEALTHY  (2 info)
```

## Commands

| Command | What it does |
| --- | --- |
| `cuda-doctor` | Same as `cuda-doctor diagnose` |
| `cuda-doctor diagnose` | Full diagnosis, terminal report |
| `cuda-doctor diagnose --verbose` | Also show info-level notes and probe errors |
| `cuda-doctor diagnose --format json` | Machine-readable report (schema v1) |
| `cuda-doctor diagnose --format markdown --output report.md` | Write a shareable report |
| `cuda-doctor info` | Tool metadata: version, checks, issue codes |
| `cuda-doctor --version` | Version |

`python -m cuda_doctor …` works identically to the `cuda-doctor` entry point.

**Exit codes:** `0` the diagnosis ran (regardless of findings) · `2` internal
fatal error. Findings never change the exit code — this is a diagnostician, not
a CI gate.

## What it checks (v0.1)

| Area | Codes | Highlights |
| --- | --- | --- |
| GPU | `GPU001`–`GPU003` | nvidia-smi missing, no GPUs, smi execution failure |
| Driver | `DRV001`–`DRV002` | unknown driver version; driver below the documented minimum for the toolkit's CUDA generation |
| CUDA | `CUDA001`–`CUDA006` | no nvcc; CUDA_HOME unset/invalid; multiple toolkits; multiple CUDA bins in PATH; nvcc ≠ CUDA_HOME |
| PyTorch | `TORCH001`–`TORCH006` | not installed; import failures (with known-signature advice); `is_available()` False; CPU-only wheel; runtime vs toolkit; runtime from a newer CUDA generation than the driver |
| Compiler | `CMP001`–`CMP002` | no host compiler; gcc/Visual Studio outside the toolkit's supported range |
| Environment | `ENV001`–`ENV004` | stale CUDA paths; duplicates; conflicting `LD_LIBRARY_PATH`; older CUDA shadowing newer in PATH |

Every finding carries a stable issue code, evidence, and concrete
recommendations. Example reports: [`examples/sample_report.md`](examples/sample_report.md)
· [`examples/sample_report.json`](examples/sample_report.json).

### The nuance that matters: three different "CUDA versions"

A machine reports three unrelated CUDA versions, and most false alarms in CUDA
debugging tools come from comparing the wrong pair:

1. **The local toolkit** (`nvcc --version`) — what you *compile* with.
2. **PyTorch's bundled runtime** (`torch.version.cuda`) — the CUDA libraries
   shipped inside the wheel; official wheels bundle their own runtime, so it
   does not have to match the local toolkit (`TORCH004` stays INFO).
3. **The driver's CUDA UMD version** (the `CUDA Version:` line in
   `nvidia-smi`) — the toolkit generation the driver was *validated with*.
   It is **not a hard ceiling**.

That third point is the one most tools get wrong. Since CUDA 11, NVIDIA
supports **CUDA minor-version compatibility**: within a CUDA major family
(e.g. any CUDA 12.x), applications built with a newer minor release run on
older drivers of the same generation, as long as the driver meets the
documented family minimum (Linux 525.60.13 / Windows 528.33 for CUDA 12.x).
So "toolkit 12.6, nvidia-smi says 12.2" is at most an informational note
(`DRV002` INFO), not an error — with the caveats that newer
driver-dependent features and newer PTX may still need a driver update.

What *is* a genuine error is a **generation gap** — a CUDA 13 toolkit or
PyTorch runtime on a CUDA 12-generation driver (`DRV002`/`TORCH006`) — and
even then only when CUDA is not observed working: if
`torch.cuda.is_available()` is `True`, observed runtime success always
overrides the static version comparison.

## Platforms

- **Linux** — first-class; developed and continuously smoke-tested on Ubuntu
  22.04 with NVIDIA H20 GPUs and 4 parallel CUDA toolkit installations.
- **Windows** — supported (separate code paths for `CUDA_PATH*` variables,
  Visual Studio detection via vswhere, `CREATE_NO_WINDOW` on subprocesses).
  Automated tests cover the Windows code paths with simulated environments;
  real-hardware validation is ongoing.
- **macOS** — runs and reports "no NVIDIA driver" as informational (Apple
  Silicon is out of scope for v0.1).

## Privacy

Reports are designed to be shareable:

- your home directory is rewritten to `~`, leftover usernames to `<user>`;
- hostname and username are never collected in the first place;
- `PATH`/`LD_LIBRARY_PATH` are shown only as the CUDA-relevant subset, never
  dumped in full;
- nvidia-smi output excerpts are truncated.

Still, a JSON report describes your hardware and software stack — read it
before posting publicly.

## Architecture

```
CLI (Typer) ─ CollectionRunner ─▶ Collectors (facts only)
                                     │  subprocess (no shell, timeouts),
                                     │  filesystem, env, torch import
                                     ▼
                               EnvironmentSnapshot
                               (dataclasses; absence = None)
                                     │
                          DiagnosisEngine (24 checks)
                             + bundled compatibility JSON
                                     ▼
                              Issues + Summary
                                     ▼
                    Reporters: terminal / JSON / Markdown
```

Key invariants (see `AGENTS.md`):

- **Collectors never judge; checks never collect.** Rules are pure functions of
  the snapshot (+ compatibility data), which makes them unit-testable without
  hardware.
- **Safe failure over crashes.** Every external probe is guarded; a failing
  collector surfaces as data, not an exception.
- **Compatibility knowledge is data**, versioned JSON inside the package
  (`src/cuda_doctor/data/`), updatable without touching logic.
- **Read-only, always.** v0.1 executes no command that modifies the system.

## Limitations (v0.1)

- Single-user, single-machine scope; no container/WSL-specific detection.
- Compatibility decisions are deliberately **conservative**: driver minimums
  follow NVIDIA's documented CUDA minor-version-compatibility baselines per
  major family, and anything the bundled knowledge does not cover — a future
  CUDA major (e.g. CUDA 14 before the data ships), an unlisted toolkit minor
  for compiler rules, or an unreported driver version — is reported as
  **unknown**, never guessed from older versions.
- Compiler compatibility tables are coarse and worded as *potential* issues;
  the definitive source is always the CUDA Installation Guide for your
  toolkit version.
- No conda-environment awareness beyond what `PATH`/env vars imply.
- nvidia-smi is the only GPU source; `NVML`/`lspci` fallbacks are future work.
- Chinese localization is planned (the maintainers are bilingual); v0.1 output
  is English.

## Development

```bash
bash scripts/dev_install.sh      # Linux/macOS   (pip install -e ".[dev]")
# or: pwsh scripts/dev_install.ps1   (Windows)

pytest                           # 291 tests: unit + integration (hermetic)
ruff check src tests             # lint
mypy                             # types (strict-ish: disallow_untyped_defs)
```

CI runs the same commands on Ubuntu and Windows across Python 3.10–3.13
(`.github/workflows/ci.yml`) — no GPU, driver, CUDA, or PyTorch required.

The test suite needs **no GPU and no CUDA**: collectors run against an injected
fake command runner, and real parser fixtures captured from actual H20 /
RTX 4090 machines live under `tests/fixtures/`.

### Repository layout

```
src/cuda_doctor/
├── cli.py                  # Typer app: diagnose / info / --version
├── core/                   # models, enums, CollectionRunner, exceptions
├── collectors/             # one module per fact source
├── checks/                 # 24 diagnostic rules (pure)
├── diagnosis/              # engine, Issue/Summary, recommendations
├── compatibility/          # driver/compiler/torch knowledge + JSON loading
├── reporters/              # terminal (Rich), JSON, Markdown
├── data/                   # bundled compatibility tables
└── utils/                  # commands, parsing, paths, redaction, versions
```

## Roadmap (v0.2+)

- NVML and `lspci` fallbacks when nvidia-smi is absent or broken.
- Conda-aware environment inspection.
- `--json-schema` self-description and a stable JSON schema contract test.
- WSL detection and WSL-specific driver mismatch explanations.
- Chinese output (`--lang zh`).
- `cuda-doctor fix --dry-run` advisory mode (still no modifications in v0.2).

## License

MIT — see `LICENSE`.
