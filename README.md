# cuda-doctor

[![CI](https://github.com/TheManWhoIsStupid/cuda-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/TheManWhoIsStupid/cuda-doctor/actions/workflows/ci.yml)

> Read-only local diagnostics for CUDA / PyTorch development environments.
>
> Detect → Analyze → Explain → Recommend. Never modifies your system.

`cuda-doctor` inspects your machine the way a colleague would before helping you
debug a CUDA problem: it looks at the GPUs, the NVIDIA driver, the CUDA toolkit(s)
on disk and on `PATH`, the host compiler, and your PyTorch install — then tells
you what looks wrong, why, and what to do about it.

Findings come in two layers. **Issues** are stable low-level findings ("no
nvcc on PATH", "driver below the documented minimum"), each with a code,
evidence, and recommendations. **Diagnoses** are an additive root-cause layer
on top: when the evidence is sufficient, cuda-doctor names the underlying
conflict — disagreeing CUDA selectors, a PyTorch build that clashes with the
local toolchain, a runtime-library conflict risk — and ranks at most one of them
as the `PRIMARY` cause. Not every run has a diagnosis; a healthy machine
simply reports none.

It is inspired by `flutter doctor`: findings are explained in plain language,
and a completed run always exits `0` (only internal tool failures exit non-zero).

## Quick start

Install from PyPI (Python 3.10+):

```bash
# Linux / macOS
python -m pip install cuda-doctor

# Windows
py -m pip install cuda-doctor
```

Then run it:

```bash
cuda-doctor                  # full diagnosis, terminal report
```

> **Install it where your PyTorch lives.** cuda-doctor imports PyTorch
> in-process from the Python environment it runs in. To diagnose the PyTorch
> in one of your conda/venv environments, install and run it *inside* that
> environment:
>
> ```bash
> conda activate my-torch-env
> python -m pip install cuda-doctor
> cuda-doctor
> ```
>
> For the same reason, prefer `pip install` over [pipx](https://pipx.pypa.io):
> a pipx install runs in its own isolated environment and cannot see the
> PyTorch installed in your conda/venv environment — it would report
> "torch not installed" even when it is.

```text
CUDA Doctor v0.2.0

System
  OS              Linux Ubuntu 22.04.5 LTS (kernel 5.15.0-91-generic)
  ...
NVIDIA Driver
  Version         580.126.09
  Reported CUDA   13.0
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
| `cuda-doctor diagnose --format json` | Machine-readable report (schema v2) |
| `cuda-doctor diagnose --format markdown --output report.md` | Write a shareable report |
| `cuda-doctor info` | Tool metadata: version, checks, issue codes |
| `cuda-doctor --version` | Version |

`python -m cuda_doctor …` works identically to the `cuda-doctor` entry point.

**Exit codes:** `0` the diagnosis ran (regardless of findings) · `2` internal
fatal error. Findings never change the exit code — this is a diagnostician, not
a CI gate.

## What it checks

Low-level checks — each produces an **Issue** with a stable code, evidence,
and concrete recommendations:

| Area | Codes | Highlights |
| --- | --- | --- |
| GPU | `GPU001`–`GPU003` | nvidia-smi missing, no GPUs, smi execution failure |
| Driver | `DRV001`–`DRV002` | unknown driver version; driver below the documented minimum for the toolkit's CUDA generation |
| CUDA | `CUDA001`–`CUDA006` | no nvcc; CUDA_HOME unset/invalid; multiple toolkits; multiple CUDA bins in PATH; nvcc ≠ CUDA_HOME |
| PyTorch | `TORCH001`–`TORCH006` | not installed; import failures (with known-signature advice); `is_available()` False; CPU-only wheel; runtime vs toolkit; runtime from a newer CUDA generation than the driver |
| Compiler | `CMP001`–`CMP002` | no host compiler; gcc/Visual Studio outside the toolkit's supported range |
| Environment | `ENV001`–`ENV004` | stale CUDA paths; duplicates; conflicting `LD_LIBRARY_PATH`; older CUDA shadowing newer in PATH |

Example reports: [`examples/sample_report.md`](examples/sample_report.md)
· [`examples/sample_report.json`](examples/sample_report.json).

### Root-cause diagnoses (v0.2)

On top of those checks, a diagnosis layer reasons over a normalized view of
the whole environment and names the underlying conflict when the evidence
supports one:

| Code | Diagnosis | What it means |
| --- | --- | --- |
| `D001` | CUDA Toolkit Resolution Conflict | The *active* CUDA development selectors disagree — e.g. `CUDA_HOME` resolves to one toolkit while `nvcc` on `PATH` reports another. Several installed toolkits alone are **not** a conflict. |
| `D002` | PyTorch / Local Toolkit Build Mismatch | The PyTorch build's CUDA version conflicts materially with the active local development toolchain. Relevant when building extensions against a local toolkit — not for wheel-only runtime use, where the local toolkit does not have to match PyTorch. |
| `D003` | PyTorch CUDA Unavailable | Explains an observed `torch.cuda.is_available() == False` when enough evidence exists: CPU-only wheel, no visible GPU, unusable or incompatible driver, a runtime-library conflict, or an explicit `CUDA_VISIBLE_DEVICES` mask. With insufficient evidence the cause is reported conservatively as unknown — never guessed. |
| `D004` | CUDA Runtime Library Conflict *(Linux only)* | Bounded static evidence of a runtime-library conflict risk: an exact versioned CUDA library name observed in `LD_LIBRARY_PATH` resolves to a file from an incompatible CUDA family, judged strictly by exact library load names. Same-name observations elsewhere (e.g. in the current Python environment) only corroborate the risk. No loader precedence across origins is inferred, and nothing is claimed about which shared object was actually loaded. |

A resolver then ranks the emitted diagnoses deterministically: at most one is
`PRIMARY`; the others are `CONTRIBUTING`, `POSSIBLE`, or `SECONDARY`, with
causal relationships made explicit (e.g. a runtime-library conflict
`explains ->` unavailable CUDA). In the human reports the Diagnoses section
leads the low-level issues, and a diagnosis may suppress the display of an
issue it supersedes (`TORCH002` under `D003`, `TORCH004` under `D002`) — the
stored Issues stay intact and the JSON report stays machine-complete.

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
documented family minimum (Linux 525.60.13 / Windows 528.33 for CUDA 12.x;
for CUDA 13.x the documented rule is the R580 driver branch, i.e. >= 580).
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
  Silicon is out of scope).

## Privacy

Reports are designed to be shareable:

- your home directory is rewritten to `~`, leftover usernames to `<user>`;
- hostname and username are never collected in the first place;
- `PATH`/`LD_LIBRARY_PATH` are shown only as the CUDA-relevant subset, never
  dumped in full;
- nvidia-smi output excerpts are truncated;
- the runtime-library inventory and CUDA probe exception details are internal —
  never serialized into any report;
- diagnosis pipeline errors are bounded and never contain raw exception text;
- redaction covers the structured diagnosis text as well.

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
                            DiagnosisFactBuilder
                                     ▼
              DiagnosisFacts (normalized, tri-state view)
                                     ▼
              D001 → D002 → D004 → D003 (deterministic rules)
                                     ▼
                DiagnosisResolver (assigns global roles)
                                     ▼
        DiagnosisResult (additive: + Diagnoses, diagnosis_errors)
                                     ▼
                 Reporters: terminal / JSON v2 / Markdown
```

Key invariants (see `AGENTS.md`):

- **Collectors never judge; checks never collect.** Collectors observe; the
  low-level checks keep their own finding semantics and are pure functions of
  the snapshot (+ compatibility data), which makes them unit-testable without
  hardware.
- **Facts normalize; rules reason.** The fact builder turns the snapshot into
  a normalized tri-state view; D001, D002, and D004 are pure functions of
  those facts, while D003 additionally receives explicit prior-candidate
  context from the rules already run (notably whether D004 was emitted). No
  rule recollects the host, and D003 never invokes D004 itself; the checks
  are *not* the root-cause rules.
- **Only the resolver assigns roles.** Rules emit candidates; the resolver
  deterministically selects at most one `PRIMARY` and ranks the rest.
- **Reporters only present.** They serialize and render; nothing downstream
  re-sorts, invents roles, or feeds back into the checks.
- **Safe failure over crashes.** Every external probe is guarded; a failing
  collector or diagnosis component surfaces as bounded data, not an exception.
- **Compatibility knowledge is data**, versioned JSON inside the package
  (`src/cuda_doctor/data/`), updatable without touching logic.
- **Read-only, always.** cuda-doctor executes no command that modifies the
  system; remediation steps are display-only text, never executed.

## Limitations

- Single-user, single-machine scope.
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
- `D004` (runtime-library conflict) is Linux-only; there is no Windows
  equivalent in this release.
- Out of scope for the current feature set: WSL-specific diagnosis,
  container/Docker detection, NCCL/distributed diagnostics, ROCm/HIP,
  TensorFlow/JAX diagnosis, and automatic fixes.
- Chinese localization is planned (the maintainers are bilingual); output is
  English.

## Development

```bash
bash scripts/dev_install.sh      # Linux/macOS   (pip install -e ".[dev]")
# or: pwsh scripts/dev_install.ps1   (Windows)

pytest                           # 1,300+ hermetic tests: unit + integration
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
├── checks/                 # 24 low-level checks (pure)
├── diagnosis/              # facts, rules (D001–D004), resolver, models, engine
├── compatibility/          # driver/compiler/torch knowledge + JSON loading
├── reporters/              # terminal (Rich), JSON, Markdown
├── data/                   # bundled compatibility tables
└── utils/                  # commands, parsing, paths, redaction, versions
```

## Roadmap (post-v0.2)

- NVML and `lspci` fallbacks when nvidia-smi is absent or broken.
- Conda-aware environment inspection.
- `--json-schema` self-description and a stable JSON schema contract test.
- WSL detection and WSL-specific driver mismatch explanations.
- Chinese output (`--lang zh`).
- `cuda-doctor fix --dry-run` advisory mode (display-only suggestions; the
  tool itself stays read-only).

## License

MIT — see `LICENSE`.
