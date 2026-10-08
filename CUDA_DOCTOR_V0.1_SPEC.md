You are the lead engineer responsible for designing and implementing an open-source CLI project named `cuda-doctor`.

Your goal is to build a high-quality v0.1.0 MVP of a local CUDA/PyTorch development environment diagnostic tool.

Do not build a toy demo.

Build a maintainable, extensible, testable project that could later become a real open-source/commercial developer tool.

# 1. Product goal

`cuda-doctor` is a local command-line diagnostic tool for developers working with NVIDIA GPUs, CUDA, PyTorch, C++, and related build tools.

The main user workflow is:

```bash
cuda-doctor
```

The tool should:

1. inspect the local machine;
2. collect CUDA/GPU/toolchain information;
3. normalize the collected information into structured models;
4. run diagnostic rules;
5. identify likely configuration problems;
6. explain whether a mismatch is actually problematic;
7. provide actionable recommendations;
8. render results in terminal, JSON, or Markdown;
9. never modify the user's system in v0.1.

The tool must be safe to run.

It MUST NOT:

- install packages;
- uninstall packages;
- modify PATH;
- modify environment variables;
- modify the registry;
- use sudo;
- modify CUDA installations;
- modify NVIDIA drivers;
- modify PyTorch;
- modify system configuration.

The v0.1 product philosophy is:

Detect → Analyze → Explain → Recommend

NOT:

Detect → Automatically modify system

---

# 2. Supported platforms

Primary support:

- Windows 10/11
- Linux

Python:

- Python >= 3.10

The code should be designed so that macOS does not crash, even though CUDA is normally unavailable.

Do not assume all tools exist.

Missing:

- nvidia-smi
- nvcc
- PyTorch
- gcc
- g++
- MSVC
- cmake
- ninja

must be handled gracefully.

A missing dependency is diagnostic information, not a fatal program error.

---

# 3. Architecture

Implement this architecture:

CLI
↓
Collectors
↓
Normalized Environment Model
↓
Diagnostic Checks / Compatibility Rules
↓
Diagnosis Engine
↓
Reporters

Collectors MUST collect facts only.

Collectors MUST NOT contain diagnostic policy.

Checks MUST evaluate the normalized environment model and generate structured issues.

Reporters MUST only render structured results.

Keep platform-specific logic isolated.

Avoid giant modules and deeply nested conditionals.

---

# 4. Repository structure

Create approximately the following structure:

```text
cuda-doctor/
│
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── .gitignore
├── .pre-commit-config.yaml
│
├── src/
│   └── cuda_doctor/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── version.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── enums.py
│       │   ├── context.py
│       │   ├── runner.py
│       │   └── exceptions.py
│       │
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── system.py
│       │   ├── gpu.py
│       │   ├── nvidia_smi.py
│       │   ├── cuda.py
│       │   ├── python_env.py
│       │   ├── pytorch.py
│       │   ├── compiler.py
│       │   ├── cmake.py
│       │   ├── ninja.py
│       │   └── environment.py
│       │
│       ├── checks/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── gpu_checks.py
│       │   ├── driver_checks.py
│       │   ├── cuda_checks.py
│       │   ├── pytorch_checks.py
│       │   ├── compiler_checks.py
│       │   └── environment_checks.py
│       │
│       ├── compatibility/
│       │   ├── __init__.py
│       │   ├── cuda_driver.py
│       │   ├── pytorch_cuda.py
│       │   └── compiler_cuda.py
│       │
│       ├── diagnosis/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── issue.py
│       │   └── recommendations.py
│       │
│       ├── reporters/
│       │   ├── __init__.py
│       │   ├── terminal.py
│       │   ├── json_report.py
│       │   └── markdown.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── commands.py
│           ├── parsing.py
│           ├── paths.py
│           ├── versions.py
│           └── platform.py
│
├── data/
│   ├── cuda_driver_compatibility.json
│   ├── cuda_compiler_compatibility.json
│   └── known_issues.json
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   ├── dev_install.sh
│   └── dev_install.ps1
│
└── examples/
    ├── sample_report.md
    └── sample_report.json
```

You may make small architectural improvements if justified.

Do not collapse everything into a few large files.

---

# 5. Core models

Use Python dataclasses or another lightweight typed structure.

Create a central:

```python
EnvironmentSnapshot
```

It should contain normalized representations of:

- system information;
- GPU information;
- NVIDIA driver;
- CUDA Toolkit;
- Python runtime;
- PyTorch;
- compiler;
- CMake;
- Ninja;
- environment variables.

Example concepts:

```python
SystemInfo
GPUInfo
DriverInfo
CUDAInfo
PythonInfo
PyTorchInfo
CompilerInfo
ToolInfo
EnvironmentInfo
EnvironmentSnapshot
```

Example GPU fields:

```python
index
name
uuid
memory_total_mb
compute_capability
```

Example CUDA fields:

```python
nvcc_found
nvcc_path
toolkit_version
cuda_home
installations
```

Example PyTorch fields:

```python
installed
version
cuda_version
cuda_available
device_count
cudnn_version
```

All optional or unavailable information must be represented safely.

Do not use exceptions for normal absence of software.

---

# 6. Command execution abstraction

Create a reusable command runner.

It should:

- use subprocess safely;
- never use shell=True unless absolutely required;
- capture stdout;
- capture stderr;
- capture return code;
- support timeout;
- gracefully handle executable-not-found;
- return structured results;
- never crash the whole application because one command failed.

Suggested model:

```python
CommandResult:
    command
    return_code
    stdout
    stderr
    success
    error
```

Use this abstraction for:

- nvidia-smi
- nvcc
- gcc
- g++
- cl
- cmake
- ninja

---

# 7. System collection

Collect:

- OS name;
- OS version;
- kernel/version;
- architecture;
- Python version;
- executable path;
- platform.

Avoid leaking unnecessary personal information.

---

# 8. NVIDIA GPU collection

Use `nvidia-smi` where possible.

Collect:

- detected GPUs;
- GPU index;
- GPU name;
- UUID if useful;
- total VRAM;
- driver version;
- CUDA compatibility version reported by nvidia-smi.

If `nvidia-smi` is unavailable or fails:

- do not crash;
- preserve stdout/stderr;
- create appropriate diagnostic information.

Prefer machine-readable querying where possible rather than fragile parsing of the pretty terminal table.

---

# 9. CUDA Toolkit collection

Detect:

- `nvcc`;
- nvcc path;
- nvcc version;
- CUDA_HOME;
- CUDA_PATH;
- Windows CUDA_PATH_V* variables;
- standard CUDA installation directories;
- multiple CUDA Toolkit installations.

Windows examples:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*
```

Linux examples:

```text
/usr/local/cuda
/usr/local/cuda-*
```

Determine whether:

- CUDA_HOME exists;
- CUDA_HOME points to a real CUDA installation;
- nvcc belongs to CUDA_HOME;
- multiple CUDA versions may be competing through PATH.

Do not assume toolkit version must equal PyTorch runtime version.

---

# 10. Python collection

Collect:

- Python version;
- Python executable;
- virtual environment presence;
- virtual environment path if relevant;
- pip version where easily available.

Redact home directory paths where possible.

---

# 11. PyTorch collection

PyTorch must be an optional dependency.

Do NOT require torch in the package dependencies.

If installed, collect:

```python
torch.__version__
torch.version.cuda
torch.cuda.is_available()
torch.cuda.device_count()
torch.backends.cudnn.version()
```

If safe, collect:

```python
torch.cuda.get_device_name()
torch.cuda.get_device_capability()
```

Any torch CUDA call can fail.

Handle all failures gracefully.

Distinguish between:

- PyTorch not installed;
- CPU-only PyTorch;
- CUDA-enabled PyTorch but CUDA unavailable;
- CUDA working normally.

---

# 12. Compiler collection

Windows:

detect MSVC if reasonably possible.

Linux:

detect:

```text
gcc
g++
clang
clang++
```

Collect versions.

Do not require compilers to exist.

---

# 13. Build tool collection

Detect:

```text
cmake
ninja
```

Collect versions and executable paths.

---

# 14. Environment variable collection

Inspect relevant variables.

Windows:

```text
PATH
CUDA_PATH
CUDA_PATH_V*
```

Linux:

```text
PATH
LD_LIBRARY_PATH
CUDA_HOME
CUDA_PATH
```

Detect:

- duplicate CUDA paths;
- nonexistent CUDA paths;
- multiple CUDA versions;
- older CUDA paths appearing before newer/selected CUDA paths;
- obvious CUDA bin/lib conflicts.

Do not dump the user's full PATH into the normal report.

Only include relevant CUDA/toolchain paths.

---

# 15. Privacy

Reports may be shared publicly or sent to support engineers.

Implement basic redaction.

At minimum:

- replace the user's home directory with `~`;
- avoid exposing usernames unnecessarily;
- avoid exposing hostname by default;
- do not dump arbitrary environment variables;
- only inspect/report variables relevant to CUDA development.

Example:

```text
C:\Users\alice\project
```

should become conceptually:

```text
~\project
```

Provide a reusable redaction utility.

---

# 16. Diagnostic issue model

Create a structured issue model.

Suggested fields:

```python
code
severity
title
description
evidence
recommendations
```

Severity:

```text
INFO
WARNING
ERROR
CRITICAL
```

Use stable issue codes.

---

# 17. Initial diagnostic rules

Implement at least these categories.

## GPU

```text
GPU001
nvidia-smi unavailable

GPU002
No NVIDIA GPU detected

GPU003
nvidia-smi execution failed
```

## Driver

```text
DRV001
Unable to determine NVIDIA driver

DRV002
Driver/runtime compatibility appears problematic
```

## CUDA

```text
CUDA001
nvcc not found

CUDA002
CUDA_HOME/CUDA_PATH not set

CUDA003
CUDA_HOME points to invalid location

CUDA004
Multiple CUDA Toolkit installations detected

CUDA005
Multiple CUDA bin directories found in PATH

CUDA006
nvcc installation differs from selected CUDA_HOME
```

## PyTorch

```text
TORCH001
PyTorch not installed

TORCH002
torch.cuda.is_available() is False

TORCH003
CPU-only PyTorch build detected

TORCH004
PyTorch CUDA runtime differs from locally installed CUDA Toolkit

TORCH005
PyTorch cannot enumerate GPU devices
```

IMPORTANT:

TORCH004 MUST NOT automatically be an ERROR.

Example:

```text
local nvcc: CUDA 12.4
torch.version.cuda: 12.1
```

is often valid because standard PyTorch binaries bundle their CUDA runtime dependencies.

The tool must explain this rather than create a false positive.

This is a core product requirement.

## Environment

Add useful rules for:

- invalid CUDA paths;
- duplicated CUDA paths;
- conflicting CUDA versions in PATH/LD_LIBRARY_PATH;
- selected CUDA_HOME not matching nvcc.

---

# 18. Compatibility logic

Separate compatibility knowledge from collectors.

Create modules/data for:

```text
NVIDIA driver ↔ CUDA runtime/toolkit
CUDA ↔ supported host compiler
PyTorch CUDA runtime interpretation
```

Do not hardcode compatibility logic throughout the codebase.

If exact compatibility information is uncertain, prefer wording such as:

```text
potential compatibility issue
```

rather than making an incorrect absolute statement.

Keep compatibility data easy to update.

---

# 19. CLI

Use Typer unless there is a strong reason not to.

Required commands:

```bash
cuda-doctor
cuda-doctor diagnose
cuda-doctor info
cuda-doctor --version
```

Required options:

```bash
cuda-doctor diagnose --verbose
cuda-doctor diagnose --format terminal
cuda-doctor diagnose --format json
cuda-doctor diagnose --format markdown
cuda-doctor diagnose --output report.md
```

Default:

```bash
cuda-doctor
```

should behave like:

```bash
cuda-doctor diagnose
```

Use Rich for terminal presentation.

---

# 20. Terminal UX

The terminal output should be clean and useful.

Use sections such as:

```text
System
GPU
NVIDIA Driver
CUDA Toolkit
Python
PyTorch
C++ Toolchain
Environment
Compatibility
Potential Issues
Summary
```

Use symbols where supported:

```text
✓ OK
⚠ Warning
✗ Error
```

But maintain readable fallback behavior.

At the end print a concise summary such as:

```text
Environment status: USABLE WITH WARNINGS

0 critical
0 errors
2 warnings
3 informational findings
```

---

# 21. JSON report

JSON output should contain:

- schema/tool version;
- timestamp;
- normalized environment snapshot;
- issues;
- summary.

It must be machine-readable and stable enough for future tooling.

---

# 22. Markdown report

Generate a human-readable Markdown report suitable for:

- GitHub issues;
- customer support;
- sending to another engineer.

Include:

```text
CUDA Doctor Report
System
GPU
Driver
CUDA
Python
PyTorch
Toolchain
Issues
Recommendations
```

Apply redaction before reporting.

---

# 23. Testing

Use pytest.

Testing is mandatory.

Do NOT require CI tests to run on a real NVIDIA GPU.

Use fixture-based parser tests.

Create fixture outputs for:

- working nvidia-smi;
- multiple GPUs;
- failed nvidia-smi;
- nvcc 11.x;
- nvcc 12.x;
- Windows-like paths;
- Linux-like paths.

Test:

- version parsing;
- nvidia-smi parsing;
- nvcc parsing;
- CUDA path detection;
- check rules;
- report generation;
- CLI basics.

Important edge cases:

1. no NVIDIA GPU;
2. no nvidia-smi;
3. no nvcc;
4. no PyTorch;
5. CPU-only PyTorch;
6. multiple CUDA installations;
7. mismatched CUDA_HOME and nvcc;
8. PyTorch CUDA version != local toolkit version;
9. external command timeout;
10. malformed command output.

The application must remain functional in all cases.

---

# 24. Development tooling

Use:

```text
pytest
ruff
mypy
```

Add useful development dependencies.

Configure them in `pyproject.toml`.

Prefer modern Python packaging.

The project should support:

```bash
pip install -e ".[dev]"
```

and eventually:

```bash
pip install cuda-doctor
```

Expose the CLI entry point through pyproject.toml.

---

# 25. README

Write a professional README.

It should include:

- what cuda-doctor is;
- why it exists;
- installation;
- quick start;
- command examples;
- example terminal output;
- JSON/Markdown report usage;
- supported platforms;
- privacy behavior;
- known limitations;
- development instructions;
- roadmap.

Clearly explain:

```text
Local CUDA Toolkit version and PyTorch CUDA runtime version do not always need to match.
```

Avoid misleading users.

---

# 26. v0.1 non-goals

Do NOT implement:

- automatic driver installation;
- automatic CUDA installation;
- automatic CUDA removal;
- automatic PyTorch installation;
- sudo operations;
- registry editing;
- remote server;
- website;
- authentication;
- database;
- telemetry;
- user tracking;
- automatic system modification.

Keep v0.1 local, private, deterministic, and safe.

---

# 27. Engineering quality requirements

Use:

- type hints;
- small focused functions;
- docstrings where useful;
- dependency injection where useful for testing;
- robust subprocess handling;
- structured models;
- deterministic checks.

Avoid:

- god classes;
- global mutable state;
- giant try/except blocks;
- shell=True;
- fragile regex when machine-readable output exists;
- duplicated parsing logic;
- unnecessary dependencies;
- premature abstractions.

---

# 28. Work process

Do not immediately write the whole project blindly.

Work in phases.

## Phase 1 — Architecture

First:

1. inspect the repository;
2. summarize the architecture you plan to implement;
3. identify any assumptions;
4. create the package skeleton.

Do not ask me for confirmation unless absolutely blocked.

Proceed using reasonable engineering judgment.

## Phase 2 — Core model and command runner

Implement:

- core data models;
- enums;
- command execution abstraction;
- redaction utilities;
- version parsing utilities.

Add unit tests.

## Phase 3 — Collectors

Implement collectors for:

- system;
- nvidia-smi;
- GPU;
- CUDA;
- Python;
- PyTorch;
- compiler;
- CMake;
- Ninja;
- environment variables.

Add tests as each collector is implemented.

## Phase 4 — Diagnosis engine

Implement:

- issue model;
- base check interface;
- initial rules;
- diagnosis engine;
- recommendations.

## Phase 5 — Reporters

Implement:

- Rich terminal output;
- JSON output;
- Markdown output.

## Phase 6 — CLI

Implement commands and options.

Ensure:

```bash
python -m cuda_doctor
```

and:

```bash
cuda-doctor
```

both work.

## Phase 7 — Tests

Run the complete test suite.

Fix failures.

Add missing edge-case tests.

## Phase 8 — Documentation

Complete:

- README;
- CHANGELOG;
- sample reports;
- development scripts.

## Phase 9 — Final engineering review

Perform a self-review.

Specifically inspect for:

- platform assumptions;
- subprocess safety;
- privacy leaks;
- false-positive diagnoses;
- missing error handling;
- duplicated logic;
- poor type usage;
- untested code paths.

Fix meaningful issues you find.

---

# 29. Acceptance criteria

The project is not complete until all of the following are true.

Running:

```bash
cuda-doctor
```

works on a normal supported Python environment.

The program does not crash when:

- there is no GPU;
- nvidia-smi is missing;
- nvcc is missing;
- PyTorch is missing;
- CMake is missing;
- Ninja is missing;
- a subprocess returns malformed output;
- a subprocess times out.

The tool supports:

```text
terminal
JSON
Markdown
```

reports.

The tool detects common CUDA/PyTorch environment problems.

The tool correctly understands that:

```text
local CUDA Toolkit != PyTorch CUDA runtime
```

does NOT automatically mean the environment is broken.

Tests pass.

Linting passes.

README is usable.

No system-modifying behavior exists.

---

# 30. Final deliverable

When implementation is complete, provide:

1. a concise architecture summary;
2. the final repository tree;
3. implemented diagnostic rules;
4. commands supported;
5. test results;
6. known limitations;
7. recommended v0.2 roadmap.

Also include exact commands I should run locally:

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src
cuda-doctor
cuda-doctor diagnose --format markdown --output report.md
```

Do not stop after merely generating files.

Run the available tests and validation commands and repair problems before considering the task complete.
