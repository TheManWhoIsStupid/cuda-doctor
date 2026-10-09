"""Rich terminal report (the default `cuda-doctor` output)."""

from __future__ import annotations

import io

from rich.console import Console

from cuda_doctor.core.enums import EnvironmentStatus, Severity
from cuda_doctor.core.models import EnvironmentSnapshot
from cuda_doctor.diagnosis.engine import DiagnosisResult
from cuda_doctor.reporters.base import Reporter, ReportInputs, format_memory
from cuda_doctor.utils.redact import redact_text
from cuda_doctor.utils.versions import CudaVersion, parse_cuda_version
from cuda_doctor.version import __version__

_STATUS_STYLE = {
    EnvironmentStatus.HEALTHY: "green",
    EnvironmentStatus.USABLE_WITH_WARNINGS: "yellow",
    EnvironmentStatus.DEGRADED: "red",
    EnvironmentStatus.BROKEN: "bold red",
}

_SEVERITY_MARK = {
    Severity.CRITICAL: "err",
    Severity.ERROR: "err",
    Severity.WARNING: "warn",
    Severity.INFO: "info",
}


class TerminalReporter(Reporter):
    """Human-oriented report with ✓/⚠/✗ symbols (ASCII fallback off-UTF-8).

    With ``verbose=False`` info-level notes are summarized away and internal
    probe failures are hidden (they remain in the JSON report regardless).
    """

    def __init__(self, console: Console | None = None, verbose: bool = True) -> None:
        self._console = console
        self._verbose = verbose

    def render(self, inputs: ReportInputs) -> str:
        console = self._console
        if console is None:
            # Capture into a buffer: render() must return text, not print it.
            console = Console(
                record=True, width=100, file=io.StringIO(), force_terminal=False,
                highlight=False,
            )
            self.draw(console, inputs)
            return console.export_text(clear=False)
        self.draw(console, inputs)
        return console.export_text(clear=False) if console.record else ""

    # -- layout ------------------------------------------------------------

    def draw(self, console: Console, inputs: ReportInputs) -> None:
        marks = _marks(console)
        snapshot, result = inputs.snapshot, inputs.result

        console.print()
        console.print(f"[bold]CUDA Doctor[/bold] [dim]v{__version__}[/dim]", justify="left")
        console.print(
            f"[dim]generated {inputs.generated_at}"
            f" · target {snapshot.system.platform.value}[/dim]"
        )
        console.print()

        self._system(console, snapshot)
        self._gpu(console, snapshot, marks)
        self._driver(console, snapshot)
        self._cuda(console, snapshot)
        self._python(console, snapshot)
        self._pytorch(console, snapshot, marks)
        self._toolchain(console, snapshot)
        self._environment(console, snapshot)
        self._compatibility(console, snapshot, marks)
        self._issues(console, result, marks)
        self._internal_errors(console, snapshot, result, marks)
        self._summary(console, result, marks)

    def _section(self, console: Console, title: str) -> None:
        console.print(f"[bold cyan]{title}[/bold cyan]")
        console.print()

    @staticmethod
    def _kv(console: Console, label: str, value: str | None, mark: str = "") -> None:
        text = value if value not in (None, "") else "[dim]not detected[/dim]"
        prefix = f"{mark} " if mark else ""
        console.print(f"  {prefix}[dim]{label:<15}[/dim] {text}")

    @staticmethod
    def _version_relation_mark(
        runtime: CudaVersion, driver_max: CudaVersion, marks: dict[str, str]
    ) -> str:
        """Mark for a runtime-vs-driver-UMD version comparison.

        A generation gap (runtime from a newer CUDA major than the driver's)
        gets the error mark; a same-family minor gap is informational only
        (CUDA minor-version compatibility); everything else is OK.
        """
        if driver_max.major < runtime.major:
            return marks["err"]
        if runtime.major == driver_max.major and runtime > driver_max:
            return marks["info"]
        return marks["ok"]

    def _system(self, console: Console, snapshot: EnvironmentSnapshot) -> None:
        self._section(console, "System")
        system = snapshot.system
        self._kv(
            console,
            "OS",
            f"{system.os_name} {system.os_version} (kernel {system.kernel_version})",
        )
        self._kv(console, "Architecture", system.architecture)
        self._kv(console, "Python", system.python_version)
        console.print()

    def _gpu(self, console: Console, snapshot: EnvironmentSnapshot, marks: dict[str, str]) -> None:
        self._section(console, "GPU")
        if not snapshot.gpus:
            self._kv(console, "Devices", "none detected", marks["err"])
        for gpu in snapshot.gpus:
            memory = format_memory(gpu.memory_total_mb)
            compute = f"compute {gpu.compute_capability}" if gpu.compute_capability else ""
            detail = "  ".join(part for part in (memory, compute) if part)
            self._kv(console, f"GPU {gpu.index}", f"{gpu.name}  {detail}".rstrip())
        console.print()

    def _driver(self, console: Console, snapshot: EnvironmentSnapshot) -> None:
        self._section(console, "NVIDIA Driver")
        driver = snapshot.driver
        if driver is None or not driver.version:
            self._kv(console, "Version", "not detected")
        else:
            self._kv(console, "Version", redact_text(driver.version))
        if driver and driver.cuda_version:
            self._kv(console, "Reported CUDA", driver.cuda_version)
        console.print()

    def _cuda(self, console: Console, snapshot: EnvironmentSnapshot) -> None:
        self._section(console, "CUDA Toolkit")
        cuda = snapshot.cuda
        if cuda.nvcc_found and cuda.nvcc_path:
            version = f" (CUDA {cuda.toolkit_version})" if cuda.toolkit_version else ""
            self._kv(console, "nvcc", f"{redact_text(cuda.nvcc_path)}{version}")
        else:
            self._kv(console, "nvcc", "not found on PATH")
        home = cuda.cuda_home or cuda.cuda_path
        self._kv(console, "CUDA_HOME", redact_text(home) if home else None)
        if cuda.installations:
            listing = ", ".join(
                f"CUDA {install.version or '?'}" for install in cuda.installations
            )
            self._kv(console, "Installations", f"{len(cuda.installations)}: {listing}")
        console.print()

    def _python(self, console: Console, snapshot: EnvironmentSnapshot) -> None:
        self._section(console, "Python")
        python = snapshot.python
        if python is None:
            self._kv(console, "Interpreter", None)
        else:
            env_note = " [virtual environment]" if python.in_virtual_env else ""
            self._kv(
                console, "Interpreter", f"{redact_text(python.executable)}{env_note}"
            )
        console.print()

    def _pytorch(
        self, console: Console, snapshot: EnvironmentSnapshot, marks: dict[str, str]
    ) -> None:
        self._section(console, "PyTorch")
        torch_info = snapshot.pytorch
        if not torch_info.installed:
            self._kv(console, "torch", "not installed")
        elif torch_info.import_error:
            self._kv(
                console,
                "torch",
                f"installed but cannot be imported {marks['err']}",
            )
            self._kv(console, "Import error", redact_text(torch_info.import_error))
        else:
            if torch_info.is_cuda_build:
                build = f"CUDA {torch_info.cuda_version} build"
            else:
                build = "CPU-only build"
            cudnn = f", cuDNN {torch_info.cudnn_version}" if torch_info.cudnn_version else ""
            self._kv(console, "torch", f"{torch_info.version} ({build}{cudnn})")
            if torch_info.cuda_available is True:
                self._kv(
                    console,
                    "CUDA",
                    f"available, {torch_info.device_count} device(s)",
                    marks["ok"],
                )
            elif torch_info.cuda_available is False:
                self._kv(console, "CUDA", "NOT available", marks["err"])
        console.print()

    def _toolchain(self, console: Console, snapshot: EnvironmentSnapshot) -> None:
        self._section(console, "C++ Toolchain")
        compilers = [tool for tool in snapshot.compiler.compilers if tool.found]
        if compilers:
            for tool in compilers:
                version = f" {tool.version}" if tool.version else ""
                path = redact_text(tool.path) if tool.path else tool.name
                self._kv(console, tool.name, f"{path}{version}")
        else:
            self._kv(console, "Compiler", "none detected")
        for label, info in (("cmake", snapshot.cmake), ("ninja", snapshot.ninja)):
            if info is None:
                continue
            version = f" {info.version}" if info.version else ""
            path = redact_text(info.path) if info.path else info.name
            self._kv(console, label, f"{path}{version}")
        console.print()

    def _environment(self, console: Console, snapshot: EnvironmentSnapshot) -> None:
        self._section(console, "Environment")
        env = snapshot.environment
        for name in sorted(env.variables):
            self._kv(console, name, redact_text(env.variables[name]))
        for index, entry in env.cuda_path_entries:
            self._kv(console, f"PATH[{index}]", redact_text(entry))
        for entry in env.cuda_ld_library_entries:
            self._kv(console, "LD_LIBRARY", redact_text(entry))
        if not (env.variables or env.cuda_path_entries or env.cuda_ld_library_entries):
            self._kv(console, "CUDA-related", "nothing set")
        console.print()

    def _compatibility(
        self, console: Console, snapshot: EnvironmentSnapshot, marks: dict[str, str]
    ) -> None:
        self._section(console, "Compatibility")
        driver = snapshot.driver
        driver_max = parse_cuda_version(driver.cuda_version) if driver else None
        toolkit = parse_cuda_version(snapshot.cuda.toolkit_version)
        torch_cuda = parse_cuda_version(snapshot.pytorch.cuda_version)

        if driver_max is None:
            self._kv(console, "Driver CUDA", "unknown")
        if toolkit is not None and driver_max is not None:
            self._kv(
                console,
                "Toolkit vs driver",
                f"toolkit {toolkit} vs driver CUDA {driver_max}",
                self._version_relation_mark(toolkit, driver_max, marks),
            )
        if torch_cuda is not None and driver_max is not None:
            # Observed runtime success outranks the static comparison.
            if snapshot.pytorch.cuda_available is True:
                mark = marks["ok"]
            else:
                mark = self._version_relation_mark(torch_cuda, driver_max, marks)
            self._kv(
                console,
                "torch vs driver",
                f"torch runtime {torch_cuda} vs driver CUDA {driver_max}",
                mark,
            )
        if torch_cuda is not None and toolkit is not None and torch_cuda != toolkit:
            self._kv(
                console,
                "torch vs toolkit",
                f"torch runtime {torch_cuda} ≠ local toolkit {toolkit} "
                "(normal for pip wheels)",
            )
        console.print()

    def _issues(
        self, console: Console, result: DiagnosisResult, marks: dict[str, str]
    ) -> None:
        self._section(console, "Potential Issues")
        shown = [
            issue
            for issue in result.issues
            if self._verbose or issue.severity is not Severity.INFO
        ]
        hidden = len(result.issues) - len(shown)
        if not shown:
            if hidden:
                self._kv(console, "None", "no warnings or errors found", marks["ok"])
                self._kv(
                    console,
                    "Hidden",
                    f"{hidden} info-level notes (use --verbose to show)",
                )
            else:
                self._kv(console, "None", "no issues found", marks["ok"])
            console.print()
            return
        for issue in shown:
            mark = marks[_SEVERITY_MARK[issue.severity]]
            console.print(f"  {mark} [bold]{issue.code}[/bold]  {issue.title}")
            if issue.description:
                console.print(f"      [dim]{issue.description}[/dim]")
            for line in issue.evidence:
                console.print(f"      [dim]· {redact_text(line)}[/dim]")
            for line in issue.recommendations:
                console.print(f"      → {redact_text(line)}")
            console.print()
        if hidden:
            self._kv(
                console, "Hidden", f"{hidden} info-level notes (use --verbose to show)"
            )
            console.print()

    def _internal_errors(
        self,
        console: Console,
        snapshot: EnvironmentSnapshot,
        result: DiagnosisResult,
        marks: dict[str, str],
    ) -> None:
        errors = {**snapshot.collection_errors, **result.check_errors}
        if not errors or not self._verbose:
            return
        self._section(console, "Internal Diagnostics")
        self._kv(
            console, "Note", "some probes failed; results may be incomplete", marks["warn"]
        )
        for source, message in sorted(errors.items()):
            self._kv(console, source, redact_text(message))
        console.print()

    def _summary(
        self, console: Console, result: DiagnosisResult, marks: dict[str, str]
    ) -> None:
        summary = result.summary
        style = _STATUS_STYLE[summary.status]
        console.print("[bold]Summary[/bold]")
        console.print()
        console.print(f"  Status: [{style}]{summary.status.value}[/{style}]", end="")
        parts = (
            (summary.critical, "critical"),
            (summary.errors, "errors"),
            (summary.warnings, "warnings"),
            (summary.info, "info"),
        )
        active = [f"{count} {label}" for count, label in parts if count]
        console.print(f"  ({', '.join(active)})" if active else "")
        console.print()


def _marks(console: Console) -> dict[str, str]:
    """Status symbols, falling back to ASCII when the console isn't UTF-8."""
    encoding = (console.encoding or "utf-8").lower()
    if "utf" in encoding:
        return {
            "ok": "[green]✓[/green]",
            "warn": "[yellow]⚠[/yellow]",
            "err": "[red]✗[/red]",
            "info": "[blue]i[/blue]",
        }
    return {
        "ok": "[green]OK[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "err": "[red]ERR[/red]",
        "info": "[blue]INFO[/blue]",
    }
