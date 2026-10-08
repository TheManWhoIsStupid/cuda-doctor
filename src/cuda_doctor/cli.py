"""cuda-doctor command-line interface (Typer)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from cuda_doctor.checks import ALL_CHECKS
from cuda_doctor.core.enums import ReportFormat
from cuda_doctor.core.models import EnvironmentSnapshot
from cuda_doctor.core.runner import CollectionRunner
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.reporters import (
    JsonReporter,
    MarkdownReporter,
    Reporter,
    ReportInputs,
    TerminalReporter,
)
from cuda_doctor.utils.commands import CommandRunner
from cuda_doctor.version import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Detect, analyze, explain, and recommend — a read-only CUDA/PyTorch"
    " environment doctor.",
)

_FORMAT_REPORTER = {
    ReportFormat.TERMINAL: TerminalReporter,
    ReportFormat.JSON: JsonReporter,
    ReportFormat.MARKDOWN: MarkdownReporter,
}

console = Console()


def _collect() -> EnvironmentSnapshot:
    """Run every collector against the local machine (isolated per module)."""
    return CollectionRunner(command_runner=CommandRunner()).collect()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_diagnose(fmt: ReportFormat, output: Path | None, verbose: bool) -> None:
    """Collect -> diagnose -> report (shared by `cuda-doctor` and `diagnose`)."""
    try:
        snapshot = _collect()
        result = DiagnosisEngine().run(snapshot)
    except Exception as exc:
        typer.secho(f"Internal error while diagnosing: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    inputs = ReportInputs(snapshot=snapshot, result=result, generated_at=_now())

    if output is not None:
        if fmt is ReportFormat.TERMINAL:
            reporter: Reporter = TerminalReporter(verbose=verbose)
        else:
            reporter = _FORMAT_REPORTER[fmt]()
        Path(output).write_text(reporter.render(inputs), encoding="utf-8")
        typer.secho(
            f"Report written to {output} — status: {result.summary.status.value}",
            fg=typer.colors.GREEN,
        )
    elif fmt is ReportFormat.TERMINAL:
        TerminalReporter(console, verbose=verbose).draw(console, inputs)
    else:
        typer.echo(_FORMAT_REPORTER[fmt]().render(inputs))


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", help="Show the cuda-doctor version and exit."
    ),
) -> None:
    if version:
        typer.echo(f"cuda-doctor {__version__}")
        raise typer.Exit
    if ctx.invoked_subcommand is None:
        _run_diagnose(ReportFormat.TERMINAL, output=None, verbose=False)


@app.command()
def diagnose(
    format: ReportFormat = typer.Option(
        ReportFormat.TERMINAL,
        "--format",
        "-f",
        help="Output format: terminal (default), json, or markdown.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the report to a file instead of stdout."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Include info-level notes and internal diagnostics."
    ),
) -> None:
    """Diagnose the local CUDA/PyTorch environment (read-only)."""
    _run_diagnose(format, output, verbose)


@app.command()
def info() -> None:
    """Show tool metadata: version, platform, checks, and bundled data."""
    typer.echo(f"cuda-doctor {__version__}")
    typer.echo(f"Platform: {sys.platform} | Python {sys.version.split()[0]}")
    typer.echo(f"Diagnostic checks: {len(ALL_CHECKS)}")
    typer.echo("Issue codes: " + ", ".join(sorted({check.code for check in ALL_CHECKS})))
    typer.echo(
        "Bundled compatibility data: cuda_driver_compatibility.json, "
        "cuda_compiler_compatibility.json, known_issues.json"
    )
    typer.echo("Read-only: cuda-doctor never modifies the system.")


def main() -> None:
    """Console-script entry point (`cuda-doctor`)."""
    app()
