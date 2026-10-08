# Install cuda-doctor in editable mode with all dev tooling.
# Usage: pwsh scripts/dev_install.ps1   (or run in PowerShell)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "==> Installing cuda-doctor (editable) with dev extras"
python -m pip install -e ".[dev]"

Write-Host ""
Write-Host "==> Sanity check"
python -c "from cuda_doctor.version import __version__; print('cuda-doctor', __version__)"
pytest -q

Write-Host ""
Write-Host "Done. Useful commands:"
Write-Host "  cuda-doctor                  # run the diagnosis"
Write-Host "  pytest                       # run the test suite"
Write-Host "  ruff check src tests         # lint"
Write-Host "  mypy                         # type check"
