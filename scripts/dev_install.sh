#!/usr/bin/env bash
# Install cuda-doctor in editable mode with all dev tooling.
# Usage: bash scripts/dev_install.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Installing cuda-doctor (editable) with dev extras"
python -m pip install -e ".[dev]"

echo
echo "==> Sanity check"
python -c "from cuda_doctor.version import __version__; print('cuda-doctor', __version__)"
pytest -q

echo
echo "Done. Useful commands:"
echo "  cuda-doctor                  # run the diagnosis"
echo "  pytest                       # run the test suite"
echo "  ruff check src tests         # lint"
echo "  mypy                         # type check"
