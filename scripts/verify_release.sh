#!/usr/bin/env bash

set -euo pipefail

echo "[1/6] Running lint checks"
ruff check .

echo "[2/6] Running regression suite"
pytest -q

echo "[3/6] Building source and wheel distributions"
python -m build --sdist --wheel

echo "[4/6] Checking built distribution metadata"
python -m twine check dist/*

echo "[5/6] Verifying wheel installation"
bash scripts/verify_wheel_install.sh

echo "[6/6] Running inverse-design smoke test"
SMOKE_OUTPUT="${TMPDIR:-/tmp}/actorob_release_smoke.pkl"
python examples/cma_es_invdes.py \
  --config configs/dog_aligator_minimal.toml \
  --tasks walk \
  --max-iterations 1 \
  --population 1 \
  --workers 1 \
  --skip-dashboard \
  --no-progress \
  --run-output "${SMOKE_OUTPUT}"

echo "Release validation completed successfully."
