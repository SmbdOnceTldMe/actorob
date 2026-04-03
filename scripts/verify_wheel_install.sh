#!/usr/bin/env bash

set -euo pipefail

if ! ls dist/actorob-*.whl >/dev/null 2>&1; then
  echo "No built wheel found in dist/. Building one now."
  python -m build --wheel
fi

WHEEL_PATH="$(ls -1t dist/actorob-*.whl | head -n 1)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/actorob-wheel-check.XXXXXX")"
VENV_PATH="${TMP_ROOT}/venv"

cleanup() {
  rm -rf "${TMP_ROOT}"
}

trap cleanup EXIT

echo "Using wheel: ${WHEEL_PATH}"
python -m venv --system-site-packages "${VENV_PATH}"

"${VENV_PATH}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir --no-deps --ignore-installed "${WHEEL_PATH}"

"${VENV_PATH}/bin/python" - <<'PY'
from importlib import import_module

root = import_module("actorob")
models = import_module("actorob.models")
utils = import_module("actorob.utils")
invdes = import_module("actorob.invdes")

assert "models" in root.__all__
assert callable(models.expand_config)
assert callable(utils.rpm_to_radsec)
assert "build_trajectory_bundle" in dir(invdes)

print("Wheel installation smoke test passed.")
PY
