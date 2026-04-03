#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/actorob-readme-check.XXXXXX")"
WORKTREE_COPY="${TMP_ROOT}/repo"

cleanup() {
  rm -rf "${TMP_ROOT}"
}

trap cleanup EXIT

echo "Creating fresh workspace copy in ${WORKTREE_COPY}"
mkdir -p "${WORKTREE_COPY}"

rsync -a \
  --exclude ".git" \
  --exclude ".pixi" \
  --exclude ".pytest_cache" \
  --exclude ".ruff_cache" \
  --exclude "__pycache__" \
  --exclude "*.egg-info" \
  --exclude "build" \
  --exclude "dist" \
  --exclude "outputs" \
  --exclude "logs" \
  "${REPO_ROOT}/" "${WORKTREE_COPY}/"

cd "${WORKTREE_COPY}"

echo "[1/3] Installing documented environments"
pixi install --all

echo "[2/3] Running documented regression suite"
pixi run pytest -q

echo "[3/3] Running documented inverse-design smoke test"
pixi run smoke-invdes

echo "README quickstart validation completed successfully."
