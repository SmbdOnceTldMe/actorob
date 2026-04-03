#!/usr/bin/env bash
set -e

if ! command -v pixi &> /dev/null; then
    echo "pixi not found. Install it first: https://pixi.sh/latest/"
    exit 1
fi

echo "Installing ACTOROB environments with pixi"
pixi install --all

echo "Installing git hooks"
pixi run -e dev pre-commit install

echo "Environment ready."
echo "Common commands:"
echo "  pixi run pytest -q"
echo "  pixi run python examples/cma_es_invdes.py --help"
echo "  pixi run -e dev ruff check ."
