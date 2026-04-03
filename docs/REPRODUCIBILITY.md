# Reproducibility Guide

This document captures the currently validated reproduction path for ACTOROB as a research
software artifact.

## Validated Environment

- Environment manager: `pixi`
- Local validation in this repository pass: `osx-arm64`
- Release CI target matrix: `osx-arm64`, `linux-64`
- Python runtime used during validation: `3.13`

See [SUPPORT.md](../SUPPORT.md) for the current support policy and what remains best-effort outside
that matrix.

## Canonical Setup

```bash
pixi install --all
```

## Validated Commands

### 1. Full Regression Suite

```bash
pixi run pytest -q
```

Expected outcome:

- all tests pass;
- the current validated repository state passes `82` tests;
- the test run completes without repository-known third-party deprecation warning noise.

### 2. Packaging Sanity

```bash
pixi run -e dev python -m build --sdist --wheel
```

Expected outcome:

- the build completes without errors;
- both an `sdist` and a wheel are produced in `dist/`.

### 3. Distribution Metadata Check

```bash
pixi run -e dev check-dist
```

Expected outcome:

- `twine check` reports no metadata/rendering problems for files in `dist/`.

### 4. Wheel Installation Smoke Test

```bash
pixi run -e dev verify-wheel-install
```

Expected outcome:

- a temporary virtual environment is created;
- the freshly built wheel installs successfully into that environment;
- lightweight package imports succeed from the installed artifact.

### 5. Inverse-Design Smoke Test

```bash
pixi run smoke-invdes
```

Expected outcome:

- the task prints an optimization summary;
- `completed_trials=1`;
- a trajectory/inverse-design record is written to the system temp directory;
- a best-candidate MJCF companion file is emitted next to the record when a candidate is available.

At the time of writing, `pixi run smoke-invdes` expands to the same single-iteration `walk`
invocation used in CI and release validation.

### 6. Combined Release Validation

```bash
pixi run -e dev verify-release
```

Expected outcome:

- the command runs lint, tests, package build validation, distribution metadata checks,
  wheel-install validation, and the minimal inverse-design smoke test;
- the command exits successfully without requiring manual cleanup of tracked files.

## Additional Entry Points

These commands are part of the documented surface of the repository and should remain runnable from
a fresh clone:

```bash
pixi run python examples/cma_es_invdes.py --help
pixi run python examples/aligator_dog_three_tasks.py --help
pixi run python examples/build_trajectory_dashboard.py --help
```

## Output Locations

- `outputs/`: inverse-design outputs written by the default example.
- `logs/trajectories/`: trajectory optimizer records and dashboards written by the three-task
  example.

These directories are intentionally ignored by git and should be treated as generated artifacts.

## What Is Still Missing Before Final Public Release

- a paper-oriented mapping from figures/tables to exact commands and configs;
- confirmation that the final GitHub Actions matrix is green on both `osx-arm64` and `linux-64`.
