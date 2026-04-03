# Contributing to ACTOROB

Thanks for your interest in improving ACTOROB.

This repository is being prepared as a public research artifact. Contributions are welcome, but the
bar for changes is a little different from a general-purpose application:

- preserve scientific reproducibility whenever possible;
- prefer small, reviewable pull requests;
- document behavior changes together with the code;
- add or update tests for any user-visible change.

## Development Setup

ACTOROB uses `pixi` as the canonical environment manager.

```bash
pixi install --all
```

Useful commands:

```bash
pixi run pytest -q
pixi run -e dev ruff check .
pixi run -e dev python -m build --sdist --wheel
pixi run -e dev check-dist
pixi run -e dev verify-wheel-install
pixi run -e dev verify-release
pixi run -e dev pre-commit run --all-files
pixi run python examples/cma_es_invdes.py --help
```

## Pull Requests

- Explain the motivation and expected effect of the change.
- Mention whether the change affects article figures, experiment outputs, or default configs.
- Include tests for fixes and new functionality.
- Keep generated artifacts, logs, and local experiment outputs out of version control.

## Reproducibility Expectations

If your change affects optimization logic, configs, reports, or examples, please include:

- the entry point you used for validation;
- the config file involved;
- whether the expected output changed intentionally.

If your change affects packaging, onboarding, or CI behavior, please also mention whether
`pixi run -e dev verify-wheel-install` and
`pixi run -e dev verify-release` still passes.

## Code Style

- Run `ruff` before submitting.
- Prefer clear APIs over clever abstractions.
- Keep example scripts runnable from a fresh clone using documented commands.
- Follow [SUPPORT.md](SUPPORT.md) when documenting platform expectations or closing support requests.
