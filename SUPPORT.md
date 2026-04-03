# Support Policy

## Validation Matrix

ACTOROB is currently maintained and validated with `pixi` in the following release matrix:

- local development path: macOS on Apple Silicon (`osx-arm64`);
- CI validation target: Linux x86_64 (`linux-64`);
- environment manager: `pixi`;
- Python: `3.11` to `3.13`, with validation currently exercised on `3.13`.

The documented local setup path remains `pixi install --all` on `osx-arm64`. Linux support is
validated through the same release commands in CI.

## Support Levels

### Fully Supported

- fresh clones set up with `pixi install --all` on `osx-arm64`;
- commands documented in `README.md`, `docs/REPRODUCIBILITY.md`, and CI.
- CI runs for the documented `linux-64` matrix entry.

### Best-Effort

- exploratory use on other platforms;
- lightweight `pip install -e ".[reporting,progress]"` workflows that do not require the full solver
  stack;
- local modifications that keep repository assumptions intact.

For best-effort scenarios, maintainers may ask for reproduction on the canonical environment before
investigating further.

### Out of Scope for Maintainer Support

- unpublished experimental branches and private local patches;
- environments that intentionally diverge from the documented setup path without a reproduction on
  the canonical one;
- generated outputs committed as if they were source files.

## Before Opening a Support Request

Please gather the following:

- the exact command that failed;
- whether the issue reproduces after `pixi install --all`;
- the platform and Python version you used;
- whether the same problem reproduces on the documented `osx-arm64` setup or the `linux-64` CI path,
  if available.

If the issue is packaging- or release-related, include the output of:

```bash
pixi run -e dev verify-release
```

## Response Expectations

The repository is maintained as a research artifact, so support is provided on a best-effort basis.
Bug reports that include a minimal reproduction and stay close to the documented setup path will be
much easier to act on quickly.
