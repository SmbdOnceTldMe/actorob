# ACTOROB

[![CI](https://github.com/SmbdOnceTldMe/actorob/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SmbdOnceTldMe/actorob/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FLRA.2026.3674006-blue)](https://doi.org/10.1109/LRA.2026.3674006)
[![Python](https://img.shields.io/badge/python-3.11--3.13-blue)](https://www.python.org/)
[![Pixi](https://img.shields.io/badge/env-pixi-0A9396)](https://pixi.sh/latest/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A**ctuators **C**o-design and **T**ask-aware **O**ptimization for **Rob**ots

ACTOROB is a research codebase for trajectory optimization and actuator co-design in robot examples
used to illustrate materials of a scientific article. The repository currently focuses on:

- trajectory optimization for configured task sets;
- actuator parameter modeling and evaluation;
- inverse-design loops built on top of Optuna + CMA-ES;
- report and dashboard generation for experiment outputs.

Project page: [Task-Aware Actuator Parameter Allocation for Multibody Robots](https://mkakanov.github.io/actorob/)
Paper DOI: [10.1109/LRA.2026.3674006](https://doi.org/10.1109/LRA.2026.3674006)

## Release Roadmap

ACTOROB is currently in a pre-release stage. The current repository snapshot already covers the
main pipeline on quadruped examples, while the broader release roadmap is:

- [x] Main research pipeline available today on the quadruped examples included in this repository.
- [ ] Full public release with humanoid robot examples and documentation.
- [ ] Unified cross-robot workflow with an additional examples.

## Current Support Status

The canonical local setup is `pixi` on `osx-arm64` (Apple Silicon macOS). The release CI matrix
also targets `linux-64` so the same documented commands are exercised on both platforms before
release.

The support policy and escalation path for unsupported environments are documented in
[SUPPORT.md](SUPPORT.md).

## Dependency Profiles

- Full trajectory-optimization and inverse-design workflows: use `pixi`. This is the supported path
  for the solver stack and research examples.
- Lightweight/reporting workflows via `pip`: the package exposes optional extras for reporting and
  terminal progress helpers, for example:

```bash
pip install -e ".[reporting,progress]"
```

This partial `pip` flow is useful for lighter API access and report generation, but it is not the
primary supported installation path for the full research stack.

## Quickstart

1. Install [`pixi`](https://pixi.sh/latest/).
2. Clone this repository.
3. Install the project environments:

```bash
pixi install --all
```

4. Run the test suite:

```bash
pixi run pytest -q
```

5. Run the quick inverse-design smoke test:

```bash
pixi run smoke-invdes
```

This smoke task runs a single-iteration `walk` example and writes its record into the system temp
directory so the repository stays clean.

For a fuller inverse-design run, start from:

```bash
pixi run python examples/cma_es_invdes.py --config configs/dog_aligator_minimal.toml --tasks walk upstairs jump_forward
```

That longer run writes artifacts into `outputs/` by default and can also emit an HTML dashboard.

## Main Entry Points

`actorob.invdes` contains the inverse-design workflow used by the repository examples:

- `ParallelAskTellOptimizer` orchestrates batched `ask/tell` optimization.
- `OptunaCmaEsStudyFactory` provides the Optuna + CMA-ES backend.
- `build_trajectory_bundle(...)` wires inverse design to the trajectory optimizer and
  actuator model.

Useful commands during development:

```bash
pixi run pytest -q
pixi run python examples/cma_es_invdes.py --help
pixi run -e dev pre-commit-all
pixi run -e dev python -m build --sdist --wheel
pixi run -e dev check-dist
pixi run -e dev verify-wheel-install
pixi run -e dev verify-release
pixi run -e dev install-git-hooks
```

To enable the git hook locally so `pre-commit` runs automatically before each commit:

```bash
pixi run -e dev install-git-hooks
```

## Repository Layout

- `src/actorob/`: library code.
- `configs/`: example experiment configurations.
- `examples/`: runnable entry points for demonstrations and smoke tests.
- `robots/`: robot models and meshes used by the examples.
- `tests/`: regression and API tests.

## Open Source Notes

- The repository is being prepared for public release as a research artifact.
- `CITATION.cff` provides a software citation entry for the repository.
- `CHANGELOG.md` tracks release-facing changes starting from the public open-source preparation.
- `SECURITY.md` describes the current vulnerability disclosure path.
- `SUPPORT.md` documents supported environments and what maintainers expect in support requests.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) documents the currently validated commands and expected
  outputs.
- [docs/ASSET_PROVENANCE.md](docs/ASSET_PROVENANCE.md) documents the bundled public asset families and the
  generated files that should stay out of version control.

## Acknowledgements

ACTOROB builds on a strong open robotics software stack. In particular, we would like to
acknowledge:

- [Aligator](https://github.com/Simple-Robotics/aligator) for constrained trajectory optimization
  components used by the motion-planning stack.
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for multibody kinematics and dynamics
  primitives used throughout the model and optimization pipeline.
- [Meshcat](https://github.com/meshcat-dev/meshcat-python) for interactive visualization used in
  dashboards and report outputs.

## Citation

If you use the scientific results behind this repository, please cite the article below. The
repository-level software citation is also provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@ARTICLE{11433790,
  author={Nasonov, Kirill and Kakanov, Mikhail and Skvortsova, Valeria and Zaliaev, Eduard and Borisov, Ivan},
  journal={IEEE Robotics and Automation Letters}, 
  title={Task-Aware Actuator Parameter Allocation for Multibody Robots}, 
  year={2026},
  volume={11},
  number={5},
  pages={5869-5874},
  keywords={Actuators;Robots;Motors;Torque;Legged locomotion;Optimization;Friction;Costs;Topology;Humanoid robots;Actuators;humanoid robots;legged locomotion;motion planning},
  doi={10.1109/LRA.2026.3674006}}
```
