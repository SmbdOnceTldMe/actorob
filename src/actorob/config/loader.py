from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import tomllib
from typing import Any

from actorob.mjcf import resolve_mjcf_path

from .resolved import (
    BaseConfig,
    ContactConfig,
    SolverConfig,
    TrajectoryConfig,
    TrajectoryOptimizerConfig,
)
from .schema import (
    parse_trajectory_optimizer_file,
)
from .task_inference import build_task_config


def load_trajectory_optimizer_config(
    path: str | Path,
    task_names: Sequence[str] | None = None,
) -> TrajectoryOptimizerConfig:
    """Load, validate, and normalize a trajectory-optimizer TOML config."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    raw = _select_requested_tasks(raw, task_names)
    parsed = parse_trajectory_optimizer_file(raw)
    mjcf_abs = resolve_mjcf_path(config_path.parent / parsed.base.mjcf_path)

    parsed_tasks = {
        task_name: build_task_config(task_name, task, parsed.trajectory.dt) for task_name, task in parsed.tasks.items()
    }

    return TrajectoryOptimizerConfig(
        base=BaseConfig(
            robot=parsed.base.robot,
            mjcf_path=str(mjcf_abs),
            init_pose=parsed.base.init_pose,
            align_feet_to_ground=parsed.base.align_feet_to_ground,
            ground_z=parsed.base.ground_z,
        ),
        trajectory=TrajectoryConfig(
            dt=parsed.trajectory.dt,
            use_control_bounds=parsed.trajectory.use_control_bounds,
            use_kinematic_constraints=parsed.trajectory.use_kinematic_constraints,
            use_friction_cones=parsed.trajectory.use_friction_cones,
            enforce_mechanical_characteristic=parsed.trajectory.enforce_mechanical_characteristic,
        ),
        solver=SolverConfig(**parsed.solver.model_dump()),
        contact=ContactConfig(**parsed.contact.model_dump()),
        tasks=parsed_tasks,
        config_path=config_path,
    )


def _select_requested_tasks(
    raw: dict[str, Any],
    task_names: Sequence[str] | None,
) -> dict[str, Any]:
    if task_names is None:
        return raw

    selected_names = tuple(dict.fromkeys(task_names))
    if not selected_names:
        raise ValueError("At least one task name must be requested.")

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, dict):
        return raw

    missing = [name for name in selected_names if name not in raw_tasks]
    if missing:
        raise ValueError(f"Unknown task names: {missing}. Available: {sorted(raw_tasks)}")

    filtered = dict(raw)
    filtered["tasks"] = {name: raw_tasks[name] for name in selected_names}
    return filtered
