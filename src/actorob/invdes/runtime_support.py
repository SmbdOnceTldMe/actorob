"""Internal runtime helpers shared by inverse-design execution paths."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from actorob.trajectories import TrajectoryRunRecord


def to_matrix(values: Any) -> list[list[float]]:
    """Convert solver output arrays or lists into nested Python float lists."""

    if hasattr(values, "tolist"):
        values = values.tolist()
    return [list(map(float, row)) for row in values]


def load_trajectory_optimizer():
    """Load the default trajectory optimizer lazily to avoid heavy eager imports."""

    from actorob.trajectories import AligatorTrajectoryOptimizer

    return AligatorTrajectoryOptimizer


def load_compute_actuator_metrics():
    """Load actuator metric evaluation lazily for optional scientific backends."""

    from actorob.actuators import compute_actuator_metrics

    return compute_actuator_metrics


def load_compute_actuator_group_metrics():
    """Load aggregate actuator metric evaluation lazily for optional scientific backends."""

    from actorob.actuators import compute_actuator_group_metrics

    return compute_actuator_group_metrics


def solve_all_with_optional_seed(
    optimizer: Any,
    task_names: Sequence[str],
    seed_record: TrajectoryRunRecord | None,
) -> list[Any]:
    """Call ``solve_all`` and pass ``seed_record`` only when supported."""

    solve_all = getattr(optimizer, "solve_all")
    if seed_record is not None:
        try:
            signature = inspect.signature(solve_all)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            params = signature.parameters
            if "seed_record" in params or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()
            ):
                return list(solve_all(task_names=list(task_names), seed_record=seed_record))
    return list(solve_all(task_names=list(task_names)))
