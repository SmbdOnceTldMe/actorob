"""Public trajectory API with lazy imports for heavyweight backends."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .footplanner import FootMotionPlanner, FootPlannerConfig, PairContactPhase, PairFootPlan, StairProfile
    from .optimizer import AligatorTrajectoryOptimizer, TaskResult
    from .record import JointTrajectoryData, TrajectoryRunRecord

_EXPORTS = {
    "AligatorTrajectoryOptimizer": ("actorob.trajectories.optimizer", "AligatorTrajectoryOptimizer"),
    "TaskResult": ("actorob.trajectories.optimizer", "TaskResult"),
    "FootMotionPlanner": ("actorob.trajectories.footplanner", "FootMotionPlanner"),
    "FootPlannerConfig": ("actorob.trajectories.footplanner", "FootPlannerConfig"),
    "PairContactPhase": ("actorob.trajectories.footplanner", "PairContactPhase"),
    "PairFootPlan": ("actorob.trajectories.footplanner", "PairFootPlan"),
    "StairProfile": ("actorob.trajectories.footplanner", "StairProfile"),
    "JointTrajectoryData": ("actorob.trajectories.record", "JointTrajectoryData"),
    "TrajectoryRunRecord": ("actorob.trajectories.record", "TrajectoryRunRecord"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AligatorTrajectoryOptimizer",
    "TaskResult",
    "FootMotionPlanner",
    "FootPlannerConfig",
    "PairContactPhase",
    "PairFootPlan",
    "StairProfile",
    "JointTrajectoryData",
    "TrajectoryRunRecord",
]
