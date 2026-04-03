from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from actorob.config import TaskConfig


@dataclass(frozen=True)
class TaskPlan:
    """Precomputed contact and reference schedules for one task."""

    contact_schedule: list[list[Any]]
    active_frames_schedule: list[tuple[str, ...]]
    phase_schedule: list[str]
    foot_refs: dict[str, np.ndarray] | None = None
    state_refs: np.ndarray | None = None
    floating_base_refs: np.ndarray | None = None


def is_walk_task(task_name: str) -> bool:
    """Return whether a task name denotes flat-ground walking."""

    lname = task_name.lower()
    return ("walk" in lname) and ("upstairs" not in lname) and ("stair" not in lname)


def is_stair_task(task_name: str, task: TaskConfig) -> bool:
    """Return whether a task should be treated as a stair-climbing task."""

    lname = task_name.lower()
    if ("upstairs" in lname) or ("stair" in lname):
        return True
    traj = task.trajectory_params
    return bool(
        traj is not None
        and traj.stair_h is not None
        and traj.stair_h > 0.0
        and traj.stair_start is not None
        and traj.stair_end is not None
        and traj.stair_end > traj.stair_start
    )


def is_jump_task(task_name: str) -> bool:
    """Return whether a task name denotes a jumping task."""

    return "jump" in task_name.lower()


def uses_footplanner(optimizer, task_name: str, task: TaskConfig) -> bool:
    """Return whether a task can be expanded with the pair-foot planner."""

    traj = task.trajectory_params
    if (
        traj is None
        or traj.n_steps is None
        or traj.step_time is None
        or traj.ds_time is None
        or traj.swing_apex is None
    ):
        return False
    required_keys = {"front_left", "front_right", "rear_left", "rear_right"}
    if not required_keys.issubset(set(optimizer.contact_role_map)):
        return False
    if is_jump_task(task_name):
        return False
    return is_walk_task(task_name) or is_stair_task(task_name, task)


def constant_state_refs(optimizer, horizon: int) -> np.ndarray:
    """Repeat the optimizer initial state across the requested horizon."""

    return np.repeat(np.asarray(optimizer.x0, dtype=float)[None, :], horizon, axis=0)
