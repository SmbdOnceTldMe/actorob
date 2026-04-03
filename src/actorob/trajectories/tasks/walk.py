from __future__ import annotations

from actorob.config import TaskConfig

from .common import TaskPlan, constant_state_refs, is_walk_task


def matches(task_name: str, task: TaskConfig) -> bool:
    """Return whether the task should use the flat-ground walk planner."""

    del task
    return is_walk_task(task_name)


def build_plan(optimizer, task_name: str, task: TaskConfig) -> TaskPlan:
    """Build the contact and state references for a walking task."""

    schedule, active_frames, phase_schedule, foot_refs = optimizer._build_contact_schedule_from_footplanner(
        task_name, task
    )
    return TaskPlan(
        contact_schedule=schedule,
        active_frames_schedule=active_frames,
        phase_schedule=phase_schedule,
        foot_refs=foot_refs,
        state_refs=constant_state_refs(optimizer, len(schedule)),
    )
