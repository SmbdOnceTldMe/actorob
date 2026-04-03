from __future__ import annotations

from typing import Any

import numpy as np

from .record import JointTrajectoryData, TrajectoryRunRecord
from .tasks import TaskPlan, is_jump_task, is_stair_task, is_walk_task


class OptimizerInitialGuessBuilder:
    """Reconstruct and blend warm-start trajectories for the solver."""

    def __init__(self, optimizer: Any) -> None:
        self._optimizer = optimizer

    def build_initial_guess(
        self,
        task_name: str,
        task_plan: TaskPlan,
        state_refs: np.ndarray,
        seed_task_data: JointTrajectoryData | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        optimizer = self._optimizer
        xs_init = [optimizer.x0.copy()]
        for x_ref in np.asarray(state_refs, dtype=float):
            xs_init.append(np.asarray(x_ref, dtype=float).copy())

        us_init = [np.zeros(optimizer.nu) for _ in range(len(state_refs))]
        if seed_task_data is None:
            return xs_init, us_init

        xs_seed, us_seed = self.reconstruct_seed_guess(seed_task_data)
        if len(xs_seed) != len(xs_init) or len(us_seed) != len(us_init):
            return xs_init, us_init

        if is_stair_task(task_name, optimizer.config.tasks[task_name]):
            xs_hot = [np.asarray(x, dtype=float).copy() for x in xs_seed]
            us_hot = [np.asarray(u, dtype=float).copy() for u in us_seed]
        elif is_walk_task(task_name):
            xs_hot = [
                0.5 * np.asarray(x0, dtype=float) + 0.5 * np.asarray(x1, dtype=float)
                for x0, x1 in zip(xs_init, xs_seed)
            ]
            us_hot = [
                0.5 * np.asarray(u0, dtype=float) + 0.5 * np.asarray(u1, dtype=float)
                for u0, u1 in zip(us_init, us_seed)
            ]
        elif is_jump_task(task_name):
            xs_hot = [
                0.5 * np.asarray(x0, dtype=float) + 0.5 * np.asarray(x1, dtype=float)
                for x0, x1 in zip(xs_init, xs_seed)
            ]
            us_hot = [np.zeros_like(np.asarray(u, dtype=float)) for u in us_seed]
        else:
            return xs_init, us_init

        xs_hot[0] = optimizer.x0.copy()
        return xs_hot, us_hot

    def reconstruct_seed_guess(self, task_data: JointTrajectoryData) -> tuple[list[np.ndarray], list[np.ndarray]]:
        q = np.hstack([task_data.floating_base_coordinates, task_data.joint_positions])
        v = np.hstack([task_data.floating_base_velocities, task_data.joint_velocities])
        xs = [np.hstack([qk, vk]).astype(float) for qk, vk in zip(q, v)]
        us = [uk.astype(float) for uk in np.asarray(task_data.joint_torques, dtype=float)]
        return xs, us

    def seed_task_data_from_record(
        self,
        task_name: str,
        seed_record: TrajectoryRunRecord | None,
    ) -> JointTrajectoryData | None:
        optimizer = self._optimizer
        if seed_record is None:
            return None
        if seed_record.robot != optimizer.config.base.robot:
            return None
        if not np.isclose(float(seed_record.dt), float(optimizer.config.trajectory.dt)):
            return None
        if tuple(seed_record.joint_names) != tuple(optimizer.rmodel.names[2:]):
            return None
        for task_data in seed_record.tasks:
            if task_data.task_name == task_name:
                return task_data
        return None
