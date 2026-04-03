from __future__ import annotations

from typing import Any

import aligator
import numpy as np

from actorob.config import TaskConfig

from .tasks import is_jump_task, is_stair_task, is_walk_task


class OptimizerCostBuilder:
    """Construct running and terminal costs for optimized tasks."""

    def __init__(self, optimizer: Any) -> None:
        self._optimizer = optimizer

    def state_tracking_weights(self, task_name: str, task: TaskConfig, use_footplanner_refs: bool) -> np.ndarray:
        optimizer = self._optimizer
        if use_footplanner_refs and (is_walk_task(task_name) or is_stair_task(task_name, task)):
            return self.original_walk_state_weights()
        if is_jump_task(task_name):
            return self.original_jump_state_weights()

        w_x = task.state_weight * np.eye(optimizer.space.ndx)
        w_x[0:3, 0:3] *= task.base_position_weight_scale
        w_x[3:6, 3:6] *= task.base_orientation_weight_scale
        vel_offset = optimizer.nv
        w_x[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] *= task.base_linear_velocity_weight_scale
        w_x[vel_offset + 3 : vel_offset + 6, vel_offset + 3 : vel_offset + 6] *= task.base_angular_velocity_weight_scale
        return w_x

    def control_tracking_weights(self, task_name: str, task: TaskConfig) -> np.ndarray:
        optimizer = self._optimizer
        if is_jump_task(task_name):
            return self.stabilized_jump_control_weights()
        return task.control_weight * np.eye(optimizer.nu)

    def terminal_state_weights(
        self,
        task_name: str,
        task: TaskConfig,
        use_footplanner_refs: bool,
        running_state_weights: np.ndarray,
    ) -> np.ndarray:
        if is_jump_task(task_name):
            return 1e4 * running_state_weights
        terminal_weight = (
            1e3
            if (use_footplanner_refs and (is_walk_task(task_name) or is_stair_task(task_name, task)))
            else task.terminal_weight
        )
        return terminal_weight * running_state_weights

    def floating_base_tracking_weight(self, task_name: str, task: TaskConfig) -> float | np.ndarray:
        if is_jump_task(task_name):
            return self.original_jump_base_translation_weights()
        return max(100.0, task.state_weight * 20.0)

    def original_walk_state_weights(self) -> np.ndarray:
        optimizer = self._optimizer
        w_x = np.zeros(optimizer.space.ndx, dtype=float)
        w_x[6 : optimizer.nv] = 10.0
        w_x[optimizer.nv : optimizer.space.ndx] = 1e-3
        return np.diag(w_x)

    @staticmethod
    def original_walking_foot_weights() -> tuple[np.ndarray, np.ndarray]:
        swing_w3 = np.diag(np.array([1e3, 1e3, 1e3], dtype=float))
        support_w3 = 1e2 * swing_w3
        return support_w3, swing_w3

    @staticmethod
    def stabilized_walk_foot_weights() -> tuple[np.ndarray, np.ndarray]:
        support_w3 = 1e2 * np.diag(np.array([1e3, 1e3, 1e3], dtype=float))
        swing_w3 = np.diag(np.array([1e3, 1e3, 6e3], dtype=float))
        return support_w3, swing_w3

    def original_jump_state_weights(self) -> np.ndarray:
        optimizer = self._optimizer
        w_x = np.full(optimizer.space.ndx, 1e-3, dtype=float)
        w_x[0:6] = 0.0
        return np.diag(w_x)

    @staticmethod
    def stabilized_jump_foot_weights() -> np.ndarray:
        return np.diag(np.array([5e3, 5e3, 5e3], dtype=float))

    @staticmethod
    def original_jump_base_translation_weights() -> np.ndarray:
        return np.diag(np.array([1.0, 1.0, 1.0], dtype=float))

    @staticmethod
    def stabilized_jump_centroidal_weights() -> np.ndarray:
        return np.diag(np.array([0.0, 0.0, 0.0, 6e-2, 6e-2, 6e-2], dtype=float))

    def stabilized_jump_control_weights(self) -> np.ndarray:
        return 1e-4 * np.eye(self._optimizer.nu)

    def add_foot_tracking_costs(
        self,
        running_cost: aligator.CostStack,
        task_name: str,
        task: TaskConfig,
        step_idx: int,
        foot_refs: dict[str, np.ndarray],
        active_frame_set: set[str],
    ) -> None:
        optimizer = self._optimizer
        is_plain_walk = is_walk_task(task_name)
        stair_task = is_stair_task(task_name, task)
        jump_task = is_jump_task(task_name)
        if is_plain_walk:
            if len(active_frame_set) == len(optimizer.config.contact.contact_frames_3d):
                return
            support_w3, swing_w3 = self.stabilized_walk_foot_weights()
        elif stair_task:
            if len(active_frame_set) == len(optimizer.config.contact.contact_frames_3d):
                return
            support_w3, swing_w3 = self.original_walking_foot_weights()
        elif jump_task:
            support_w3 = self.stabilized_jump_foot_weights()
            swing_w3 = support_w3
        else:
            support_weight = max(80.0, task.state_weight * 40.0)
            swing_weight = max(20.0, task.state_weight * 10.0)
            support_diag = np.ones(3, dtype=float)
            swing_diag = np.ones(3, dtype=float)

        for frame_name in optimizer.config.contact.contact_frames_3d:
            if frame_name not in foot_refs:
                continue
            frame_id = optimizer.contact_frame_ids.get(frame_name)
            if frame_id is None:
                continue
            target = np.asarray(foot_refs[frame_name][step_idx], dtype=float).reshape(3)
            if is_plain_walk or stair_task:
                w3 = support_w3 if frame_name in active_frame_set else swing_w3
            elif jump_task:
                w3 = support_w3 if frame_name in active_frame_set else swing_w3
            elif frame_name in active_frame_set:
                w3 = float(support_weight) * np.diag(support_diag)
            else:
                w3 = float(swing_weight) * np.diag(swing_diag)
            running_cost.addCost(
                f"foot_track_{frame_name}",
                aligator.QuadraticResidualCost(
                    optimizer.space,
                    aligator.FrameTranslationResidual(
                        optimizer.space.ndx, optimizer.nu, optimizer.rmodel, target, frame_id
                    ),
                    w3,
                ),
            )

    def add_floating_base_tracking_cost(
        self,
        running_cost: aligator.CostStack,
        target: np.ndarray,
        weight: float | np.ndarray,
    ) -> None:
        optimizer = self._optimizer
        frame_name = (
            "floating_base_joint"
            if optimizer.rmodel.existFrame("floating_base_joint")
            else ("body_link" if optimizer.rmodel.existFrame("body_link") else None)
        )
        if frame_name is None:
            return
        frame_id = optimizer.rmodel.getFrameId(frame_name)
        if np.isscalar(weight):
            w3 = float(weight) * np.eye(3)
        else:
            w3 = np.asarray(weight, dtype=float).reshape(3, 3)
        running_cost.addCost(
            "floating_base_track",
            aligator.QuadraticResidualCost(
                optimizer.space,
                aligator.FrameTranslationResidual(
                    optimizer.space.ndx,
                    optimizer.nu,
                    optimizer.rmodel,
                    np.asarray(target, dtype=float).reshape(3),
                    frame_id,
                ),
                w3,
            ),
        )

    def add_centroidal_momentum_cost(
        self,
        running_cost: aligator.CostStack,
        target: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        optimizer = self._optimizer
        running_cost.addCost(
            "centroidal_momentum",
            aligator.QuadraticResidualCost(
                optimizer.space,
                aligator.CentroidalMomentumResidual(
                    optimizer.space.ndx,
                    optimizer.nu,
                    optimizer.rmodel,
                    np.asarray(target, dtype=float).reshape(6),
                ),
                np.asarray(weight, dtype=float).reshape(6, 6),
            ),
        )
