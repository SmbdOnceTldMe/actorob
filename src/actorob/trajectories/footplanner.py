from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class PairContactPhase(StrEnum):
    """Contact phase labels for a pair of diagonal support feet."""

    DOUBLE = "double"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class StairProfile:
    """Stair geometry used by the foot planner to raise footholds."""

    height: float = 0.0
    start_step: int = 0
    end_step: int = 0
    offset_x: float | None = None
    step_length: float | None = None
    step_count: int | None = None
    offset_z: float = 0.0

    def __post_init__(self) -> None:
        if self.height < 0:
            raise ValueError(f"height must be >= 0, got {self.height}.")
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}.")
        if self.end_step < self.start_step:
            raise ValueError(
                f"end_step must be >= start_step, got start_step={self.start_step}, end_step={self.end_step}."
            )
        if self.step_length is not None and self.step_length <= 0:
            raise ValueError(f"step_length must be positive, got {self.step_length}.")
        if self.step_count is not None and self.step_count < 0:
            raise ValueError(f"step_count must be >= 0, got {self.step_count}.")


@dataclass(frozen=True)
class FootPlannerConfig:
    """Runtime parameters for pair-foot gait and stair planning."""

    dt: float
    step_time: float
    ds_time: float
    swing_apex: float
    initial_left: np.ndarray
    initial_right: np.ndarray
    stair_profile: StairProfile = field(default_factory=StairProfile)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}.")
        if self.step_time <= 0:
            raise ValueError(f"step_time must be positive, got {self.step_time}.")
        if self.ds_time < 0:
            raise ValueError(f"ds_time must be >= 0, got {self.ds_time}.")
        if self.ds_time >= self.step_time:
            raise ValueError(
                f"ds_time must be smaller than step_time, got ds_time={self.ds_time}, step_time={self.step_time}."
            )
        if self.swing_apex < 0:
            raise ValueError(f"swing_apex must be >= 0, got {self.swing_apex}.")

        initial_left = np.asarray(self.initial_left, dtype=float).reshape(3)
        initial_right = np.asarray(self.initial_right, dtype=float).reshape(3)
        object.__setattr__(self, "initial_left", initial_left)
        object.__setattr__(self, "initial_right", initial_right)


@dataclass(frozen=True)
class PairFootPlan:
    """Discrete contact phases and Cartesian trajectories for both feet."""

    contact_phases: tuple[PairContactPhase, ...]
    left_traj: np.ndarray
    right_traj: np.ndarray


class FootMotionPlanner:
    """Pure pair-foot motion planner for walk and stair foothold generation."""

    def __init__(self, config: FootPlannerConfig) -> None:
        self.config = config
        self.dt = config.dt
        self.step_time = config.step_time
        self.ds_time = config.ds_time
        self.ss_time = max(self.step_time - self.ds_time, self.dt)
        self.swing_apex = config.swing_apex
        self.initial_left = config.initial_left
        self.initial_right = config.initial_right
        self.stair_profile = config.stair_profile
        self.half_width = 0.5 * abs(float(self.initial_left[1] - self.initial_right[1]))
        self.ss_steps = max(1, int(round(self.ss_time / self.dt)))
        self.ds_steps = max(1, int(round(max(self.ds_time, self.dt) / self.dt)))
        self.init_ds_steps = 3 * self.ds_steps

    def generate_plan(self, v_des: np.ndarray, n_steps: int, start_left: bool) -> PairFootPlan:
        """Generate footstep and swing trajectories for a walking sequence."""

        desired_velocity = np.asarray(v_des, dtype=float).reshape(3)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}.")

        left_stance, right_stance = self._generate_stance_points(
            desired_velocity=desired_velocity,
            n_steps=n_steps,
            start_left=start_left,
        )
        contact_phases = self._generate_contact_phases(n_steps=n_steps, start_left=start_left)
        left_traj, right_traj = self._interpolate_foot_trajectories(contact_phases, left_stance, right_stance)
        return PairFootPlan(contact_phases=contact_phases, left_traj=left_traj, right_traj=right_traj)

    def _generate_stance_points(
        self,
        desired_velocity: np.ndarray,
        n_steps: int,
        start_left: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        step_vec = np.array(
            [float(desired_velocity[0]) * self.step_time, float(desired_velocity[1]) * self.step_time],
            dtype=float,
        )
        if self.half_width > 0 and step_vec[1] >= 2.0 * self.half_width:
            step_vec[1] = 1.5 * self.half_width

        left_stance = [self.initial_left.copy()]
        right_stance = [self.initial_right.copy()]
        mid_stance = [0.5 * (self.initial_left + self.initial_right)]
        yaw = 0.0
        dyaw = float(desired_velocity[2]) * self.step_time
        new_x = float(mid_stance[-1][0])
        new_y = float(mid_stance[-1][1])
        new_z = float(mid_stance[-1][2])

        for step_idx in range(n_steps + 1):
            if step_idx < n_steps:
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                step_x = step_vec[0] * cos_yaw - step_vec[1] * sin_yaw
                step_y = step_vec[0] * sin_yaw + step_vec[1] * cos_yaw
                new_x = float(mid_stance[-1][0] + step_x)
                new_y = float(mid_stance[-1][1] + step_y)
                new_x, new_z = self._snap_to_stair_if_needed(new_x, float(mid_stance[-1][2]), step_idx)
                yaw = float(yaw + dyaw)
            mid_stance.append(np.array([new_x, new_y, new_z], dtype=float))

            is_left_moving = (step_idx % 2 == 0) == start_left
            sign = -1.0 if is_left_moving else 1.0
            offset_x = sign * self.half_width * np.sin(yaw)
            offset_y = -sign * self.half_width * np.cos(yaw)
            foot_xyz = np.array([new_x + offset_x, new_y + offset_y, new_z], dtype=float)
            if is_left_moving:
                left_stance.append(foot_xyz)
            else:
                right_stance.append(foot_xyz)
        return left_stance, right_stance

    def _snap_to_stair_if_needed(self, x: float, base_z: float, step_idx: int) -> tuple[float, float]:
        stair = self.stair_profile
        if (
            stair.height <= 0.0
            or stair.offset_x is None
            or stair.step_length is None
            or stair.step_count is None
            or stair.step_count <= 0
        ):
            if stair.height > 0.0 and stair.start_step <= step_idx < stair.end_step:
                return float(x), float(base_z + stair.height)
            return float(x), float(base_z)

        x_rel = float(x - stair.offset_x)
        if x_rel < 0.0:
            return float(x), float(base_z)

        stair_idx = int(np.floor(x_rel / stair.step_length)) + 1
        stair_idx = max(0, min(stair_idx, stair.step_count))
        if stair_idx <= 0:
            return float(x), float(base_z)

        snapped_x = stair.offset_x + (stair_idx - 0.5) * stair.step_length
        snapped_z = stair.offset_z + stair_idx * stair.height
        return float(snapped_x), float(snapped_z)

    def _generate_contact_phases(self, n_steps: int, start_left: bool) -> tuple[PairContactPhase, ...]:
        contact_phases: list[PairContactPhase] = [PairContactPhase.DOUBLE] * self.init_ds_steps
        swing_left = start_left
        for phase_idx in range(2 * n_steps + 1):
            if phase_idx % 2 == 0:
                stance_foot = PairContactPhase.RIGHT if swing_left else PairContactPhase.LEFT
                contact_phases.extend([stance_foot] * self.ss_steps)
                swing_left = not swing_left
            else:
                contact_phases.extend([PairContactPhase.DOUBLE] * self.ds_steps)
        contact_phases.extend([PairContactPhase.DOUBLE] * self.init_ds_steps)
        return tuple(contact_phases)

    def _interpolate_foot_trajectories(
        self,
        contact_phases: tuple[PairContactPhase, ...],
        left_stance: list[np.ndarray],
        right_stance: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        ts = 0
        left_idx, right_idx = 0, 0
        left_traj: list[np.ndarray] = []
        right_traj: list[np.ndarray] = []
        prev_cp = PairContactPhase.DOUBLE
        for contact_phase in contact_phases:
            ts += 1
            if contact_phase == PairContactPhase.DOUBLE:
                ts = 0
                left_idx += int(prev_cp == PairContactPhase.RIGHT)
                right_idx += int(prev_cp == PairContactPhase.LEFT)
                left_traj.append(left_stance[left_idx].copy())
                right_traj.append(right_stance[right_idx].copy())
            elif contact_phase == PairContactPhase.RIGHT:
                left_traj.append(self._swing_interp(left_stance[left_idx], left_stance[left_idx + 1], ts))
                right_traj.append(right_stance[right_idx].copy())
            elif contact_phase == PairContactPhase.LEFT:
                right_traj.append(self._swing_interp(right_stance[right_idx], right_stance[right_idx + 1], ts))
                left_traj.append(left_stance[left_idx].copy())
            prev_cp = contact_phase
        return np.asarray(left_traj, dtype=float), np.asarray(right_traj, dtype=float)

    def _swing_interp(self, p0: np.ndarray, p1: np.ndarray, ts: int) -> np.ndarray:
        alpha = min(max(float(ts) / float(self.ss_steps), 0.0), 1.0)
        cos_phase = np.cos(np.pi * alpha)
        planar = 0.5 * ((cos_phase + 1.0) * p0[:2] + (1.0 - cos_phase) * p1[:2])

        z0 = float(p0[2])
        z1 = float(p1[2])
        swing_h = self.swing_apex + max(z0, z1) - 0.5 * (z0 + z1)
        z = 0.5 * (1.0 - cos_phase) * (z1 - z0) + z0 + swing_h * np.sin(np.pi * alpha)
        return np.array([planar[0], planar[1], z], dtype=float)
