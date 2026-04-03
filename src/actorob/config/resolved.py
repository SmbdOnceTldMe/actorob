from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaseConfig:
    """Resolved robot-level settings shared by all trajectory tasks."""

    robot: str
    mjcf_path: str
    init_pose: tuple[float, ...] | None = None
    align_feet_to_ground: bool = True
    ground_z: float = 0.0


@dataclass(frozen=True)
class TrajectoryConfig:
    """Global trajectory discretization and constraint toggles."""

    dt: float
    use_control_bounds: bool = True
    use_kinematic_constraints: bool = True
    use_friction_cones: bool = True
    enforce_mechanical_characteristic: bool = True


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings for the Aligator trajectory solver."""

    tol: float
    mu_init: float
    max_iter: int
    verbose: str
    rollout_type: str = "ROLLOUT_LINEAR"
    sa_strategy: str = "SA_FILTER"
    linear_solver_choice: str = "LQ_SOLVER_PARALLEL"
    filter_beta: float = 1e-5
    force_initial_condition: bool = True
    reg_min: float = 1e-6
    num_threads: int = 1


@dataclass(frozen=True)
class ContactConfig:
    """Contact model settings shared across all optimized tasks."""

    contact_frames_3d: tuple[str, ...]
    contact_mu: float
    prox_accuracy: float
    prox_mu: float
    prox_max_iter: int


@dataclass(frozen=True)
class TrajectoryParams:
    """High-level gait parameters used to infer task schedules and targets."""

    n_steps: int | None = None
    step_time: float | None = None
    ds_time: float | None = None
    dx: float | None = None
    dy: float | None = None
    swing_apex: float | None = None
    stair_start: int | None = None
    stair_end: int | None = None
    stair_h: float | None = None
    t_st: int | None = None
    t_ft: int | None = None
    jump_h: float | None = None
    jump_l: float | None = None


@dataclass(frozen=True)
class TaskConfig:
    """Resolved per-task objective, contact, and reference settings."""

    horizon: int
    target_dx: float = 0.0
    target_dy: float = 0.0
    target_dz: float = 0.0
    apex_dz: float = 0.0
    touchdown_dx: float = 0.0
    touchdown_dy: float = 0.0
    touchdown_dz: float = 0.0
    state_weight: float = 1.0
    control_weight: float = 1e-3
    terminal_weight: float = 10.0
    base_position_weight_scale: float = 40.0
    base_orientation_weight_scale: float = 5.0
    base_linear_velocity_weight_scale: float = 10.0
    base_angular_velocity_weight_scale: float = 2.0
    joint_targets: dict[str, float] = field(default_factory=dict)
    contact_phases: tuple[str, ...] | None = None
    phase_durations: tuple[int, ...] | None = None
    stairs: "StairConfig | None" = None
    trajectory_params: TrajectoryParams | None = None


@dataclass(frozen=True)
class StairConfig:
    """Geometric description of a staircase used by a task."""

    step_length: float
    step_height: float
    step_count: int
    width: float = 0.5
    offset_x: float | None = None
    offset_y: float = 0.0
    offset_z: float = 0.0
    flat_length: float | None = None


@dataclass(frozen=True)
class TrajectoryOptimizerConfig:
    """Top-level runtime configuration for the trajectory optimizer."""

    base: BaseConfig
    trajectory: TrajectoryConfig
    solver: SolverConfig
    contact: ContactConfig
    tasks: dict[str, TaskConfig]
    config_path: Path
    actuators: dict[str, Any] | None = None


__all__ = [
    "BaseConfig",
    "ContactConfig",
    "SolverConfig",
    "StairConfig",
    "TaskConfig",
    "TrajectoryConfig",
    "TrajectoryOptimizerConfig",
    "TrajectoryParams",
]
