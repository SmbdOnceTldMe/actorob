from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import pickle

import numpy as np


@dataclass(frozen=True)
class StairVisualizationData:
    """Geometry metadata used to visualize stairs alongside a trajectory."""

    step_length: float
    step_height: float
    step_count: int
    width: float
    offset_x: float | None
    offset_y: float
    offset_z: float
    flat_length: float | None
    start_step: int | None = None
    total_steps: int | None = None


@dataclass(frozen=True)
class JointTrajectoryData:
    """Per-task trajectory arrays and derived actuator metrics."""

    task_name: str
    converged: bool
    iterations: int
    trajectory_cost: float
    state_time: np.ndarray
    control_time: np.ndarray
    floating_base_coordinates: np.ndarray
    floating_base_velocities: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    electrical_power: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    friction_power: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    electrical_energy: float = 0.0
    friction_energy: float = 0.0
    contact_frame_names: tuple[str, ...] = ()
    contact_phase_labels: tuple[str, ...] = ()
    contact_active: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    contact_forces: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=float))
    foot_target_refs: dict[str, np.ndarray] = field(default_factory=dict)
    stairs: StairVisualizationData | None = None


@dataclass(frozen=True)
class TrajectoryRunRecord:
    """Serializable record of one multi-task trajectory-optimization run."""

    robot: str
    mjcf_path: str
    dt: float
    created_at_utc: str
    contact_mu: float
    joint_names: tuple[str, ...]
    joint_position_lower_limits: np.ndarray
    joint_position_upper_limits: np.ndarray
    joint_velocity_lower_limits: np.ndarray
    joint_velocity_upper_limits: np.ndarray
    joint_torque_lower_limits: np.ndarray
    joint_torque_upper_limits: np.ndarray
    tasks: tuple[JointTrajectoryData, ...]

    def save(self, path: str | Path) -> Path:
        """Serialize the trajectory record to a pickle file."""

        out_path = Path(path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(self, f)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryRunRecord":
        """Load a previously serialized trajectory run record."""

        in_path = Path(path).expanduser().resolve()
        with in_path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, cls):
            raise TypeError(f"Invalid record type in '{in_path}': {type(data)}")
        return data

    @classmethod
    def now(
        cls,
        robot: str,
        mjcf_path: str,
        dt: float,
        contact_mu: float,
        joint_names: tuple[str, ...],
        joint_position_lower_limits: np.ndarray,
        joint_position_upper_limits: np.ndarray,
        joint_velocity_lower_limits: np.ndarray,
        joint_velocity_upper_limits: np.ndarray,
        joint_torque_lower_limits: np.ndarray,
        joint_torque_upper_limits: np.ndarray,
        tasks: tuple[JointTrajectoryData, ...],
    ) -> "TrajectoryRunRecord":
        """Create a trajectory record stamped with the current UTC timestamp."""

        ts = datetime.now(timezone.utc).isoformat()
        return cls(
            robot=robot,
            mjcf_path=mjcf_path,
            dt=dt,
            created_at_utc=ts,
            contact_mu=contact_mu,
            joint_names=joint_names,
            joint_position_lower_limits=joint_position_lower_limits,
            joint_position_upper_limits=joint_position_upper_limits,
            joint_velocity_lower_limits=joint_velocity_lower_limits,
            joint_velocity_upper_limits=joint_velocity_upper_limits,
            joint_torque_lower_limits=joint_torque_lower_limits,
            joint_torque_upper_limits=joint_torque_upper_limits,
            tasks=tasks,
        )
