"""Actuation reconstruction helpers for inverse-design dashboards."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from actorob.actuators import ActuatorParameters, GearboxParameters, MotorParameters
from actorob.config import load_trajectory_optimizer_config
from actorob.invdes.record import InverseDesignRunRecord
from actorob.mjcf import resolve_mjcf_path
from actorob.trajectories import TrajectoryRunRecord
from actorob.trajectories.actuation_constraints import (
    MechanicalCharacteristicSpec,
    build_mechanical_characteristic_spec,
)
from ..plotly_support import go


@dataclass(frozen=True)
class _ActuationSpecs:
    mechanical: MechanicalCharacteristicSpec | None


def _load_actuation_specs(
    run: InverseDesignRunRecord,
    record: TrajectoryRunRecord,
) -> _ActuationSpecs:
    try:
        config = load_trajectory_optimizer_config(Path(run.config_path), task_names=run.task_names)
    except Exception:
        return _ActuationSpecs(mechanical=None)

    try:
        config = replace(
            config,
            base=replace(config.base, mjcf_path=str(resolve_mjcf_path(record.mjcf_path))),
            actuators=_reconstruct_actuators(run, record.joint_names),
        )
    except Exception:
        return _ActuationSpecs(mechanical=None)

    mechanical = None
    if bool(config.trajectory.enforce_mechanical_characteristic):
        mechanical = build_mechanical_characteristic_spec(config, record.joint_names)

    return _ActuationSpecs(mechanical=mechanical)


def _reconstruct_actuators(
    run: InverseDesignRunRecord,
    joint_names: tuple[str, ...],
) -> dict[str, Any] | None:
    if run.best_params is None or len(run.best_params) != len(run.parameter_names):
        return None

    motor_mass_by_family: dict[str, float] = {}
    gear_ratio_by_family: dict[str, float] = {}
    for name, value in zip(run.parameter_names, run.best_params):
        if name.startswith("m_"):
            motor_mass_by_family[name[2:]] = float(value)
        elif name.startswith("g_"):
            gear_ratio_by_family[name[2:]] = float(value)

    families = set(motor_mass_by_family).intersection(gear_ratio_by_family)
    if len(families) == 0:
        return None

    actuators: dict[str, Any] = {}
    for joint_name in joint_names:
        family = next((candidate for candidate in families if joint_name.endswith(f"{candidate}_joint")), None)
        if family is None:
            return None
        actuators[joint_name] = ActuatorParameters(
            motor=MotorParameters(motor_mass_by_family[family]),
            gearbox=GearboxParameters(gear_ratio_by_family[family]),
            name=family,
        ).actuator_unit
    return actuators


def _mechanical_speed_envelope(
    mechanical_spec: MechanicalCharacteristicSpec | None,
    joint_idx: int,
    torque_grid: np.ndarray,
) -> np.ndarray | None:
    if mechanical_spec is None:
        return None
    no_load = float(mechanical_spec.no_load_velocity[joint_idx])
    slope = float(mechanical_spec.slope[joint_idx])
    return np.maximum(0.0, no_load - slope * torque_grid)


def _add_phase_envelope(
    fig: go.Figure,
    row: int,
    col: int,
    torque_grid: np.ndarray,
    speed_limit: np.ndarray,
    *,
    color: str,
    name: str,
    showlegend: bool,
) -> None:
    for idx, (x_values, y_values) in enumerate(
        (
            (torque_grid, speed_limit),
            (-torque_grid, speed_limit),
            (torque_grid, -speed_limit),
            (-torque_grid, -speed_limit),
        )
    ):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                name=name,
                legendgroup=name,
                showlegend=bool(showlegend and idx == 0),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
