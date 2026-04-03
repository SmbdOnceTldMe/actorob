from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import aligator
import numpy as np

from actorob.actuators.inference import infer_motor_mass_from_total_actuator_mass, parse_gear_ratio_from_name
from actorob.mjcf import resolve_mjcf_path


class MechanicalCharacteristicResidual(aligator.StageFunction):
    """Residual enforcing the linear motor torque-speed characteristic."""

    class Data:
        """Workspace buffers for ``MechanicalCharacteristicResidual``."""

        def __init__(self, residual_model: "MechanicalCharacteristicResidual") -> None:
            ndx = getattr(residual_model, "ndx", getattr(residual_model, "ndx1"))
            self.value = np.zeros(residual_model.nr)
            self.Jx = np.zeros((residual_model.nr, ndx))
            self.Ju = np.zeros((residual_model.nr, residual_model.nu))

    def __init__(self, pin_model, space, no_load_velocity: np.ndarray, slope: np.ndarray):
        ndx = space.ndx
        nu = pin_model.nv - 6
        nr = nu
        super().__init__(ndx, nu, nr)
        self.pin_model = pin_model
        self.space = space
        self.nq = pin_model.nq
        self.nv = pin_model.nv
        self.no_load_velocity = np.asarray(no_load_velocity, dtype=float).reshape(nu)
        self.slope = np.asarray(slope, dtype=float).reshape(nu)
        self._joint_velocity_jac_offset = self.nv + 6

    def create_data(self):
        """Allocate workspace arrays for residual value and Jacobians."""

        return MechanicalCharacteristicResidual.Data(self)

    def evaluate(self, x, u, data):
        """Evaluate torque-speed margin violations for the current stage."""

        velocity = np.asarray(x[self.nq + 6 :], dtype=float).reshape(self.nu)
        torque = np.asarray(u, dtype=float).reshape(self.nu)
        data.value[:] = np.abs(velocity) - (self.no_load_velocity - self.slope * np.abs(torque))

    def computeJacobians(self, x, u, data):
        """Compute Jacobians of the torque-speed characteristic residual."""

        velocity = np.asarray(x[self.nq + 6 :], dtype=float).reshape(self.nu)
        torque = np.asarray(u, dtype=float).reshape(self.nu)
        data.Jx.fill(0.0)
        data.Ju.fill(0.0)
        data.Jx[:, self._joint_velocity_jac_offset :] = np.diag(np.sign(velocity))
        data.Ju[:, :] = np.diag(self.slope * np.sign(torque))

    def __deepcopy__(self, memo):
        return MechanicalCharacteristicResidual(
            self.pin_model,
            self.space,
            self.no_load_velocity.copy(),
            self.slope.copy(),
        )


@dataclass(frozen=True)
class MechanicalCharacteristicSpec:
    """Per-joint parameters of the linear torque-speed envelope."""

    no_load_velocity: np.ndarray
    slope: np.ndarray


def build_mechanical_characteristic_spec(
    config: Any, joint_names: tuple[str, ...]
) -> MechanicalCharacteristicSpec | None:
    """Build per-joint torque-speed limits from configured actuators or MJCF data."""

    actuators = _joint_actuator_units_from_config(config, joint_names)
    if actuators is None:
        return None

    no_load_velocity = np.array([float(actuators[name].max_velocity_sim) for name in joint_names], dtype=float)
    max_torque = np.array([float(actuators[name].max_torque) for name in joint_names], dtype=float)
    slope = np.divide(
        no_load_velocity,
        max_torque,
        out=np.zeros_like(no_load_velocity),
        where=np.abs(max_torque) > 1.0e-12,
    )
    return MechanicalCharacteristicSpec(no_load_velocity=no_load_velocity, slope=slope)


def _joint_actuator_units_from_config(config: Any, joint_names: tuple[str, ...]):
    configured = getattr(config, "actuators", None)
    if configured is not None:
        if all(name in configured for name in joint_names):
            return configured

    return _infer_joint_actuators_from_mjcf(config.base.mjcf_path, joint_names)


def _infer_joint_actuators_from_mjcf(mjcf_path: str | Path, joint_names: tuple[str, ...]):
    from actorob.actuators import ActuatorParameters, GearboxParameters, MotorParameters

    root = ET.parse(resolve_mjcf_path(mjcf_path)).getroot()
    actuator_names_by_joint: dict[str, str] = {}
    for actuator in root.iter("general"):
        joint_name = actuator.attrib.get("joint")
        if joint_name is None:
            continue
        actuator_names_by_joint[joint_name] = actuator.attrib.get("name", "")

    body_mass_by_joint: dict[str, float] = {}
    for joint_name in joint_names:
        actuator_body_name = joint_name.replace("_joint", "_actuator")
        body = next((body for body in root.iter("body") if body.attrib.get("name") == actuator_body_name), None)
        if body is None:
            return None
        inertial = body.find("inertial")
        if inertial is None or "mass" not in inertial.attrib:
            return None
        body_mass_by_joint[joint_name] = float(inertial.attrib["mass"])

    actuator_units = {}
    for joint_name in joint_names:
        actuator_name = actuator_names_by_joint.get(joint_name, "")
        gear_ratio = parse_gear_ratio_from_name(actuator_name)
        if gear_ratio is None:
            return None
        motor_mass = infer_motor_mass_from_total_actuator_mass(body_mass_by_joint[joint_name], gear_ratio)
        actuator_units[joint_name] = ActuatorParameters(
            motor=MotorParameters(motor_mass),
            gearbox=GearboxParameters(gear_ratio),
            name=joint_name,
        ).actuator_unit

    return actuator_units
