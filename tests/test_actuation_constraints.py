from dataclasses import dataclass

import numpy as np

from actorob.actuators import ActuatorUnit
from actorob.trajectories.actuation_constraints import build_mechanical_characteristic_spec


@dataclass(frozen=True)
class _FakeBaseConfig:
    mjcf_path: str


@dataclass(frozen=True)
class _FakeConfig:
    base: _FakeBaseConfig
    actuators: dict[str, ActuatorUnit]


def _make_actuator(name: str) -> ActuatorUnit:
    return ActuatorUnit(
        name=name,
        gear_ratio=12.0,
        nominal_torque=18.0,
        max_torque=40.0,
        nominal_velocity=120.0,
        max_velocity=160.0,
        torque_constant=1.0,
        motor_constant=0.9,
        rotor_inertia=0.01,
        damping=0.05,
        friction_loss=0.01,
        gearbox_efficiency=0.85,
        resistance=1.0,
        voltage=48.0,
        diameter=0.1,
        length=0.2,
        mass=1.0,
    )


def test_build_specs_from_configured_actuators():
    actuators = {
        "joint_a": _make_actuator("joint_a"),
        "joint_b": _make_actuator("joint_b"),
    }
    config = _FakeConfig(base=_FakeBaseConfig(mjcf_path="unused.xml"), actuators=actuators)

    mech = build_mechanical_characteristic_spec(config, ("joint_a", "joint_b"))
    assert mech is not None
    assert mech.no_load_velocity.shape == (2,)
    expected_no_load = np.array([actuators["joint_a"].max_velocity_sim, actuators["joint_b"].max_velocity_sim])
    expected_slope = expected_no_load / np.array([actuators["joint_a"].max_torque, actuators["joint_b"].max_torque])
    assert np.allclose(mech.no_load_velocity, expected_no_load)
    assert np.allclose(mech.slope, expected_slope)
    assert np.allclose(mech.no_load_velocity - mech.slope * np.array([40.0, 40.0]), 0.0)
