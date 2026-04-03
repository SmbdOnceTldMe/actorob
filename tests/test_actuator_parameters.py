import pytest
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from actorob.actuators.actuator import (
    DEFAULT_OUTPUT_DAMPING_COEFFICIENT,
    DEFAULT_OUTPUT_FRICTION_COEFFICIENT,
    ActuatorParameters,
    ActuatorUnit,
)
from actorob.actuators import GearboxParameters, MotorParameters
from actorob.invdes.design_space import _expand_actuator_config
from actorob.invdes.trajectory_bundle import _extract_nominal_joint_family_params
from actorob.models import ModelFactory


def test_actuator_geometry(real_motor, real_gearbox):
    """Check total actuator geometry calculations."""
    act = ActuatorParameters(real_motor, real_gearbox)

    assert act.mass > real_motor.mass
    assert act.length > real_motor.axial_length
    assert act.diameter > real_motor.stator_diameter


def test_post_init(real_motor, real_gearbox):
    """Check gearbox properties are computed after initialization."""
    act = ActuatorParameters(real_motor, real_gearbox)

    assert act.gearbox.volume > 0
    assert act.gearbox.mass > 0
    assert act.gearbox.length > 0


def test_actuator_dynamics(real_motor, real_gearbox):
    """Check reflected inertia and torque scaling."""
    act = ActuatorParameters(real_motor, real_gearbox)

    assert act.reflected_inertia == pytest.approx(real_motor.rotor_inertia * real_gearbox.gear_ratio**2)
    assert act.peak_torque == pytest.approx(real_motor.peak_torque * real_gearbox.gear_ratio)


def test_actuator_unit_from_parameters(real_motor, real_gearbox):
    """Check ActuatorUnit creation from ActuatorParameters."""

    # Create ActuatorParameters instance
    act_params = ActuatorParameters(motor=real_motor, gearbox=real_gearbox, name="hip_motor")
    # Create ActuatorUnit from ActuatorParameters
    unit = act_params.actuator_unit

    # Check that the unit is an instance of ActuatorUnit
    assert isinstance(unit, ActuatorUnit)

    # Check basic numerical parameters
    assert unit.name == "hip_motor"
    assert unit.gear_ratio == pytest.approx(10.0)
    assert unit.nominal_torque == pytest.approx(act_params.nominal_torque)
    assert unit.max_torque == pytest.approx(act_params.peak_torque)

    # Check correspondence of nested structures
    assert unit.diameter == pytest.approx(act_params.diameter)
    assert unit.mass == pytest.approx(act_params.mass)
    assert unit.rotor_inertia == pytest.approx(real_motor.rotor_inertia)
    assert unit.armature == pytest.approx(act_params.reflected_inertia)


def test_actuator_placeholder_dynamics_are_explicit_and_forwarded(real_motor, real_gearbox):
    act_params = ActuatorParameters(motor=real_motor, gearbox=real_gearbox, name="hip_motor")
    unit = act_params.actuator_unit

    assert act_params.damping_coefficient == pytest.approx(DEFAULT_OUTPUT_DAMPING_COEFFICIENT)
    assert act_params.friction_coefficient == pytest.approx(DEFAULT_OUTPUT_FRICTION_COEFFICIENT)
    assert unit.damping == pytest.approx(DEFAULT_OUTPUT_DAMPING_COEFFICIENT)
    assert unit.friction_loss == pytest.approx(DEFAULT_OUTPUT_FRICTION_COEFFICIENT)


def _nominal_joint_and_actuator_attrs(mjcf_path: Path, family: str) -> tuple[dict[str, float], dict[str, float], float]:
    root = ET.parse(mjcf_path).getroot()
    joint = next(
        element for element in root.iter("joint") if element.attrib.get("name", "").endswith(f"{family}_joint")
    )
    actuator = next(element for element in root.iter("general") if family in element.attrib.get("name", ""))
    actuator_body = next(
        element for element in root.iter("body") if element.attrib.get("name", "").endswith(f"{family}_actuator")
    )
    inertial = actuator_body.find("inertial")
    if inertial is None:
        raise AssertionError(f"Actuator body for family '{family}' is missing inertial data.")

    ctrlrange = tuple(float(value) for value in actuator.attrib["ctrlrange"].split())
    return (
        {
            "armature": float(joint.attrib["armature"]),
            "damping": float(joint.attrib["damping"]),
            "frictionloss": float(joint.attrib["frictionloss"]),
        },
        {
            "max_torque": max(abs(ctrlrange[0]), abs(ctrlrange[1])),
        },
        float(inertial.attrib["mass"]),
    )


def test_nominal_actuator_inference_requires_explicit_gear_ratio_metadata():
    root = Path(__file__).resolve().parents[1]
    mjcf_path = root / "robots" / "dog" / "dog.xml"
    nominal = _extract_nominal_joint_family_params(mjcf_path)
    assert nominal == {}


def test_model_factory_roundtrip_preserves_nominal_joint_armature():
    root = Path(__file__).resolve().parents[1]
    robot_dir = root / "robots" / "dog"
    mjcf_path = robot_dir / "dog.xml"
    nominal = _extract_nominal_joint_family_params(mjcf_path)
    grouped = {
        family: ActuatorParameters(
            motor=MotorParameters(params["motor_mass"]),
            gearbox=GearboxParameters(params["gear_ratio"]),
            name=family,
        ).actuator_unit
        for family, params in nominal.items()
    }

    with TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        tmp_mjcf = tmp_root / "robot.xml"
        shutil.copy2(mjcf_path, tmp_mjcf)
        shutil.copytree(robot_dir / "meshes", tmp_root / "meshes")

        factory = ModelFactory(str(tmp_mjcf), _expand_actuator_config(grouped))
        factory.postfix = "roundtrip"
        factory.save()

        generated_path = tmp_root / f"{factory.name}_{factory.postfix}.xml"
        generated_root = ET.parse(generated_path).getroot()
        source_root = ET.parse(mjcf_path).getroot()

        for family in ("hip_roll", "hip_pitch", "knee_pitch"):
            source_joint = next(
                element
                for element in source_root.iter("joint")
                if element.attrib.get("name", "") == f"front_left_{family}_joint"
            )
            generated_joint = next(
                element
                for element in generated_root.iter("joint")
                if element.attrib.get("name", "") == f"front_left_{family}_joint"
            )
            assert float(generated_joint.attrib["armature"]) == pytest.approx(
                float(source_joint.attrib["armature"]), rel=0.25
            )
