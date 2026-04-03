from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from actorob.invdes.design_space import ActuatorDesignVariable, ActuatorPreparer


@dataclass(frozen=True)
class FakeBaseConfig:
    mjcf_path: str


@dataclass(frozen=True)
class FakeTrajectoryConfig:
    dt: float


@dataclass(frozen=True)
class FakeOptimizerConfig:
    base: FakeBaseConfig
    trajectory: FakeTrajectoryConfig
    tasks: dict[str, object]


class FakeModelFactory:
    def __init__(self, xml_path: str, actuators):
        self.xml_path = xml_path
        self.actuators = actuators
        self.xml_dir = Path(xml_path).parent
        self.name = Path(xml_path).stem
        self.postfix = "actorob"

    def save(self) -> None:
        target = self.xml_dir / f"{self.name}_{self.postfix}.xml"
        target.write_text("<mujoco/>", encoding="utf-8")


class ActuatorPreparerTest(unittest.TestCase):
    def test_prepare_creates_internal_model_and_updates_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "dog.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")
            config = FakeOptimizerConfig(
                base=FakeBaseConfig(mjcf_path=str(xml_path)),
                trajectory=FakeTrajectoryConfig(dt=0.02),
                tasks={"walk": object()},
            )
            preparer = ActuatorPreparer(
                config=config,
                design_variables=(
                    ActuatorDesignVariable(
                        joint_family="hip_pitch",
                        motor_mass_bounds=(0.5, 2.0),
                        gear_ratio_bounds=(5.0, 20.0),
                        initial_motor_mass=1.0,
                        initial_gear_ratio=10.0,
                    ),
                ),
                model_factory_cls=FakeModelFactory,
                actuator_builder=lambda joint_family, mass, gear_ratio: {
                    "joint_family": joint_family,
                    "mass": mass,
                    "gear_ratio": gear_ratio,
                },
                expand_config_fn=lambda grouped: {
                    f"front_left_{joint}_joint": actuator for joint, actuator in grouped.items()
                }
                | {f"front_right_{joint}_joint": actuator for joint, actuator in grouped.items()}
                | {f"rear_left_{joint}_joint": actuator for joint, actuator in grouped.items()}
                | {f"rear_right_{joint}_joint": actuator for joint, actuator in grouped.items()},
            )

            prepared = preparer.prepare((1.2, 11.0))

            self.assertEqual(prepared.solution, (1.2, 11.0))
            self.assertNotEqual(prepared.config.base.mjcf_path, str(xml_path))
            self.assertTrue(Path(prepared.config.base.mjcf_path).exists())
            actuators = prepared.metadata["actuators"]
            self.assertEqual(
                set(actuators),
                {
                    "front_left_hip_pitch_joint",
                    "front_right_hip_pitch_joint",
                    "rear_left_hip_pitch_joint",
                    "rear_right_hip_pitch_joint",
                },
            )
