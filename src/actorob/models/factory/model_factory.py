"""High-level MuJoCo model assembly around parameterized actuators."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco

from actorob.actuators import ActuatorPosition, ActuatorUnit
from actorob.models.factory.body_ops import (
    create_actuator_body,
    update_actuator_body,
    update_joint_by_actuator,
)
from actorob.models.factory.coloring import apply_group_colors


class ModelFactory:
    """Factory for constructing and modifying MuJoCo models with parameterized actuators."""

    def __init__(
        self,
        xml_path: str,
        actuators: dict[str, ActuatorUnit],
        actuators_position: Optional[dict[str, ActuatorPosition]] = None,
    ):
        self.xml_path = xml_path
        self.xml_dir = Path(self.xml_path).parent
        self.xml_name = Path(self.xml_path).name
        self.actuators = actuators
        self.actuators_position = actuators_position
        self._load_model()
        self._load_actuators()

    def _load_model(self) -> None:
        self.spec = mujoco.MjSpec.from_file(self.xml_path)
        self.name = self.spec.modelname
        self.postfix = "actorob"
        self.spec.modelname = self.name
        self.postfixed_name = self.name
        self._update_sim_elements()
        for key in self.spec.keys:
            self.spec.delete(key)
        for actuator in self.spec.actuators:
            self.spec.delete(actuator)

    def _update_sim_elements(self) -> None:
        self.bodies = {body.name: body for body in self.spec.bodies}
        self.joints = {joint.name: joint for joint in self.spec.joints}

    def upd_joint_by_actuator(self, joint_name, actuator_unit) -> None:
        """Update joint parameters based on an actuator definition."""

        update_joint_by_actuator(self.joints, joint_name, actuator_unit)

    def _load_actuators(self) -> None:
        for joint_name, actuator_unit in self.actuators.items():
            if joint_name not in self.joints:
                raise ValueError(f"Joint {joint_name} out of model {self.xml_name}")

            self.upd_joint_by_actuator(joint_name, actuator_unit)

            actuator_name = f"{joint_name.replace('_joint', '')}_actuator"
            if actuator_name in self.bodies:
                actuator_body = self.bodies[actuator_name]
                update_actuator_body(self.spec, actuator_body, actuator_unit)
            else:
                if self.actuators_position is None:
                    raise ValueError("Add actuators positions to create additional bodies for actuators")
                if joint_name not in self.actuators_position:
                    raise ValueError(f"Add actuator position to control joint {joint_name}")
                actuator_body = create_actuator_body(
                    bodies=self.bodies,
                    joints=self.joints,
                    joint_name=joint_name,
                    actuator_name=actuator_name,
                    unit=actuator_unit,
                    position=self.actuators_position[joint_name],
                )
            self.spec.add_actuator(
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                name=actuator_name,
                target=joint_name,
                forcerange=[-actuator_unit.max_torque, actuator_unit.max_torque],
                ctrlrange=[-actuator_unit.max_torque, actuator_unit.max_torque],
                ctrllimited=1,
            )

        self.spec.compile()
        self._update_sim_elements()
        self.actuators_sim = {actuator.target: actuator for actuator in self.spec.actuators}

    def build(self):
        """Compile the MuJoCo model and create runtime data structures."""

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        return self.model, self.data

    def save(self) -> None:
        """Build and save the modified model to disk."""

        self.build()
        xml_content = self.spec.to_xml()
        xml_path = self.xml_dir / f"{self.name}_{self.postfix}.xml"
        with open(xml_path, "w") as file_obj:
            file_obj.write(xml_content)

    def add_scene(self) -> None:
        """Attach a predefined scene to the current model."""

        scene_path = Path(__file__).resolve().parents[1] / "assets" / "scene.xml"
        scene_spec = mujoco.MjSpec.from_file(str(scene_path))
        robot_frame = self.spec.worldbody.add_frame()
        robot_frame.attach_body(scene_spec.worldbody, "scene", "")

    def upd_actuator(
        self,
        actuator: ActuatorUnit,
        joint_name: str,
        postfix: Optional[str] = None,
    ) -> None:
        """Update actuator parameters for an existing joint."""

        if postfix is not None:
            self.postfix = postfix
        if joint_name not in self.actuators:
            raise ValueError(f"There is no actuator in joint {joint_name}")

        self.actuators[joint_name] = actuator
        actuator_name = joint_name.replace("joint", "actuator")
        body = self.bodies[actuator_name]
        update_actuator_body(self.spec, body, actuator)
        actuator_sim = self.actuators_sim[joint_name]
        actuator_sim.forcerange = [-actuator.max_torque, actuator.max_torque]
        actuator_sim.ctrlrange = [-actuator.max_torque, actuator.max_torque]
        self.upd_joint_by_actuator(joint_name, actuator)

    def set_actuators(self, config: dict[str, ActuatorUnit], postfix: str = "custom") -> None:
        """Set a full actuator configuration for the robot."""

        self.postfix = postfix
        for joint_name, actuator in config.items():
            self.upd_actuator(actuator, joint_name=joint_name)
        self.spec.compile()

    def colorize_similar_actuators(self) -> None:
        """Assign consistent colors to geometries of similar actuators."""

        apply_group_colors(actuators=self.actuators, bodies=self.bodies)


__all__ = ["ModelFactory"]
