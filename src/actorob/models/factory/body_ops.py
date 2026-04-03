"""Low-level actuator body updates for :class:`ModelFactory`."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import mujoco
import numpy as np

from actorob.utils import get_fullinertia, quat2mat


def update_joint_by_actuator(joints: dict[str, Any], joint_name: str, actuator_unit: Any) -> None:
    """Update joint limits and dissipative terms from one actuator definition."""

    actuator_joint = joints[joint_name]
    actuator_joint.actfrcrange = [
        -actuator_unit.max_torque,
        actuator_unit.max_torque,
    ]
    actuator_joint.damping = actuator_unit.damping
    actuator_joint.frictionloss = actuator_unit.friction_loss
    actuator_joint.armature = actuator_unit.armature


def get_axis_idx(body: Any) -> int:
    """Return the principal axis index best suited for actuator alignment."""

    diagonal = np.diag(get_fullinertia(body))
    pairs = list(combinations(range(len(diagonal)), 2))
    diffs = {(i, j): abs(diagonal[i] - diagonal[j]) for i, j in pairs}
    closest_pair = min(diffs, key=diffs.get)
    all_indices = set(range(len(diagonal)))
    return list(all_indices - set(closest_pair))[0]


def get_cylinder_body_properties(actuator: Any, idx: int) -> tuple[list[float], float, list[float], list[float]]:
    """Compute cylinder geometry, mass, inertia and orientation for an actuator body."""

    quats = [[0.7071, 0, 0.7071, 0], [0.7071, 0.7071, 0, 0], [1, 0, 0, 0]]
    iquat = quats[idx]
    mass = actuator.mass
    radius = actuator.diameter / 2
    length = actuator.length
    size = [radius, length / 2, 0]
    inertia = [
        mass * (3 * radius**2 + length**2) / 12,
        mass * (3 * radius**2 + length**2) / 12,
        mass * radius**2 / 2,
    ]
    return size, mass, inertia, iquat


def check_inertia_tensor(inertia: np.ndarray) -> bool:
    """Check whether an inertia tensor is positive definite."""

    eigvals, _ = np.linalg.eig(inertia)
    return np.min(eigvals) > 0


def create_actuator_body(
    *,
    bodies: dict[str, Any],
    joints: dict[str, Any],
    joint_name: str,
    actuator_name: str,
    unit: Any,
    position: Any,
) -> Any:
    """Create an actuator body and rebalance the parent body inertia."""

    joint = joints[joint_name]
    if position.body_name is None:
        parent_body = joint.parent.parent
    else:
        if position.body_name not in bodies:
            raise ValueError(
                f"parent_body_name {position.body_name} for actuator in joint {joint_name} out of bodies in model"
            )
        parent_body = bodies[position.body_name]

    quat = joint.parent.quat if position.quat is None else position.quat

    size, mass, inertia, iquat = get_cylinder_body_properties(unit, int(np.argmax(np.abs(joint.axis))))
    actuator_body = parent_body.add_body(
        name=actuator_name,
        pos=position.pos,
        quat=quat,
        ipos=unit.ipos or [0, 0, 0],
        mass=mass,
        iquat=iquat,
        inertia=inertia,
        explicitinertial=1,
    )
    actuator_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=size,
        contype=0,
        conaffinity=0,
        quat=iquat,
    )

    updated_parent_mass = parent_body.mass - actuator_body.mass
    if updated_parent_mass < 0:
        raise ValueError(
            f"Invalid mass update for body '{parent_body.name}': "
            f"resulting mass is negative after adding actuator '{actuator_body.name}'. "
            f"The actuator mass exceeds the parent body mass."
        )

    rotation = quat2mat(actuator_body.quat)
    actuator_ipos = actuator_body.pos + rotation @ actuator_body.ipos
    updated_parent_ipos = (
        parent_body.mass * parent_body.ipos - actuator_body.mass * actuator_ipos
    ) / updated_parent_mass

    parent_inertia = get_fullinertia(parent_body)
    actuator_inertia = rotation @ get_fullinertia(actuator_body) @ rotation.T
    actuator_offset = parent_body.ipos - actuator_ipos
    identity = np.eye(3)
    actuator_inertia_parent = actuator_inertia + actuator_body.mass * (
        np.dot(actuator_offset, actuator_offset) * identity - np.outer(actuator_offset, actuator_offset)
    )

    distributed_parent_inertia = parent_inertia - actuator_inertia_parent
    parent_offset = parent_body.ipos - updated_parent_ipos
    updated_parent_inertia = distributed_parent_inertia - updated_parent_mass * (
        np.dot(parent_offset, parent_offset) * identity - np.outer(parent_offset, parent_offset)
    )

    if not check_inertia_tensor(updated_parent_inertia):
        raise ValueError(
            f"Invalid inertia tensor after adding actuator '{actuator_body.name}' to body '{parent_body.name}'. "
            f"The resulting inertia is not positive definite. This is likely caused by an actuator whose "
            f"current size and mass cannot be physically accommodated inside the parent body."
        )

    parent_body.mass = updated_parent_mass
    parent_body.ipos = updated_parent_ipos
    parent_body.iquat = [1, 0, 0, 0]
    parent_body.inertia = [0, 0, 0]
    parent_body.fullinertia = [
        updated_parent_inertia[0, 0],
        updated_parent_inertia[1, 1],
        updated_parent_inertia[2, 2],
        updated_parent_inertia[0, 1],
        updated_parent_inertia[0, 2],
        updated_parent_inertia[1, 2],
    ]
    return actuator_body


def update_actuator_body(spec: Any, body: Any, actuator: Any) -> None:
    """Refresh an existing actuator body geometry and inertial parameters."""

    size, mass, inertia, quat = get_cylinder_body_properties(actuator, get_axis_idx(body))

    for geom in body.geoms:
        spec.delete(geom)

    pos = body.ipos if actuator.ipos is None else actuator.ipos
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=pos,
        size=size,
        quat=quat,
        contype=0,
        conaffinity=0,
    )
    body.mass = mass
    body.inertia = inertia
    body.iquat = quat
    body.ipos = pos


__all__ = [
    "check_inertia_tensor",
    "create_actuator_body",
    "get_axis_idx",
    "get_cylinder_body_properties",
    "update_actuator_body",
    "update_joint_by_actuator",
]
