"""Visualization-oriented helpers for actuator grouping and coloring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from actorob.models.utils import random_rgba


def group_similar_actuators(actuators: Mapping[str, Any]) -> list[list[str]]:
    """Group joints whose actuator definitions compare equal."""

    actuator_groups: list[list[str]] = []
    for joint_name, actuator in actuators.items():
        matched_group = None
        for actuator_group in actuator_groups:
            if actuator == actuators[actuator_group[0]]:
                matched_group = actuator_group
                break
        if matched_group is None:
            actuator_groups.append([joint_name])
        else:
            matched_group.append(joint_name)
    return actuator_groups


def apply_group_colors(
    *,
    actuators: Mapping[str, Any],
    bodies: Mapping[str, Any],
    color_factory: Callable[[], list[float]] = random_rgba,
) -> None:
    """Apply a single color to each group of equivalent actuators."""

    for actuator_group in group_similar_actuators(actuators):
        rgba = np.asarray(color_factory(), dtype=float)
        for joint_name in actuator_group:
            actuator_name = f"{joint_name.replace('_joint', '')}_actuator"
            for geom in bodies[actuator_name].geoms:
                geom.rgba = rgba


__all__ = ["apply_group_colors", "group_similar_actuators"]
