from __future__ import annotations

from typing import Any

import numpy as np


def random_rgba(*, rng: Any | None = None) -> list[float]:
    """Generate an RGBA color without mutating NumPy's global RNG state.

    Returns:
        list[float]: A 4-element list representing color components in the
        range [0.0, 1.0], formatted as ``[R, G, B, A]`` where ``A`` is fixed
        to 1.0 (fully opaque).
    """
    random_source = np.random.default_rng() if rng is None else rng
    if hasattr(random_source, "random"):
        rgb = np.asarray(random_source.random(3), dtype=float)
    elif hasattr(random_source, "rand"):
        rgb = np.asarray(random_source.rand(3), dtype=float)
    else:
        raise TypeError("rng must expose either random(size) or rand(size).")
    rgba = np.append(rgb, 1.0)
    return rgba.tolist()


def expand_config(
    config_to_add: dict[str, Any],
    config: dict[str, Any] | None = None,
    mirror: bool = True,
    front_rear: bool = False,
) -> dict[str, Any]:
    """Expand actuator configuration across symmetric joints.

    Adds actuator definitions for left/right or single-side joints to the
    configuration dictionary.

    Args:
        config_to_add (dict[str, ActuatorUnit]):
            Base configuration mapping (e.g., one leg or one arm).
        config (dict[str, ActuatorUnit], optional):
            Existing configuration to extend. Defaults to an empty dict.
        mirror (bool, optional):
            If True, creates both 'left_' and 'right_' joint entries.
            If False, adds entries without prefix. Defaults to True.
        front_rear (bool, optional):
            If True, creates both 'front_' and 'rear_' joint entries.
            If False, adds entries without prefix. Defaults to False.
            Method is userful if you want to create config for robot dog.

    Returns:
        dict[str, ActuatorUnit]: Updated configuration dictionary.
    """
    expanded = {} if config is None else config
    for joint_name, actuator in config_to_add.items():
        if mirror:
            if front_rear:
                expanded[f"front_left_{joint_name}_joint"] = actuator
                expanded[f"front_right_{joint_name}_joint"] = actuator
                expanded[f"rear_left_{joint_name}_joint"] = actuator
                expanded[f"rear_right_{joint_name}_joint"] = actuator
            else:
                expanded[f"left_{joint_name}_joint"] = actuator
                expanded[f"right_{joint_name}_joint"] = actuator
        else:
            if front_rear:
                expanded[f"front_{joint_name}_joint"] = actuator
                expanded[f"rear_{joint_name}_joint"] = actuator
            else:
                expanded[f"{joint_name}_joint"] = actuator

    return expanded
