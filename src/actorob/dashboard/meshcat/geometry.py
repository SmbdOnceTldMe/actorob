"""Geometry helpers shared by meshcat dashboard rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StairRenderSpec:
    step_length: float
    step_height: float
    step_count: int
    width: float
    offset_x: float
    offset_y: float
    offset_z: float
    flat_length: float


def stair_step_center(spec: StairRenderSpec, step_idx: int) -> np.ndarray:
    step_x = spec.offset_x + step_idx * spec.step_length
    step_z = spec.offset_z + step_idx * spec.step_height
    return np.asarray(
        [
            step_x + spec.step_length / 2.0,
            spec.offset_y,
            step_z + spec.step_height / 2.0,
        ],
        dtype=float,
    )


def stair_flat_center(spec: StairRenderSpec) -> np.ndarray:
    flat_x = spec.offset_x + spec.step_count * spec.step_length
    flat_top_z = spec.offset_z + spec.step_count * spec.step_height
    return np.asarray(
        [
            flat_x + spec.flat_length / 2.0,
            spec.offset_y,
            flat_top_z - spec.step_height / 2.0,
        ],
        dtype=float,
    )


def sampled_foot_target_refs(
    foot_target_refs: dict[str, np.ndarray] | None,
    sampled_indices: list[int],
) -> dict[str, np.ndarray]:
    if not foot_target_refs:
        return {}

    sampled: dict[str, np.ndarray] = {}
    for frame_name, points in foot_target_refs.items():
        points_array = np.asarray(points, dtype=float)
        if points_array.ndim != 2 or points_array.shape[1] != 3 or points_array.shape[0] == 0:
            continue
        clipped = np.asarray(
            [points_array[min(max(idx, 0), points_array.shape[0] - 1)] for idx in sampled_indices],
            dtype=float,
        )
        sampled[frame_name] = clipped
    return sampled


def foot_target_color(frame_name: str) -> int:
    lname = frame_name.lower()
    if "front" in lname and "left" in lname:
        return int(0xE76F51)
    if "front" in lname and "right" in lname:
        return int(0x2A9D8F)
    if "rear" in lname and "left" in lname:
        return int(0xF4A261)
    if "rear" in lname and "right" in lname:
        return int(0x457B9D)
    return int(0x6B7280)


__all__ = [
    "StairRenderSpec",
    "foot_target_color",
    "sampled_foot_target_refs",
    "stair_flat_center",
    "stair_step_center",
]
