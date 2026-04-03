"""Meshcat scene rendering helpers for dashboard views."""

from actorob.dashboard.meshcat.geometry import (
    StairRenderSpec,
    sampled_foot_target_refs,
    stair_flat_center,
    stair_step_center,
)
from actorob.dashboard.meshcat.html import build_meshcat_section_html

__all__ = [
    "StairRenderSpec",
    "build_meshcat_section_html",
    "sampled_foot_target_refs",
    "stair_flat_center",
    "stair_step_center",
]
