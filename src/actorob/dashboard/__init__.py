"""Dashboard entry points with lazy imports for Plotly-backed rendering."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .invdes.page import build_inverse_design_dashboard_html
    from .trajectory.page import build_trajectory_dashboard_from_file, build_trajectory_dashboard_html

_EXPORTS = {
    "build_inverse_design_dashboard_html": ("actorob.dashboard.invdes.page", "build_inverse_design_dashboard_html"),
    "build_trajectory_dashboard_html": ("actorob.dashboard.trajectory.page", "build_trajectory_dashboard_html"),
    "build_trajectory_dashboard_from_file": (
        "actorob.dashboard.trajectory.page",
        "build_trajectory_dashboard_from_file",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "build_inverse_design_dashboard_html",
    "build_trajectory_dashboard_html",
    "build_trajectory_dashboard_from_file",
]
