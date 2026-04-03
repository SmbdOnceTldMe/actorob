"""Public utility API with lazy imports for optional MuJoCo-backed helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .render import render
    from .scaling import power_law
    from .transform import get_fullinertia, mat2quat, quat2mat
    from .units import radsec_to_rpm, rpm_to_radsec

_EXPORTS = {
    "power_law": ("actorob.utils.scaling", "power_law"),
    "rpm_to_radsec": ("actorob.utils.units", "rpm_to_radsec"),
    "radsec_to_rpm": ("actorob.utils.units", "radsec_to_rpm"),
    "quat2mat": ("actorob.utils.transform", "quat2mat"),
    "mat2quat": ("actorob.utils.transform", "mat2quat"),
    "get_fullinertia": ("actorob.utils.transform", "get_fullinertia"),
    "render": ("actorob.utils.render", "render"),
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
    "power_law",
    "rpm_to_radsec",
    "radsec_to_rpm",
    "quat2mat",
    "mat2quat",
    "get_fullinertia",
    "render",
]
