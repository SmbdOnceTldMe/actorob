"""Public model API with lazy imports for MuJoCo-backed helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .factory import ModelFactory
    from .mjcf_assets import copy_mjcf_with_resolved_assets
    from .utils import expand_config

_EXPORTS = {
    "ModelFactory": ("actorob.models.factory", "ModelFactory"),
    "copy_mjcf_with_resolved_assets": ("actorob.models.mjcf_assets", "copy_mjcf_with_resolved_assets"),
    "expand_config": ("actorob.models.utils", "expand_config"),
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


__all__ = ["ModelFactory", "copy_mjcf_with_resolved_assets", "expand_config"]
