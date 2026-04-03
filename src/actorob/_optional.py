"""Helpers for optional dependency messaging."""

from __future__ import annotations


def missing_dependency_error(
    feature: str,
    dependency: str,
    *,
    extra: str | None = None,
) -> RuntimeError:
    """Build a consistent error for an unavailable optional dependency."""

    if extra is None:
        hint = "Use the documented pixi environment to provision this feature."
    else:
        hint = f"Install ACTOROB with the '{extra}' extra or use the documented pixi environment."
    return RuntimeError(f"{feature} requires optional dependency '{dependency}'. {hint}")
