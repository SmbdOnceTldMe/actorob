from __future__ import annotations

from pathlib import Path


def resolve_mjcf_path(path: str | Path) -> Path:
    """Expand and resolve an MJCF path to an absolute filesystem path."""

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"MJCF file does not exist: {candidate}")
