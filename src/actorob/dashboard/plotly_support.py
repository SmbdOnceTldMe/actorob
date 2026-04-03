"""Optional Plotly dependency helpers for dashboard modules."""

from __future__ import annotations

from actorob._optional import missing_dependency_error

try:
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from plotly.io import to_html
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - exercised in dependency-missing environments
    go = None
    qualitative = None
    to_html = None
    make_subplots = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None


def require_plotly(feature: str) -> None:
    if _PLOTLY_IMPORT_ERROR is not None:
        raise missing_dependency_error(feature, "plotly", extra="reporting") from _PLOTLY_IMPORT_ERROR


__all__ = [
    "go",
    "make_subplots",
    "qualitative",
    "require_plotly",
    "to_html",
]
