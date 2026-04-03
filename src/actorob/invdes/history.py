"""Optimization history records for inverse design runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrialHistoryEntry:
    """Serializable summary of one evaluated optimization trial."""

    trial_number: int | None
    params: tuple[float, ...]
    status: str
    value: float | None
    error: str | None
    generation: int | None = None
    sigma: float | None = None


@dataclass(frozen=True)
class BatchHistoryEntry:
    """Serializable summary of one ask/tell batch."""

    batch_index: int
    total_batches: int
    batch_size: int
    evaluated_trials: int
    completed_trials: int
    failed_trials: int
    duration_seconds: float
    best_value: float | None
    generation: int | None
    sigma: float | None
    trials: tuple[TrialHistoryEntry, ...]


@dataclass(frozen=True)
class OptimizationHistory:
    """Ordered history of all batches evaluated during one run."""

    parameter_names: tuple[str, ...]
    batches: tuple[BatchHistoryEntry, ...]


def extract_trial_metadata(trial: Any) -> dict[str, Any]:
    """Best-effort extraction of backend-specific trial metadata."""

    metadata_getter = getattr(trial, "metadata", None)
    if callable(metadata_getter):
        metadata = metadata_getter()
        if isinstance(metadata, dict):
            return metadata
    return {}
