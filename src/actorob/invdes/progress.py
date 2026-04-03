"""Progress reporting utilities for inverse design optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from actorob._optional import missing_dependency_error
from actorob.invdes.evaluation.types import ObjectiveWeights
from actorob.invdes.problem import OptimizationSettings


@dataclass(frozen=True)
class BatchProgress:
    """Summary of one completed ask/tell batch."""

    batch_index: int
    total_batches: int
    batch_size: int
    total_trials: int
    evaluated_trials: int
    completed_trials: int
    failed_trials: int
    best_value: float | None
    batch_duration_seconds: float


@dataclass(frozen=True)
class BatchHeartbeat:
    """Lightweight in-batch progress snapshot."""

    batch_index: int
    total_batches: int
    batch_size: int
    completed_in_batch: int
    total_trials: int
    observed_trials: int
    completed_trials: int
    failed_trials: int
    best_value: float | None
    elapsed_seconds: float


class OptimizationProgressReporter(Protocol):
    """Receives optimization progress updates."""

    def start(self, total_batches: int, total_trials: int) -> None:
        """Called before the first batch is evaluated."""

    def advance(self, progress: BatchProgress) -> None:
        """Called after each completed batch."""

    def close(self) -> None:
        """Called when the optimization loop finishes or aborts."""


class TqdmProgressReporter:
    """Terminal progress bar based on tqdm."""

    def __init__(self, *, description: str = "Inverse design") -> None:
        self._description = description
        self._bar = None

    def start(self, total_batches: int, total_trials: int) -> None:
        """Initialize the underlying tqdm progress bar."""

        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise missing_dependency_error("Optimization progress reporting", "tqdm", extra="progress") from exc

        self._bar = tqdm(
            total=total_batches,
            desc=self._description,
            unit="batch",
            dynamic_ncols=True,
        )
        self._bar.set_postfix(
            trials=f"0/{total_trials}",
            iter_s="running",
            best="pending",
            ok=0,
            fail=0,
            refresh=True,
        )

    def advance(self, progress: BatchProgress) -> None:
        """Advance the bar after one completed batch."""

        if self._bar is None:
            raise RuntimeError("Progress reporter must be started before advance().")

        best_value = "pending" if progress.best_value is None else f"{progress.best_value:.4g}"
        self._bar.set_postfix(
            trials=f"{progress.evaluated_trials}/{progress.total_trials}",
            iter_s=f"{progress.batch_duration_seconds:.2f}",
            best=best_value,
            ok=progress.completed_trials,
            fail=progress.failed_trials,
            refresh=False,
        )
        self._bar.update(1)
        self._bar.refresh()

    def heartbeat(self, progress: BatchHeartbeat) -> None:
        """Refresh the bar with partial progress inside the current batch."""

        if self._bar is None:
            raise RuntimeError("Progress reporter must be started before heartbeat().")

        best_value = "pending" if progress.best_value is None else f"{progress.best_value:.4g}"
        self._bar.set_postfix(
            trials=f"{progress.observed_trials}/{progress.total_trials}",
            batch=f"{progress.completed_in_batch}/{progress.batch_size}",
            iter_s=f"{progress.elapsed_seconds:.2f}",
            best=best_value,
            ok=progress.completed_trials,
            fail=progress.failed_trials,
            refresh=False,
        )
        self._bar.refresh()

    def close(self) -> None:
        """Close the underlying tqdm instance if it was created."""

        if self._bar is not None:
            self._bar.close()
            self._bar = None


@dataclass(frozen=True)
class RunSummary:
    """Static information printed before an optimization run starts."""

    config_path: str
    tasks: tuple[str, ...]
    parameter_count: int
    settings: OptimizationSettings
    seed: int | None
    sigma: float
    progress_enabled: bool
    weights: ObjectiveWeights

    @property
    def max_trials(self) -> int:
        """Return the maximum number of candidate evaluations in the run."""

        return self.settings.max_trials

    @property
    def expected_batches(self) -> int:
        """Return the number of ask/tell batches implied by the settings."""

        return int(self.settings.max_iterations)


def format_run_summary(summary: RunSummary) -> str:
    """Render a human-readable summary of the optimization setup."""

    tasks = ", ".join(summary.tasks)
    progress_mode = "tqdm" if summary.progress_enabled else "disabled"
    lines = [
        "Inverse design run",
        f"  config: {summary.config_path}",
        f"  tasks ({len(summary.tasks)}): {tasks}",
        f"  parameters: {summary.parameter_count}",
        f"  max iterations: {summary.settings.max_iterations}",
        f"  total trials: {summary.max_trials}",
        f"  population size: {summary.settings.population_size}",
        f"  parallel threads: {summary.settings.parallelism}",
        f"  expected batches: {summary.expected_batches}",
        f"  seed: {summary.seed}",
        f"  sigma0: {summary.sigma}",
        (
            "  weights: "
            f"traj={summary.weights.traj_cost}, "
            f"energy={summary.weights.energy}, "
            f"friction={summary.weights.friction}"
        ),
        f"  progress: {progress_mode}",
    ]
    return "\n".join(lines)
