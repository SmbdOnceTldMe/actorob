"""Inverse-design run record serialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

from actorob.invdes.evaluation.types import ObjectiveWeights
from actorob.invdes.problem import OptimizationResult, OptimizationSettings

if TYPE_CHECKING:
    from actorob.invdes.evaluation.types import ScenarioEvaluation
    from actorob.trajectories import TrajectoryRunRecord


@dataclass(frozen=True)
class InverseDesignRunRecord:
    """Serializable artifact capturing one inverse-design optimization run."""

    config_path: str
    task_names: tuple[str, ...]
    created_at_utc: str
    settings: OptimizationSettings
    seed: int | None
    sigma0: float
    weights: ObjectiveWeights
    result: OptimizationResult
    best_trajectory_record: TrajectoryRunRecord | None = None
    normalization_stats: dict[str, dict[str, dict[str, float]]] | None = None
    best_scenarios: tuple[ScenarioEvaluation, ...] = ()

    def __post_init__(self) -> None:
        if self.result.history is None:
            raise ValueError("InverseDesignRunRecord requires OptimizationResult.history.")

    @property
    def max_trials(self) -> int:
        """Return the maximum number of trials planned for the run."""

        return self.settings.max_trials

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the ordered optimization parameter names."""

        return self.history.parameter_names

    @property
    def best_params(self) -> tuple[float, ...] | None:
        """Return the best parameter vector found by the optimization."""

        return self.result.best_params

    @property
    def best_value(self) -> float | None:
        """Return the best objective value found by the optimization."""

        return self.result.best_value

    @property
    def history(self):
        """Return the optimization history captured in the run result."""

        if self.result.history is None:
            raise RuntimeError("Optimization history is unavailable.")
        return self.result.history

    def save(self, path: str | Path) -> Path:
        """Serialize the run record to a pickle file."""

        out_path = Path(path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as stream:
            pickle.dump(self, stream)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "InverseDesignRunRecord":
        """Load a previously serialized inverse-design run record."""

        in_path = Path(path).expanduser().resolve()
        with in_path.open("rb") as stream:
            data = pickle.load(stream)
        if not isinstance(data, cls):
            raise TypeError(f"Invalid inverse-design record type in '{in_path}': {type(data)}")
        return data

    @classmethod
    def now(
        cls,
        *,
        config_path: str,
        task_names: tuple[str, ...],
        settings: OptimizationSettings,
        seed: int | None,
        sigma0: float,
        weights: ObjectiveWeights,
        result: OptimizationResult,
        best_trajectory_record: TrajectoryRunRecord | None,
        normalization_stats: dict[str, dict[str, dict[str, float]]] | None = None,
        best_scenarios: tuple[ScenarioEvaluation, ...] = (),
    ) -> "InverseDesignRunRecord":
        """Create a run record stamped with the current UTC timestamp."""

        return cls(
            config_path=config_path,
            task_names=task_names,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            settings=settings,
            seed=seed,
            sigma0=sigma0,
            weights=weights,
            result=result,
            best_trajectory_record=best_trajectory_record,
            normalization_stats=normalization_stats,
            best_scenarios=best_scenarios,
        )


__all__ = ["InverseDesignRunRecord"]
