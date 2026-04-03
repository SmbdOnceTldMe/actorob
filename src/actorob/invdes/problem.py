"""Domain objects for inverse design optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from actorob.invdes.history import OptimizationHistory


ObjectiveFunction = Callable[[tuple[float, ...]], float]


@dataclass(frozen=True)
class FloatParameter:
    """Continuous optimization variable."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError("Parameter lower bound must be smaller than the upper bound.")


@dataclass(frozen=True)
class OptimizationProblem:
    """Optimization task described by bounded variables and an objective."""

    parameters: tuple[FloatParameter, ...]
    objective: ObjectiveFunction | None = None

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        if not parameters:
            raise ValueError("Optimization problem must define at least one parameter.")
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ValueError("Optimization parameter names must be unique.")
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True)
class OptimizationSettings:
    """Execution settings for the ask/tell optimization loop."""

    max_iterations: int
    parallelism: int = 1
    population_size: int | None = None

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.parallelism <= 0:
            raise ValueError("parallelism must be positive.")
        if self.population_size is None:
            object.__setattr__(self, "population_size", self.parallelism)
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")

    @property
    def max_trials(self) -> int:
        """Return the total number of trials implied by the settings."""

        return int(self.max_iterations) * int(self.population_size)


@dataclass(frozen=True)
class CompletedTrial:
    """Successfully evaluated trial."""

    params: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class FailedTrial:
    """Trial that failed during objective evaluation."""

    params: tuple[float, ...]
    error: str


@dataclass(frozen=True)
class OptimizationResult:
    """Summary of the optimization run."""

    best_params: tuple[float, ...] | None
    best_value: float | None
    completed_trials: tuple[CompletedTrial, ...]
    failed_trials: tuple[FailedTrial, ...]
    history: OptimizationHistory | None = None

    @property
    def completed_trials_count(self) -> int:
        """Return how many trials finished successfully."""

        return len(self.completed_trials)

    @property
    def failed_trials_count(self) -> int:
        """Return how many trials failed during evaluation."""

        return len(self.failed_trials)

    @classmethod
    def from_trials(
        cls,
        *,
        best_params: tuple[float, ...] | None,
        best_value: float | None,
        completed_trials: Iterable[CompletedTrial],
        failed_trials: Iterable[FailedTrial],
        history: OptimizationHistory | None = None,
    ) -> "OptimizationResult":
        """Create a result object from arbitrary completed and failed trial iterables."""

        return cls(
            best_params=best_params,
            best_value=best_value,
            completed_trials=tuple(completed_trials),
            failed_trials=tuple(failed_trials),
            history=history,
        )
