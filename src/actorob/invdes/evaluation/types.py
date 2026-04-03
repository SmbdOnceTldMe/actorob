"""Shared inverse-design evaluation types and protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ObjectiveWeights:
    """Relative weights for trajectory, energy, and friction objectives."""

    traj_cost: float = 1.0
    energy: float = 1.0
    friction: float = 1.0


@dataclass(frozen=True)
class PreparedCandidate:
    """Prepared design candidate together with derived config and metadata."""

    config: Any
    solution: tuple[float, ...]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScenarioEvaluation:
    """Trajectory evaluation metrics for one scenario or task variant."""

    mode: str
    traj_cost: float
    electrical_energy: float
    friction_loss: float
    converged: bool
    case_name: str | None = None
    iterations: int | None = None
    load: bool | None = None


class CandidateEvaluator(Protocol):
    """Protocol for evaluating one design candidate into scenario metrics."""

    def evaluate(self, candidate: tuple[float, ...]) -> Sequence[ScenarioEvaluation]:
        """Evaluate one design candidate across one or more scenarios."""


class CandidatePreparer(Protocol):
    """Protocol for turning a candidate vector into runtime-ready inputs."""

    def prepare(self, candidate: tuple[float, ...]) -> PreparedCandidate:
        """Prepare config and metadata for one design candidate."""


__all__ = [
    "CandidateEvaluator",
    "CandidatePreparer",
    "ObjectiveWeights",
    "PreparedCandidate",
    "ScenarioEvaluation",
]
