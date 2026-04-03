"""Batch-oriented inverse-design evaluators."""

from __future__ import annotations

from concurrent.futures import Executor, FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from math import sqrt
import multiprocessing as mp
from typing import Callable, Sequence

from actorob.invdes.evaluation.types import CandidateEvaluator, ObjectiveWeights, ScenarioEvaluation
from actorob.invdes.problem import CompletedTrial, FailedTrial

ExecutorFactory = Callable[[int], Executor]
CandidateProgressCallback = Callable[[int, int], None]


def process_pool_executor_factory(max_workers: int) -> ProcessPoolExecutor:
    """Create a process pool using ``spawn`` for safer scientific backends."""

    return ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))


class ThreadedBatchEvaluator:
    """Default evaluator that parallelizes an objective over a batch of candidates."""

    def __init__(
        self,
        objective: Callable[[tuple[float, ...]], float],
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self._objective = objective
        self._executor_factory = executor_factory or self._default_executor_factory
        self._progress_callback: CandidateProgressCallback | None = None

    def set_progress_callback(self, callback: CandidateProgressCallback | None) -> None:
        """Register a callback invoked as candidate evaluations complete."""

        self._progress_callback = callback

    def evaluate(
        self,
        candidates: Sequence[tuple[float, ...]],
        parallelism: int,
    ) -> tuple[CompletedTrial | FailedTrial, ...]:
        """Evaluate a batch of scalar objectives in parallel threads."""

        with self._executor_factory(parallelism) as executor:
            future_by_index = {
                executor.submit(self._objective, candidate): index for index, candidate in enumerate(candidates)
            }
            outcomes: list[CompletedTrial | FailedTrial | None] = [None] * len(candidates)
            pending = set(future_by_index)

            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                if not done:
                    if self._progress_callback is not None:
                        self._progress_callback(0, 0)
                    continue

                for future in done:
                    index = future_by_index[future]
                    candidate = candidates[index]
                    try:
                        value = float(future.result())
                    except Exception as exc:
                        outcomes[index] = FailedTrial(params=tuple(candidate), error=str(exc))
                        if self._progress_callback is not None:
                            self._progress_callback(1, 1)
                    else:
                        outcomes[index] = CompletedTrial(params=tuple(candidate), value=value)
                        if self._progress_callback is not None:
                            self._progress_callback(1, 0)

            if any(outcome is None for outcome in outcomes):
                raise RuntimeError("Batch evaluation finished without producing all outcomes.")

            return tuple(outcome for outcome in outcomes if outcome is not None)

    @staticmethod
    def _default_executor_factory(max_workers: int) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=max_workers)


@dataclass(frozen=True)
class _MetricStats:
    mean: float
    std: float


class BatchEvaluator:
    """Batch evaluator with first-batch normalization for batch-relative scoring."""

    def __init__(
        self,
        candidate_evaluator: CandidateEvaluator,
        weights: ObjectiveWeights,
        *,
        non_convergence_penalty: float = 10.0,
        min_std_fraction: float = 0.05,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self._candidate_evaluator = candidate_evaluator
        self._weights = weights
        self._non_convergence_penalty = non_convergence_penalty
        self._min_std_fraction = float(min_std_fraction)
        self._executor_factory = executor_factory or ThreadedBatchEvaluator._default_executor_factory
        self._stats: dict[str, dict[str, _MetricStats]] | None = None
        self._progress_callback: CandidateProgressCallback | None = None

    def set_progress_callback(self, callback: CandidateProgressCallback | None) -> None:
        """Register a callback invoked as candidate evaluations complete."""

        self._progress_callback = callback

    def evaluate(
        self,
        candidates: Sequence[tuple[float, ...]],
        parallelism: int,
    ) -> tuple[CompletedTrial | FailedTrial, ...]:
        """Evaluate a batch and convert scenario metrics into scalar fitness."""

        raw_results = self._evaluate_candidates(candidates, parallelism)
        successful = [(candidate, scenarios) for candidate, scenarios, error in raw_results if error is None]

        if self._stats is None and successful:
            self._stats = self._build_stats([scenarios for _, scenarios in successful])

        outcomes: list[CompletedTrial | FailedTrial] = []
        for candidate, scenarios, error in raw_results:
            if error is not None:
                outcomes.append(FailedTrial(params=tuple(candidate), error=error))
                continue

            value = self._compute_fitness(scenarios)
            outcomes.append(CompletedTrial(params=tuple(candidate), value=value))

        return tuple(outcomes)

    def stats_snapshot(self) -> dict[str, dict[str, dict[str, float]]] | None:
        """Return the normalization statistics captured from the first batch."""

        if self._stats is None:
            return None
        snapshot: dict[str, dict[str, dict[str, float]]] = {}
        for mode, metrics in self._stats.items():
            snapshot[mode] = {}
            for metric_name, stats in metrics.items():
                snapshot[mode][metric_name] = {
                    "mean": stats.mean,
                    "std": stats.std,
                }
        return snapshot

    def _evaluate_candidates(
        self,
        candidates: Sequence[tuple[float, ...]],
        parallelism: int,
    ) -> list[tuple[tuple[float, ...], Sequence[ScenarioEvaluation], str | None]]:
        with self._executor_factory(parallelism) as executor:
            future_by_index = {
                executor.submit(self._candidate_evaluator.evaluate, tuple(candidate)): index
                for index, candidate in enumerate(candidates)
            }
            outputs: list[tuple[tuple[float, ...], Sequence[ScenarioEvaluation], str | None] | None] = [None] * len(
                candidates
            )
            pending = set(future_by_index)

            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                if not done:
                    if self._progress_callback is not None:
                        self._progress_callback(0, 0)
                    continue

                for future in done:
                    index = future_by_index[future]
                    candidate = tuple(candidates[index])
                    try:
                        scenarios = future.result()
                    except Exception as exc:
                        outputs[index] = (candidate, (), str(exc))
                        if self._progress_callback is not None:
                            self._progress_callback(1, 1)
                    else:
                        outputs[index] = (candidate, tuple(scenarios), None)
                        if self._progress_callback is not None:
                            self._progress_callback(1, 0)

            if any(output is None for output in outputs):
                raise RuntimeError("Candidate evaluation finished without producing all outputs.")

            return [output for output in outputs if output is not None]

    def _build_stats(
        self,
        evaluations: Sequence[Sequence[ScenarioEvaluation]],
    ) -> dict[str, dict[str, _MetricStats]]:
        grouped: dict[str, dict[str, list[float]]] = {}

        for candidate_evaluations in evaluations:
            for scenario in candidate_evaluations:
                metrics = grouped.setdefault(
                    scenario.mode,
                    {
                        "traj_cost": [],
                        "electrical_energy": [],
                        "friction_loss": [],
                    },
                )
                metrics["traj_cost"].append(scenario.traj_cost)
                metrics["electrical_energy"].append(scenario.electrical_energy)
                metrics["friction_loss"].append(scenario.friction_loss)

        stats: dict[str, dict[str, _MetricStats]] = {}
        for mode, metrics in grouped.items():
            stats[mode] = {}
            for metric_name, values in metrics.items():
                mean = sum(values) / len(values)
                std = self._regularized_std(values, mean)
                stats[mode][metric_name] = _MetricStats(mean=mean, std=std)
        return stats

    def _regularized_std(self, values: Sequence[float], mean: float) -> float:
        std = self._sample_std(values)
        scale = max(max((abs(value) for value in values), default=0.0), abs(mean), 1.0)
        return max(std, self._min_std_fraction * scale)

    @staticmethod
    def _sample_std(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return sqrt(variance) or 1.0

    def _compute_fitness(self, scenarios: Sequence[ScenarioEvaluation]) -> float:
        total = 0.0
        for scenario in scenarios:
            total += self._z_score(scenario.mode, "traj_cost", scenario.traj_cost) * self._weights.traj_cost
            total += (
                self._z_score(scenario.mode, "electrical_energy", scenario.electrical_energy) * self._weights.energy
            )
            total += self._z_score(scenario.mode, "friction_loss", scenario.friction_loss) * self._weights.friction
            if not scenario.converged:
                total += self._non_convergence_penalty
        return total

    def _z_score(self, mode: str, metric: str, value: float) -> float:
        if self._stats is None:
            return 0.0
        mode_stats = self._stats.get(mode)
        if mode_stats is None:
            return 0.0
        stats = mode_stats.get(metric)
        if stats is None or stats.std == 0:
            return 0.0
        return (value - stats.mean) / stats.std


__all__ = [
    "BatchEvaluator",
    "CandidateProgressCallback",
    "ExecutorFactory",
    "ThreadedBatchEvaluator",
    "process_pool_executor_factory",
]
