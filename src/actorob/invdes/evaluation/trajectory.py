"""Trajectory-optimizer-backed candidate evaluation for inverse design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from actorob.invdes.evaluation.types import ScenarioEvaluation
from actorob.invdes.runtime_support import (
    load_compute_actuator_group_metrics,
    load_trajectory_optimizer,
    solve_all_with_optional_seed,
    to_matrix,
)
from actorob.trajectories import TrajectoryRunRecord


@dataclass(frozen=True)
class _TaskActuationMetrics:
    electrical_power: np.ndarray
    friction_power: np.ndarray
    electrical_energy: float
    friction_energy: float


@dataclass(frozen=True)
class TrajectoryCandidateReport:
    """Detailed evaluation output for one actuator candidate."""

    scenarios: tuple[ScenarioEvaluation, ...]
    trajectory_record: TrajectoryRunRecord
    generated_mjcf_path: str | None = None


class TrajectoryCandidateEvaluator:
    """Evaluate candidates by running the trajectory optimizer on selected tasks."""

    def __init__(
        self,
        *,
        candidate_preparer: Any,
        task_names: Sequence[str] | None = None,
        optimizer_cls: type | None = None,
        actuator_metrics_fn: Callable[..., Mapping[str, Any]] | None = None,
        cleanup_generated_models: bool = True,
        seed_record: TrajectoryRunRecord | None = None,
    ) -> None:
        self._candidate_preparer = candidate_preparer
        self._task_names = None if task_names is None else tuple(task_names)
        self._optimizer_cls = optimizer_cls or load_trajectory_optimizer()
        self._actuator_metrics_fn = actuator_metrics_fn or load_compute_actuator_group_metrics()
        self._cleanup_generated_models = cleanup_generated_models
        self._seed_record = seed_record
        self._evaluation_lock = Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_evaluation_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._evaluation_lock = Lock()

    def evaluate(self, candidate: tuple[float, ...]) -> tuple[ScenarioEvaluation, ...]:
        """Evaluate one candidate and return only scalar scenario summaries."""

        scenarios, _, _, _, generated_mjcf_path = self._run_candidate(candidate)
        try:
            return scenarios
        finally:
            self._cleanup_generated_model(generated_mjcf_path)

    def evaluate_with_record(
        self,
        candidate: tuple[float, ...],
        *,
        cleanup_generated_model: bool | None = None,
    ) -> TrajectoryCandidateReport:
        """Evaluate one candidate and keep the full trajectory record."""

        scenarios, optimizer, results, task_metrics, generated_mjcf_path = self._run_candidate(candidate)
        try:
            if not hasattr(optimizer, "build_record"):
                raise AttributeError("Optimizer does not support build_record().")
            record = optimizer.build_record(results, task_metrics=task_metrics)
            return TrajectoryCandidateReport(
                scenarios=tuple(scenarios),
                trajectory_record=record,
                generated_mjcf_path=generated_mjcf_path,
            )
        finally:
            should_cleanup = (
                self._cleanup_generated_models if cleanup_generated_model is None else cleanup_generated_model
            )
            if should_cleanup:
                self._cleanup_generated_model(generated_mjcf_path)

    def _run_candidate(
        self,
        candidate: tuple[float, ...],
    ) -> tuple[tuple[ScenarioEvaluation, ...], Any, list[Any], dict[str, dict[str, np.ndarray | float]], str | None]:
        with self._evaluation_lock:
            prepared = self._candidate_preparer.prepare(candidate)
            generated_mjcf_path = None if prepared.metadata is None else prepared.metadata.get("generated_mjcf_path")
            optimizer = self._optimizer_cls(prepared.config)
            task_names = list(prepared.config.tasks) if self._task_names is None else list(self._task_names)
            results = solve_all_with_optional_seed(optimizer, task_names, self._seed_record)
            actuators = {} if prepared.metadata is None else dict(prepared.metadata.get("actuators", {}))
            scenarios: list[ScenarioEvaluation] = []
            task_metrics: dict[str, dict[str, np.ndarray | float]] = {}
            for result in results:
                scenario, metrics = self._build_scenario(result, optimizer, actuators, prepared.config.trajectory.dt)
                scenarios.append(scenario)
                task_metrics[str(result.task_name)] = {
                    "electrical_power": metrics.electrical_power,
                    "friction_power": metrics.friction_power,
                    "electrical_energy": metrics.electrical_energy,
                    "friction_energy": metrics.friction_energy,
                }
            return tuple(scenarios), optimizer, results, task_metrics, generated_mjcf_path

    def _cleanup_generated_model(self, generated_mjcf_path: str | None) -> None:
        if self._cleanup_generated_models and generated_mjcf_path:
            generated_path = Path(generated_mjcf_path)
            if generated_path.exists():
                generated_path.unlink()

    def _build_scenario(
        self,
        result: Any,
        optimizer: Any,
        actuators: Mapping[str, Any],
        dt: float,
    ) -> tuple[ScenarioEvaluation, _TaskActuationMetrics]:
        torques = to_matrix(result.us)
        states = to_matrix(result.xs)
        joint_names = tuple(optimizer.rmodel.names[2:])
        velocities = [row[optimizer.nq + 6 :] for row in states[: len(torques)]]
        electrical_energy = 0.0
        friction_loss = 0.0
        electrical_power = np.zeros(len(result.us), dtype=float)
        friction_power = np.zeros(len(result.us), dtype=float)

        if torques and velocities:
            joint_count = min(len(joint_names), len(torques[0]), len(velocities[0]))
            metrics = self._actuator_metrics_fn(
                [actuators.get(joint_name) for joint_name in joint_names[:joint_count]],
                np.asarray([row[:joint_count] for row in torques], dtype=float),
                np.asarray([row[:joint_count] for row in velocities], dtype=float),
                dt=dt,
            )
            electrical_power = np.asarray(metrics.get("P_elec", electrical_power), dtype=float)
            friction_power = np.asarray(metrics.get("P_gearbox_losses", friction_power), dtype=float)
            electrical_energy = float(metrics["E_motor_losses"] + metrics["E_mech"])
            friction_loss = float(metrics["E_gearbox_losses"])

        return (
            ScenarioEvaluation(
                case_name=str(result.task_name),
                mode=str(result.task_name).upper(),
                traj_cost=float(result.trajectory_cost),
                electrical_energy=electrical_energy,
                friction_loss=friction_loss,
                converged=bool(result.converged),
                iterations=int(result.iterations),
            ),
            _TaskActuationMetrics(
                electrical_power=electrical_power,
                friction_power=friction_power,
                electrical_energy=electrical_energy,
                friction_energy=friction_loss,
            ),
        )


__all__ = [
    "TrajectoryCandidateEvaluator",
    "TrajectoryCandidateReport",
]
