from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass

from actorob.invdes.problem import (
    CompletedTrial,
    FailedTrial,
    FloatParameter,
    OptimizationProblem,
    OptimizationSettings,
)
from actorob.invdes.progress import BatchProgress
from actorob.invdes.runner import ParallelAskTellOptimizer, TrialStatus


@dataclass
class FakeTrial:
    values: dict[str, float]
    trial_number: int | None = None
    generation: int | None = None
    sigma: float | None = None

    def suggest_float(self, name: str, low: float, high: float) -> float:
        return self.values[name]

    def metadata(self) -> dict[str, float | int | None]:
        return {
            "trial_number": self.trial_number,
            "generation": self.generation,
            "sigma": self.sigma,
        }


class FakeStudy:
    def __init__(self, trials: list[FakeTrial]):
        self._trials = list(trials)
        self.tell_calls: list[tuple[FakeTrial, float | None, TrialStatus]] = []

    def ask(self) -> FakeTrial:
        return self._trials.pop(0)

    def tell(self, trial: FakeTrial, value: float | None = None, status: TrialStatus = TrialStatus.COMPLETE) -> None:
        self.tell_calls.append((trial, value, status))


class FakeStudyFactory:
    def __init__(self, study: FakeStudy):
        self._study = study

    def create(self):
        return self._study


class RecordingProgressReporter:
    def __init__(self) -> None:
        self.started: tuple[int, int] | None = None
        self.events: list[BatchProgress] = []
        self.heartbeats: list[dict[str, float | int | None]] = []
        self.closed = False

    def start(self, total_batches: int, total_trials: int) -> None:
        self.started = (total_batches, total_trials)

    def advance(self, progress: BatchProgress) -> None:
        self.events.append(progress)

    def heartbeat(self, progress) -> None:
        self.heartbeats.append(
            {
                "batch_index": progress.batch_index,
                "completed_in_batch": progress.completed_in_batch,
                "observed_trials": progress.observed_trials,
                "completed_trials": progress.completed_trials,
                "failed_trials": progress.failed_trials,
            }
        )

    def close(self) -> None:
        self.closed = True


class ParallelAskTellOptimizerTest(unittest.TestCase):
    def test_optimizer_reports_best_result_for_completed_trials(self) -> None:
        study = FakeStudy(
            [
                FakeTrial({"mass": 2.0, "gear": 2.0}),
                FakeTrial({"mass": 1.0, "gear": 1.0}),
                FakeTrial({"mass": 0.0, "gear": 0.0}),
                FakeTrial({"mass": 3.0, "gear": 3.0}),
            ]
        )
        optimizer = ParallelAskTellOptimizer(FakeStudyFactory(study))
        problem = OptimizationProblem(
            parameters=(
                FloatParameter("mass", 0.0, 10.0),
                FloatParameter("gear", 0.0, 10.0),
            ),
            objective=lambda params: sum(value * value for value in params),
        )

        result = optimizer.optimize(problem, OptimizationSettings(max_iterations=2, parallelism=2))

        self.assertEqual(result.best_params, (0.0, 0.0))
        self.assertEqual(result.best_value, 0.0)
        self.assertEqual(result.completed_trials_count, 4)
        self.assertEqual(result.failed_trials_count, 0)
        self.assertEqual([status for _, _, status in study.tell_calls], [TrialStatus.COMPLETE] * 4)

    def test_optimizer_evaluates_batch_in_parallel(self) -> None:
        study = FakeStudy(
            [
                FakeTrial({"mass": 1.0}),
                FakeTrial({"mass": 2.0}),
                FakeTrial({"mass": 3.0}),
                FakeTrial({"mass": 4.0}),
            ]
        )
        optimizer = ParallelAskTellOptimizer(FakeStudyFactory(study))
        active = 0
        max_active = 0
        lock = threading.Lock()

        def objective(params: tuple[float, ...]) -> float:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return params[0]

        problem = OptimizationProblem(
            parameters=(FloatParameter("mass", 0.0, 10.0),),
            objective=objective,
        )

        optimizer.optimize(problem, OptimizationSettings(max_iterations=2, parallelism=2))

        self.assertGreaterEqual(max_active, 2)

    def test_optimizer_marks_failed_trials(self) -> None:
        bad_trial = FakeTrial({"mass": 13.0})
        good_trial = FakeTrial({"mass": 3.0})
        study = FakeStudy([bad_trial, good_trial])
        optimizer = ParallelAskTellOptimizer(FakeStudyFactory(study))

        def objective(params: tuple[float, ...]) -> float:
            if params[0] == 13.0:
                raise RuntimeError("boom")
            return params[0]

        problem = OptimizationProblem(
            parameters=(FloatParameter("mass", 0.0, 20.0),),
            objective=objective,
        )

        result = optimizer.optimize(problem, OptimizationSettings(max_iterations=1, parallelism=2))

        self.assertEqual(result.best_params, (3.0,))
        self.assertEqual(result.best_value, 3.0)
        self.assertEqual(result.completed_trials_count, 1)
        self.assertEqual(result.failed_trials_count, 1)
        self.assertEqual(study.tell_calls[0][2], TrialStatus.FAIL)
        self.assertIsNone(study.tell_calls[0][1])
        self.assertEqual(study.tell_calls[1][2], TrialStatus.COMPLETE)
        self.assertEqual(study.tell_calls[1][1], 3.0)

    def test_optimizer_reports_batch_progress(self) -> None:
        study = FakeStudy(
            [
                FakeTrial({"mass": 5.0}),
                FakeTrial({"mass": 4.0}),
                FakeTrial({"mass": 3.0}),
                FakeTrial({"mass": 2.0}),
                FakeTrial({"mass": 1.0}),
                FakeTrial({"mass": 0.0}),
            ]
        )
        clock_values = iter([index / 10.0 for index in range(20)])
        optimizer = ParallelAskTellOptimizer(FakeStudyFactory(study), clock=lambda: next(clock_values))
        reporter = RecordingProgressReporter()
        problem = OptimizationProblem(
            parameters=(FloatParameter("mass", 0.0, 10.0),),
            objective=lambda params: params[0],
        )

        optimizer.optimize(
            problem,
            OptimizationSettings(max_iterations=3, parallelism=2, population_size=2),
            progress_reporter=reporter,
        )

        self.assertEqual(reporter.started, (3, 6))
        self.assertTrue(reporter.closed)
        self.assertEqual([event.batch_index for event in reporter.events], [1, 2, 3])
        self.assertEqual([event.batch_size for event in reporter.events], [2, 2, 2])
        self.assertEqual([event.evaluated_trials for event in reporter.events], [2, 4, 6])
        self.assertEqual([event.completed_trials for event in reporter.events], [2, 4, 6])
        self.assertEqual([event.failed_trials for event in reporter.events], [0, 0, 0])
        self.assertEqual([event.best_value for event in reporter.events], [4.0, 2.0, 0.0])
        self.assertEqual(
            [round(event.batch_duration_seconds, 2) for event in reporter.events],
            [0.3, 0.3, 0.3],
        )

    def test_optimizer_captures_history_entries(self) -> None:
        study = FakeStudy(
            [
                FakeTrial({"mass": 5.0}, trial_number=0, generation=0, sigma=0.25),
                FakeTrial({"mass": 4.0}, trial_number=1, generation=0, sigma=0.25),
                FakeTrial({"mass": 3.0}, trial_number=2, generation=1, sigma=0.2),
                FakeTrial({"mass": 2.0}, trial_number=3, generation=1, sigma=0.2),
            ]
        )
        optimizer = ParallelAskTellOptimizer(FakeStudyFactory(study), clock=iter((0.0, 0.5, 1.0, 1.6)).__next__)
        problem = OptimizationProblem(
            parameters=(FloatParameter("mass", 0.0, 10.0),),
            objective=lambda params: params[0],
        )

        result = optimizer.optimize(problem, OptimizationSettings(max_iterations=2, parallelism=2, population_size=2))

        self.assertIsNotNone(result.history)
        assert result.history is not None
        self.assertEqual(result.history.parameter_names, ("mass",))
        self.assertEqual(len(result.history.batches), 2)
        self.assertEqual(result.history.batches[0].generation, 0)
        self.assertEqual(result.history.batches[0].sigma, 0.25)
        self.assertEqual(result.history.batches[0].trials[0].trial_number, 0)
        self.assertEqual(result.history.batches[0].trials[0].status, TrialStatus.COMPLETE.value)
        self.assertEqual(result.history.batches[1].generation, 1)
        self.assertEqual(result.history.batches[1].sigma, 0.2)

    def test_optimizer_emits_batch_snapshots_for_callbacks(self) -> None:
        study = FakeStudy(
            [
                FakeTrial({"mass": 5.0}),
                FakeTrial({"mass": 4.0}),
                FakeTrial({"mass": 3.0}),
                FakeTrial({"mass": 2.0}),
            ]
        )
        optimizer = ParallelAskTellOptimizer(FakeStudyFactory(study))
        snapshots = []
        problem = OptimizationProblem(
            parameters=(FloatParameter("mass", 0.0, 10.0),),
            objective=lambda params: params[0],
        )

        optimizer.optimize(
            problem,
            OptimizationSettings(max_iterations=2, parallelism=2, population_size=2),
            batch_complete_callback=snapshots.append,
        )

        self.assertEqual(len(snapshots), 2)
        self.assertEqual([len(snapshot.history.batches) for snapshot in snapshots], [1, 2])
        self.assertEqual([snapshot.completed_trials_count for snapshot in snapshots], [2, 4])
        self.assertEqual([snapshot.best_value for snapshot in snapshots], [4.0, 2.0])
        self.assertEqual([snapshot.history.batches[-1].batch_index for snapshot in snapshots], [1, 2])

    def test_optimizer_emits_in_batch_heartbeats_when_evaluator_supports_them(self) -> None:
        study = FakeStudy([FakeTrial({"mass": 2.0}), FakeTrial({"mass": 1.0})])
        optimizer = ParallelAskTellOptimizer(FakeStudyFactory(study), clock=iter((0.0, 0.1, 0.2, 0.3, 0.4)).__next__)
        reporter = RecordingProgressReporter()

        class HeartbeatEvaluator:
            def __init__(self) -> None:
                self._callback = None

            def set_progress_callback(self, callback) -> None:
                self._callback = callback

            def evaluate(self, candidates, parallelism):
                assert self._callback is not None
                self._callback(0, 0)
                self._callback(1, 0)
                self._callback(1, 1)
                return (
                    CompletedTrial(params=tuple(candidates[0]), value=2.0),
                    FailedTrial(params=tuple(candidates[1]), error="boom"),
                )

        problem = OptimizationProblem(
            parameters=(FloatParameter("mass", 0.0, 10.0),),
            objective=lambda params: params[0],
        )

        optimizer.optimize(
            problem,
            OptimizationSettings(max_iterations=1, parallelism=1, population_size=2),
            batch_evaluator=HeartbeatEvaluator(),
            progress_reporter=reporter,
        )

        self.assertEqual(reporter.started, (1, 2))
        self.assertEqual(
            reporter.heartbeats,
            [
                {
                    "batch_index": 1,
                    "completed_in_batch": 0,
                    "observed_trials": 0,
                    "completed_trials": 0,
                    "failed_trials": 0,
                },
                {
                    "batch_index": 1,
                    "completed_in_batch": 1,
                    "observed_trials": 1,
                    "completed_trials": 1,
                    "failed_trials": 0,
                },
                {
                    "batch_index": 1,
                    "completed_in_batch": 2,
                    "observed_trials": 2,
                    "completed_trials": 1,
                    "failed_trials": 1,
                },
            ],
        )
