"""Threaded ask/tell optimizer used by inverse design packaging."""

from __future__ import annotations

from enum import Enum
from time import perf_counter
from typing import Callable, Protocol, Sequence, Union

from actorob.invdes.history import BatchHistoryEntry, OptimizationHistory, TrialHistoryEntry, extract_trial_metadata
from actorob.invdes.problem import (
    CompletedTrial,
    FailedTrial,
    OptimizationProblem,
    OptimizationResult,
    OptimizationSettings,
)
from actorob.invdes.progress import BatchHeartbeat, BatchProgress, OptimizationProgressReporter


class TrialStatus(str, Enum):
    """Outcome reported back to the optimization backend for a trial."""

    COMPLETE = "complete"
    FAIL = "fail"


class TrialHandle(Protocol):
    """Protocol implemented by optimization-backend trial wrappers."""

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Ask the optimization backend for a bounded float parameter."""


class StudyHandle(Protocol):
    """Protocol implemented by optimization-backend study wrappers."""

    def ask(self) -> TrialHandle:
        """Request the next trial from the backend."""

    def tell(self, trial: TrialHandle, value: float | None = None, status: TrialStatus = TrialStatus.COMPLETE) -> None:
        """Report the trial outcome back to the backend."""


class StudyFactory(Protocol):
    """Factory interface for creating backend studies."""

    def create(self) -> StudyHandle:
        """Create a study instance for one optimization run."""


EvaluationOutcome = Union[CompletedTrial, FailedTrial]


class BatchEvaluator(Protocol):
    """Protocol for evaluating a batch of candidate parameter vectors."""

    def evaluate(
        self,
        candidates: Sequence[tuple[float, ...]],
        parallelism: int,
    ) -> Sequence[EvaluationOutcome]:
        """Evaluate one ask/tell batch."""


class ParallelAskTellOptimizer:
    """Runs batch ask/tell optimization with thread-based objective evaluation."""

    def __init__(
        self,
        study_factory: StudyFactory,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._study_factory = study_factory
        self._clock = clock

    def optimize(
        self,
        problem: OptimizationProblem,
        settings: OptimizationSettings,
        batch_evaluator: BatchEvaluator | None = None,
        progress_reporter: OptimizationProgressReporter | None = None,
        batch_complete_callback: Callable[[OptimizationResult], None] | None = None,
    ) -> OptimizationResult:
        """Run the full ask/tell loop and collect completed run metadata."""

        study = self._study_factory.create()
        completed_trials: list[CompletedTrial] = []
        failed_trials: list[FailedTrial] = []
        best_params: tuple[float, ...] | None = None
        best_value: float | None = None
        remaining_trials = settings.max_trials
        evaluated_trials = 0
        batch_index = 0
        generation_size = int(settings.population_size)
        total_batches = int(settings.max_iterations)
        parameter_names = tuple(parameter.name for parameter in problem.parameters)
        history_batches: list[BatchHistoryEntry] = []
        evaluator = batch_evaluator or self._build_default_batch_evaluator(problem)

        if progress_reporter is not None:
            progress_reporter.start(total_batches=total_batches, total_trials=settings.max_trials)

        try:
            while remaining_trials > 0:
                batch_size = min(generation_size, remaining_trials)
                batch_index += 1
                batch_started_at = self._clock()
                batch = [self._ask_trial(study, problem) for _ in range(batch_size)]
                candidates = tuple(params for _, params in batch)
                progress_callback_set = False
                if (
                    progress_reporter is not None
                    and hasattr(evaluator, "set_progress_callback")
                    and hasattr(progress_reporter, "heartbeat")
                ):
                    observed_in_batch = 0
                    completed_preview = len(completed_trials)
                    failed_preview = len(failed_trials)

                    def _on_candidate_progress(completed_delta: int, failed_delta: int) -> None:
                        nonlocal observed_in_batch, completed_preview, failed_preview
                        observed_in_batch += int(completed_delta)
                        failed_preview += int(failed_delta)
                        completed_preview += max(int(completed_delta) - int(failed_delta), 0)
                        progress_reporter.heartbeat(
                            BatchHeartbeat(
                                batch_index=batch_index,
                                total_batches=total_batches,
                                batch_size=batch_size,
                                completed_in_batch=observed_in_batch,
                                total_trials=settings.max_trials,
                                observed_trials=evaluated_trials + observed_in_batch,
                                completed_trials=completed_preview,
                                failed_trials=failed_preview,
                                best_value=best_value,
                                elapsed_seconds=self._clock() - batch_started_at,
                            )
                        )

                    evaluator.set_progress_callback(_on_candidate_progress)
                    progress_callback_set = True

                try:
                    outcomes = tuple(evaluator.evaluate(candidates, parallelism=settings.parallelism))
                finally:
                    if progress_callback_set:
                        evaluator.set_progress_callback(None)

                if len(outcomes) != batch_size:
                    raise ValueError("Batch evaluator must return one outcome per candidate.")

                for (trial, params), outcome in zip(batch, outcomes):
                    if isinstance(outcome, FailedTrial):
                        study.tell(trial, value=None, status=TrialStatus.FAIL)
                        failed_trials.append(outcome)
                        continue

                    if not isinstance(outcome, CompletedTrial):
                        raise TypeError("Unsupported batch evaluation outcome.")

                    study.tell(trial, value=outcome.value, status=TrialStatus.COMPLETE)
                    completed_trials.append(outcome)
                    if best_value is None or outcome.value < best_value:
                        best_value = outcome.value
                        best_params = params

                remaining_trials -= batch_size
                evaluated_trials += batch_size
                batch_duration = self._clock() - batch_started_at

                if progress_reporter is not None:
                    progress_reporter.advance(
                        BatchProgress(
                            batch_index=batch_index,
                            total_batches=total_batches,
                            batch_size=batch_size,
                            total_trials=settings.max_trials,
                            evaluated_trials=evaluated_trials,
                            completed_trials=len(completed_trials),
                            failed_trials=len(failed_trials),
                            best_value=best_value,
                            batch_duration_seconds=batch_duration,
                        )
                    )

                history_trials: list[TrialHistoryEntry] = []
                for (trial, params), outcome in zip(batch, outcomes):
                    metadata = extract_trial_metadata(trial)
                    if isinstance(outcome, FailedTrial):
                        history_trials.append(
                            TrialHistoryEntry(
                                trial_number=metadata.get("trial_number"),
                                params=params,
                                status=TrialStatus.FAIL.value,
                                value=None,
                                error=outcome.error,
                                generation=metadata.get("generation"),
                                sigma=metadata.get("sigma"),
                            )
                        )
                    else:
                        history_trials.append(
                            TrialHistoryEntry(
                                trial_number=metadata.get("trial_number"),
                                params=params,
                                status=TrialStatus.COMPLETE.value,
                                value=outcome.value,
                                error=None,
                                generation=metadata.get("generation"),
                                sigma=metadata.get("sigma"),
                            )
                        )

                batch_generation = next(
                    (entry.generation for entry in history_trials if entry.generation is not None), None
                )
                batch_sigma = next((entry.sigma for entry in history_trials if entry.sigma is not None), None)
                history_batches.append(
                    BatchHistoryEntry(
                        batch_index=batch_index,
                        total_batches=total_batches,
                        batch_size=batch_size,
                        evaluated_trials=evaluated_trials,
                        completed_trials=len(completed_trials),
                        failed_trials=len(failed_trials),
                        duration_seconds=batch_duration,
                        best_value=best_value,
                        generation=batch_generation,
                        sigma=batch_sigma,
                        trials=tuple(history_trials),
                    )
                )

                if batch_complete_callback is not None:
                    batch_complete_callback(
                        OptimizationResult.from_trials(
                            best_params=best_params,
                            best_value=best_value,
                            completed_trials=completed_trials,
                            failed_trials=failed_trials,
                            history=OptimizationHistory(
                                parameter_names=parameter_names,
                                batches=tuple(history_batches),
                            ),
                        )
                    )
        finally:
            if progress_reporter is not None:
                progress_reporter.close()

        return OptimizationResult.from_trials(
            best_params=best_params,
            best_value=best_value,
            completed_trials=completed_trials,
            failed_trials=failed_trials,
            history=OptimizationHistory(
                parameter_names=parameter_names,
                batches=tuple(history_batches),
            ),
        )

    @staticmethod
    def _ask_trial(study: StudyHandle, problem: OptimizationProblem) -> tuple[TrialHandle, tuple[float, ...]]:
        trial = study.ask()
        params = tuple(
            trial.suggest_float(parameter.name, parameter.low, parameter.high) for parameter in problem.parameters
        )
        return trial, params

    @staticmethod
    def _build_default_batch_evaluator(problem: OptimizationProblem) -> BatchEvaluator:
        if problem.objective is None:
            raise ValueError("OptimizationProblem.objective is required when no batch_evaluator is provided.")

        from actorob.invdes.evaluation.batch import ThreadedBatchEvaluator

        return ThreadedBatchEvaluator(problem.objective)
