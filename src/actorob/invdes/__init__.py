"""Public inverse-design API.

The package uses lazy imports so lightweight metadata access does not require optional
visualization or optimization dependencies to be imported eagerly.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actorob.invdes.evaluation.batch import BatchEvaluator, ThreadedBatchEvaluator
    from actorob.invdes.evaluation.trajectory import TrajectoryCandidateEvaluator
    from actorob.invdes.evaluation.types import ObjectiveWeights, PreparedCandidate, ScenarioEvaluation
    from actorob.invdes.history import BatchHistoryEntry, OptimizationHistory, TrialHistoryEntry
    from actorob.invdes.optuna_adapter import OptunaCmaEsStudyFactory
    from actorob.invdes.problem import (
        CompletedTrial,
        FailedTrial,
        FloatParameter,
        OptimizationProblem,
        OptimizationResult,
        OptimizationSettings,
    )
    from actorob.invdes.progress import (
        BatchProgress,
        OptimizationProgressReporter,
        RunSummary,
        TqdmProgressReporter,
        format_run_summary,
    )
    from actorob.invdes.record import InverseDesignRunRecord
    from actorob.invdes.runner import ParallelAskTellOptimizer, TrialStatus
    from actorob.invdes.trajectory_bundle import (
        ActuatorDesignVariable,
        ActuatorPreparer,
        TrajectoryBundle,
        build_trajectory_bundle,
        default_actuator_design_variables,
    )

_EXPORTS = {
    "ActuatorDesignVariable": ("actorob.invdes.trajectory_bundle", "ActuatorDesignVariable"),
    "ActuatorPreparer": ("actorob.invdes.trajectory_bundle", "ActuatorPreparer"),
    "TrajectoryCandidateEvaluator": ("actorob.invdes.evaluation.trajectory", "TrajectoryCandidateEvaluator"),
    "TrajectoryBundle": ("actorob.invdes.trajectory_bundle", "TrajectoryBundle"),
    "build_trajectory_bundle": ("actorob.invdes.trajectory_bundle", "build_trajectory_bundle"),
    "default_actuator_design_variables": ("actorob.invdes.trajectory_bundle", "default_actuator_design_variables"),
    "BatchEvaluator": ("actorob.invdes.evaluation.batch", "BatchEvaluator"),
    "ObjectiveWeights": ("actorob.invdes.evaluation.types", "ObjectiveWeights"),
    "PreparedCandidate": ("actorob.invdes.evaluation.types", "PreparedCandidate"),
    "ScenarioEvaluation": ("actorob.invdes.evaluation.types", "ScenarioEvaluation"),
    "ThreadedBatchEvaluator": ("actorob.invdes.evaluation.batch", "ThreadedBatchEvaluator"),
    "BatchHistoryEntry": ("actorob.invdes.history", "BatchHistoryEntry"),
    "OptimizationHistory": ("actorob.invdes.history", "OptimizationHistory"),
    "TrialHistoryEntry": ("actorob.invdes.history", "TrialHistoryEntry"),
    "OptunaCmaEsStudyFactory": ("actorob.invdes.optuna_adapter", "OptunaCmaEsStudyFactory"),
    "CompletedTrial": ("actorob.invdes.problem", "CompletedTrial"),
    "FailedTrial": ("actorob.invdes.problem", "FailedTrial"),
    "FloatParameter": ("actorob.invdes.problem", "FloatParameter"),
    "OptimizationProblem": ("actorob.invdes.problem", "OptimizationProblem"),
    "OptimizationResult": ("actorob.invdes.problem", "OptimizationResult"),
    "OptimizationSettings": ("actorob.invdes.problem", "OptimizationSettings"),
    "BatchProgress": ("actorob.invdes.progress", "BatchProgress"),
    "OptimizationProgressReporter": ("actorob.invdes.progress", "OptimizationProgressReporter"),
    "RunSummary": ("actorob.invdes.progress", "RunSummary"),
    "TqdmProgressReporter": ("actorob.invdes.progress", "TqdmProgressReporter"),
    "format_run_summary": ("actorob.invdes.progress", "format_run_summary"),
    "InverseDesignRunRecord": ("actorob.invdes.record", "InverseDesignRunRecord"),
    "ParallelAskTellOptimizer": ("actorob.invdes.runner", "ParallelAskTellOptimizer"),
    "TrialStatus": ("actorob.invdes.runner", "TrialStatus"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ActuatorDesignVariable",
    "ActuatorPreparer",
    "BatchEvaluator",
    "BatchProgress",
    "BatchHistoryEntry",
    "CompletedTrial",
    "InverseDesignRunRecord",
    "FailedTrial",
    "FloatParameter",
    "ObjectiveWeights",
    "OptimizationHistory",
    "OptimizationProblem",
    "OptimizationProgressReporter",
    "OptimizationResult",
    "OptimizationSettings",
    "OptunaCmaEsStudyFactory",
    "ParallelAskTellOptimizer",
    "PreparedCandidate",
    "RunSummary",
    "ScenarioEvaluation",
    "ThreadedBatchEvaluator",
    "TqdmProgressReporter",
    "TrajectoryCandidateEvaluator",
    "TrajectoryBundle",
    "TrialHistoryEntry",
    "TrialStatus",
    "build_trajectory_bundle",
    "default_actuator_design_variables",
    "format_run_summary",
]
