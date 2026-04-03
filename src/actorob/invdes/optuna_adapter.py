"""Optuna-backed study factory configured for CMA-ES."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
import pickle
from typing import Any

from actorob.invdes.runner import StudyHandle, TrialHandle, TrialStatus


class OptunaTrialHandle:
    """Adapter exposing an Optuna trial through the runner protocol."""

    def __init__(self, trial) -> None:
        self._trial = trial

    @property
    def raw_trial(self):
        """Return the wrapped Optuna trial object."""

        return self._trial

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Delegate bounded float suggestion to Optuna."""

        return self._trial.suggest_float(name, low, high)

    def metadata(self) -> dict[str, Any]:
        """Expose Optuna-specific CMA-ES metadata for reporting."""

        system_attrs = self._trial.storage.get_trial_system_attrs(self._trial._trial_id)
        generation = system_attrs.get("cma:generation")
        sigma = _extract_cma_sigma(system_attrs)
        return {
            "trial_number": self._trial.number,
            "generation": generation,
            "sigma": sigma,
        }


class OptunaStudyHandle:
    """Adapter exposing an Optuna study through the runner protocol."""

    def __init__(self, study, fail_state) -> None:
        self._study = study
        self._fail_state = fail_state

    def ask(self) -> TrialHandle:
        """Request the next Optuna trial."""

        return OptunaTrialHandle(self._study.ask())

    def tell(self, trial: TrialHandle, value: float | None = None, status: TrialStatus = TrialStatus.COMPLETE) -> None:
        """Report a completed or failed trial back to Optuna."""

        if not isinstance(trial, OptunaTrialHandle):
            raise TypeError("OptunaStudyHandle expects OptunaTrialHandle instances.")

        if status is TrialStatus.FAIL:
            self._study.tell(trial.raw_trial, state=self._fail_state)
            return

        self._study.tell(trial.raw_trial, value)


@dataclass(frozen=True)
class OptunaCmaEsStudyFactory:
    """Factory that creates Optuna studies with a CMA-ES sampler."""

    seed: int | None = None
    direction: str = "minimize"
    popsize: int | None = None
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)

    def create(self) -> StudyHandle:
        """Create an Optuna study configured with the CMA-ES sampler."""

        try:
            import optuna
            from optuna.trial import TrialState
        except ImportError as exc:
            raise RuntimeError("Install 'optuna' and 'cmaes' to use the Optuna CMA-ES adapter.") from exc

        sampler_kwargs = {"seed": self.seed}
        if self.popsize is not None and "popsize" in signature(optuna.samplers.CmaEsSampler).parameters:
            sampler_kwargs["popsize"] = self.popsize
        for key, value in self.sampler_kwargs.items():
            if key in signature(optuna.samplers.CmaEsSampler).parameters:
                sampler_kwargs[key] = value

        sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        return OptunaStudyHandle(study, fail_state=TrialState.FAIL)


def _extract_cma_sigma(system_attrs: dict[str, Any]) -> float | None:
    optimizer_keys = [key for key in system_attrs if key.startswith("cma:optimizer:")]
    if not optimizer_keys:
        return None

    optimizer_str = "".join(system_attrs[key] for key in sorted(optimizer_keys))
    optimizer = pickle.loads(bytes.fromhex(optimizer_str))
    sigma = getattr(optimizer, "sigma", None)
    if sigma is None:
        sigma = getattr(optimizer, "_sigma", None)
    return None if sigma is None else float(sigma)
