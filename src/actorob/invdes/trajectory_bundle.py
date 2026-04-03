"""Inverse-design bundle assembly for trajectory-optimization driven studies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import inspect
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

from actorob.actuators.inference import infer_motor_mass_from_total_actuator_mass, parse_gear_ratio_from_name
from actorob.mjcf import resolve_mjcf_path

from actorob.invdes.evaluation.batch import BatchEvaluator, process_pool_executor_factory
from actorob.invdes.evaluation.types import ObjectiveWeights
from actorob.invdes.problem import OptimizationProblem
from actorob.trajectories import TrajectoryRunRecord

from .design_space import (
    ActuatorDesignVariable,
    ActuatorPreparer,
    default_actuator_design_variables,
)
from .evaluation.trajectory import TrajectoryCandidateEvaluator
from .runtime_support import load_trajectory_optimizer, solve_all_with_optional_seed


@dataclass(frozen=True)
class TrajectoryBundle:
    """Ready-to-run inverse-design package built around trajectory optimization."""

    problem: OptimizationProblem
    batch_evaluator: BatchEvaluator
    initial_guess: dict[str, float]
    candidate_evaluator: TrajectoryCandidateEvaluator


def build_trajectory_bundle(
    *,
    config_path: str | Path,
    task_names: Sequence[str] | None = None,
    design_variables: Sequence[ActuatorDesignVariable] | None = None,
    weights: ObjectiveWeights | None = None,
    load_config_fn: Callable[[str | Path], Any] | None = None,
    warm_start: bool = True,
) -> TrajectoryBundle:
    """Assemble the optimization problem, evaluator, and defaults for inverse design."""

    load_config = load_config_fn or _load_trajectory_optimizer_config
    selected_task_names = None if task_names is None else tuple(task_names)
    config = _load_bundle_config(load_config, config_path, selected_task_names)
    selected_task_names = tuple(config.tasks) if selected_task_names is None else selected_task_names
    _validate_task_names(config.tasks, selected_task_names)
    config = _replace_tasks(config, selected_task_names)

    design_space = tuple(design_variables) if design_variables is not None else default_actuator_design_variables()
    design_space = _apply_nominal_initial_guess(config.base.mjcf_path, design_space)
    preparer = ActuatorPreparer(config=config, design_variables=design_space)
    optimizer_cls = load_trajectory_optimizer()
    seed_record = _build_nominal_seed_record(config, selected_task_names, optimizer_cls) if warm_start else None
    candidate_evaluator = TrajectoryCandidateEvaluator(
        candidate_preparer=preparer,
        task_names=selected_task_names,
        optimizer_cls=optimizer_cls,
        seed_record=seed_record,
    )
    batch_evaluator = BatchEvaluator(
        candidate_evaluator=candidate_evaluator,
        weights=weights or ObjectiveWeights(),
        executor_factory=_load_process_pool_executor_factory(),
    )
    return TrajectoryBundle(
        problem=OptimizationProblem(parameters=preparer.build_parameters()),
        batch_evaluator=batch_evaluator,
        initial_guess=preparer.initial_guess_by_name(),
        candidate_evaluator=candidate_evaluator,
    )


def _replace_tasks(config: Any, task_names: Sequence[str]) -> Any:
    selected = {name: config.tasks[name] for name in task_names}
    if hasattr(config, "__dataclass_fields__"):
        return replace(config, tasks=selected)

    cloned = deepcopy(config)
    cloned.tasks = selected
    return cloned


def _validate_task_names(tasks: Mapping[str, Any], task_names: Sequence[str]) -> None:
    missing = [name for name in task_names if name not in tasks]
    if missing:
        raise ValueError(f"Unknown task names: {missing}. Available: {sorted(tasks)}")


def _load_bundle_config(
    load_config_fn: Callable[..., Any],
    config_path: str | Path,
    task_names: Sequence[str] | None,
) -> Any:
    if task_names is None:
        return load_config_fn(config_path)

    try:
        signature = inspect.signature(load_config_fn)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        params = signature.parameters
        if "task_names" in params or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()
        ):
            return load_config_fn(config_path, task_names=task_names)

    return load_config_fn(config_path)


def _load_trajectory_optimizer_config(path: str | Path, task_names: Sequence[str] | None = None):
    from actorob.config import load_trajectory_optimizer_config

    return load_trajectory_optimizer_config(path, task_names=task_names)


def _load_process_pool_executor_factory():
    return process_pool_executor_factory


def _build_nominal_seed_record(
    config: Any,
    task_names: Sequence[str],
    optimizer_cls: type,
) -> TrajectoryRunRecord | None:
    optimizer = optimizer_cls(config)
    if not hasattr(optimizer, "build_record"):
        return None
    results = solve_all_with_optional_seed(optimizer, task_names, None)
    return optimizer.build_record(results)


def _apply_nominal_initial_guess(
    mjcf_path: str | Path,
    design_space: Sequence[ActuatorDesignVariable],
) -> tuple[ActuatorDesignVariable, ...]:
    nominal = _extract_nominal_joint_family_params(mjcf_path)
    updated: list[ActuatorDesignVariable] = []
    for variable in design_space:
        params = nominal.get(variable.joint_family)
        if params is None:
            updated.append(variable)
            continue
        updated.append(
            replace(
                variable,
                initial_motor_mass=params["motor_mass"],
                initial_gear_ratio=params["gear_ratio"],
            )
        )
    return tuple(updated)


def _extract_nominal_joint_family_params(mjcf_path: str | Path) -> dict[str, dict[str, float]]:
    root = ET.parse(resolve_mjcf_path(mjcf_path)).getroot()
    body_masses: dict[str, list[float]] = {}
    gear_ratios: dict[str, list[float]] = {}

    for body in root.iter("body"):
        name = body.attrib.get("name", "")
        if not name.endswith("_actuator"):
            continue
        family = _joint_family_from_name(name)
        if family is None:
            continue
        inertial = body.find("inertial")
        if inertial is None or "mass" not in inertial.attrib:
            continue
        body_masses.setdefault(family, []).append(float(inertial.attrib["mass"]))

    for actuator in root.iter("general"):
        name = actuator.attrib.get("name", "")
        family = _joint_family_from_name(name)
        if family is None:
            continue
        gear_ratio = parse_gear_ratio_from_name(name)
        if gear_ratio is not None:
            gear_ratios.setdefault(family, []).append(float(gear_ratio))

    nominal: dict[str, dict[str, float]] = {}
    for family, masses in body_masses.items():
        if family not in gear_ratios or not masses:
            continue
        avg_total_mass = sum(masses) / len(masses)
        avg_gear_ratio = sum(gear_ratios[family]) / len(gear_ratios[family])
        nominal[family] = {
            "motor_mass": infer_motor_mass_from_total_actuator_mass(avg_total_mass, avg_gear_ratio),
            "gear_ratio": avg_gear_ratio,
        }
    return nominal


def _joint_family_from_name(name: str) -> str | None:
    match = re.search(r"(hip_roll|hip_pitch|knee_pitch)", name)
    return None if match is None else match.group(1)


__all__ = [
    "ActuatorDesignVariable",
    "ActuatorPreparer",
    "TrajectoryCandidateEvaluator",
    "TrajectoryBundle",
    "build_trajectory_bundle",
    "default_actuator_design_variables",
]
