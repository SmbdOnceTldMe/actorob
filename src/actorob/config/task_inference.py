from __future__ import annotations

import numpy as np

from .resolved import StairConfig, TaskConfig, TrajectoryParams
from .schema import StairSectionModel, TaskInputModel, TrajectoryParamsModel


SUPPORTED_CONTACT_PHASES = frozenset(
    {
        "stance",
        "flight",
        "diag_a",
        "diag_b",
        "front_pair",
        "rear_pair",
        "left_pair",
        "right_pair",
    }
)


def build_task_config(task_name: str, task: TaskInputModel, dt: float) -> TaskConfig:
    """Normalize a raw task section into a resolved runtime task config."""

    trajectory_params = _to_trajectory_params(task.trajectory_params)
    horizon = _resolve_task_horizon(task_name, task, trajectory_params, dt)
    contact_phases, phase_durations = _resolve_task_contact_schedule(task_name, task, trajectory_params, horizon)
    stairs = _resolve_task_stairs(task, trajectory_params)
    defaults = _task_default_targets(task_name, trajectory_params)

    return TaskConfig(
        horizon=horizon,
        target_dx=defaults["target_dx"] if task.target_dx is None else task.target_dx,
        target_dy=defaults["target_dy"] if task.target_dy is None else task.target_dy,
        target_dz=defaults["target_dz"] if task.target_dz is None else task.target_dz,
        apex_dz=defaults["apex_dz"] if task.apex_dz is None else task.apex_dz,
        touchdown_dx=defaults["touchdown_dx"] if task.touchdown_dx is None else task.touchdown_dx,
        touchdown_dy=defaults["touchdown_dy"] if task.touchdown_dy is None else task.touchdown_dy,
        touchdown_dz=defaults["touchdown_dz"] if task.touchdown_dz is None else task.touchdown_dz,
        state_weight=task.state_weight,
        control_weight=task.control_weight,
        terminal_weight=task.terminal_weight,
        base_position_weight_scale=task.base_position_weight_scale,
        base_orientation_weight_scale=task.base_orientation_weight_scale,
        base_linear_velocity_weight_scale=task.base_linear_velocity_weight_scale,
        base_angular_velocity_weight_scale=task.base_angular_velocity_weight_scale,
        joint_targets=dict(task.joint_targets),
        contact_phases=contact_phases,
        phase_durations=phase_durations,
        stairs=stairs,
        trajectory_params=trajectory_params,
    )


def _to_trajectory_params(model: TrajectoryParamsModel | None) -> TrajectoryParams | None:
    if model is None:
        return None
    return TrajectoryParams(**model.model_dump())


def _to_stair_config(model: StairSectionModel | None) -> StairConfig | None:
    if model is None:
        return None
    return StairConfig(**model.model_dump())


def _integer_durations_from_weights(weights: list[float], total_steps: int) -> tuple[int, ...]:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if len(weights) == 0:
        raise ValueError("weights must be non-empty.")
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"all weights must be positive, got {weights}.")
    if total_steps < len(weights):
        raise ValueError(f"Cannot allocate {len(weights)} positive phase durations into total_steps={total_steps}.")

    durations = np.ones(len(weights), dtype=int)
    remaining = total_steps - len(weights)
    if remaining == 0:
        return tuple(int(v) for v in durations.tolist())

    weights_arr = np.asarray(weights, dtype=float)
    normalized = weights_arr / float(np.sum(weights_arr))
    extra_raw = normalized * float(remaining)
    extra_floor = np.floor(extra_raw).astype(int)
    durations += extra_floor

    still_remaining = remaining - int(np.sum(extra_floor))
    if still_remaining > 0:
        frac = extra_raw - extra_floor
        order = np.argsort(-frac)
        for idx in order[:still_remaining]:
            durations[int(idx)] += 1

    return tuple(int(v) for v in durations.tolist())


def _infer_contact_schedule_from_trajectory_params(
    task_name: str,
    trajectory_params: TrajectoryParams | None,
    horizon: int,
) -> tuple[tuple[str, ...] | None, tuple[int, ...] | None]:
    if trajectory_params is None:
        return None, None

    lname = task_name.lower()
    if "jump" in lname:
        if trajectory_params.t_st is None or trajectory_params.t_ft is None:
            return None, None
        t_st = int(trajectory_params.t_st)
        t_ft = int(trajectory_params.t_ft)
        if t_st <= 0 or t_ft <= 0:
            return None, None
        if horizon != 2 * t_st + t_ft:
            return None, None
        return ("stance", "flight", "stance"), (t_st, t_ft, t_st)

    if trajectory_params.n_steps is None or trajectory_params.n_steps <= 0:
        return None, None
    if trajectory_params.step_time is None or trajectory_params.step_time <= 0:
        return None, None

    ds_time = 0.0 if trajectory_params.ds_time is None else trajectory_params.ds_time
    if ds_time < 0:
        return None, None

    swing_time = trajectory_params.step_time - ds_time
    if swing_time <= 0:
        swing_time = trajectory_params.step_time
        ds_time = 0.0

    stance_time = ds_time if ds_time > 0 else trajectory_params.step_time * 0.5

    phase_names: list[str] = ["stance"]
    phase_weights: list[float] = [stance_time]
    for step_idx in range(trajectory_params.n_steps):
        phase_names.append("diag_a" if (step_idx % 2 == 0) else "diag_b")
        phase_weights.append(swing_time)
        phase_names.append("stance")
        phase_weights.append(stance_time)

    phase_durations = _integer_durations_from_weights(phase_weights, horizon)
    return tuple(phase_names), phase_durations


def _infer_horizon_from_trajectory_params(
    task_name: str,
    trajectory_params: TrajectoryParams | None,
    dt: float,
) -> tuple[int | None, int | None]:
    horizon_from_steps = None
    horizon_from_jump = None

    if (
        "jump" in task_name.lower()
        and trajectory_params is not None
        and trajectory_params.t_st is not None
        and trajectory_params.t_ft is not None
    ):
        horizon_from_jump = 2 * int(trajectory_params.t_st) + int(trajectory_params.t_ft)

    if (
        trajectory_params is not None
        and trajectory_params.n_steps is not None
        and trajectory_params.step_time is not None
    ):
        if trajectory_params.ds_time is not None:
            ds_time = max(float(trajectory_params.ds_time), 0.0)
            ss_time = max(float(trajectory_params.step_time) - ds_time, dt)
            ss_steps = max(1, int(round(ss_time / dt)))
            ds_steps = max(1, int(round(max(ds_time, dt) / dt)))
            init_ds_steps = 3 * ds_steps
            horizon_from_steps = (
                2 * init_ds_steps
                + (int(trajectory_params.n_steps) + 1) * ss_steps
                + int(trajectory_params.n_steps) * ds_steps
            )
        else:
            horizon_from_steps = int(round((trajectory_params.n_steps * trajectory_params.step_time) / dt))

    return horizon_from_steps, horizon_from_jump


def _explicit_schedule_horizon(task: TaskInputModel) -> int | None:
    if task.contact_phases is None or task.phase_durations is None:
        return None
    return int(sum(task.phase_durations))


def _validate_redundant_horizon(
    task_name: str, explicit_horizon: int | None, derived_horizon: int, source: str
) -> None:
    if explicit_horizon is None:
        return
    if int(explicit_horizon) != int(derived_horizon):
        raise ValueError(
            f"Task '{task_name}': horizon={explicit_horizon} does not match {source}={derived_horizon}. "
            "Remove 'horizon' or make it consistent."
        )


def _resolve_task_horizon(
    task_name: str, task: TaskInputModel, trajectory_params: TrajectoryParams | None, dt: float
) -> int:
    schedule_horizon = _explicit_schedule_horizon(task)
    horizon_from_steps, horizon_from_jump = _infer_horizon_from_trajectory_params(task_name, trajectory_params, dt)

    if schedule_horizon is not None:
        _validate_redundant_horizon(task_name, task.horizon, schedule_horizon, "sum(phase_durations)")
        horizon = schedule_horizon
    elif horizon_from_jump is not None:
        _validate_redundant_horizon(task_name, task.horizon, horizon_from_jump, "inferred jump horizon")
        horizon = horizon_from_jump
    elif horizon_from_steps is not None:
        _validate_redundant_horizon(task_name, task.horizon, horizon_from_steps, "inferred trajectory horizon")
        horizon = horizon_from_steps
    elif task.horizon is not None:
        horizon = task.horizon
    else:
        raise ValueError(
            f"Task '{task_name}' must define 'horizon', provide explicit 'phase_durations', "
            f"or provide inferable 'trajectory_params' (either t_st+t_ft, or n_steps+step_time)."
        )

    if horizon <= 0:
        raise ValueError(f"Task '{task_name}' has non-positive horizon: {horizon}.")
    return horizon


def _resolve_task_contact_schedule(
    task_name: str,
    task: TaskInputModel,
    trajectory_params: TrajectoryParams | None,
    horizon: int,
) -> tuple[tuple[str, ...] | None, tuple[int, ...] | None]:
    if task.contact_phases is None or task.phase_durations is None:
        return _infer_contact_schedule_from_trajectory_params(task_name, trajectory_params, horizon)

    unsupported = [
        phase
        for phase in task.contact_phases
        if phase not in SUPPORTED_CONTACT_PHASES and not phase.startswith("custom:")
    ]
    if unsupported:
        raise ValueError(
            f"Task '{task_name}': unsupported contact phase(s) {unsupported}. Supported: {sorted(SUPPORTED_CONTACT_PHASES)}."
        )
    if sum(task.phase_durations) != horizon:
        raise ValueError(
            f"Task '{task_name}': sum(phase_durations)={sum(task.phase_durations)} must match horizon={horizon}."
        )

    return task.contact_phases, task.phase_durations


def _resolve_task_stairs(task: TaskInputModel, trajectory_params: TrajectoryParams | None) -> StairConfig | None:
    stairs = _to_stair_config(task.stairs)
    if stairs is not None:
        return stairs

    if (
        trajectory_params is None
        or trajectory_params.n_steps is None
        or trajectory_params.step_time is None
        or trajectory_params.dx is None
        or trajectory_params.stair_start is None
        or trajectory_params.stair_end is None
        or trajectory_params.stair_h is None
    ):
        return None

    step_count = trajectory_params.stair_end - trajectory_params.stair_start
    if step_count <= 0 or trajectory_params.stair_h <= 0:
        return None

    step_length = trajectory_params.dx * trajectory_params.step_time
    flat_length = max(trajectory_params.n_steps - trajectory_params.stair_end, 0) * step_length
    return StairConfig(
        step_length=step_length,
        step_height=trajectory_params.stair_h,
        step_count=step_count,
        width=0.5,
        offset_x=None,
        offset_y=0.0,
        offset_z=0.0,
        flat_length=flat_length,
    )


def _task_default_targets(task_name: str, trajectory_params: TrajectoryParams | None) -> dict[str, float]:
    defaults = {
        "target_dx": 0.0,
        "target_dy": 0.0,
        "target_dz": 0.0,
        "apex_dz": 0.0,
        "touchdown_dx": 0.0,
        "touchdown_dy": 0.0,
        "touchdown_dz": 0.0,
    }

    if trajectory_params is None:
        return defaults

    if trajectory_params.dx is not None:
        if trajectory_params.step_time is not None and trajectory_params.n_steps is not None:
            defaults["target_dx"] = trajectory_params.dx * trajectory_params.step_time * trajectory_params.n_steps
            defaults["touchdown_dx"] = trajectory_params.dx * trajectory_params.step_time
        else:
            defaults["target_dx"] = trajectory_params.dx
    elif "jump" in task_name.lower() and trajectory_params.jump_l is not None:
        defaults["target_dx"] = trajectory_params.jump_l

    if trajectory_params.dy is not None:
        if trajectory_params.step_time is not None and trajectory_params.n_steps is not None:
            defaults["target_dy"] = trajectory_params.dy * trajectory_params.step_time * trajectory_params.n_steps
        else:
            defaults["target_dy"] = trajectory_params.dy

    if trajectory_params.jump_h is not None:
        defaults["apex_dz"] = trajectory_params.jump_h
    elif trajectory_params.swing_apex is not None:
        defaults["apex_dz"] = trajectory_params.swing_apex

    if (
        trajectory_params.stair_h is not None
        and trajectory_params.stair_start is not None
        and trajectory_params.stair_end is not None
    ):
        defaults["target_dz"] = trajectory_params.stair_h * max(
            trajectory_params.stair_end - trajectory_params.stair_start,
            0,
        )
        defaults["touchdown_dz"] = trajectory_params.stair_h

    return defaults


__all__ = [
    "SUPPORTED_CONTACT_PHASES",
    "build_task_config",
]
