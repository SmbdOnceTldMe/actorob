from __future__ import annotations

import numpy as np

from actorob.config import TaskConfig

from .common import TaskPlan, constant_state_refs, is_jump_task


_LANDING_COMPLIANCE_WINDOW_FRACTION = 0.07
_LANDING_COMPLIANCE_WINDOW_MAX_STEPS = 7
_LANDING_COMPLIANCE_REAR_HIP_DELTA = 0.08
_LANDING_COMPLIANCE_REAR_KNEE_DELTA = -0.10


def matches(task_name: str, task: TaskConfig) -> bool:
    """Return whether the task should use the jump planner."""

    del task
    return is_jump_task(task_name)


def _landing_start_index(phase_schedule: list[str]) -> int | None:
    flight_indices = [idx for idx, phase_name in enumerate(phase_schedule) if phase_name == "flight"]
    if len(flight_indices) == 0 or flight_indices[-1] >= len(phase_schedule) - 1:
        return None
    return flight_indices[-1] + 1


def _landing_compliance_window_size(phase_schedule: list[str], landing_start: int) -> int:
    landing_horizon = max(len(phase_schedule) - landing_start, 0)
    if landing_horizon <= 0:
        return 0
    return min(
        _LANDING_COMPLIANCE_WINDOW_MAX_STEPS,
        max(1, int(round(_LANDING_COMPLIANCE_WINDOW_FRACTION * float(landing_horizon)))),
    )


def _jump_state_refs(optimizer, phase_schedule: list[str]) -> np.ndarray:
    state_refs = constant_state_refs(optimizer, len(phase_schedule))
    landing_start = _landing_start_index(phase_schedule)
    if landing_start is None:
        return state_refs

    window = _landing_compliance_window_size(phase_schedule, landing_start)
    if window <= 0:
        return state_refs

    rear_joint_deltas = {
        "rear_left_hip_pitch_joint": _LANDING_COMPLIANCE_REAR_HIP_DELTA,
        "rear_right_hip_pitch_joint": _LANDING_COMPLIANCE_REAR_HIP_DELTA,
        "rear_left_knee_pitch_joint": _LANDING_COMPLIANCE_REAR_KNEE_DELTA,
        "rear_right_knee_pitch_joint": _LANDING_COMPLIANCE_REAR_KNEE_DELTA,
    }
    for local_idx, step_idx in enumerate(range(landing_start, min(landing_start + window, len(phase_schedule)))):
        alpha = 1.0 - float(local_idx) / float(window)
        for joint_name, delta_q in rear_joint_deltas.items():
            q_idx = optimizer.joint_to_q_index.get(joint_name)
            if q_idx is None:
                continue
            state_refs[step_idx, q_idx] += delta_q * alpha
    return state_refs


def build_plan(optimizer, task_name: str, task: TaskConfig) -> TaskPlan:
    """Build contact, foot, and floating-base references for a jump task."""

    del task_name
    if task.contact_phases is None or task.phase_durations is None:
        raise ValueError("Jump task requires explicit contact_phases and phase_durations.")

    horizon = int(task.horizon)
    if horizon <= 0:
        raise ValueError(f"Jump task horizon must be positive, got {horizon}.")

    traj = task.trajectory_params
    jump_height = float(
        task.apex_dz if task.apex_dz != 0.0 else (0.0 if traj is None or traj.jump_h is None else traj.jump_h)
    )
    jump_length = float(
        task.target_dx if task.target_dx != 0.0 else (0.0 if traj is None or traj.jump_l is None else traj.jump_l)
    )

    frame_names = tuple(optimizer.config.contact.contact_frames_3d)
    footholds = {name: optimizer.contact_specs[name].world_translation.copy() for name in frame_names}
    foot_refs = {name: np.zeros((horizon, 3), dtype=float) for name in frame_names}
    base_refs = np.zeros((horizon, 3), dtype=float)

    body_frame_name = (
        "floating_base_joint"
        if optimizer.rmodel.existFrame("floating_base_joint")
        else (
            "body_link" if optimizer.rmodel.existFrame("body_link") else optimizer.config.contact.contact_frames_3d[0]
        )
    )
    body_frame_id = optimizer.rmodel.getFrameId(body_frame_name)
    body_translation0 = np.asarray(optimizer.rdata.oMf[body_frame_id].translation, dtype=float).reshape(3)

    schedule: list[list[object]] = []
    active_frames_schedule: list[tuple[str, ...]] = []
    phase_schedule: list[str] = []

    step_idx = 0
    flight_horizon = max(
        sum(duration for phase, duration in zip(task.contact_phases, task.phase_durations) if phase == "flight"), 1
    )
    flight_step = 0
    flight_z = jump_height * np.sin(np.linspace(0.0, np.pi, flight_horizon, dtype=float))
    flight_x = np.linspace(0.0, jump_length, flight_horizon, dtype=float)

    for phase_name, duration in zip(task.contact_phases, task.phase_durations):
        active_frames = optimizer._phase_active_frames(phase_name)
        for _ in range(duration):
            if phase_name == "flight":
                x_disp = float(flight_x[flight_step])
                z_disp = float(flight_z[flight_step])
                flight_step += 1
            elif flight_step > 0:
                x_disp = jump_length
                z_disp = 0.0
            else:
                x_disp = 0.0
                z_disp = 0.0

            base_refs[step_idx, :] = body_translation0 + np.array([x_disp, 0.0, z_disp], dtype=float)
            for frame_name in frame_names:
                shifted = footholds[frame_name] + np.array([x_disp, 0.0, 0.0], dtype=float)
                foot_refs[frame_name][step_idx, :] = shifted

            schedule.append(
                [optimizer._make_contact_model(frame, foot_refs[frame][step_idx, :]) for frame in active_frames]
            )
            active_frames_schedule.append(active_frames)
            phase_schedule.append(phase_name)
            step_idx += 1

    if step_idx != horizon:
        raise RuntimeError(f"Internal error: generated jump plan has {step_idx} stages, expected {horizon}.")

    return TaskPlan(
        contact_schedule=schedule,
        active_frames_schedule=active_frames_schedule,
        phase_schedule=phase_schedule,
        foot_refs=foot_refs,
        state_refs=_jump_state_refs(optimizer, phase_schedule),
        floating_base_refs=base_refs,
    )
