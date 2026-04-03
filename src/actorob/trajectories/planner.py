from __future__ import annotations

from typing import Any

import numpy as np
import pinocchio as pin

from actorob.config import TaskConfig

from .footplanner import FootMotionPlanner, FootPlannerConfig, StairProfile
from .tasks import (
    TaskPlan,
    build_jump_plan,
    build_upstairs_plan,
    build_walk_plan,
    is_jump_task,
    is_stair_task,
    is_walk_task,
    uses_footplanner,
)


class OptimizerTaskPlanner:
    """Build task plans and contact schedules for the trajectory optimizer."""

    def __init__(self, optimizer: Any) -> None:
        self._optimizer = optimizer

    def should_use_footplanner(self, task_name: str, task: TaskConfig) -> bool:
        return uses_footplanner(self._optimizer, task_name, task)

    def build_task_plan(self, task_name: str, task: TaskConfig) -> TaskPlan:
        if is_walk_task(task_name):
            return build_walk_plan(self._optimizer, task_name, task)
        if is_stair_task(task_name, task):
            return build_upstairs_plan(self._optimizer, task_name, task)
        if is_jump_task(task_name):
            return build_jump_plan(self._optimizer, task_name, task)

        contact_schedule, active_frames_schedule, phase_schedule, foot_refs = self.build_generic_contact_schedule(task)
        return TaskPlan(
            contact_schedule=contact_schedule,
            active_frames_schedule=active_frames_schedule,
            phase_schedule=phase_schedule,
            foot_refs=foot_refs,
            state_refs=np.asarray(
                [self.reference_state(task, k, len(contact_schedule)) for k in range(len(contact_schedule))],
                dtype=float,
            ),
        )

    @staticmethod
    def resample_indices(src_len: int, dst_len: int) -> np.ndarray:
        if src_len <= 0 or dst_len <= 0:
            raise ValueError(f"Invalid resample sizes src_len={src_len}, dst_len={dst_len}.")
        if src_len == dst_len:
            return np.arange(dst_len, dtype=int)
        if dst_len == 1:
            return np.array([0], dtype=int)
        xs = np.linspace(0.0, float(src_len - 1), num=dst_len)
        return np.rint(xs).astype(int)

    def phase_active_frames(self, phase: str) -> tuple[str, ...]:
        optimizer = self._optimizer
        all_frames = tuple(optimizer.config.contact.contact_frames_3d)
        if phase == "stance":
            return all_frames
        if phase == "flight":
            return ()

        support_map = {
            "diag_a": ("front_left", "rear_right"),
            "diag_b": ("front_right", "rear_left"),
            "front_pair": ("front_left", "front_right"),
            "rear_pair": ("rear_left", "rear_right"),
            "left_pair": ("front_left", "rear_left"),
            "right_pair": ("front_right", "rear_right"),
        }
        if phase in support_map:
            required_keys = support_map[phase]
            missing = [key for key in required_keys if key not in optimizer.contact_role_map]
            if missing:
                raise ValueError(
                    f"Phase '{phase}' requires frames {required_keys}, but missing inferred keys {missing}. "
                    f"Configured contact frames: {all_frames}"
                )
            active_set = {optimizer.contact_role_map[key] for key in required_keys}
            return tuple(frame for frame in all_frames if frame in active_set)

        if phase.startswith("custom:"):
            names = tuple(name.strip() for name in phase.split(":", maxsplit=1)[1].split(",") if name.strip())
            if len(names) == 0:
                return ()
            unknown = [name for name in names if name not in optimizer.contact_specs]
            if unknown:
                raise ValueError(
                    f"Unknown frame(s) in custom contact phase '{phase}': {unknown}. "
                    f"Available: {tuple(optimizer.contact_specs)}"
                )
            active_set = set(names)
            return tuple(frame for frame in all_frames if frame in active_set)

        raise ValueError(f"Unsupported contact phase '{phase}'.")

    def pair_phase_to_active_frames(self, front_phase: str, rear_phase: str) -> tuple[str, ...]:
        optimizer = self._optimizer
        active_set: set[str] = set()

        if front_phase in {"double", "left"}:
            active_set.add(optimizer.contact_role_map["front_left"])
        if front_phase in {"double", "right"}:
            active_set.add(optimizer.contact_role_map["front_right"])
        if rear_phase in {"double", "left"}:
            active_set.add(optimizer.contact_role_map["rear_left"])
        if rear_phase in {"double", "right"}:
            active_set.add(optimizer.contact_role_map["rear_right"])

        all_frames = tuple(optimizer.config.contact.contact_frames_3d)
        return tuple(frame for frame in all_frames if frame in active_set)

    def active_frames_to_phase_label(self, active_frames: tuple[str, ...]) -> str:
        optimizer = self._optimizer
        active_set = set(active_frames)
        if len(active_set) == len(optimizer.config.contact.contact_frames_3d):
            return "stance"
        diag_a = {optimizer.contact_role_map["front_left"], optimizer.contact_role_map["rear_right"]}
        diag_b = {optimizer.contact_role_map["front_right"], optimizer.contact_role_map["rear_left"]}
        if active_set == diag_a:
            return "diag_a"
        if active_set == diag_b:
            return "diag_b"
        if len(active_set) == 0:
            return "flight"
        return "custom:" + ",".join(active_frames)

    def build_contact_schedule_from_footplanner(
        self,
        task_name: str,
        task: TaskConfig,
    ) -> tuple[list[list[pin.RigidConstraintModel]], list[tuple[str, ...]], list[str], dict[str, np.ndarray]]:
        optimizer = self._optimizer
        traj = task.trajectory_params
        if (
            traj is None
            or traj.n_steps is None
            or traj.step_time is None
            or traj.ds_time is None
            or traj.swing_apex is None
        ):
            raise ValueError("Foot planner requires trajectory_params with n_steps, step_time, ds_time, swing_apex.")

        n_steps = int(traj.n_steps)
        vx = float(traj.dx or 0.0)
        vy = float(traj.dy or 0.0)
        lname = task_name.lower()
        is_plain_walk = ("walk" in lname) and ("upstairs" not in lname) and ("stair" not in lname)
        v_des = np.array([vx, vy, 0.0], dtype=float)
        stair_start = int(traj.stair_start or 0)
        stair_end = int(traj.stair_end or 0)
        stair_h = float(traj.stair_h or 0.0)
        stair_task = stair_h > 0.0 and stair_end > stair_start

        fl = optimizer.contact_specs[optimizer.contact_role_map["front_left"]].world_translation
        fr = optimizer.contact_specs[optimizer.contact_role_map["front_right"]].world_translation
        rl = optimizer.contact_specs[optimizer.contact_role_map["rear_left"]].world_translation
        rr = optimizer.contact_specs[optimizer.contact_role_map["rear_right"]].world_translation
        stair_offset_x, stair_step_length, stair_step_count, stair_offset_z = self.stair_geometry(task)
        if stair_task:
            front_stair_profile = StairProfile(
                height=stair_h,
                start_step=stair_start,
                end_step=stair_end,
            )
            rear_stair_profile = StairProfile(
                height=stair_h,
                start_step=stair_start + 2,
                end_step=stair_end + 2,
            )
        else:
            stair_profile = StairProfile(
                height=stair_h,
                start_step=stair_start,
                end_step=stair_end,
                offset_x=stair_offset_x,
                step_length=stair_step_length,
                step_count=stair_step_count,
                offset_z=stair_offset_z,
            )
            front_stair_profile = stair_profile
            rear_stair_profile = stair_profile

        front_plan = FootMotionPlanner(
            FootPlannerConfig(
                dt=optimizer.config.trajectory.dt,
                step_time=float(traj.step_time),
                ds_time=float(traj.ds_time),
                swing_apex=float(traj.swing_apex),
                initial_left=fl,
                initial_right=fr,
                stair_profile=front_stair_profile,
            )
        ).generate_plan(v_des=v_des, n_steps=n_steps, start_left=True)

        rear_plan = FootMotionPlanner(
            FootPlannerConfig(
                dt=optimizer.config.trajectory.dt,
                step_time=float(traj.step_time),
                ds_time=float(traj.ds_time),
                swing_apex=float(traj.swing_apex),
                initial_left=rl,
                initial_right=rr,
                stair_profile=rear_stair_profile,
            )
        ).generate_plan(v_des=v_des, n_steps=n_steps, start_left=False)

        src_len = min(len(front_plan.contact_phases), len(rear_plan.contact_phases))
        if is_plain_walk or stair_task:
            dst_horizon = max(src_len, 1)
        else:
            dst_horizon = max(int(task.horizon), 1)
        indices = self.resample_indices(src_len=src_len, dst_len=dst_horizon)

        frame_names = tuple(optimizer.config.contact.contact_frames_3d)
        foot_refs = {frame: np.zeros((dst_horizon, 3), dtype=float) for frame in frame_names}
        schedule: list[list[pin.RigidConstraintModel]] = []
        active_frames_schedule: list[tuple[str, ...]] = []
        phase_schedule: list[str] = []

        for k, src_idx in enumerate(indices.tolist()):
            frame_targets = {
                optimizer.contact_role_map["front_left"]: front_plan.left_traj[src_idx].copy(),
                optimizer.contact_role_map["front_right"]: front_plan.right_traj[src_idx].copy(),
                optimizer.contact_role_map["rear_left"]: rear_plan.left_traj[src_idx].copy(),
                optimizer.contact_role_map["rear_right"]: rear_plan.right_traj[src_idx].copy(),
            }
            for frame_name in frame_names:
                if frame_name in frame_targets:
                    foot_refs[frame_name][k, :] = frame_targets[frame_name]
                elif k > 0:
                    foot_refs[frame_name][k, :] = foot_refs[frame_name][k - 1, :]
                else:
                    foot_refs[frame_name][k, :] = optimizer.contact_specs[frame_name].world_translation

            front_phase = front_plan.contact_phases[src_idx].value
            rear_phase = rear_plan.contact_phases[src_idx].value
            active_frames = self.pair_phase_to_active_frames(front_phase, rear_phase)
            phase_schedule.append(self.active_frames_to_phase_label(active_frames))
            active_frames_schedule.append(active_frames)
            schedule.append([optimizer._make_contact_model(frame, foot_refs[frame][k, :]) for frame in active_frames])

        return schedule, active_frames_schedule, phase_schedule, foot_refs

    def stair_geometry(self, task: TaskConfig) -> tuple[float | None, float | None, int | None, float]:
        optimizer = self._optimizer
        if task.stairs is None:
            return None, None, None, 0.0

        offset_x = task.stairs.offset_x
        if offset_x is None:
            first_contact_name = optimizer.config.contact.contact_frames_3d[0]
            if optimizer.rmodel.existFrame(first_contact_name):
                frame_id = optimizer.rmodel.getFrameId(first_contact_name)
                foot_init_x = float(optimizer.rdata.oMf[frame_id].translation[0])
            else:
                foot_init_x = float(optimizer.q0[0])
            stair_start = (
                0
                if task.trajectory_params is None or task.trajectory_params.stair_start is None
                else int(task.trajectory_params.stair_start)
            )
            offset_x = foot_init_x + 0.15 + stair_start * task.stairs.step_length

        return (
            float(offset_x),
            float(task.stairs.step_length),
            int(task.stairs.step_count),
            float(task.stairs.offset_z),
        )

    def build_generic_contact_schedule(
        self,
        task: TaskConfig,
    ) -> tuple[list[list[pin.RigidConstraintModel]], list[tuple[str, ...]], list[str], dict[str, np.ndarray] | None]:
        optimizer = self._optimizer
        footholds = {name: spec.world_translation.copy() for name, spec in optimizer.contact_specs.items()}

        if task.contact_phases is None or task.phase_durations is None:
            all_frames = tuple(optimizer.config.contact.contact_frames_3d)
            models_schedule = []
            active_frames_schedule: list[tuple[str, ...]] = []
            phase_schedule: list[str] = []
            for _ in range(task.horizon):
                models_schedule.append(
                    [optimizer._make_contact_model(frame_name, footholds[frame_name]) for frame_name in all_frames]
                )
                active_frames_schedule.append(all_frames)
                phase_schedule.append("stance")
            return models_schedule, active_frames_schedule, phase_schedule, None

        schedule: list[list[pin.RigidConstraintModel]] = []
        active_frames_schedule: list[tuple[str, ...]] = []
        phase_schedule: list[str] = []
        phase_frames = [self.phase_active_frames(phase) for phase in task.contact_phases]
        traj = task.trajectory_params
        use_stair_touchdown_profile = (
            traj is not None
            and traj.stair_start is not None
            and traj.stair_end is not None
            and traj.stair_h is not None
            and traj.stair_end > traj.stair_start
        )

        prev_active = set(phase_frames[0]) if len(phase_frames) > 0 else set()
        touchdown_event_idx = 0
        last_touchdown_event_idx: dict[str, int | None] = {
            frame: None for frame in optimizer.config.contact.contact_frames_3d
        }
        for phase_idx, (phase_name, active_frames, duration) in enumerate(
            zip(task.contact_phases, phase_frames, task.phase_durations)
        ):
            active_set = set(active_frames)
            if phase_idx > 0:
                touchdown_frames = active_set - prev_active
                for frame_name in touchdown_frames:
                    touchdown_shift_x = task.touchdown_dx
                    touchdown_shift_y = task.touchdown_dy
                    touchdown_shift_z = task.touchdown_dz
                    if use_stair_touchdown_profile:
                        last_idx = last_touchdown_event_idx[frame_name]
                        elapsed_events = 1 if last_idx is None else max(touchdown_event_idx - last_idx, 1)
                        touchdown_shift_x *= float(elapsed_events)
                        touchdown_shift_y *= float(elapsed_events)
                        z_from_event = 0 if last_idx is None else last_idx + 1
                        z_to_event = touchdown_event_idx
                        stair_events_count = 0
                        for stair_event_idx in range(z_from_event, z_to_event + 1):
                            if traj.stair_start <= stair_event_idx < traj.stair_end:
                                stair_events_count += 1
                        touchdown_shift_z = float(traj.stair_h) * float(stair_events_count)
                    touchdown_shift = np.array([touchdown_shift_x, touchdown_shift_y, touchdown_shift_z], dtype=float)
                    footholds[frame_name] = footholds[frame_name] + touchdown_shift
                    if use_stair_touchdown_profile:
                        last_touchdown_event_idx[frame_name] = touchdown_event_idx
                if len(touchdown_frames) > 0:
                    touchdown_event_idx += 1

            for _ in range(duration):
                schedule.append(
                    [optimizer._make_contact_model(frame_name, footholds[frame_name]) for frame_name in active_frames]
                )
                active_frames_schedule.append(active_frames)
                phase_schedule.append(phase_name)
            prev_active = active_set

        if len(schedule) != task.horizon:
            raise RuntimeError("Internal error: generated contact schedule length does not match task horizon.")
        return schedule, active_frames_schedule, phase_schedule, None

    def build_contact_schedule(
        self,
        task_name: str,
        task: TaskConfig,
    ) -> tuple[list[list[pin.RigidConstraintModel]], list[tuple[str, ...]], list[str], dict[str, np.ndarray] | None]:
        task_plan = self.build_task_plan(task_name, task)
        return (
            task_plan.contact_schedule,
            task_plan.active_frames_schedule,
            task_plan.phase_schedule,
            task_plan.foot_refs,
        )

    def reference_state(self, task: TaskConfig, step_idx: int, horizon: int | None = None) -> np.ndarray:
        optimizer = self._optimizer
        total_horizon = task.horizon if horizon is None else horizon
        alpha = float(step_idx + 1) / float(total_horizon)
        x_ref = optimizer.x0.copy()
        x_ref[0] = optimizer.x0[0] + task.target_dx * alpha
        x_ref[1] = optimizer.x0[1] + task.target_dy * alpha
        z_linear = optimizer.x0[2] + task.target_dz * alpha
        z_arc = task.apex_dz * np.sin(np.pi * alpha)
        x_ref[2] = z_linear + z_arc

        for joint_name, target_q in task.joint_targets.items():
            if joint_name not in optimizer.joint_to_q_index:
                raise ValueError(f"Unknown joint name in task config: {joint_name}")
            q_idx = optimizer.joint_to_q_index[joint_name]
            x_ref[q_idx] = (1.0 - alpha) * optimizer.x0[q_idx] + alpha * target_q
        return x_ref
