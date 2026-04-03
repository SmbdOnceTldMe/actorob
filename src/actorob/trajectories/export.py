from __future__ import annotations

from typing import Any

import numpy as np
import pinocchio as pin

from .record import JointTrajectoryData, StairVisualizationData, TrajectoryRunRecord


class OptimizerRecordExporter:
    """Build serializable records and derived contact-force summaries."""

    def __init__(self, optimizer: Any) -> None:
        self._optimizer = optimizer

    def compute_contact_forces(
        self,
        xs: np.ndarray,
        us: np.ndarray,
        contact_models_schedule: list[list[pin.RigidConstraintModel]],
        active_frames_schedule: list[tuple[str, ...]],
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        optimizer = self._optimizer
        frame_names = tuple(optimizer.config.contact.contact_frames_3d)
        frame_index = {name: idx for idx, name in enumerate(frame_names)}
        n_steps = us.shape[0]
        n_frames = len(frame_names)

        active = np.zeros((n_steps, n_frames), dtype=bool)
        forces = np.zeros((n_steps, n_frames, 3), dtype=float)

        rdata = optimizer.rmodel.createData()
        for step_idx in range(n_steps):
            q = xs[step_idx, : optimizer.nq]
            v = xs[step_idx, optimizer.nq :]
            tau = optimizer.act_matrix @ us[step_idx]

            models = contact_models_schedule[step_idx]
            active_frames = active_frames_schedule[step_idx]
            if len(models) == 0:
                continue

            for frame_name in active_frames:
                active[step_idx, frame_index[frame_name]] = True

            datas = [model.createData() for model in models]
            try:
                pin.initConstraintDynamics(optimizer.rmodel, rdata, models)
                pin.constraintDynamics(optimizer.rmodel, rdata, q, v, tau, models, datas, optimizer.prox_settings)
                contact_forces = list(getattr(rdata, "contact_forces", []))
                for local_idx, frame_name in enumerate(active_frames):
                    frame_idx = frame_index[frame_name]
                    if local_idx < len(contact_forces):
                        force_obj = contact_forces[local_idx]
                    else:
                        force_obj = datas[local_idx].contact_force
                    forces[step_idx, frame_idx, :] = np.asarray(force_obj.linear, dtype=float).reshape(3)
            except Exception:
                for frame_name in active_frames:
                    frame_idx = frame_index[frame_name]
                    forces[step_idx, frame_idx, :] = np.nan

        return frame_names, active, forces

    def build_record(
        self,
        results: list[Any],
        task_metrics: dict[str, dict[str, np.ndarray | float]] | None = None,
    ) -> TrajectoryRunRecord:
        optimizer = self._optimizer
        joint_names = tuple(optimizer.rmodel.names[2:])
        tasks_data: list[JointTrajectoryData] = []

        for result in results:
            xs = np.asarray(result.xs)
            us = np.asarray(result.us)

            qs = xs[:, : optimizer.nq]
            dqs = xs[:, optimizer.nq :]

            state_time = np.arange(qs.shape[0], dtype=float) * optimizer.config.trajectory.dt
            control_time = np.arange(us.shape[0], dtype=float) * optimizer.config.trajectory.dt
            task_cfg = optimizer.config.tasks[result.task_name]
            metrics = {} if task_metrics is None else dict(task_metrics.get(result.task_name, {}))
            task_plan = optimizer._build_task_plan(result.task_name, task_cfg)
            contact_models_schedule = task_plan.contact_schedule
            active_frames_schedule = task_plan.active_frames_schedule
            phase_schedule = task_plan.phase_schedule
            foot_target_refs = (
                {name: np.asarray(points, dtype=float).copy() for name, points in task_plan.foot_refs.items()}
                if task_plan.foot_refs is not None
                else {}
            )
            contact_frame_names, contact_active, contact_forces = self.compute_contact_forces(
                xs=xs,
                us=us,
                contact_models_schedule=contact_models_schedule,
                active_frames_schedule=active_frames_schedule,
            )
            stair_data = self._build_stair_visualization(task_cfg)

            tasks_data.append(
                JointTrajectoryData(
                    task_name=result.task_name,
                    converged=result.converged,
                    iterations=result.iterations,
                    trajectory_cost=result.trajectory_cost,
                    state_time=state_time,
                    control_time=control_time,
                    floating_base_coordinates=qs[:, :7],
                    floating_base_velocities=dqs[:, :6],
                    joint_positions=qs[:, 7:],
                    joint_velocities=dqs[:, 6:],
                    joint_torques=us,
                    electrical_power=np.asarray(
                        metrics.get("electrical_power", np.zeros(us.shape[0], dtype=float)), dtype=float
                    ),
                    friction_power=np.asarray(
                        metrics.get("friction_power", np.zeros(us.shape[0], dtype=float)), dtype=float
                    ),
                    electrical_energy=float(metrics.get("electrical_energy", 0.0)),
                    friction_energy=float(metrics.get("friction_energy", 0.0)),
                    contact_frame_names=contact_frame_names,
                    contact_phase_labels=tuple(phase_schedule),
                    contact_active=contact_active,
                    contact_forces=contact_forces,
                    foot_target_refs=foot_target_refs,
                    stairs=stair_data,
                )
            )

        position_lower = np.asarray(optimizer.rmodel.lowerPositionLimit[7:], dtype=float)
        position_upper = np.asarray(optimizer.rmodel.upperPositionLimit[7:], dtype=float)
        velocity_abs = np.asarray(optimizer.rmodel.velocityLimit[6:], dtype=float)
        velocity_lower = -velocity_abs
        velocity_upper = velocity_abs
        torque_abs = np.asarray(optimizer.rmodel.effortLimit[6:], dtype=float)
        torque_lower = -torque_abs
        torque_upper = torque_abs

        return TrajectoryRunRecord.now(
            robot=optimizer.config.base.robot,
            mjcf_path=optimizer.config.base.mjcf_path,
            dt=optimizer.config.trajectory.dt,
            contact_mu=optimizer.config.contact.contact_mu,
            joint_names=joint_names,
            joint_position_lower_limits=position_lower,
            joint_position_upper_limits=position_upper,
            joint_velocity_lower_limits=velocity_lower,
            joint_velocity_upper_limits=velocity_upper,
            joint_torque_lower_limits=torque_lower,
            joint_torque_upper_limits=torque_upper,
            tasks=tuple(tasks_data),
        )

    def _build_stair_visualization(self, task_cfg: Any) -> StairVisualizationData | None:
        optimizer = self._optimizer
        if task_cfg.stairs is None:
            return None

        stairs_offset_x, _, _, _ = optimizer._stair_geometry(task_cfg)
        return StairVisualizationData(
            step_length=task_cfg.stairs.step_length,
            step_height=task_cfg.stairs.step_height,
            step_count=task_cfg.stairs.step_count,
            width=task_cfg.stairs.width,
            offset_x=stairs_offset_x,
            offset_y=task_cfg.stairs.offset_y,
            offset_z=task_cfg.stairs.offset_z,
            flat_length=task_cfg.stairs.flat_length,
            start_step=(
                None
                if task_cfg.trajectory_params is None or task_cfg.trajectory_params.stair_start is None
                else int(task_cfg.trajectory_params.stair_start)
            ),
            total_steps=(
                None
                if task_cfg.trajectory_params is None or task_cfg.trajectory_params.n_steps is None
                else int(task_cfg.trajectory_params.n_steps)
            ),
        )
