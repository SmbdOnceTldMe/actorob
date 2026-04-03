from __future__ import annotations

from dataclasses import dataclass

import aligator
import mujoco
import numpy as np
import pinocchio as pin

from actorob.config import TaskConfig, TrajectoryOptimizerConfig

from .actuation_constraints import build_mechanical_characteristic_spec
from .constraints import OptimizerConstraintBuilder
from .costs import OptimizerCostBuilder
from .export import OptimizerRecordExporter
from .planner import OptimizerTaskPlanner
from .record import TrajectoryRunRecord
from .tasks import TaskPlan, is_jump_task, is_stair_task, is_walk_task
from .warm_start import OptimizerInitialGuessBuilder


@dataclass(frozen=True)
class TaskResult:
    """Trajectory solver output for one named task."""

    task_name: str
    converged: bool
    iterations: int
    trajectory_cost: float
    xs: np.ndarray
    us: np.ndarray


@dataclass(frozen=True)
class _ContactFrameSpec:
    frame_name: str
    joint_id: int
    frame_placement: pin.SE3
    world_translation: np.ndarray


class AligatorTrajectoryOptimizer:
    """Minimal Aligator-based trajectory optimizer."""

    def __init__(self, config: TrajectoryOptimizerConfig):
        self.config = config

        self.rmodel = pin.buildModelFromMJCF(self.config.base.mjcf_path)
        self._mechanical_characteristic = build_mechanical_characteristic_spec(
            self.config, tuple(self.rmodel.names[2:])
        )
        self._apply_mujoco_joint_properties()
        self.space = aligator.manifolds.MultibodyPhaseSpace(self.rmodel)

        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nu = self.nv - 6
        self.act_matrix = np.eye(self.nv, self.nu, -6)
        self.joint_to_q_index = {name: 7 + idx for idx, name in enumerate(self.rmodel.names[2:])}

        self.q0 = self._get_initial_configuration()
        self.rdata = self.rmodel.createData()
        pin.forwardKinematics(self.rmodel, self.rdata, self.q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        self._align_floating_base_to_ground()
        self.x0 = np.concatenate([self.q0, np.zeros(self.nv)])

        self.prox_settings = pin.ProximalSettings(
            self.config.contact.prox_accuracy,
            self.config.contact.prox_mu,
            self.config.contact.prox_max_iter,
        )
        self.contact_specs = self._build_contact_specs()
        self.contact_frame_ids = {name: self.rmodel.getFrameId(name) for name in self.config.contact.contact_frames_3d}
        self.contact_role_map = self._infer_contact_role_map()

        self._task_planner = OptimizerTaskPlanner(self)
        self._constraint_builder = OptimizerConstraintBuilder(self)
        self._cost_builder = OptimizerCostBuilder(self)
        self._record_exporter = OptimizerRecordExporter(self)
        self._initial_guess_builder = OptimizerInitialGuessBuilder(self)

    def _apply_mujoco_joint_properties(self) -> None:
        try:
            mj_model = mujoco.MjModel.from_xml_path(self.config.base.mjcf_path)
        except Exception:
            return

        joint_dofs = self.rmodel.nv - 6
        armature = np.asarray(getattr(mj_model, "dof_armature", ()), dtype=float)
        friction = np.asarray(getattr(mj_model, "dof_frictionloss", ()), dtype=float)
        if armature.size == self.rmodel.nv:
            self.rmodel.armature = armature.copy()
        if friction.size == self.rmodel.nv:
            with np.errstate(all="ignore"):
                try:
                    self.rmodel.friction = friction.copy()
                except Exception:
                    pass
        ctrlrange = np.asarray(getattr(mj_model, "actuator_ctrlrange", ()), dtype=float)
        if ctrlrange.ndim == 2 and ctrlrange.shape == (joint_dofs, 2):
            effort_limit = np.full(self.rmodel.nv, np.inf, dtype=float)
            effort_limit[6:] = np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1]))
            self.rmodel.effortLimit = effort_limit

        mech_spec = self._mechanical_characteristic
        if mech_spec is not None and mech_spec.no_load_velocity.shape == (joint_dofs,):
            velocity_limit = np.full(self.rmodel.nv, np.inf, dtype=float)
            velocity_limit[6:] = mech_spec.no_load_velocity
            self.rmodel.velocityLimit = velocity_limit

    def _get_initial_configuration(self) -> np.ndarray:
        if self.config.base.init_pose is None:
            return pin.neutral(self.rmodel)

        init_pose = np.asarray(self.config.base.init_pose, dtype=float)
        if init_pose.size != self.nq:
            raise ValueError(f"init_pose size {init_pose.size} does not match robot nq={self.nq}.")
        return pin.normalize(self.rmodel, init_pose)

    def _align_floating_base_to_ground(self) -> None:
        if not self.config.base.align_feet_to_ground:
            return
        if self.nq < 3:
            return

        foot_z: list[float] = []
        for frame_name in self.config.contact.contact_frames_3d:
            if not self.rmodel.existFrame(frame_name):
                continue
            frame_id = self.rmodel.getFrameId(frame_name)
            foot_z.append(float(self.rdata.oMf[frame_id].translation[2]))

        if len(foot_z) == 0:
            return

        z_shift = min(foot_z) - float(self.config.base.ground_z)
        if abs(z_shift) <= 1e-12:
            return

        self.q0 = self.q0.copy()
        self.q0[2] -= z_shift
        self.q0 = pin.normalize(self.rmodel, self.q0)
        self.rdata = self.rmodel.createData()
        pin.forwardKinematics(self.rmodel, self.rdata, self.q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

    def _build_solver(
        self,
        tol_override: float | None = None,
        max_iter_override: int | None = None,
        reg_min_override: float | None = None,
    ) -> aligator.SolverProxDDP:
        verbose = self.config.solver.verbose
        level = aligator.VerboseLevel.__dict__[verbose]
        solver = aligator.SolverProxDDP(
            self.config.solver.tol if tol_override is None else float(tol_override),
            self.config.solver.mu_init,
            self.config.solver.max_iter if max_iter_override is None else int(max_iter_override),
            level,
        )
        if hasattr(solver, "rollout_type"):
            solver.rollout_type = aligator.__dict__[self.config.solver.rollout_type]
        if hasattr(solver, "sa_strategy"):
            solver.sa_strategy = aligator.__dict__[self.config.solver.sa_strategy]
        if hasattr(solver, "linear_solver_choice"):
            solver.linear_solver_choice = aligator.__dict__[self.config.solver.linear_solver_choice]
        if hasattr(solver, "filter") and hasattr(solver.filter, "beta"):
            solver.filter.beta = float(self.config.solver.filter_beta)
        if hasattr(solver, "force_initial_condition"):
            solver.force_initial_condition = bool(self.config.solver.force_initial_condition)
        if hasattr(solver, "reg_min"):
            solver.reg_min = float(self.config.solver.reg_min) if reg_min_override is None else float(reg_min_override)
        solver.max_iters = self.config.solver.max_iter if max_iter_override is None else int(max_iter_override)
        if hasattr(solver, "setNumThreads") and int(self.config.solver.num_threads) > 1:
            solver.setNumThreads(int(self.config.solver.num_threads))
        return solver

    def _build_contact_specs(self) -> dict[str, _ContactFrameSpec]:
        contact_specs: dict[str, _ContactFrameSpec] = {}
        for frame_name in self.config.contact.contact_frames_3d:
            if not self.rmodel.existFrame(frame_name):
                raise ValueError(f"Contact frame '{frame_name}' does not exist in model.")
            frame_id = self.rmodel.getFrameId(frame_name)
            joint_id = self.rmodel.frames[frame_id].parentJoint
            frame_placement = self.rmodel.frames[frame_id].placement
            world_translation = self.rdata.oMf[frame_id].translation.copy()
            contact_specs[frame_name] = _ContactFrameSpec(
                frame_name=frame_name,
                joint_id=joint_id,
                frame_placement=frame_placement,
                world_translation=world_translation,
            )
        return contact_specs

    def _infer_contact_role_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for frame_name in self.config.contact.contact_frames_3d:
            lname = frame_name.lower()
            if "front" in lname and "left" in lname:
                mapping["front_left"] = frame_name
            elif "front" in lname and "right" in lname:
                mapping["front_right"] = frame_name
            elif "rear" in lname and "left" in lname:
                mapping["rear_left"] = frame_name
            elif "rear" in lname and "right" in lname:
                mapping["rear_right"] = frame_name
        return mapping

    def _make_contact_model(self, frame_name: str, world_translation: np.ndarray) -> pin.RigidConstraintModel:
        spec = self.contact_specs[frame_name]
        world_placement = pin.SE3(np.eye(3), world_translation)
        model = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            self.rmodel,
            spec.joint_id,
            spec.frame_placement,
            0,
            world_placement,
            pin.LOCAL_WORLD_ALIGNED,
        )
        model.name = frame_name
        return model

    def _solver_reg_min_for_task(self, task_name: str) -> float:
        if is_jump_task(task_name):
            return max(float(self.config.solver.reg_min), 8.0e-3)
        return float(self.config.solver.reg_min)

    def _apply_task_specific_solver_settings(self, solver: aligator.SolverProxDDP, task_name: str) -> None:
        if is_stair_task(task_name, self.config.tasks[task_name]):
            if hasattr(solver, "sa_strategy"):
                solver.sa_strategy = aligator.SA_FILTER
            if hasattr(solver, "rollout_type"):
                solver.rollout_type = aligator.ROLLOUT_LINEAR
            if hasattr(solver, "linear_solver_choice"):
                solver.linear_solver_choice = aligator.LQ_SOLVER_SERIAL

    def _phase_active_frames(self, phase: str) -> tuple[str, ...]:
        return self._task_planner.phase_active_frames(phase)

    def _build_task_plan(self, task_name: str, task: TaskConfig) -> TaskPlan:
        return self._task_planner.build_task_plan(task_name, task)

    def _build_contact_schedule_from_footplanner(
        self,
        task_name: str,
        task: TaskConfig,
    ) -> tuple[list[list[pin.RigidConstraintModel]], list[tuple[str, ...]], list[str], dict[str, np.ndarray]]:
        return self._task_planner.build_contact_schedule_from_footplanner(task_name, task)

    def _stair_geometry(
        self,
        task: TaskConfig,
    ) -> tuple[float | None, float | None, int | None, float]:
        return self._task_planner.stair_geometry(task)

    def _reference_state(self, task: TaskConfig, step_idx: int, horizon: int | None = None) -> np.ndarray:
        return self._task_planner.reference_state(task, step_idx, horizon)

    def build_problem(self, task_name: str) -> aligator.TrajOptProblem:
        """Construct the trajectory-optimization problem for one configured task."""

        if task_name not in self.config.tasks:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(self.config.tasks)}")

        task = self.config.tasks[task_name]
        task_plan = self._build_task_plan(task_name, task)
        contact_schedule = task_plan.contact_schedule
        active_frames_schedule = task_plan.active_frames_schedule
        phase_schedule = task_plan.phase_schedule
        foot_refs = task_plan.foot_refs
        horizon = len(contact_schedule)
        if horizon <= 0:
            raise RuntimeError(f"Empty contact schedule for task '{task_name}'.")

        use_footplanner_refs = foot_refs is not None
        is_plain_walk = is_walk_task(task_name)
        stair_task = is_stair_task(task_name, task)
        jump_task = is_jump_task(task_name)
        state_refs = task_plan.state_refs
        if state_refs is None:
            state_refs = np.asarray([self._reference_state(task, k, horizon) for k in range(horizon)], dtype=float)

        w_x = self._cost_builder.state_tracking_weights(task_name, task, use_footplanner_refs)
        w_u = self._cost_builder.control_tracking_weights(task_name, task)
        u_ref = np.zeros(self.nu)

        stages = []
        prev_active_frames: set[str] = set(active_frames_schedule[0]) if len(active_frames_schedule) > 0 else set()
        for k in range(horizon):
            active_contacts = contact_schedule[k]
            active_frame_set = set(active_frames_schedule[k]) if k < len(active_frames_schedule) else set()
            dyn_model = self._constraint_builder.build_dynamics_model(active_contacts)
            running_cost = aligator.CostStack(self.space, self.nu)
            running_cost.addCost(
                "state",
                aligator.QuadraticStateCost(self.space, self.nu, np.asarray(state_refs[k], dtype=float), w_x),
            )
            running_cost.addCost(
                "control",
                aligator.QuadraticControlCost(self.space, u_ref, w_u),
            )
            if jump_task:
                self._cost_builder.add_centroidal_momentum_cost(
                    running_cost,
                    np.zeros(6, dtype=float),
                    self._cost_builder.stabilized_jump_centroidal_weights(),
                )
            if foot_refs is not None and (not jump_task or phase_schedule[k] == "stance"):
                self._cost_builder.add_foot_tracking_costs(
                    running_cost=running_cost,
                    task_name=task_name,
                    task=task,
                    step_idx=k,
                    foot_refs=foot_refs,
                    active_frame_set=active_frame_set,
                )
            if task_plan.floating_base_refs is not None and phase_schedule[k] == "flight":
                self._cost_builder.add_floating_base_tracking_cost(
                    running_cost,
                    task_plan.floating_base_refs[k],
                    weight=(
                        self._cost_builder.original_jump_base_translation_weights()
                        if jump_task
                        else max(100.0, task.state_weight * 20.0)
                    ),
                )

            stage_model = aligator.StageModel(running_cost, dyn_model)
            self._constraint_builder.add_control_constraints(stage_model)
            self._constraint_builder.add_kinematic_constraints(stage_model)
            self._constraint_builder.add_mechanical_characteristic_constraint(stage_model)
            self._constraint_builder.add_friction_cone_constraints(stage_model, active_contacts)

            if use_footplanner_refs:
                if is_plain_walk:
                    should_pin = True
                    tol_xy, tol_z = 5.0e-3, 3.0e-3
                    as_equality = True
                elif stair_task:
                    should_pin = True
                    tol_xy, tol_z = 1.0e-2, 5.0e-3
                    as_equality = True
                else:
                    should_pin = (k % 5 == 0) or (active_frame_set != prev_active_frames)
                    tol_xy, tol_z = 6.0e-2, 2.5e-2
                    as_equality = False
                if should_pin:
                    self._constraint_builder.add_hard_foot_constraints(
                        stage_model,
                        active_contacts,
                        tol_xy=tol_xy,
                        tol_z=tol_z,
                        as_equality=as_equality,
                    )
            else:
                touchdown_frames = active_frame_set - prev_active_frames if k > 0 else set()
                if len(touchdown_frames) > 0 and phase_schedule[k] == "stance":
                    self._constraint_builder.add_hard_foot_constraints(
                        stage_model, active_contacts, frame_names=touchdown_frames
                    )
            stages.append(stage_model)
            prev_active_frames = active_frame_set

        terminal_wx = self._cost_builder.terminal_state_weights(task_name, task, use_footplanner_refs, w_x)
        terminal_cost = aligator.CostStack(self.space, self.nu)
        terminal_cost.addCost(
            "terminal_state",
            aligator.QuadraticStateCost(
                self.space,
                self.nu,
                np.asarray(state_refs[horizon - 1], dtype=float),
                terminal_wx,
            ),
        )
        return aligator.TrajOptProblem(self.x0, stages, terminal_cost)

    def solve_task(self, task_name: str, seed_record: TrajectoryRunRecord | None = None) -> TaskResult:
        """Solve one task and return its converged state and control trajectories."""

        task = self.config.tasks[task_name]
        is_plain_walk = is_walk_task(task_name)
        jump_task = is_jump_task(task_name)
        task_plan = self._build_task_plan(task_name, task)
        state_refs = task_plan.state_refs
        if state_refs is None:
            state_refs = np.asarray(
                [
                    self._reference_state(task, k, len(task_plan.contact_schedule))
                    for k in range(len(task_plan.contact_schedule))
                ],
                dtype=float,
            )

        problem = self.build_problem(task_name)
        if is_plain_walk:
            solver = self._build_solver(
                tol_override=min(float(self.config.solver.tol), 1e-3),
                max_iter_override=max(int(self.config.solver.max_iter), 200),
            )
        elif jump_task:
            solver = self._build_solver(
                tol_override=float(self.config.solver.tol),
                max_iter_override=max(int(self.config.solver.max_iter), 400),
                reg_min_override=self._solver_reg_min_for_task(task_name),
            )
        else:
            solver = self._build_solver()
        self._apply_task_specific_solver_settings(solver, task_name)

        seed_task_data = self._initial_guess_builder.seed_task_data_from_record(task_name, seed_record)
        xs_init, us_init = self._initial_guess_builder.build_initial_guess(
            task_name,
            task_plan,
            state_refs,
            seed_task_data=seed_task_data,
        )

        solver.setup(problem)
        solver.run(problem, xs_init, us_init)
        result = solver.results

        return TaskResult(
            task_name=task_name,
            converged=bool(result.conv),
            iterations=int(result.num_iters),
            trajectory_cost=float(result.traj_cost),
            xs=np.asarray(result.xs),
            us=np.asarray(result.us),
        )

    def solve_all(
        self,
        task_names: list[str] | None = None,
        seed_record: TrajectoryRunRecord | None = None,
    ) -> list[TaskResult]:
        """Solve all requested tasks in sequence and return their results."""

        names = list(self.config.tasks.keys()) if task_names is None else task_names
        return [self.solve_task(name, seed_record=seed_record) for name in names]

    def build_record(
        self,
        results: list[TaskResult],
        task_metrics: dict[str, dict[str, np.ndarray | float]] | None = None,
    ) -> TrajectoryRunRecord:
        """Convert solved task results into a serializable trajectory record."""

        return self._record_exporter.build_record(results, task_metrics=task_metrics)
