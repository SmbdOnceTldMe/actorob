from pathlib import Path

import numpy as np
import pytest

from actorob.config import load_trajectory_optimizer_config
from actorob.trajectories import AligatorTrajectoryOptimizer
from actorob.trajectories.footplanner import FootMotionPlanner, FootPlannerConfig, StairProfile


def test_walk_and_upstairs_use_footplanner_horizon_from_trajectory_params():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    walk = cfg.tasks["walk"]
    walk_plan = optimizer._build_task_plan("walk", walk)
    walk_schedule = walk_plan.contact_schedule
    walk_phases = walk_plan.phase_schedule
    walk_refs = walk_plan.foot_refs
    walk_ss_steps = max(
        1,
        int(
            round(
                max(walk.trajectory_params.step_time - walk.trajectory_params.ds_time, cfg.trajectory.dt)
                / cfg.trajectory.dt
            )
        ),
    )
    walk_ds_steps = max(1, int(round(max(walk.trajectory_params.ds_time, cfg.trajectory.dt) / cfg.trajectory.dt)))
    walk_init_ds_steps = 3 * walk_ds_steps
    expected_walk_horizon = (
        2 * walk_init_ds_steps
        + (walk.trajectory_params.n_steps + 1) * walk_ss_steps
        + walk.trajectory_params.n_steps * walk_ds_steps
    )
    assert len(walk_schedule) == expected_walk_horizon
    assert walk_refs is not None
    assert walk_refs["front_left_foot"].shape == (expected_walk_horizon, 3)
    assert {"stance", "diag_a", "diag_b"}.issubset(set(walk_phases))

    upstairs = cfg.tasks["upstairs"]
    upstairs_plan = optimizer._build_task_plan("upstairs", upstairs)
    upstairs_schedule = upstairs_plan.contact_schedule
    upstairs_phases = upstairs_plan.phase_schedule
    upstairs_refs = upstairs_plan.foot_refs
    upstairs_ss_steps = max(
        1,
        int(
            round(
                max(upstairs.trajectory_params.step_time - upstairs.trajectory_params.ds_time, cfg.trajectory.dt)
                / cfg.trajectory.dt
            )
        ),
    )
    upstairs_ds_steps = max(
        1, int(round(max(upstairs.trajectory_params.ds_time, cfg.trajectory.dt) / cfg.trajectory.dt))
    )
    upstairs_init_ds_steps = 3 * upstairs_ds_steps
    expected_upstairs_horizon = (
        2 * upstairs_init_ds_steps
        + (upstairs.trajectory_params.n_steps + 1) * upstairs_ss_steps
        + upstairs.trajectory_params.n_steps * upstairs_ds_steps
    )
    assert len(upstairs_schedule) == expected_upstairs_horizon
    assert upstairs_refs is not None
    assert upstairs_refs["front_left_foot"].shape == (expected_upstairs_horizon, 3)
    assert {"stance", "diag_a", "diag_b"}.issubset(set(upstairs_phases))


def test_stair_end_is_treated_as_exclusive_in_footplanner():
    planner = FootMotionPlanner(
        FootPlannerConfig(
            dt=0.02,
            step_time=0.2,
            ds_time=0.05,
            swing_apex=0.1,
            initial_left=np.array([-0.3, 0.1, 0.0]),
            initial_right=np.array([-0.3, -0.1, 0.0]),
            stair_profile=StairProfile(
                height=0.12,
                start_step=0,
                end_step=2,
                offset_x=0.05,
                step_length=0.12,
                step_count=2,
            ),
        )
    )

    left_stance, right_stance = planner._generate_stance_points(
        desired_velocity=np.array([0.6, 0.0, 0.0]),
        n_steps=4,
        start_left=True,
    )

    assert left_stance[0][2] == pytest.approx(0.0)
    assert right_stance[0][2] == pytest.approx(0.0)
    assert left_stance[1][2] == pytest.approx(0.0)
    assert right_stance[1][2] == pytest.approx(0.0)
    assert left_stance[-1][0] == pytest.approx(0.23)
    assert right_stance[-1][0] == pytest.approx(0.23)
    assert left_stance[-1][2] == pytest.approx(0.24)
    assert right_stance[-1][2] == pytest.approx(0.24)


def test_jump_plan_uses_flight_phase_and_shifted_landing_targets():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    jump = cfg.tasks["jump_forward"]
    jump_plan = optimizer._build_task_plan("jump_forward", jump)
    jump_schedule = jump_plan.contact_schedule
    active_frames = jump_plan.active_frames_schedule
    jump_phases = jump_plan.phase_schedule
    jump_refs = jump_plan.foot_refs

    assert len(jump_schedule) == jump.horizon
    t_st = int(jump.trajectory_params.t_st)
    t_ft = int(jump.trajectory_params.t_ft)
    assert jump_phases == ["stance"] * t_st + ["flight"] * t_ft + ["stance"] * t_st
    assert all(len(frames) == 4 for frames in active_frames[:t_st])
    assert all(len(frames) == 0 for frames in active_frames[t_st : t_st + t_ft])
    assert all(len(frames) == 4 for frames in active_frames[t_st + t_ft :])
    assert jump_refs is not None
    assert jump_refs["front_left_foot"][0][0] == pytest.approx(jump_refs["front_left_foot"][t_st - 1][0])
    assert jump_refs["front_left_foot"][t_st][0] == pytest.approx(jump_refs["front_left_foot"][0][0])
    assert jump_refs["front_left_foot"][t_st + t_ft - 1][0] > jump_refs["front_left_foot"][0][0]
    assert jump_refs["front_left_foot"][t_st + t_ft - 1][0] == pytest.approx(jump_refs["front_left_foot"][-1][0])


def test_jump_plan_adds_short_rear_landing_compliance_window():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    jump = cfg.tasks["jump_forward"]
    task_plan = optimizer._build_task_plan("jump_forward", jump)
    t_st = int(jump.trajectory_params.t_st)
    t_ft = int(jump.trajectory_params.t_ft)
    landing_start = t_st + t_ft
    window = 7

    rear_left_hip_idx = optimizer.joint_to_q_index["rear_left_hip_pitch_joint"]
    rear_left_knee_idx = optimizer.joint_to_q_index["rear_left_knee_pitch_joint"]
    nominal_hip = optimizer.x0[rear_left_hip_idx]
    nominal_knee = optimizer.x0[rear_left_knee_idx]

    assert task_plan.state_refs is not None
    np.testing.assert_allclose(task_plan.state_refs[landing_start - 1], optimizer.x0)
    assert task_plan.state_refs[landing_start, rear_left_hip_idx] == pytest.approx(nominal_hip + 0.08)
    assert task_plan.state_refs[landing_start, rear_left_knee_idx] == pytest.approx(nominal_knee - 0.10)
    assert task_plan.state_refs[landing_start + window - 1, rear_left_hip_idx] == pytest.approx(
        nominal_hip + (0.08 / 7.0)
    )
    assert task_plan.state_refs[landing_start + window - 1, rear_left_knee_idx] == pytest.approx(
        nominal_knee - (0.10 / 7.0)
    )
    assert task_plan.state_refs[landing_start + window, rear_left_hip_idx] == pytest.approx(nominal_hip)
    assert task_plan.state_refs[landing_start + window, rear_left_knee_idx] == pytest.approx(nominal_knee)
