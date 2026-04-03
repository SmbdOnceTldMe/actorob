import numpy as np
import pytest

from actorob.trajectories.footplanner import (
    FootMotionPlanner,
    FootPlannerConfig,
    PairContactPhase,
    StairProfile,
)


def make_planner(**overrides) -> FootMotionPlanner:
    if "stair_profile" not in overrides:
        overrides["stair_profile"] = StairProfile()
    config = FootPlannerConfig(
        dt=0.02,
        step_time=0.2,
        ds_time=0.05,
        swing_apex=0.1,
        initial_left=np.array([-0.3, 0.1, 0.0]),
        initial_right=np.array([-0.3, -0.1, 0.0]),
        **overrides,
    )
    return FootMotionPlanner(config)


def test_rejects_non_positive_timing_parameters():
    with pytest.raises(ValueError, match="dt must be positive"):
        FootPlannerConfig(
            dt=0.0,
            step_time=0.2,
            ds_time=0.05,
            swing_apex=0.1,
            initial_left=np.zeros(3),
            initial_right=np.zeros(3),
        )

    with pytest.raises(ValueError, match="ds_time must be smaller than step_time"):
        FootPlannerConfig(
            dt=0.02,
            step_time=0.2,
            ds_time=0.2,
            swing_apex=0.1,
            initial_left=np.zeros(3),
            initial_right=np.zeros(3),
        )


def test_generates_expected_pair_contact_phase_sequence():
    planner = make_planner()

    plan = planner.generate_plan(v_des=np.array([0.4, 0.0, 0.0]), n_steps=2, start_left=True)

    assert plan.contact_phases[: planner.init_ds_steps] == (PairContactPhase.DOUBLE,) * planner.init_ds_steps
    assert plan.contact_phases[-planner.init_ds_steps :] == (PairContactPhase.DOUBLE,) * planner.init_ds_steps
    assert plan.contact_phases.count(PairContactPhase.RIGHT) == 2 * planner.ss_steps
    assert plan.contact_phases.count(PairContactPhase.LEFT) == planner.ss_steps


def test_stair_geometry_snaps_last_footsteps_to_step_centers():
    planner = make_planner(
        stair_profile=StairProfile(
            height=0.12,
            start_step=0,
            end_step=2,
            offset_x=0.05,
            step_length=0.12,
            step_count=2,
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


def test_swing_trajectory_lifts_above_both_endpoints():
    planner = make_planner()

    swing = planner._swing_interp(
        np.array([0.0, 0.1, 0.0]),
        np.array([0.2, 0.1, 0.05]),
        ts=planner.ss_steps // 2,
    )

    assert swing[0] > 0.0
    assert swing[0] < 0.2
    assert swing[2] > 0.05


def test_final_double_support_keeps_last_foothold_instead_of_resetting_to_initial():
    planner = make_planner(
        stair_profile=StairProfile(
            height=0.12,
            start_step=0,
            end_step=2,
            offset_x=0.05,
            step_length=0.12,
            step_count=2,
        )
    )

    plan = planner.generate_plan(v_des=np.array([0.6, 0.0, 0.0]), n_steps=4, start_left=True)

    np.testing.assert_allclose(plan.left_traj[-1], np.array([0.23, 0.1, 0.24]), atol=1e-9)
    np.testing.assert_allclose(plan.right_traj[-1], np.array([0.23, -0.1, 0.24]), atol=1e-9)
