from pathlib import Path

import aligator
import numpy as np
import pytest

from actorob.config import load_trajectory_optimizer_config
from actorob.trajectories.costs import OptimizerCostBuilder
from actorob.trajectories import AligatorTrajectoryOptimizer, TaskResult


def test_trajectory_optimizer_applies_effort_limits_even_without_velocity_metadata():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    assert np.all(np.isfinite(optimizer.rmodel.effortLimit[6:]))
    assert np.all(optimizer.rmodel.effortLimit[6:] > 0.0)
    assert np.all(np.isinf(optimizer.rmodel.velocityLimit[6:]))


def test_jump_foot_tracking_uses_stabilized_xyz_weights_under_actuator_limits():
    weights = OptimizerCostBuilder.stabilized_jump_foot_weights()
    np.testing.assert_allclose(weights, np.diag([5e3, 5e3, 5e3]))


def test_jump_centroidal_weights_use_stronger_angular_regularization_under_actuator_limits():
    weights = OptimizerCostBuilder.stabilized_jump_centroidal_weights()
    np.testing.assert_allclose(weights, np.diag([0.0, 0.0, 0.0, 6e-2, 6e-2, 6e-2]))


def test_jump_control_weights_stay_uniform_under_actuator_limits():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    control_weights = OptimizerCostBuilder(optimizer).stabilized_jump_control_weights()

    np.testing.assert_allclose(np.diag(control_weights), np.full(optimizer.nu, 1e-4))


def test_jump_solver_uses_regularization_floor_for_stability():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    assert optimizer._solver_reg_min_for_task("jump_forward") == pytest.approx(8.0e-3)
    assert optimizer._solver_reg_min_for_task("walk") == pytest.approx(cfg.solver.reg_min)


def test_walk_swing_foot_weights_emphasize_vertical_clearance():
    support_w3, swing_w3 = OptimizerCostBuilder.stabilized_walk_foot_weights()
    np.testing.assert_allclose(np.diag(support_w3), np.array([1e5, 1e5, 1e5]))
    np.testing.assert_allclose(np.diag(swing_w3), np.array([1e3, 1e3, 6e3]))


def test_upstairs_solver_uses_filter_globalization_for_stability():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    solver = optimizer._build_solver()
    optimizer._apply_task_specific_solver_settings(solver, "upstairs")

    assert solver.sa_strategy == aligator.SA_FILTER
    assert solver.rollout_type == aligator.ROLLOUT_LINEAR
    assert solver.linear_solver_choice == aligator.LQ_SOLVER_SERIAL


def test_upstairs_record_keeps_original_stair_render_metadata():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)

    horizon = cfg.tasks["upstairs"].horizon
    xs = np.repeat(optimizer.x0[None, :], horizon + 1, axis=0)
    us = np.zeros((horizon, optimizer.nu), dtype=float)
    record = optimizer.build_record(
        [
            TaskResult(
                task_name="upstairs",
                converged=True,
                iterations=0,
                trajectory_cost=0.0,
                xs=xs,
                us=us,
            )
        ]
    )

    stairs = record.tasks[0].stairs
    assert stairs is not None
    assert stairs.width == pytest.approx(0.5)
    assert stairs.start_step == cfg.tasks["upstairs"].trajectory_params.stair_start
    assert stairs.total_steps == cfg.tasks["upstairs"].trajectory_params.n_steps
    assert stairs.offset_x == pytest.approx(optimizer._stair_geometry(cfg.tasks["upstairs"])[0])
