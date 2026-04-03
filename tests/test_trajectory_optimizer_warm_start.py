from pathlib import Path

import numpy as np

from actorob.config import load_trajectory_optimizer_config
from actorob.trajectories import AligatorTrajectoryOptimizer, JointTrajectoryData, TrajectoryRunRecord


def _seed_task_data_from_guess(
    optimizer,
    task_name: str,
    xs: list[np.ndarray],
    us: list[np.ndarray],
) -> JointTrajectoryData:
    xs_arr = np.asarray(xs, dtype=float)
    us_arr = np.asarray(us, dtype=float)
    qs = xs_arr[:, : optimizer.nq]
    dqs = xs_arr[:, optimizer.nq :]
    state_time = np.arange(xs_arr.shape[0], dtype=float) * optimizer.config.trajectory.dt
    control_time = np.arange(us_arr.shape[0], dtype=float) * optimizer.config.trajectory.dt
    return JointTrajectoryData(
        task_name=task_name,
        converged=True,
        iterations=0,
        trajectory_cost=0.0,
        state_time=state_time,
        control_time=control_time,
        floating_base_coordinates=qs[:, :7],
        floating_base_velocities=dqs[:, :6],
        joint_positions=qs[:, 7:],
        joint_velocities=dqs[:, 6:],
        joint_torques=us_arr,
    )


def test_jump_initial_guess_tracks_floating_base_refs():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)
    guess_builder = optimizer._initial_guess_builder

    jump = cfg.tasks["jump_forward"]
    task_plan = optimizer._build_task_plan("jump_forward", jump)
    xs_init, us_init = guess_builder.build_initial_guess("jump_forward", task_plan, task_plan.state_refs)

    assert len(xs_init) == jump.horizon + 1
    assert len(us_init) == jump.horizon
    np.testing.assert_allclose(xs_init[0], optimizer.x0)
    np.testing.assert_allclose(xs_init[1], task_plan.state_refs[0])
    np.testing.assert_allclose(xs_init[-1], task_plan.state_refs[-1])


def test_walk_seeded_initial_guess_blends_default_and_saved_solution():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)
    guess_builder = optimizer._initial_guess_builder

    walk = cfg.tasks["walk"]
    task_plan = optimizer._build_task_plan("walk", walk)
    xs_default, us_default = guess_builder.build_initial_guess("walk", task_plan, task_plan.state_refs)
    xs_seed = [x.copy() + 2.0 for x in xs_default]
    us_seed = [np.ones_like(u) * 4.0 for u in us_default]
    seed_task_data = _seed_task_data_from_guess(optimizer, "walk", xs_seed, us_seed)

    xs_hot, us_hot = guess_builder.build_initial_guess(
        "walk", task_plan, task_plan.state_refs, seed_task_data=seed_task_data
    )

    np.testing.assert_allclose(xs_hot[0], optimizer.x0)
    np.testing.assert_allclose(xs_hot[1], 0.5 * (xs_default[1] + xs_seed[1]))
    np.testing.assert_allclose(us_hot[0], 0.5 * us_seed[0])


def test_jump_seeded_initial_guess_blends_states_and_zeros_controls():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)
    guess_builder = optimizer._initial_guess_builder

    jump = cfg.tasks["jump_forward"]
    task_plan = optimizer._build_task_plan("jump_forward", jump)
    xs_default, us_default = guess_builder.build_initial_guess("jump_forward", task_plan, task_plan.state_refs)
    xs_seed = [x.copy() + 3.0 for x in xs_default]
    us_seed = [np.ones_like(u) * 5.0 for u in us_default]
    seed_task_data = _seed_task_data_from_guess(optimizer, "jump_forward", xs_seed, us_seed)

    xs_hot, us_hot = guess_builder.build_initial_guess(
        "jump_forward",
        task_plan,
        task_plan.state_refs,
        seed_task_data=seed_task_data,
    )

    np.testing.assert_allclose(xs_hot[0], optimizer.x0)
    np.testing.assert_allclose(xs_hot[1], 0.5 * (xs_default[1] + xs_seed[1]))
    np.testing.assert_allclose(us_hot[0], np.zeros_like(us_seed[0]))


def test_upstairs_seeded_initial_guess_replays_saved_solution():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)
    guess_builder = optimizer._initial_guess_builder

    upstairs = cfg.tasks["upstairs"]
    task_plan = optimizer._build_task_plan("upstairs", upstairs)
    xs_default, us_default = guess_builder.build_initial_guess("upstairs", task_plan, task_plan.state_refs)
    xs_seed = [x.copy() + 1.5 for x in xs_default]
    us_seed = [np.ones_like(u) * 2.5 for u in us_default]
    seed_task_data = _seed_task_data_from_guess(optimizer, "upstairs", xs_seed, us_seed)

    xs_hot, us_hot = guess_builder.build_initial_guess(
        "upstairs",
        task_plan,
        task_plan.state_refs,
        seed_task_data=seed_task_data,
    )

    np.testing.assert_allclose(xs_hot[0], optimizer.x0)
    np.testing.assert_allclose(xs_hot[1], xs_seed[1])
    np.testing.assert_allclose(us_hot[0], us_seed[0])


def test_seed_record_lookup_rejects_mismatched_dt():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")
    optimizer = AligatorTrajectoryOptimizer(cfg)
    guess_builder = optimizer._initial_guess_builder

    walk = cfg.tasks["walk"]
    task_plan = optimizer._build_task_plan("walk", walk)
    xs_default, us_default = guess_builder.build_initial_guess("walk", task_plan, task_plan.state_refs)
    seed_task_data = _seed_task_data_from_guess(optimizer, "walk", xs_default, us_default)
    record = TrajectoryRunRecord.now(
        robot=cfg.base.robot,
        mjcf_path=cfg.base.mjcf_path,
        dt=cfg.trajectory.dt * 2.0,
        contact_mu=cfg.contact.contact_mu,
        joint_names=tuple(optimizer.rmodel.names[2:]),
        joint_position_lower_limits=np.zeros(optimizer.nu),
        joint_position_upper_limits=np.zeros(optimizer.nu),
        joint_velocity_lower_limits=np.zeros(optimizer.nu),
        joint_velocity_upper_limits=np.zeros(optimizer.nu),
        joint_torque_lower_limits=np.zeros(optimizer.nu),
        joint_torque_upper_limits=np.zeros(optimizer.nu),
        tasks=(seed_task_data,),
    )

    assert guess_builder.seed_task_data_from_record("walk", record) is None
