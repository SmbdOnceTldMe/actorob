import pytest
from pathlib import Path

from actorob.config import load_trajectory_optimizer_config


def test_load_minimal_trajectory_optimizer_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")

    assert cfg.base.robot == "dog"
    assert Path(cfg.base.mjcf_path).exists()
    assert cfg.trajectory.dt > 0
    assert cfg.solver.num_threads == 1
    assert cfg.solver.rollout_type == "ROLLOUT_LINEAR"
    assert cfg.solver.sa_strategy == "SA_LINESEARCH_ARMIJO"
    assert cfg.solver.linear_solver_choice == "LQ_SOLVER_SERIAL"
    assert cfg.solver.filter_beta == pytest.approx(1e-5)
    assert cfg.solver.force_initial_condition is True
    assert cfg.solver.reg_min == pytest.approx(1e-6)
    assert len(cfg.contact.contact_frames_3d) == 4
    assert cfg.trajectory.use_friction_cones is True
    assert cfg.trajectory.enforce_mechanical_characteristic is True
    assert set(cfg.tasks.keys()) == {"walk", "upstairs", "jump_forward"}
    assert cfg.tasks["walk"].horizon > 0
    assert cfg.tasks["walk"].horizon == 50
    assert cfg.tasks["walk"].contact_phases == ("stance", "diag_a", "stance", "diag_b", "stance", "diag_a", "stance")
    assert cfg.tasks["walk"].touchdown_dx > 0
    assert cfg.tasks["walk"].trajectory_params is not None
    assert cfg.tasks["walk"].trajectory_params.n_steps == 3
    assert cfg.tasks["upstairs"].stairs is not None
    assert cfg.tasks["upstairs"].trajectory_params is not None
    assert cfg.tasks["upstairs"].trajectory_params.n_steps == 8
    assert cfg.tasks["upstairs"].stairs.step_count == 4
    assert cfg.tasks["upstairs"].stairs.step_height > 0
    assert cfg.tasks["upstairs"].stairs.offset_x is None
    assert cfg.tasks["upstairs"].stairs.step_length == pytest.approx(0.3)
    assert cfg.tasks["upstairs"].stairs.width == pytest.approx(0.5)
    assert cfg.tasks["upstairs"].stairs.flat_length == pytest.approx(0.6)
    assert cfg.tasks["upstairs"].horizon == 274
    assert cfg.tasks["upstairs"].contact_phases is not None
    assert cfg.tasks["upstairs"].phase_durations is not None
    assert sum(cfg.tasks["upstairs"].phase_durations) == cfg.tasks["upstairs"].horizon
    assert len(cfg.tasks["upstairs"].contact_phases) == 17
    assert cfg.tasks["jump_forward"].horizon == 220
    assert cfg.tasks["jump_forward"].contact_phases == ("stance", "flight", "stance")
    assert cfg.tasks["jump_forward"].phase_durations == (100, 20, 100)
    assert cfg.tasks["jump_forward"].apex_dz > 0
    assert cfg.tasks["jump_forward"].target_dx == pytest.approx(0.6)


def test_selected_tasks_skip_validation_for_unrequested_tasks(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    mjcf_path = (root / "robots" / "dog" / "dog.xml").resolve()
    cfg_path = tmp_path / "selected_tasks.toml"
    cfg_path.write_text(
        f"""
[base]
robot = "dog"
mjcf_path = "{mjcf_path}"

[trajectory]
dt = 0.02

[solver]
tol = 1e-4
mu_init = 1e-4
max_iter = 20
verbose = "QUIET"

[contact]
contact_frames_3d = ["front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"]

[tasks.walk]
horizon = 20

[tasks.upstairs]
[tasks.upstairs.trajectory_params]
n_steps = 6
step_time = 0.5
stair_start = 3
stair_end = 1
stair_h = 0.1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stair_end must be >= stair_start"):
        load_trajectory_optimizer_config(cfg_path)

    cfg = load_trajectory_optimizer_config(cfg_path, task_names=("walk",))

    assert set(cfg.tasks) == {"walk"}
    assert cfg.tasks["walk"].horizon == 20


def test_selected_task_names_must_exist(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    mjcf_path = (root / "robots" / "dog" / "dog.xml").resolve()
    cfg_path = tmp_path / "unknown_task.toml"
    cfg_path.write_text(
        f"""
[base]
robot = "dog"
mjcf_path = "{mjcf_path}"

[trajectory]
dt = 0.02

[solver]
tol = 1e-4
mu_init = 1e-4
max_iter = 20
verbose = "QUIET"

[contact]
contact_frames_3d = ["front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"]

[tasks.walk]
horizon = 20
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown task names"):
        load_trajectory_optimizer_config(cfg_path, task_names=("jump_forward",))


def test_task_inference_from_trajectory_params(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    mjcf_path = (root / "robots" / "dog" / "dog.xml").resolve()
    cfg_path = tmp_path / "trajectory_params.toml"
    cfg_path.write_text(
        f"""
[base]
robot = "dog"
mjcf_path = "{mjcf_path}"

[trajectory]
dt = 0.02

[solver]
tol = 1e-4
mu_init = 1e-4
max_iter = 20
verbose = "QUIET"

[contact]
contact_frames_3d = ["front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"]

[tasks.upstairs]
[tasks.upstairs.trajectory_params]
n_steps = 6
step_time = 0.5
dx = 0.6
stair_start = 1
stair_end = 3
stair_h = 0.1
""".strip(),
        encoding="utf-8",
    )

    cfg = load_trajectory_optimizer_config(cfg_path)
    upstairs = cfg.tasks["upstairs"]
    assert upstairs.horizon == 150
    assert upstairs.target_dx == pytest.approx(1.8)
    assert upstairs.target_dz == pytest.approx(0.2)
    assert upstairs.touchdown_dx == pytest.approx(0.3)
    assert upstairs.touchdown_dz == pytest.approx(0.1)
    assert upstairs.stairs is not None
    assert upstairs.stairs.step_count == 2
    assert upstairs.stairs.step_length == pytest.approx(0.3)
    assert upstairs.stairs.width == pytest.approx(0.5)
    assert upstairs.stairs.flat_length == pytest.approx(0.9)


def test_horizon_is_derived_from_explicit_phase_durations(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    mjcf_path = (root / "robots" / "dog" / "dog.xml").resolve()
    cfg_path = tmp_path / "explicit_schedule.toml"
    cfg_path.write_text(
        f"""
[base]
robot = "dog"
mjcf_path = "{mjcf_path}"

[trajectory]
dt = 0.02

[solver]
tol = 1e-4
mu_init = 1e-4
max_iter = 20
verbose = "QUIET"

[contact]
contact_frames_3d = ["front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"]

[tasks.custom_walk]
contact_phases = ["stance", "flight", "stance"]
phase_durations = [5, 3, 7]
""".strip(),
        encoding="utf-8",
    )

    cfg = load_trajectory_optimizer_config(cfg_path)

    assert cfg.tasks["custom_walk"].horizon == 15
    assert cfg.tasks["custom_walk"].phase_durations == (5, 3, 7)


def test_conflicting_horizon_and_trajectory_params_are_rejected(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    mjcf_path = (root / "robots" / "dog" / "dog.xml").resolve()
    cfg_path = tmp_path / "conflicting_horizon.toml"
    cfg_path.write_text(
        f"""
[base]
robot = "dog"
mjcf_path = "{mjcf_path}"

[trajectory]
dt = 0.02

[solver]
tol = 1e-4
mu_init = 1e-4
max_iter = 20
verbose = "QUIET"

[contact]
contact_frames_3d = ["front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"]

[tasks.walk]
horizon = 20

[tasks.walk.trajectory_params]
n_steps = 3
step_time = 0.2
ds_time = 0.05
dx = 0.4
swing_apex = 0.1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match inferred trajectory horizon=50"):
        load_trajectory_optimizer_config(cfg_path)


def test_invalid_solver_strategy_values(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    mjcf_path = (root / "robots" / "dog" / "dog.xml").resolve()
    cfg_path = tmp_path / "bad_solver.toml"
    cfg_path.write_text(
        f"""
[base]
robot = "dog"
mjcf_path = "{mjcf_path}"

[trajectory]
dt = 0.02

[solver]
tol = 1e-4
mu_init = 1e-4
max_iter = 20
verbose = "QUIET"
rollout_type = "BAD_ROLLOUT"

[contact]
contact_frames_3d = ["front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"]

[tasks.jump_forward]
[tasks.jump_forward.trajectory_params]
t_st = 10
t_ft = 4
jump_h = 0.2
jump_l = 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="solver.rollout_type"):
        load_trajectory_optimizer_config(cfg_path)
