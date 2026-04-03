from pathlib import Path

from actorob.config import (
    StairConfig,
    TaskConfig,
    TrajectoryOptimizerConfig,
    TrajectoryParams,
    load_trajectory_optimizer_config,
)
from actorob.trajectories import AligatorTrajectoryOptimizer, TaskResult


def test_optimizer_exposes_generic_contact_role_map():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")

    optimizer = AligatorTrajectoryOptimizer(cfg)

    assert {"front_left", "front_right", "rear_left", "rear_right"}.issubset(set(optimizer.contact_role_map))


def test_generic_types_are_instantiable_from_public_api():
    root = Path(__file__).resolve().parents[1]
    cfg = load_trajectory_optimizer_config(root / "configs" / "dog_aligator_minimal.toml")

    assert isinstance(cfg, TrajectoryOptimizerConfig)
    assert isinstance(cfg.tasks["walk"], TaskConfig)
    assert isinstance(cfg.tasks["walk"].trajectory_params, TrajectoryParams)
    assert isinstance(cfg.tasks["upstairs"].stairs, StairConfig)
    assert TaskResult.__name__ == "TaskResult"
