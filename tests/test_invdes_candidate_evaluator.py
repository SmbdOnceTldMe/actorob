from __future__ import annotations

import pickle
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from actorob.invdes.evaluation.trajectory import TrajectoryCandidateEvaluator
from actorob.invdes.evaluation.types import PreparedCandidate
from actorob.trajectories import TrajectoryRunRecord


@dataclass(frozen=True)
class FakeBaseConfig:
    mjcf_path: str


@dataclass(frozen=True)
class FakeTrajectoryConfig:
    dt: float


@dataclass(frozen=True)
class FakeOptimizerConfig:
    base: FakeBaseConfig
    trajectory: FakeTrajectoryConfig
    tasks: dict[str, object]


@dataclass(frozen=True)
class FakeTaskResult:
    task_name: str
    converged: bool
    iterations: int
    trajectory_cost: float
    xs: list[list[float]]
    us: list[list[float]]


class FakeOptimizer:
    def __init__(self, config):
        self.config = config
        self.nq = 0
        self.rmodel = type(
            "FakeModel",
            (),
            {"names": ["universe", "root", "front_left_hip_pitch_joint", "front_right_hip_pitch_joint"]},
        )()

    def solve_all(self, task_names):
        return [
            FakeTaskResult(
                task_name="walk",
                converged=True,
                iterations=7,
                trajectory_cost=12.5,
                xs=[
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 2.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
                ],
                us=[
                    [10.0, 20.0],
                    [15.0, 25.0],
                ],
            )
        ]

    def build_record(self, results, task_metrics=None):
        return object()


class FakeCandidatePreparer:
    def __init__(self, generated_path: Path):
        self.generated_path = generated_path
        self.generated_path.write_text("<mujoco/>", encoding="utf-8")

    def prepare(self, candidate: tuple[float, ...]) -> PreparedCandidate:
        config = FakeOptimizerConfig(
            base=FakeBaseConfig(mjcf_path=str(self.generated_path)),
            trajectory=FakeTrajectoryConfig(dt=0.05),
            tasks={"walk": object()},
        )
        return PreparedCandidate(
            config=config,
            solution=candidate,
            metadata={
                "actuators": {
                    "front_left_hip_pitch_joint": object(),
                    "front_right_hip_pitch_joint": object(),
                },
                "generated_mjcf_path": str(self.generated_path),
            },
        )


def fake_actuator_metrics_fn(actuators, _torque, _velocity, dt=None):
    del dt
    active_count = sum(1 for actuator in actuators if actuator is not None)
    return {
        "E_motor_losses": float(active_count),
        "E_mech": 0.5 * float(active_count),
        "E_gearbox_losses": 0.25 * float(active_count),
    }


class TrajectoryCandidateEvaluatorTest(unittest.TestCase):
    def test_evaluator_aggregates_task_metrics_and_cleans_generated_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_path = Path(tmp_dir) / "candidate.xml"
            evaluator = TrajectoryCandidateEvaluator(
                candidate_preparer=FakeCandidatePreparer(generated_path),
                optimizer_cls=FakeOptimizer,
                actuator_metrics_fn=lambda actuators, torque, velocity, dt=None: {
                    "E_motor_losses": 2.0 * float(sum(actuator is not None for actuator in actuators)),
                    "E_mech": 1.0 * float(sum(actuator is not None for actuator in actuators)),
                    "E_gearbox_losses": 0.5 * float(sum(actuator is not None for actuator in actuators)),
                },
            )

            scenarios = evaluator.evaluate((1.0, 10.0))

            self.assertEqual(len(scenarios), 1)
            scenario = scenarios[0]
            self.assertEqual(scenario.case_name, "walk")
            self.assertEqual(scenario.mode, "WALK")
            self.assertTrue(scenario.converged)
            self.assertEqual(scenario.iterations, 7)
            self.assertEqual(scenario.traj_cost, 12.5)
            self.assertEqual(scenario.electrical_energy, 6.0)
            self.assertEqual(scenario.friction_loss, 1.0)
            self.assertFalse(generated_path.exists())

    def test_evaluate_with_record_can_keep_generated_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_path = Path(tmp_dir) / "candidate.xml"
            evaluator = TrajectoryCandidateEvaluator(
                candidate_preparer=FakeCandidatePreparer(generated_path),
                optimizer_cls=FakeOptimizer,
                actuator_metrics_fn=fake_actuator_metrics_fn,
            )

            report = evaluator.evaluate_with_record((1.0, 10.0), cleanup_generated_model=False)

            self.assertEqual(report.generated_mjcf_path, str(generated_path))
            self.assertTrue(generated_path.exists())

    def test_evaluator_is_picklable_for_process_pool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_path = Path(tmp_dir) / "candidate.xml"
            evaluator = TrajectoryCandidateEvaluator(
                candidate_preparer=FakeCandidatePreparer(generated_path),
                optimizer_cls=FakeOptimizer,
                actuator_metrics_fn=fake_actuator_metrics_fn,
            )

            restored = pickle.loads(pickle.dumps(evaluator))
            scenarios = restored.evaluate((1.0, 10.0))

            self.assertEqual(len(scenarios), 1)
            self.assertEqual(scenarios[0].case_name, "walk")

    def test_evaluator_passes_seed_record_to_optimizer_when_supported(self) -> None:
        @dataclass(frozen=True)
        class SeedAwareConfig:
            base: FakeBaseConfig
            trajectory: FakeTrajectoryConfig
            tasks: dict[str, object]

        class SeedAwareCandidatePreparer:
            def prepare(self, candidate: tuple[float, ...]) -> PreparedCandidate:
                del candidate
                return PreparedCandidate(
                    config=SeedAwareConfig(
                        base=FakeBaseConfig(mjcf_path="seed.xml"),
                        trajectory=FakeTrajectoryConfig(dt=0.05),
                        tasks={"walk": object()},
                    ),
                    solution=(1.0, 10.0),
                    metadata={"actuators": {}},
                )

        class SeedAwareOptimizer(FakeOptimizer):
            received_seed_record = None

            def solve_all(self, task_names, seed_record=None):
                del task_names
                type(self).received_seed_record = seed_record
                return super().solve_all(("walk",))

        seed_record = TrajectoryRunRecord.now(
            robot="dog",
            mjcf_path="robot.xml",
            dt=0.05,
            contact_mu=0.8,
            joint_names=("front_left_hip_pitch_joint", "front_right_hip_pitch_joint"),
            joint_position_lower_limits=np.zeros(2),
            joint_position_upper_limits=np.zeros(2),
            joint_velocity_lower_limits=np.zeros(2),
            joint_velocity_upper_limits=np.zeros(2),
            joint_torque_lower_limits=np.zeros(2),
            joint_torque_upper_limits=np.zeros(2),
            tasks=(),
        )
        evaluator = TrajectoryCandidateEvaluator(
            candidate_preparer=SeedAwareCandidatePreparer(),
            optimizer_cls=SeedAwareOptimizer,
            actuator_metrics_fn=fake_actuator_metrics_fn,
            cleanup_generated_models=False,
            seed_record=seed_record,
        )

        evaluator.evaluate((1.0, 10.0))

        self.assertIs(SeedAwareOptimizer.received_seed_record, seed_record)
