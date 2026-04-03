from __future__ import annotations

import unittest

from actorob.invdes.evaluation.batch import BatchEvaluator
from actorob.invdes.evaluation.types import ObjectiveWeights, ScenarioEvaluation
from actorob.invdes.problem import CompletedTrial, FailedTrial


class StubCandidateEvaluator:
    def __init__(self, responses):
        self._responses = list(responses)

    def evaluate(self, candidate: tuple[float, ...]):
        del candidate
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class BatchEvaluatorTest(unittest.TestCase):
    def test_reuses_first_batch_statistics_for_following_batches(self) -> None:
        candidate_evaluator = StubCandidateEvaluator(
            [
                [
                    ScenarioEvaluation(
                        mode="WALK", traj_cost=0.0, electrical_energy=0.0, friction_loss=0.0, converged=True
                    )
                ],
                [
                    ScenarioEvaluation(
                        mode="WALK", traj_cost=2.0, electrical_energy=2.0, friction_loss=2.0, converged=True
                    )
                ],
                [
                    ScenarioEvaluation(
                        mode="WALK", traj_cost=3.0, electrical_energy=3.0, friction_loss=3.0, converged=True
                    )
                ],
            ]
        )
        evaluator = BatchEvaluator(candidate_evaluator, ObjectiveWeights(traj_cost=1.0, energy=1.0, friction=1.0))

        first_batch = evaluator.evaluate(((0.0,), (1.0,)), parallelism=2)
        second_batch = evaluator.evaluate(((2.0,),), parallelism=1)

        self.assertIsInstance(first_batch[0], CompletedTrial)
        self.assertIsInstance(first_batch[1], CompletedTrial)
        self.assertAlmostEqual(first_batch[0].value, -2.1213203435596424)
        self.assertAlmostEqual(first_batch[1].value, 2.1213203435596424)
        self.assertIsInstance(second_batch[0], CompletedTrial)
        self.assertAlmostEqual(second_batch[0].value, 4.242640687119285)

    def test_adds_penalty_for_non_converged_scenario(self) -> None:
        candidate_evaluator = StubCandidateEvaluator(
            [
                [
                    ScenarioEvaluation(
                        mode="WALK", traj_cost=1.0, electrical_energy=1.0, friction_loss=1.0, converged=False
                    )
                ]
            ]
        )
        evaluator = BatchEvaluator(candidate_evaluator, ObjectiveWeights(traj_cost=1.0, energy=1.0, friction=1.0))

        batch = evaluator.evaluate(((1.0,),), parallelism=1)

        self.assertIsInstance(batch[0], CompletedTrial)
        self.assertEqual(batch[0].value, 10.0)

    def test_marks_candidate_failed_when_candidate_evaluator_raises(self) -> None:
        candidate_evaluator = StubCandidateEvaluator([RuntimeError("solver exploded")])
        evaluator = BatchEvaluator(candidate_evaluator, ObjectiveWeights())

        batch = evaluator.evaluate(((5.0,),), parallelism=1)

        self.assertIsInstance(batch[0], FailedTrial)
        self.assertIn("solver exploded", batch[0].error)

    def test_regularizes_tiny_first_batch_std_to_avoid_exploding_z_scores(self) -> None:
        candidate_evaluator = StubCandidateEvaluator(
            [
                [
                    ScenarioEvaluation(
                        mode="JUMP_FORWARD",
                        traj_cost=1000.0,
                        electrical_energy=100.0,
                        friction_loss=10.0,
                        converged=True,
                    )
                ],
                [
                    ScenarioEvaluation(
                        mode="JUMP_FORWARD",
                        traj_cost=1000.1,
                        electrical_energy=100.1,
                        friction_loss=10.1,
                        converged=True,
                    )
                ],
                [
                    ScenarioEvaluation(
                        mode="JUMP_FORWARD",
                        traj_cost=1300.0,
                        electrical_energy=120.0,
                        friction_loss=12.0,
                        converged=True,
                    )
                ],
            ]
        )
        evaluator = BatchEvaluator(
            candidate_evaluator,
            ObjectiveWeights(traj_cost=1.0, energy=0.0, friction=0.0),
            min_std_fraction=0.05,
        )

        evaluator.evaluate(((0.0,), (1.0,)), parallelism=2)
        second_batch = evaluator.evaluate(((2.0,),), parallelism=1)

        self.assertIsInstance(second_batch[0], CompletedTrial)
        self.assertLess(second_batch[0].value, 10.0)
