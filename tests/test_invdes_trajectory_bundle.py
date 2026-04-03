from __future__ import annotations

import unittest
from unittest.mock import patch

from actorob.invdes import build_trajectory_bundle


class TrajectoryBundleTest(unittest.TestCase):
    def test_bundle_uses_nominal_initial_guess_and_source_repo_bounds(self) -> None:
        bundle = build_trajectory_bundle(
            config_path="configs/dog_aligator_minimal.toml",
            warm_start=False,
        )

        self.assertEqual(
            bundle.initial_guess,
            {
                "m_hip_roll": 1.0,
                "m_hip_pitch": 1.0,
                "m_knee_pitch": 1.0,
                "g_hip_roll": 24.0,
                "g_hip_pitch": 24.0,
                "g_knee_pitch": 24.0,
            },
        )

        parameters = {parameter.name: parameter for parameter in bundle.problem.parameters}
        self.assertEqual(parameters["m_hip_roll"].low, 0.15)
        self.assertEqual(parameters["m_hip_roll"].high, 1.5)
        self.assertEqual(parameters["g_hip_roll"].low, 10.0)
        self.assertEqual(parameters["g_hip_roll"].high, 70.0)

    def test_bundle_can_disable_nominal_warm_start(self) -> None:
        with patch(
            "actorob.invdes.trajectory_bundle._build_nominal_seed_record",
            side_effect=AssertionError("warm-start should be skipped"),
        ):
            bundle = build_trajectory_bundle(
                config_path="configs/dog_aligator_minimal.toml",
                warm_start=False,
            )

        self.assertIsNone(bundle.candidate_evaluator._seed_record)
