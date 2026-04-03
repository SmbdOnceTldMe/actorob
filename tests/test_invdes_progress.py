from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from actorob.invdes.progress import (
    BatchHeartbeat,
    BatchProgress,
    RunSummary,
    TqdmProgressReporter,
    format_run_summary,
)
from actorob.invdes.evaluation.types import ObjectiveWeights
from actorob.invdes.problem import OptimizationSettings


class FakeTqdmBar:
    def __init__(self) -> None:
        self.updated_by: list[int] = []
        self.postfix_calls: list[dict[str, object]] = []
        self.closed = False
        self.refresh_count = 0

    def update(self, count: int) -> None:
        self.updated_by.append(count)

    def set_postfix(self, **kwargs) -> None:
        self.postfix_calls.append(kwargs)

    def refresh(self) -> None:
        self.refresh_count += 1

    def close(self) -> None:
        self.closed = True


class TqdmProgressReporterTest(unittest.TestCase):
    def test_reporter_updates_tqdm_bar(self) -> None:
        bar = FakeTqdmBar()
        calls: list[dict[str, object]] = []

        def fake_tqdm(**kwargs):
            calls.append(kwargs)
            return bar

        fake_module = types.ModuleType("tqdm.auto")
        fake_module.tqdm = fake_tqdm

        with patch.dict(sys.modules, {"tqdm.auto": fake_module}):
            reporter = TqdmProgressReporter(description="Inverse design")
            reporter.start(total_batches=4, total_trials=16)
            reporter.advance(
                BatchProgress(
                    batch_index=1,
                    total_batches=4,
                    batch_size=4,
                    total_trials=16,
                    evaluated_trials=4,
                    completed_trials=3,
                    failed_trials=1,
                    best_value=12.345,
                    batch_duration_seconds=8.4,
                )
            )
            reporter.close()

        self.assertEqual(calls[0]["total"], 4)
        self.assertEqual(calls[0]["desc"], "Inverse design")
        self.assertEqual(calls[0]["unit"], "batch")
        self.assertEqual(bar.updated_by, [1])
        self.assertEqual(bar.postfix_calls[0]["trials"], "0/16")
        self.assertTrue(bar.postfix_calls[0]["refresh"])
        self.assertEqual(bar.postfix_calls[0]["iter_s"], "running")
        self.assertEqual(bar.postfix_calls[0]["best"], "pending")
        self.assertEqual(bar.postfix_calls[1]["trials"], "4/16")
        self.assertEqual(bar.postfix_calls[1]["iter_s"], "8.40")
        self.assertEqual(bar.postfix_calls[1]["best"], "12.35")
        self.assertEqual(bar.postfix_calls[1]["ok"], 3)
        self.assertEqual(bar.postfix_calls[1]["fail"], 1)
        self.assertFalse(bar.postfix_calls[1]["refresh"])
        self.assertEqual(bar.refresh_count, 1)
        self.assertTrue(bar.closed)

    def test_reporter_heartbeat_updates_postfix_without_advancing_bar(self) -> None:
        bar = FakeTqdmBar()

        def fake_tqdm(**kwargs):
            return bar

        fake_module = types.ModuleType("tqdm.auto")
        fake_module.tqdm = fake_tqdm

        with patch.dict(sys.modules, {"tqdm.auto": fake_module}):
            reporter = TqdmProgressReporter(description="Inverse design")
            reporter.start(total_batches=2, total_trials=8)
            reporter.heartbeat(
                BatchHeartbeat(
                    batch_index=1,
                    total_batches=2,
                    batch_size=4,
                    completed_in_batch=2,
                    total_trials=8,
                    observed_trials=2,
                    completed_trials=1,
                    failed_trials=1,
                    best_value=None,
                    elapsed_seconds=1.25,
                )
            )
            reporter.close()

        self.assertEqual(bar.updated_by, [])
        self.assertEqual(bar.postfix_calls[1]["trials"], "2/8")
        self.assertEqual(bar.postfix_calls[1]["batch"], "2/4")
        self.assertEqual(bar.postfix_calls[1]["iter_s"], "1.25")
        self.assertEqual(bar.postfix_calls[1]["best"], "pending")
        self.assertEqual(bar.postfix_calls[1]["ok"], 1)
        self.assertEqual(bar.postfix_calls[1]["fail"], 1)
        self.assertEqual(bar.refresh_count, 1)

    def test_format_run_summary_includes_key_run_statistics(self) -> None:
        summary = RunSummary(
            config_path="configs/dog_aligator_minimal.toml",
            tasks=("walk", "jump_forward"),
            parameter_count=12,
            settings=OptimizationSettings(max_iterations=2, parallelism=4, population_size=8),
            seed=12345,
            sigma=0.25,
            progress_enabled=True,
            weights=ObjectiveWeights(traj_cost=1.0, energy=2.0, friction=3.0),
        )

        rendered = format_run_summary(summary)

        self.assertIn("max iterations: 2", rendered)
        self.assertIn("total trials: 16", rendered)
        self.assertIn("population size: 8", rendered)
        self.assertIn("parallel threads: 4", rendered)
        self.assertIn("expected batches: 2", rendered)
        self.assertIn("tasks (2): walk, jump_forward", rendered)
        self.assertIn("parameters: 12", rendered)
        self.assertIn("progress: tqdm", rendered)
