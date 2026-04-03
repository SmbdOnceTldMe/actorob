from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import examples.cma_es_invdes as invdes_example
from actorob.invdes.history import TrialHistoryEntry
from actorob.invdes.evaluation.types import ObjectiveWeights, ScenarioEvaluation
from actorob.invdes.history import BatchHistoryEntry, OptimizationHistory
from actorob.invdes.problem import FailedTrial
from actorob.invdes.problem import OptimizationResult, OptimizationSettings
from actorob.invdes.record import InverseDesignRunRecord


class _FakeCandidateEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[float, ...]] = []

    def evaluate_with_record(self, candidate: tuple[float, ...], *, cleanup_generated_model: bool | None = None):
        self.calls.append(tuple(candidate))
        return SimpleNamespace(
            scenarios=(
                ScenarioEvaluation(
                    case_name="walk",
                    mode="WALK",
                    traj_cost=float(candidate[0]),
                    electrical_energy=0.5,
                    friction_loss=0.1,
                    converged=True,
                    iterations=3,
                ),
            ),
            trajectory_record=SimpleNamespace(label=f"candidate={candidate}"),
            generated_mjcf_path=None,
        )


class _FakeBatchEvaluator:
    def stats_snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            "WALK": {
                "traj_cost": {"mean": 1.0, "std": 0.1},
                "electrical_energy": {"mean": 0.5, "std": 0.1},
                "friction_loss": {"mean": 0.1, "std": 0.05},
            }
        }


def _make_result(
    batch_count: int,
    *,
    best_params: tuple[float, ...],
    best_value: float,
) -> OptimizationResult:
    history = OptimizationHistory(
        parameter_names=("mass",),
        batches=tuple(
            BatchHistoryEntry(
                batch_index=index,
                total_batches=4,
                batch_size=1,
                evaluated_trials=index,
                completed_trials=index,
                failed_trials=0,
                duration_seconds=0.1,
                best_value=best_value,
                generation=index - 1,
                sigma=0.25,
                trials=(),
            )
            for index in range(1, batch_count + 1)
        ),
    )
    return OptimizationResult(
        best_params=best_params,
        best_value=best_value,
        completed_trials=(),
        failed_trials=(),
        history=history,
    )


def test_periodic_dashboard_writer_updates_on_configured_cadence(tmp_path: Path):
    args = SimpleNamespace(
        config=tmp_path / "config.toml",
        tasks=["walk"],
        seed=123,
        sigma=0.25,
        run_output=tmp_path / "run.pkl",
        dashboard_output=tmp_path / "dashboard.html",
        best_model_output=None,
        skip_dashboard=False,
        dashboard_update_every=2,
        skip_meshcat=False,
        meshcat_frame_stride=2,
        meshcat_fps=30,
    )
    bundle = SimpleNamespace(
        candidate_evaluator=_FakeCandidateEvaluator(),
        batch_evaluator=_FakeBatchEvaluator(),
    )
    writer = invdes_example._PeriodicDashboardWriter(
        args=args,
        bundle=bundle,
        settings=OptimizationSettings(max_iterations=4, parallelism=1, population_size=1),
        weights=ObjectiveWeights(),
        created_at_utc="2026-04-02T12:00:00+00:00",
    )

    dashboard_calls: list[tuple[float | None, bool]] = []

    def _fake_dashboard_builder(
        run, output_path, *, include_meshcat: bool, meshcat_frame_stride: int, meshcat_fps: int
    ):
        output = Path(output_path)
        output.write_text(f"best_value={run.best_value}", encoding="utf-8")
        dashboard_calls.append((run.best_value, include_meshcat))
        return output

    with patch.object(invdes_example, "build_inverse_design_dashboard_html", side_effect=_fake_dashboard_builder):
        writer.maybe_write_checkpoint(_make_result(1, best_params=(2.0,), best_value=2.0))
        assert not args.run_output.exists()
        assert bundle.candidate_evaluator.calls == []

        writer.maybe_write_checkpoint(_make_result(2, best_params=(2.0,), best_value=2.0))
        assert args.run_output.exists()
        checkpoint_run = InverseDesignRunRecord.load(args.run_output)
        assert checkpoint_run.best_value == 2.0
        assert bundle.candidate_evaluator.calls == [(2.0,)]
        assert dashboard_calls == [(2.0, False)]

        writer.maybe_write_checkpoint(_make_result(4, best_params=(2.0,), best_value=2.0))
        assert bundle.candidate_evaluator.calls == [(2.0,)]
        assert dashboard_calls == [(2.0, False), (2.0, False)]

        _, saved_run_path, dashboard_path = writer.finalize(_make_result(4, best_params=(1.0,), best_value=1.0))

    final_run = InverseDesignRunRecord.load(saved_run_path)
    assert final_run.best_value == 1.0
    assert bundle.candidate_evaluator.calls == [(2.0,), (1.0,)]
    assert dashboard_path == args.dashboard_output
    assert dashboard_calls == [(2.0, False), (2.0, False), (1.0, True)]


def test_print_failed_trial_errors_groups_duplicate_messages(capsys):
    result = OptimizationResult(
        best_params=None,
        best_value=None,
        completed_trials=(),
        failed_trials=(
            FailedTrial(params=(1.0,), error="solver exploded"),
            FailedTrial(params=(2.0,), error="solver exploded"),
            FailedTrial(params=(3.0,), error="bad config"),
        ),
        history=OptimizationHistory(
            parameter_names=("mass",),
            batches=(
                BatchHistoryEntry(
                    batch_index=1,
                    total_batches=1,
                    batch_size=3,
                    evaluated_trials=3,
                    completed_trials=0,
                    failed_trials=3,
                    duration_seconds=0.1,
                    best_value=None,
                    generation=0,
                    sigma=0.25,
                    trials=(
                        TrialHistoryEntry(
                            trial_number=0,
                            params=(1.0,),
                            status="fail",
                            value=None,
                            error="solver exploded",
                            generation=0,
                            sigma=0.25,
                        ),
                    ),
                ),
            ),
        ),
    )

    invdes_example._print_failed_trial_errors(result)

    captured = capsys.readouterr()
    assert "failed_trial_errors:" in captured.out
    assert "[1] solver exploded (count=2)" in captured.out
    assert "[2] bad config" in captured.out
