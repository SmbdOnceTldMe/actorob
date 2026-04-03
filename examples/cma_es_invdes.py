"""Minimal inverse-design example for trajectory tasks via Optuna + CMA-ES."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from actorob.dashboard import build_inverse_design_dashboard_html
from actorob.invdes import (
    InverseDesignRunRecord,
    ObjectiveWeights,
    OptimizationResult,
    OptimizationSettings,
    OptunaCmaEsStudyFactory,
    ParallelAskTellOptimizer,
    RunSummary,
    TqdmProgressReporter,
    build_trajectory_bundle,
    default_actuator_design_variables,
    format_run_summary,
)
from actorob.models import copy_mjcf_with_resolved_assets


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "dog_aligator_minimal.toml"


def _default_output_paths() -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    return out_dir / f"invdes_run_{stamp}.pkl", out_dir / f"invdes_run_{stamp}.html"


def _default_best_model_path(run_output: Path) -> Path:
    return run_output.with_suffix(".mjcf.xml")


def _best_model_output_path(run_output: Path, best_model_output: Path | None) -> Path:
    return (
        (_default_best_model_path(run_output) if best_model_output is None else best_model_output)
        .expanduser()
        .resolve()
    )


def _print_failed_trial_errors(result: OptimizationResult) -> None:
    if not result.failed_trials:
        return

    grouped_errors: dict[str, int] = {}
    ordered_errors: list[str] = []
    for failed_trial in result.failed_trials:
        error = failed_trial.error.strip() or "unknown error"
        if error not in grouped_errors:
            grouped_errors[error] = 0
            ordered_errors.append(error)
        grouped_errors[error] += 1

    print("failed_trial_errors:")
    for index, error in enumerate(ordered_errors, start=1):
        count = grouped_errors[error]
        suffix = "" if count == 1 else f" (count={count})"
        print(f"  [{index}] {error}{suffix}")


def _materialize_best_candidate(
    candidate_report,
    *,
    run_output: Path,
    best_model_output: Path | None,
) -> tuple[object, tuple[object, ...]]:
    best_record = candidate_report.trajectory_record
    best_scenarios = tuple(candidate_report.scenarios)
    generated_mjcf_path = (
        None
        if candidate_report.generated_mjcf_path is None
        else Path(candidate_report.generated_mjcf_path).expanduser().resolve()
    )
    if generated_mjcf_path is not None and generated_mjcf_path.exists():
        model_output = _best_model_output_path(run_output, best_model_output)
        copy_mjcf_with_resolved_assets(generated_mjcf_path, model_output)
        best_record = replace(best_record, mjcf_path=str(model_output))
        generated_mjcf_path.unlink()
    return best_record, best_scenarios


def _build_run_record(
    *,
    created_at_utc: str,
    args: argparse.Namespace,
    settings: OptimizationSettings,
    weights: ObjectiveWeights,
    result: OptimizationResult,
    normalization_stats: dict[str, dict[str, dict[str, float]]] | None,
    best_record: object | None,
    best_scenarios: tuple[object, ...],
) -> InverseDesignRunRecord:
    return InverseDesignRunRecord(
        config_path=str(args.config),
        task_names=tuple(args.tasks),
        created_at_utc=created_at_utc,
        settings=settings,
        seed=args.seed,
        sigma0=args.sigma,
        weights=weights,
        result=result,
        best_trajectory_record=best_record,
        normalization_stats=normalization_stats,
        best_scenarios=best_scenarios,
    )


class _PeriodicDashboardWriter:
    """Persist intermediate inverse-design artifacts during optimization."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        bundle,
        settings: OptimizationSettings,
        weights: ObjectiveWeights,
        created_at_utc: str,
    ) -> None:
        self._args = args
        self._bundle = bundle
        self._settings = settings
        self._weights = weights
        self._created_at_utc = created_at_utc
        self._cached_best_params: tuple[float, ...] | None = None
        self._cached_best_record: object | None = None
        self._cached_best_scenarios: tuple[object, ...] = ()

    def maybe_write_checkpoint(self, result: OptimizationResult) -> None:
        update_every = int(self._args.dashboard_update_every)
        if self._args.skip_dashboard or update_every <= 0:
            return
        if result.history is None:
            raise RuntimeError("Optimization did not produce run history.")

        batch_count = len(result.history.batches)
        if batch_count == 0 or batch_count % update_every != 0:
            return

        try:
            _, saved_run_path, dashboard_path = self._write_outputs(
                result,
                render_dashboard=True,
                include_meshcat=False,
            )
        except Exception as exc:
            print(f"warning: intermediate dashboard update failed after iteration {batch_count}: {exc}")
            return

        print(f"checkpoint_iteration={batch_count}")
        print(f"checkpoint_run_record={saved_run_path}")
        if dashboard_path is not None:
            print(f"checkpoint_dashboard_html={dashboard_path}")

    def finalize(self, result: OptimizationResult) -> tuple[InverseDesignRunRecord, Path, Path | None]:
        return self._write_outputs(
            result,
            render_dashboard=(not self._args.skip_dashboard),
            include_meshcat=(not self._args.skip_meshcat),
        )

    def _write_outputs(
        self,
        result: OptimizationResult,
        *,
        render_dashboard: bool,
        include_meshcat: bool,
    ) -> tuple[InverseDesignRunRecord, Path, Path | None]:
        self._refresh_best_candidate(result.best_params)
        run_record = _build_run_record(
            created_at_utc=self._created_at_utc,
            args=self._args,
            settings=self._settings,
            weights=self._weights,
            result=result,
            normalization_stats=self._bundle.batch_evaluator.stats_snapshot(),
            best_record=self._cached_best_record,
            best_scenarios=self._cached_best_scenarios,
        )
        saved_run_path = run_record.save(self._args.run_output)
        dashboard_path = None
        if render_dashboard:
            dashboard_path = build_inverse_design_dashboard_html(
                run_record,
                self._args.dashboard_output,
                include_meshcat=include_meshcat,
                meshcat_frame_stride=self._args.meshcat_frame_stride,
                meshcat_fps=self._args.meshcat_fps,
            )
        return run_record, saved_run_path, dashboard_path

    def _refresh_best_candidate(self, best_params: tuple[float, ...] | None) -> None:
        if best_params is None:
            self._cached_best_params = None
            self._cached_best_record = None
            self._cached_best_scenarios = ()
            return
        if self._cached_best_params == best_params and self._cached_best_record is not None:
            return

        best_candidate_report = self._bundle.candidate_evaluator.evaluate_with_record(
            best_params,
            cleanup_generated_model=False,
        )
        best_record, best_scenarios = _materialize_best_candidate(
            best_candidate_report,
            run_output=Path(self._args.run_output).expanduser().resolve(),
            best_model_output=self._args.best_model_output,
        )
        self._cached_best_params = tuple(best_params)
        self._cached_best_record = best_record
        self._cached_best_scenarios = best_scenarios


def parse_args() -> argparse.Namespace:
    default_record_path, default_dashboard_path = _default_output_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_default_config_path(), help="Path to trajectory optimizer TOML config."
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["walk", "upstairs", "jump_forward"],
        help="Task names from the config to include in inverse design.",
    )
    parser.add_argument("--max-iterations", type=int, default=4, help="Number of CMA-ES generations to run.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes for evaluation.")
    parser.add_argument("--population", type=int, default=4, help="CMA-ES population size per generation.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for CMA-ES.")
    parser.add_argument("--sigma", type=float, default=0.25, help="Initial CMA-ES step size.")
    parser.add_argument("--traj-cost", type=float, default=1.0, help="Weight of trajectory cost in fitness.")
    parser.add_argument("--energy", type=float, default=1.0, help="Weight of electrical energy in fitness.")
    parser.add_argument("--friction", type=float, default=1.0, help="Weight of friction loss in fitness.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    parser.add_argument(
        "--run-output", type=Path, default=default_record_path, help="Path to save inverse-design run record (.pkl)."
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=default_dashboard_path,
        help="Path to save inverse-design dashboard HTML.",
    )
    parser.add_argument(
        "--best-model-output",
        type=Path,
        default=None,
        help="Path to save best-candidate MJCF used by meshcat/dashboard.",
    )
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip HTML dashboard generation.")
    parser.add_argument(
        "--dashboard-update-every",
        type=int,
        default=0,
        help="If > 0, rewrite the run record and dashboard every N CMA-ES generations during optimization.",
    )
    parser.add_argument("--skip-meshcat", action="store_true", help="Skip embedded meshcat in HTML dashboard.")
    parser.add_argument(
        "--meshcat-frame-stride", type=int, default=2, help="Take every N-th frame for meshcat animation."
    )
    parser.add_argument("--meshcat-fps", type=int, default=30, help="Meshcat animation frame rate.")
    parser.add_argument(
        "--no-warm-start", action="store_true", help="Disable nominal warm-start trajectory precomputation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dashboard_update_every < 0:
        raise ValueError("--dashboard-update-every must be non-negative.")

    weights = ObjectiveWeights(
        traj_cost=args.traj_cost,
        energy=args.energy,
        friction=args.friction,
    )
    settings = OptimizationSettings(
        max_iterations=args.max_iterations, parallelism=args.workers, population_size=args.population
    )
    parameter_count = 2 * len(default_actuator_design_variables())
    print(
        format_run_summary(
            RunSummary(
                config_path=str(args.config),
                tasks=tuple(args.tasks),
                parameter_count=parameter_count,
                settings=settings,
                seed=args.seed,
                sigma=args.sigma,
                progress_enabled=not args.no_progress,
                weights=weights,
            )
        )
    )
    bundle = build_trajectory_bundle(
        config_path=args.config,
        task_names=args.tasks,
        weights=weights,
        warm_start=(not args.no_warm_start),
    )
    sampler_kwargs = {"sigma0": args.sigma, "x0": bundle.initial_guess}
    optimizer = ParallelAskTellOptimizer(
        OptunaCmaEsStudyFactory(seed=args.seed, popsize=settings.population_size, sampler_kwargs=sampler_kwargs),
    )
    progress_reporter = None if args.no_progress else TqdmProgressReporter(description="Inverse design")
    checkpoint_writer = _PeriodicDashboardWriter(
        args=args,
        bundle=bundle,
        settings=settings,
        weights=weights,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    result = optimizer.optimize(
        bundle.problem,
        settings,
        batch_evaluator=bundle.batch_evaluator,
        progress_reporter=progress_reporter,
        batch_complete_callback=checkpoint_writer.maybe_write_checkpoint,
    )
    print(f"best_value={result.best_value}")
    print(f"best_params={result.best_params}")
    print(f"completed_trials={result.completed_trials_count}")
    print(f"failed_trials={result.failed_trials_count}")
    _print_failed_trial_errors(result)

    if result.history is None:
        raise RuntimeError("Optimization did not produce run history.")

    _, saved_run_path, dashboard_path = checkpoint_writer.finalize(result)
    print(f"run_record={saved_run_path}")
    if dashboard_path is not None:
        print(f"dashboard_html={dashboard_path}")


if __name__ == "__main__":
    main()
