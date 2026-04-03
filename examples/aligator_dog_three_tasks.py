from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from actorob.config import load_trajectory_optimizer_config
from actorob.dashboard import build_trajectory_dashboard_html
from actorob.trajectories import AligatorTrajectoryOptimizer


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "dog_aligator_minimal.toml"


def _default_output_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "logs" / "trajectories"
    record = out_dir / f"aligator_three_tasks_{stamp}.pkl"
    dashboard = out_dir / f"aligator_three_tasks_{stamp}.html"
    return record, dashboard


def main() -> None:
    default_record_path, default_dashboard_path = _default_output_paths()
    parser = argparse.ArgumentParser(description="Minimal Aligator trajectory optimization for three locomotion tasks.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to TOML config.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["walk", "upstairs", "jump_forward"],
        help="Task names from [tasks] section.",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        default=default_record_path,
        help="Path to save serialized trajectory record (.pkl).",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=default_dashboard_path,
        help="Path to save dashboard HTML.",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="If set, dashboard HTML is not generated.",
    )
    parser.add_argument(
        "--skip-meshcat",
        action="store_true",
        help="If set, meshcat simulation is not embedded into dashboard.",
    )
    parser.add_argument(
        "--meshcat-frame-stride",
        type=int,
        default=2,
        help="Take every N-th frame for meshcat animation to control HTML size.",
    )
    parser.add_argument(
        "--meshcat-fps",
        type=int,
        default=30,
        help="Meshcat animation framerate.",
    )
    args = parser.parse_args()

    config = load_trajectory_optimizer_config(args.config)
    optimizer = AligatorTrajectoryOptimizer(config)
    results = optimizer.solve_all(task_names=args.tasks)
    record = optimizer.build_record(results)
    saved_record_path = record.save(args.record_output)

    for res in results:
        final_xyz = res.xs[-1][:3]
        final_front_left_hip_pitch = res.xs[-1][8]
        final_front_left_knee_pitch = res.xs[-1][9]
        print(
            f"[{res.task_name}] conv={res.converged} iters={res.iterations} "
            f"cost={res.trajectory_cost:.6f} final_base_xyz={final_xyz} "
            f"q(front_left_hip_pitch)={final_front_left_hip_pitch:.3f} "
            f"q(front_left_knee_pitch)={final_front_left_knee_pitch:.3f}"
        )

    print(f"Trajectory record saved to: {saved_record_path}")

    if not args.skip_dashboard:
        saved_dashboard_path = build_trajectory_dashboard_html(
            record=record,
            output_path=args.dashboard_output,
            include_meshcat=(not args.skip_meshcat),
            meshcat_frame_stride=args.meshcat_frame_stride,
            meshcat_fps=args.meshcat_fps,
        )
        print(f"Dashboard HTML saved to: {saved_dashboard_path}")


if __name__ == "__main__":
    main()
