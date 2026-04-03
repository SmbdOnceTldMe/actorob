from __future__ import annotations

import argparse
from pathlib import Path

from actorob.dashboard import build_trajectory_dashboard_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Plotly dashboard HTML from saved trajectory record.")
    parser.add_argument("record", type=Path, help="Path to trajectory record (.pkl).")
    parser.add_argument("output", type=Path, help="Output HTML path.")
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

    out = build_trajectory_dashboard_from_file(
        record_path=args.record,
        output_html_path=args.output,
        include_meshcat=(not args.skip_meshcat),
        meshcat_frame_stride=args.meshcat_frame_stride,
        meshcat_fps=args.meshcat_fps,
    )
    print(f"Dashboard HTML saved to: {out}")


if __name__ == "__main__":
    main()
