from __future__ import annotations

from pathlib import Path

from actorob.trajectories import TrajectoryRunRecord
from ..meshcat import build_meshcat_section_html
from ..page_fragments import MESHCAT_EMBED_STYLES, PLOTLY_EMBED_STYLES, meshcat_case_tabs_script, plotly_resize_script
from ..plotly_support import require_plotly as _require_plotly, to_html
from ..tiles import build_floating_base_tiles, build_joint_tiles
from .figures import _build_grf_section_html


def build_trajectory_dashboard_html(
    record: TrajectoryRunRecord,
    output_path: str | Path,
    include_meshcat: bool = True,
    meshcat_frame_stride: int = 2,
    meshcat_fps: int = 30,
) -> Path:
    """Render a self-contained HTML dashboard for a trajectory record."""

    _require_plotly("Trajectory dashboard rendering")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    floating_coordinates_html = to_html(
        build_floating_base_tiles(record, "coordinates"),
        include_plotlyjs="cdn",
        full_html=False,
    )
    floating_velocities_html = to_html(
        build_floating_base_tiles(record, "velocities"),
        include_plotlyjs=False,
        full_html=False,
    )
    positions_html = to_html(build_joint_tiles(record, "positions"), include_plotlyjs=False, full_html=False)
    velocities_html = to_html(build_joint_tiles(record, "velocities"), include_plotlyjs=False, full_html=False)
    torques_html = to_html(build_joint_tiles(record, "torques"), include_plotlyjs=False, full_html=False)
    grf_section_html = _build_grf_section_html(record)
    meshcat_section_html = build_meshcat_section_html(
        record=record,
        include_meshcat=include_meshcat,
        meshcat_frame_stride=meshcat_frame_stride,
        meshcat_fps=meshcat_fps,
    )

    html_page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trajectory Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: #f4f7fb;
      color: #111827;
    }}
    .container {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px 24px 32px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
    }}
    .meta {{
      margin-bottom: 16px;
      color: #4b5563;
      font-size: 14px;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .tab-button {{
      border: 1px solid #c7d2fe;
      background: #eef2ff;
      color: #1e3a8a;
      font-weight: 600;
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
    }}
    .tab-button.active {{
      background: #1e3a8a;
      color: #ffffff;
      border-color: #1e3a8a;
    }}
    .tab-panel {{
      display: none;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 12px;
    }}
    .tab-panel.active {{
      display: block;
    }}
    .floating-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }}
    .meshcat-iframe {{
      width: 100%;
      height: 82vh;
      min-height: 860px;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #fff;
    }}
{MESHCAT_EMBED_STYLES}
{PLOTLY_EMBED_STYLES}
  </style>
</head>
<body>
  <div class="container">
    <h1>Trajectory Dashboard</h1>
    <div class="meta">
      robot={record.robot},
      dt={record.dt},
      tasks={", ".join([task.task_name for task in record.tasks])},
      created_at_utc={record.created_at_utc}
    </div>
    <div class="tabs">
      <button class="tab-button active" data-tab="floating-base">Floating Base</button>
      <button class="tab-button" data-tab="positions">Joint Positions</button>
      <button class="tab-button" data-tab="velocities">Joint Velocities</button>
      <button class="tab-button" data-tab="torques">Joint Torques</button>
      <button class="tab-button" data-tab="grf">Ground Reaction Forces</button>
      <button class="tab-button" data-tab="meshcat">Meshcat Simulation</button>
    </div>

    <section id="floating-base" class="tab-panel active">
      <div class="floating-grid">
        {floating_coordinates_html}
        {floating_velocities_html}
      </div>
    </section>
    <section id="positions" class="tab-panel">{positions_html}</section>
    <section id="velocities" class="tab-panel">{velocities_html}</section>
    <section id="torques" class="tab-panel">{torques_html}</section>
    <section id="grf" class="tab-panel">{grf_section_html}</section>
    <section id="meshcat" class="tab-panel">{meshcat_section_html}</section>
  </div>

  <script>
    {plotly_resize_script()}
    const buttons = document.querySelectorAll('.tab-button');
    const panels = document.querySelectorAll('.tab-panel');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const target = button.getAttribute('data-tab');
        buttons.forEach((b) => b.classList.remove('active'));
        panels.forEach((p) => p.classList.remove('active'));
        button.classList.add('active');
        const panel = document.getElementById(target);
        panel.classList.add('active');
        requestAnimationFrame(() => resizePlotlyIn(panel));
      }});
    }});

    {meshcat_case_tabs_script(lookup_expr='document.querySelector(`[data-meshcat-panel="${target}"]`)')}

    resizePlotlyIn(document.querySelector('.tab-panel.active'));
  </script>
</body>
</html>
"""

    output.write_text(html_page, encoding="utf-8")
    return output


def build_trajectory_dashboard_from_file(
    record_path: str | Path,
    output_html_path: str | Path,
    include_meshcat: bool = True,
    meshcat_frame_stride: int = 2,
    meshcat_fps: int = 30,
) -> Path:
    """Load a trajectory record from disk and render its HTML dashboard."""

    _require_plotly("Trajectory dashboard rendering")
    record = TrajectoryRunRecord.load(record_path)
    return build_trajectory_dashboard_html(
        record=record,
        output_path=output_html_path,
        include_meshcat=include_meshcat,
        meshcat_frame_stride=meshcat_frame_stride,
        meshcat_fps=meshcat_fps,
    )


__all__ = [
    "build_trajectory_dashboard_from_file",
    "build_trajectory_dashboard_html",
]
