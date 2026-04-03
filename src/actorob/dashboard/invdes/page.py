"""Inverse-design HTML dashboard rendering."""

from __future__ import annotations

import html
import json
from pathlib import Path

from actorob.invdes.record import InverseDesignRunRecord
from .candidate import (
    _build_best_candidate_section_html,
)
from .overview import (
    _build_fitness_figure,
    _build_iteration_table_payload,
    _build_sigma_and_duration_figure,
    _build_solver_cards_html,
    _build_summary_cards_html,
)
from ..page_fragments import MESHCAT_EMBED_STYLES, PLOTLY_EMBED_STYLES, meshcat_case_tabs_script, plotly_resize_script
from ..plotly_support import require_plotly as _require_plotly, to_html


def build_inverse_design_dashboard_html(
    run: InverseDesignRunRecord,
    output_path: str | Path,
    *,
    include_meshcat: bool = True,
    meshcat_frame_stride: int = 2,
    meshcat_fps: int = 30,
) -> Path:
    """Render a self-contained HTML dashboard for an inverse-design run."""

    _require_plotly("Inverse-design dashboard rendering")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    overview_fitness_html = to_html(_build_fitness_figure(run), include_plotlyjs="cdn", full_html=False)
    overview_sigma_html = to_html(_build_sigma_and_duration_figure(run), include_plotlyjs=False, full_html=False)
    solver_cards_html = _build_solver_cards_html(run)
    best_candidate_html = _build_best_candidate_section_html(
        run,
        include_meshcat=include_meshcat,
        meshcat_frame_stride=meshcat_frame_stride,
        meshcat_fps=meshcat_fps,
    )
    iteration_table_data = json.dumps(_build_iteration_table_payload(run))
    summary_cards_html = _build_summary_cards_html(run)

    html_page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Inverse Design Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: #14213d;
      background:
        radial-gradient(circle at top left, rgba(142, 202, 230, 0.28), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
    }}
    .container {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 24px 24px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .meta {{
      color: #516072;
      margin-bottom: 18px;
      line-height: 1.5;
    }}
    .section {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid rgba(191, 219, 254, 0.9);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
      backdrop-filter: blur(6px);
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .card {{
      padding: 14px;
      border-radius: 14px;
      background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
      border: 1px solid #d7e6f7;
    }}
    .card-label {{
      color: #5b6b7f;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .card-value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .two-up {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .overview-stack {{
      display: grid;
      gap: 14px;
    }}
    .dashboard-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .dashboard-tab-button {{
      border: 1px solid #c7d2fe;
      background: #eef2ff;
      color: #1e3a8a;
      font-weight: 600;
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
    }}
    .dashboard-tab-button.active {{
      background: #1e3a8a;
      color: #ffffff;
      border-color: #1e3a8a;
    }}
    .dashboard-tab-panel {{
      display: none;
      background: rgba(255,255,255,0.78);
      border: 1px solid #dbe7f5;
      border-radius: 14px;
      padding: 12px;
    }}
    .dashboard-tab-panel.active {{
      display: block;
    }}
    .solver-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .solver-card {{
      padding: 14px;
      border-radius: 14px;
      background: #f8fbff;
      border: 1px solid #d7e6f7;
    }}
    .solver-card strong {{
      display: block;
      margin-bottom: 6px;
      color: #274c77;
    }}
    .table-controls {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .range-wrap {{
      flex: 1 1 320px;
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .range-wrap input {{
      width: 100%;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: #fff;
      border-radius: 14px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #e5edf6;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eff6ff;
      color: #274c77;
      position: sticky;
      top: 0;
    }}
    .status-complete {{
      color: #166534;
      font-weight: 700;
    }}
    .status-fail {{
      color: #b91c1c;
      font-weight: 700;
    }}
    .meshcat-iframe {{
      width: 100%;
      height: 82vh;
      min-height: 860px;
    }}
{MESHCAT_EMBED_STYLES}
{PLOTLY_EMBED_STYLES}
    @media (max-width: 1100px) {{
      .two-up {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Inverse Design Dashboard</h1>
    <div class="meta">
      config={html.escape(run.config_path)}<br>
      tasks={html.escape(", ".join(run.task_names))}<br>
      created_at_utc={html.escape(run.created_at_utc)}
    </div>

    <section class="section">
      <h2>Overview</h2>
      <div class="overview-stack">
        <div>
          <h3>Run Summary</h3>
          {summary_cards_html}
        </div>
        <div>
          <h3>Solver Setup</h3>
          {solver_cards_html}
        </div>
        <div>
          <h3>Optimization History</h3>
        </div>
        <div class="two-up">
          <div>{overview_fitness_html}</div>
          <div>{overview_sigma_html}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Best Candidate Dashboard</h2>
      {best_candidate_html}
    </section>

    <section class="section">
      <h2>Iteration Explorer</h2>
      <div class="table-controls">
        <div class="range-wrap">
          <label for="iteration-slider"><strong>Iteration</strong></label>
          <input id="iteration-slider" type="range" min="1" max="{max(len(run.history.batches), 1)}" value="1" />
        </div>
        <div id="iteration-label">Iteration 1 / {max(len(run.history.batches), 1)}</div>
      </div>
      <div id="iteration-meta" class="meta"></div>
      <div style="overflow:auto;">
        <table>
          <thead id="iteration-table-head"></thead>
          <tbody id="iteration-table-body"></tbody>
        </table>
      </div>
    </section>

  </div>

  <script>
    {plotly_resize_script()}

    const iterationData = {iteration_table_data};
    const slider = document.getElementById('iteration-slider');
    const label = document.getElementById('iteration-label');
    const meta = document.getElementById('iteration-meta');
    const head = document.getElementById('iteration-table-head');
    const body = document.getElementById('iteration-table-body');

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('\"', '&quot;')
        .replaceAll(\"'\", '&#039;');
    }}

    function renderIteration(index) {{
      const item = iterationData[index];
      label.textContent = `Iteration ${{item.batch_index}} / ${{iterationData.length}}`;
      meta.textContent =
        `generation=${{item.generation_text}}, sigma=${{item.sigma_text}}, duration=${{item.duration_text}}, completed=${{item.completed_trials}}, failed=${{item.failed_trials}}, best=${{item.best_value_text}}`;

      head.innerHTML = `<tr>${{item.headers.map((value) => `<th>${{escapeHtml(value)}}</th>`).join('')}}</tr>`;
      body.innerHTML = item.rows
        .map((row) => `<tr>${{row.map((value, idx) => idx === 1 ? `<td class="status-${{String(value).toLowerCase()}}">${{escapeHtml(value)}}</td>` : `<td>${{escapeHtml(value)}}</td>`).join('')}}</tr>`)
        .join('');
    }}

    slider.addEventListener('input', () => {{
      renderIteration(Number(slider.value) - 1);
    }});

    const dashboardTabButtons = document.querySelectorAll('.dashboard-tab-button');
    const dashboardTabPanels = document.querySelectorAll('.dashboard-tab-panel');
    dashboardTabButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const target = button.getAttribute('data-dashboard-tab');
        dashboardTabButtons.forEach((item) => item.classList.remove('active'));
        dashboardTabPanels.forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        const panel = document.querySelector(`[data-dashboard-panel="${{target}}"]`);
        panel.classList.add('active');
        requestAnimationFrame(() => resizePlotlyIn(panel));
      }});
    }});

    {meshcat_case_tabs_script(lookup_expr='document.querySelector(`[data-meshcat-panel="${target}"]`)')}

    renderIteration(0);
    resizePlotlyIn(document.querySelector('.dashboard-tab-panel.active'));
  </script>
</body>
</html>
"""

    output.write_text(html_page, encoding="utf-8")
    return output


__all__ = [
    "build_inverse_design_dashboard_html",
]
