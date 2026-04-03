"""Best-candidate sections and figures for inverse-design dashboards."""

from __future__ import annotations

import html

import numpy as np

from actorob.invdes.record import InverseDesignRunRecord
from actorob.trajectories import TrajectoryRunRecord
from .actuation import (
    _ActuationSpecs,
    _add_phase_envelope,
    _load_actuation_specs,
    _mechanical_speed_envelope,
)
from .overview import _format_parameter_label
from ..meshcat import build_meshcat_section_html
from ..plotly_support import go, make_subplots, require_plotly as _require_plotly, to_html
from ..tiles import build_floating_base_tiles, build_joint_tiles, color_map, subplot_titles, subplots_shape


def _build_best_candidate_section_html(
    run: InverseDesignRunRecord,
    *,
    include_meshcat: bool,
    meshcat_frame_stride: int,
    meshcat_fps: int,
) -> str:
    _require_plotly("Inverse-design candidate dashboard rendering")
    record = run.best_trajectory_record
    if record is None:
        return '<div class="meshcat-note">Best candidate trajectory record is unavailable.</div>'

    actuation_specs = _load_actuation_specs(run, record)
    floating_coordinates_html = to_html(
        build_floating_base_tiles(record, "coordinates"),
        include_plotlyjs=False,
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
    power_html = to_html(_build_record_power_figure(record), include_plotlyjs=False, full_html=False)
    phase_portrait_html = to_html(
        _build_torque_speed_phase_portrait(record, actuation_specs),
        include_plotlyjs=False,
        full_html=False,
    )
    meshcat_html = build_meshcat_section_html(
        record=record,
        include_meshcat=include_meshcat,
        meshcat_frame_stride=meshcat_frame_stride,
        meshcat_fps=meshcat_fps,
    )

    params_html = "n/a"
    if run.best_params is not None:
        params_html = ", ".join(
            f"{_format_parameter_label(name)}={value:.4g}" for name, value in zip(run.parameter_names, run.best_params)
        )

    raw_metrics_html = _build_best_scenario_metrics_html(run)

    return f"""
<div class="meta">best_fitness={html.escape("n/a" if run.best_value is None else f"{run.best_value:.4g}")}<br>best_params={html.escape(params_html)}</div>
{raw_metrics_html}
<div class="dashboard-tabs">
  <button class="dashboard-tab-button active" data-dashboard-tab="floating-base">Floating Base</button>
  <button class="dashboard-tab-button" data-dashboard-tab="positions">Joint Positions</button>
  <button class="dashboard-tab-button" data-dashboard-tab="velocities">Joint Velocities</button>
  <button class="dashboard-tab-button" data-dashboard-tab="torques">Joint Torques</button>
  <button class="dashboard-tab-button" data-dashboard-tab="power">Power</button>
  <button class="dashboard-tab-button" data-dashboard-tab="phase-portraits">Phase Portraits</button>
  <button class="dashboard-tab-button" data-dashboard-tab="meshcat">Meshcat Simulation</button>
</div>
<div class="dashboard-tab-panel active" data-dashboard-panel="floating-base">
  <div class="floating-grid">
    {floating_coordinates_html}
    {floating_velocities_html}
  </div>
</div>
<div class="dashboard-tab-panel" data-dashboard-panel="positions">{positions_html}</div>
<div class="dashboard-tab-panel" data-dashboard-panel="velocities">{velocities_html}</div>
<div class="dashboard-tab-panel" data-dashboard-panel="torques">{torques_html}</div>
<div class="dashboard-tab-panel" data-dashboard-panel="power">{power_html}</div>
<div class="dashboard-tab-panel" data-dashboard-panel="phase-portraits">{phase_portrait_html}</div>
<div class="dashboard-tab-panel" data-dashboard-panel="meshcat">{meshcat_html}</div>
"""


def _build_record_power_figure(record: TrajectoryRunRecord) -> go.Figure:
    _require_plotly("Inverse-design candidate dashboard rendering")
    fig = make_subplots(
        rows=max(len(record.tasks), 1),
        cols=1,
        subplot_titles=tuple(task.task_name for task in record.tasks) or ("power",),
        shared_xaxes=False,
    )
    for row, task in enumerate(record.tasks, start=1):
        electrical_power = np.asarray(task.electrical_power, dtype=float)
        friction_power = np.asarray(task.friction_power, dtype=float)
        x = np.asarray(task.control_time, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x[: len(electrical_power)],
                y=electrical_power,
                mode="lines",
                name=f"{task.task_name}: electrical",
                legendgroup=task.task_name,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x[: len(friction_power)],
                y=friction_power,
                mode="lines",
                name=f"{task.task_name}: friction",
                legendgroup=task.task_name,
                line=dict(dash="dash"),
            ),
            row=row,
            col=1,
        )
        fig.update_xaxes(title_text="t [s]", row=row, col=1)
        fig.update_yaxes(title_text="Power [W]", row=row, col=1)
    fig.update_layout(
        template="plotly_white",
        title="Electrical Power and Friction Losses by Task",
        margin=dict(l=50, r=20, t=60, b=40),
        height=max(360, 280 * max(len(record.tasks), 1)),
    )
    return fig


def _build_torque_speed_phase_portrait(
    record: TrajectoryRunRecord,
    actuation_specs: _ActuationSpecs,
) -> go.Figure:
    _require_plotly("Inverse-design candidate dashboard rendering")
    joint_names = record.joint_names
    rows, cols = subplots_shape(len(joint_names), n_cols=3)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles(joint_names, rows, cols),
        shared_xaxes=False,
        shared_yaxes=False,
    )

    colors = color_map(record)
    shown_cases: set[str] = set()
    shown_constraint_groups: set[str] = set()

    for joint_idx, _joint_name in enumerate(joint_names):
        row = joint_idx // cols + 1
        col = joint_idx % cols + 1

        for task in record.tasks:
            torques = np.asarray(task.joint_torques[:, joint_idx], dtype=float)
            velocities = np.asarray(task.joint_velocities[: len(torques), joint_idx], dtype=float)
            showlegend = task.task_name not in shown_cases
            fig.add_trace(
                go.Scatter(
                    x=torques,
                    y=velocities,
                    mode="markers",
                    name=task.task_name,
                    legendgroup=task.task_name,
                    marker=dict(color=colors[task.task_name], size=7, opacity=0.8),
                    showlegend=showlegend,
                    hovertemplate=f"case={task.task_name}<br>tau=%{{x:.4f}}<br>dq=%{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            if showlegend:
                shown_cases.add(task.task_name)

        torque_limit = float(record.joint_torque_upper_limits[joint_idx])
        velocity_limit = float(record.joint_velocity_upper_limits[joint_idx])
        torque_grid = np.linspace(0.0, max(torque_limit, 1.0e-6), 180)

        mech_envelope = _mechanical_speed_envelope(actuation_specs.mechanical, joint_idx, torque_grid)
        if mech_envelope is not None:
            _add_phase_envelope(
                fig,
                row,
                col,
                torque_grid,
                mech_envelope,
                color="#6b7280",
                name="Mechanical characteristic",
                showlegend=("mechanical" not in shown_constraint_groups),
            )
            shown_constraint_groups.add("mechanical")

        fig.update_xaxes(title_text="tau [Nm]", row=row, col=col)
        fig.update_yaxes(title_text="dq [rad/s]", row=row, col=col)
        fig.update_xaxes(range=[-torque_limit, torque_limit], row=row, col=col)
        fig.update_yaxes(range=[-velocity_limit, velocity_limit], row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title="Torque-Speed Phase Portrait with OCP Actuation Limits",
        height=max(460, 310 * rows),
        margin=dict(l=50, r=25, t=70, b=40),
        legend_title_text="Cases and Limits",
    )
    return fig


def _build_best_scenario_metrics_html(run: InverseDesignRunRecord) -> str:
    if not run.best_scenarios:
        return '<div class="meshcat-note">Raw metrics for the best candidate are unavailable.</div>'

    headers = [
        "Task",
        "Converged",
        "traj_cost",
        "energy",
        "friction",
        "traj_z",
        "energy_z",
        "friction_z",
    ]
    rows = []
    for scenario in run.best_scenarios:
        rows.append(
            [
                scenario.case_name or scenario.mode,
                "yes" if scenario.converged else "no",
                f"{scenario.traj_cost:.6g}",
                f"{scenario.electrical_energy:.6g}",
                f"{scenario.friction_loss:.6g}",
                _format_z(run.normalization_stats, scenario.mode, "traj_cost", scenario.traj_cost),
                _format_z(run.normalization_stats, scenario.mode, "electrical_energy", scenario.electrical_energy),
                _format_z(run.normalization_stats, scenario.mode, "friction_loss", scenario.friction_loss),
            ]
        )

    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    row_html = "".join("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>" for row in rows)
    note = (
        "Fitness is a weighted sum of z-scores, so it cannot be uniquely inverted into one raw scalar. "
        "The table below shows the raw metrics and their normalized contributions for the best candidate."
    )
    return f"""
<div class="meta">{html.escape(note)}</div>
<div style="overflow:auto;">
  <table>
    <thead><tr>{header_html}</tr></thead>
    <tbody>{row_html}</tbody>
  </table>
</div>
"""


def _format_z(
    stats: dict[str, dict[str, dict[str, float]]] | None,
    mode: str,
    metric: str,
    value: float,
) -> str:
    if stats is None:
        return "n/a"
    metric_stats = stats.get(mode, {}).get(metric)
    if metric_stats is None:
        metric_stats = stats.get(mode.upper(), {}).get(metric)
    if metric_stats is None:
        return "n/a"
    std = float(metric_stats["std"])
    if std == 0:
        return "0"
    mean = float(metric_stats["mean"])
    return f"{(value - mean) / std:.6g}"
