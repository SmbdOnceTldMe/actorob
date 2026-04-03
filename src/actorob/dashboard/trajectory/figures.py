"""Trajectory-specific figure builders used by the trajectory dashboard page."""

from __future__ import annotations

import html

import numpy as np

from actorob.trajectories import TrajectoryRunRecord
from ..plotly_support import go, make_subplots, require_plotly as _require_plotly, to_html
from ..tiles import color_map as _color_map, subplot_titles as _subplot_titles, subplots_shape as _subplots_shape


def _available_contact_frame_names(record: TrajectoryRunRecord) -> tuple[str, ...]:
    for task in record.tasks:
        frame_names = tuple(getattr(task, "contact_frame_names", ()))
        if len(frame_names) > 0:
            return frame_names
    return ()


def _build_grf_normal_tiles(record: TrajectoryRunRecord) -> go.Figure:
    _require_plotly("Ground-reaction-force figure rendering")
    frame_names = _available_contact_frame_names(record)
    if len(frame_names) == 0:
        raise ValueError("No contact frame data found in record.")

    rows, cols = _subplots_shape(len(frame_names), n_cols=2)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=_subplot_titles(frame_names, rows, cols),
        shared_xaxes=False,
    )

    colors = _color_map(record)
    shown_cases: set[str] = set()
    n_added = 0

    for frame_idx, _frame_name in enumerate(frame_names):
        row = frame_idx // cols + 1
        col = frame_idx % cols + 1

        for task in record.tasks:
            contact_forces = np.asarray(getattr(task, "contact_forces", np.zeros((0, 0, 3), dtype=float)))
            if contact_forces.ndim != 3 or contact_forces.shape[0] == 0 or contact_forces.shape[1] <= frame_idx:
                continue
            contact_active = np.asarray(getattr(task, "contact_active", np.zeros((0, 0), dtype=bool)))

            x = task.control_time[: contact_forces.shape[0]]
            fz = np.asarray(contact_forces[:, frame_idx, 2], dtype=float)
            if contact_active.shape == contact_forces.shape[:2]:
                fz = np.where(contact_active[:, frame_idx], fz, np.nan)

            showlegend = task.task_name not in shown_cases
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=fz,
                    mode="lines",
                    name=task.task_name,
                    legendgroup=task.task_name,
                    line=dict(color=colors[task.task_name], width=2),
                    showlegend=showlegend,
                    hovertemplate=f"case={task.task_name}<br>t=%{{x:.3f}}<br>Fz=%{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            n_added += 1
            if showlegend:
                shown_cases.add(task.task_name)

        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="#6b7280", row=row, col=col)
        fig.update_xaxes(title_text="t [s]", row=row, col=col)
        fig.update_yaxes(title_text="Fz [N]", row=row, col=col)

    if n_added == 0:
        raise ValueError("Contact force tensors are empty in all tasks.")

    fig.update_layout(
        template="plotly_white",
        title="Ground Reaction Forces: Normal Component Fz (per-foot tiles)",
        height=max(420, 320 * rows),
        margin=dict(l=50, r=25, t=70, b=40),
        legend_title_text="Cases",
    )
    return fig


def _build_grf_friction_ratio_tiles(record: TrajectoryRunRecord) -> go.Figure:
    _require_plotly("Ground-reaction-force figure rendering")
    frame_names = _available_contact_frame_names(record)
    if len(frame_names) == 0:
        raise ValueError("No contact frame data found in record.")

    mu = float(getattr(record, "contact_mu", 0.0))
    if mu <= 0:
        raise ValueError(f"Invalid contact_mu in record: {mu}.")

    rows, cols = _subplots_shape(len(frame_names), n_cols=2)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=_subplot_titles(frame_names, rows, cols),
        shared_xaxes=False,
    )

    colors = _color_map(record)
    shown_cases: set[str] = set()
    n_added = 0

    for frame_idx, _frame_name in enumerate(frame_names):
        row = frame_idx // cols + 1
        col = frame_idx % cols + 1

        for task in record.tasks:
            contact_forces = np.asarray(getattr(task, "contact_forces", np.zeros((0, 0, 3), dtype=float)))
            if contact_forces.ndim != 3 or contact_forces.shape[0] == 0 or contact_forces.shape[1] <= frame_idx:
                continue
            contact_active = np.asarray(getattr(task, "contact_active", np.zeros((0, 0), dtype=bool)))

            x = task.control_time[: contact_forces.shape[0]]
            force_xy = np.linalg.norm(contact_forces[:, frame_idx, :2], axis=1)
            force_z = np.asarray(contact_forces[:, frame_idx, 2], dtype=float)
            friction_limit = mu * np.maximum(force_z, 0.0)
            ratio = force_xy / np.maximum(friction_limit, 1e-8)
            if contact_active.shape == contact_forces.shape[:2]:
                ratio = np.where(contact_active[:, frame_idx], ratio, np.nan)

            showlegend = task.task_name not in shown_cases
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ratio,
                    mode="lines",
                    name=task.task_name,
                    legendgroup=task.task_name,
                    line=dict(color=colors[task.task_name], width=2),
                    showlegend=showlegend,
                    hovertemplate=(f"case={task.task_name}<br>t=%{{x:.3f}}<br>||Ft||/(μFz)=%{{y:.4f}}<extra></extra>"),
                ),
                row=row,
                col=col,
            )
            n_added += 1
            if showlegend:
                shown_cases.add(task.task_name)

        fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color="#dc2626", row=row, col=col)
        fig.update_xaxes(title_text="t [s]", row=row, col=col)
        fig.update_yaxes(title_text="ratio", row=row, col=col)

    if n_added == 0:
        raise ValueError("Contact force tensors are empty in all tasks.")

    fig.update_layout(
        template="plotly_white",
        title=f"Friction Cone Check: ||Ft|| <= μFz (μ={mu:.3f}, limit line=1.0)",
        height=max(420, 320 * rows),
        margin=dict(l=50, r=25, t=70, b=40),
        legend_title_text="Cases",
    )
    return fig


def _build_grf_section_html(record: TrajectoryRunRecord) -> str:
    _require_plotly("Ground-reaction-force figure rendering")
    try:
        normal_html = to_html(_build_grf_normal_tiles(record), include_plotlyjs=False, full_html=False)
        ratio_html = to_html(_build_grf_friction_ratio_tiles(record), include_plotlyjs=False, full_html=False)
    except Exception as exc:
        return f'<div class="meshcat-note">Ground reaction force data unavailable: {html.escape(str(exc))}</div>'

    return f"""
<div class="floating-grid">
  {normal_html}
  {ratio_html}
</div>
"""
