"""Shared Plotly tile builders for dashboard views."""

from __future__ import annotations

from math import ceil

import numpy as np

from actorob.trajectories import TrajectoryRunRecord
from .plotly_support import go, make_subplots, qualitative, require_plotly as _require_plotly


def color_map(record: TrajectoryRunRecord) -> dict[str, str]:
    _require_plotly("Trajectory/dashboard figure rendering")
    palette = qualitative.Plotly
    return {task.task_name: palette[idx % len(palette)] for idx, task in enumerate(record.tasks)}


def subplots_shape(n_items: int, n_cols: int = 3) -> tuple[int, int]:
    n_rows = int(ceil(n_items / n_cols))
    return n_rows, n_cols


def subplot_titles(names: tuple[str, ...], n_rows: int, n_cols: int) -> list[str]:
    total = n_rows * n_cols
    titles = list(names)
    if len(titles) < total:
        titles.extend([""] * (total - len(titles)))
    return titles


def _add_limits(
    fig: go.Figure,
    row: int,
    col: int,
    x: np.ndarray,
    lower: float,
    upper: float,
) -> None:
    if np.isfinite(lower):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full_like(x, lower, dtype=float),
                mode="lines",
                line=dict(color="#6b7280", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    if np.isfinite(upper):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full_like(x, upper, dtype=float),
                mode="lines",
                line=dict(color="#6b7280", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _longest_time_axis(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) == 0:
        return np.zeros(0, dtype=float)

    def _axis_key(values: np.ndarray) -> tuple[float, int]:
        if values.size == 0:
            return (float("-inf"), 0)
        return (float(values[-1]), int(values.size))

    return max(arrays, key=_axis_key)


def build_joint_tiles(record: TrajectoryRunRecord, metric: str) -> go.Figure:
    _require_plotly("Joint figure rendering")

    joint_names = record.joint_names
    rows, cols = subplots_shape(len(joint_names), n_cols=3)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles(joint_names, rows, cols),
        shared_xaxes=False,
    )

    colors = color_map(record)
    shown_cases: set[str] = set()

    if metric == "positions":
        title = "Joint Positions (per-joint tiles, color=case)"
        y_label = "q [rad]"
        lower_limits = record.joint_position_lower_limits
        upper_limits = record.joint_position_upper_limits
    elif metric == "velocities":
        title = "Joint Velocities (per-joint tiles, color=case)"
        y_label = "dq [rad/s]"
        lower_limits = record.joint_velocity_lower_limits
        upper_limits = record.joint_velocity_upper_limits
    elif metric == "torques":
        title = "Joint Torques (per-joint tiles, color=case)"
        y_label = "tau [Nm]"
        lower_limits = record.joint_torque_lower_limits
        upper_limits = record.joint_torque_upper_limits
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    for joint_idx, _joint_name in enumerate(joint_names):
        row = joint_idx // cols + 1
        col = joint_idx % cols + 1

        for task in record.tasks:
            if metric == "positions":
                x = task.state_time
                y = task.joint_positions[:, joint_idx]
            elif metric == "velocities":
                x = task.state_time
                y = task.joint_velocities[:, joint_idx]
            else:
                x = task.control_time
                y = task.joint_torques[:, joint_idx]

            showlegend = task.task_name not in shown_cases
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=task.task_name,
                    legendgroup=task.task_name,
                    line=dict(color=colors[task.task_name], width=2),
                    showlegend=showlegend,
                    hovertemplate=f"case={task.task_name}<br>t=%{{x:.3f}}<br>value=%{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            if showlegend:
                shown_cases.add(task.task_name)

        reference_x = _longest_time_axis(
            [
                np.asarray(task.state_time if metric in {"positions", "velocities"} else task.control_time, dtype=float)
                for task in record.tasks
            ]
        )
        _add_limits(
            fig=fig,
            row=row,
            col=col,
            x=reference_x,
            lower=float(lower_limits[joint_idx]),
            upper=float(upper_limits[joint_idx]),
        )

        if metric == "torques":
            rms_lines = []
            for task in record.tasks:
                rms = float(np.sqrt(np.mean(task.joint_torques[:, joint_idx] ** 2)))
                rms_lines.append(f"{task.task_name}: {rms:.3f}")
            fig.add_annotation(
                x=0.01,
                y=0.99,
                xref="x domain",
                yref="y domain",
                text="<b>RMS</b><br>" + "<br>".join(rms_lines),
                showarrow=False,
                align="left",
                bordercolor="#d1d5db",
                borderwidth=1,
                borderpad=3,
                bgcolor="rgba(255,255,255,0.85)",
                font=dict(size=10),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="t [s]", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(460, 310 * rows),
        margin=dict(l=50, r=25, t=70, b=40),
        legend_title_text="Cases",
    )
    return fig


def build_floating_base_tiles(record: TrajectoryRunRecord, metric: str) -> go.Figure:
    _require_plotly("Floating-base figure rendering")

    if metric == "coordinates":
        component_names = ("x", "y", "z", "qx", "qy", "qz", "qw")
        title = "Floating Base Coordinates (color=case)"
        y_label = "value"

        def data_getter(task):
            return task.state_time, task.floating_base_coordinates

    elif metric == "velocities":
        component_names = ("vx", "vy", "vz", "wx", "wy", "wz")
        title = "Floating Base Velocities (color=case)"
        y_label = "value"

        def data_getter(task):
            return task.state_time, task.floating_base_velocities

    else:
        raise ValueError(f"Unsupported floating base metric: {metric}")

    rows, cols = subplots_shape(len(component_names), n_cols=3)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles(component_names, rows, cols),
        shared_xaxes=False,
    )

    colors = color_map(record)
    shown_cases: set[str] = set()

    for comp_idx, _comp_name in enumerate(component_names):
        row = comp_idx // cols + 1
        col = comp_idx % cols + 1
        for task in record.tasks:
            x, values = data_getter(task)
            showlegend = task.task_name not in shown_cases
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=values[:, comp_idx],
                    mode="lines",
                    name=task.task_name,
                    legendgroup=task.task_name,
                    line=dict(color=colors[task.task_name], width=2),
                    showlegend=showlegend,
                    hovertemplate=f"case={task.task_name}<br>t=%{{x:.3f}}<br>value=%{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            if showlegend:
                shown_cases.add(task.task_name)
        fig.update_xaxes(title_text="t [s]", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(420, 290 * rows),
        margin=dict(l=50, r=25, t=70, b=40),
        legend_title_text="Cases",
    )
    return fig


__all__ = [
    "build_floating_base_tiles",
    "build_joint_tiles",
    "color_map",
    "subplot_titles",
    "subplots_shape",
]
