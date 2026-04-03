"""Overview and history sections for inverse-design dashboards."""

from __future__ import annotations

import html

from actorob.invdes.record import InverseDesignRunRecord
from ..plotly_support import go, make_subplots, require_plotly as _require_plotly


def _build_summary_cards_html(run: InverseDesignRunRecord) -> str:
    total_duration = sum(batch.duration_seconds for batch in run.history.batches)
    cards = [
        ("Best Fitness", "n/a" if run.best_value is None else f"{run.best_value:.4g}"),
        ("Iterations", str(run.settings.max_iterations)),
        ("Trials", str(run.max_trials)),
        ("Population", str(run.settings.population_size)),
        ("Workers", str(run.settings.parallelism)),
        ("Tasks / Offspring", str(len(run.task_names))),
        ("Parameters", str(len(run.parameter_names))),
        ("Total Optimization", _format_seconds(total_duration)),
    ]
    return (
        '<div class="card-grid">'
        + "".join(
            f'<div class="card"><div class="card-label">{html.escape(label)}</div><div class="card-value">{html.escape(value)}</div></div>'
            for label, value in cards
        )
        + "</div>"
    )


def _build_solver_cards_html(run: InverseDesignRunRecord) -> str:
    cards = [
        ("Sigma0", f"{run.sigma0:.6g}"),
        ("Seed", "n/a" if run.seed is None else str(run.seed)),
        ("Objective", "weighted sum of z-scores"),
        (
            "Objective Weights",
            html.escape(
                f"traj={run.weights.traj_cost:.3g}, energy={run.weights.energy:.3g}, friction={run.weights.friction:.3g}"
            ),
        ),
    ]
    return (
        '<div class="solver-grid">'
        + "".join(
            f'<div class="solver-card"><strong>{label}</strong><div>{value}</div></div>' for label, value in cards
        )
        + "</div>"
    )


def _build_fitness_figure(run: InverseDesignRunRecord) -> go.Figure:
    _require_plotly("Inverse-design overview figure rendering")
    trial_indices: list[int] = []
    values: list[float] = []
    best_so_far: list[float] = []
    current_best: float | None = None
    cursor = 0
    for batch in run.history.batches:
        for trial in batch.trials:
            if trial.status != "complete" or trial.value is None:
                continue
            cursor += 1
            trial_indices.append(cursor)
            values.append(trial.value)
            current_best = trial.value if current_best is None else min(current_best, trial.value)
            best_so_far.append(current_best)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trial_indices,
            y=values,
            mode="markers",
            name="Trial fitness",
            marker=dict(color="#60a5fa", size=10, opacity=0.85),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trial_indices,
            y=best_so_far,
            mode="lines+markers",
            name="Best so far",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=7),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Fitness Landscape by Trial",
        xaxis_title="Completed trial",
        yaxis_title="Fitness",
        margin=dict(l=50, r=20, t=60, b=45),
    )
    return fig


def _build_sigma_and_duration_figure(run: InverseDesignRunRecord) -> go.Figure:
    _require_plotly("Inverse-design overview figure rendering")
    x = [batch.batch_index for batch in run.history.batches]
    sigma = []
    current_sigma = run.sigma0
    for batch in run.history.batches:
        if batch.sigma is not None:
            current_sigma = batch.sigma
        sigma.append(current_sigma)

    durations = [batch.duration_seconds for batch in run.history.batches]
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Sigma by Iteration", "Batch Duration"), shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=x, y=sigma, mode="lines+markers", name="sigma", line=dict(color="#0f766e", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=x, y=durations, name="duration", marker_color="#f59e0b"),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Batch", row=2, col=1)
    fig.update_yaxes(title_text="sigma", row=1, col=1)
    fig.update_yaxes(title_text="seconds", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        title="CMA-ES State",
        margin=dict(l=50, r=20, t=70, b=45),
        height=620,
        showlegend=False,
    )
    return fig


def _build_iteration_table_payload(run: InverseDesignRunRecord) -> list[dict[str, object]]:
    headers = ["Trial", "Status", "Value", "Error"] + [_format_parameter_label(name) for name in run.parameter_names]
    payload: list[dict[str, object]] = []
    for batch in run.history.batches:
        rows = []
        ordered_trials = sorted(
            batch.trials,
            key=lambda trial: (trial.value is None, float("inf") if trial.value is None else trial.value),
        )
        for trial in ordered_trials:
            value = "n/a" if trial.value is None else f"{trial.value:.6g}"
            error = "" if trial.error is None else trial.error
            row = [trial.trial_number if trial.trial_number is not None else "-", trial.status.upper(), value, error]
            row.extend(f"{parameter_value:.6g}" for parameter_value in trial.params)
            rows.append(row)
        payload.append(
            {
                "batch_index": batch.batch_index,
                "headers": headers,
                "rows": rows,
                "generation_text": "n/a" if batch.generation is None else str(batch.generation),
                "sigma_text": "n/a" if batch.sigma is None else f"{batch.sigma:.6g}",
                "duration_text": f"{batch.duration_seconds:.2f}s",
                "completed_trials": batch.completed_trials,
                "failed_trials": batch.failed_trials,
                "best_value_text": "n/a" if batch.best_value is None else f"{batch.best_value:.6g}",
            }
        )

    if not payload:
        payload.append(
            {
                "batch_index": 1,
                "headers": headers,
                "rows": [],
                "generation_text": "n/a",
                "sigma_text": "n/a",
                "duration_text": "0.00s",
                "completed_trials": 0,
                "failed_trials": 0,
                "best_value_text": "n/a",
            }
        )
    return payload


def _format_parameter_label(name: str) -> str:
    if name.startswith("m_"):
        return f"{name[2:]} m"
    if name.startswith("g_"):
        return f"{name[2:]} g"
    return name


def _format_seconds(seconds: float) -> str:
    if seconds >= 60.0:
        minutes = int(seconds // 60.0)
        remainder = seconds - 60.0 * minutes
        return f"{minutes}m {remainder:.1f}s"
    return f"{seconds:.2f}s"
