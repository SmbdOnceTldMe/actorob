"""HTML assembly for embedded meshcat dashboard sections."""

from __future__ import annotations

import html

from actorob.dashboard.meshcat.scene import build_meshcat_task_html
from actorob.trajectories import TrajectoryRunRecord


def build_meshcat_section_html(
    record: TrajectoryRunRecord,
    include_meshcat: bool,
    meshcat_frame_stride: int,
    meshcat_fps: int,
) -> str:
    if not include_meshcat:
        return '<div class="meshcat-note">Meshcat simulation disabled by flag.</div>'

    iframe_items: list[tuple[str, str]] = []
    errors: list[str] = []
    for task_idx, task in enumerate(record.tasks):
        try:
            meshcat_html = build_meshcat_task_html(
                record=record,
                task_idx=task_idx,
                frame_stride=meshcat_frame_stride,
                fps=meshcat_fps,
            )
            iframe_items.append((task.task_name, html.escape(meshcat_html, quote=True)))
        except Exception as exc:
            errors.append(f"{task.task_name}: {exc}")

    if len(iframe_items) == 0:
        joined = "<br>".join(errors) if errors else "unknown error"
        return f'<div class="meshcat-note">Meshcat simulation unavailable.<br>{joined}</div>'

    buttons_html = []
    panels_html = []
    for idx, (task_name, srcdoc) in enumerate(iframe_items):
        active_class = " active" if idx == 0 else ""
        buttons_html.append(
            f'<button class="meshcat-case-button{active_class}" data-meshcat-case="{idx}">{task_name}</button>'
        )
        panels_html.append(
            f"""
<div class="meshcat-case-panel{active_class}" data-meshcat-panel="{idx}">
  <iframe class="meshcat-iframe" srcdoc="{srcdoc}"></iframe>
</div>
"""
        )

    errors_html = ""
    if errors:
        errors_html = '<div class="meshcat-note">Failed to render some cases:<br>' + "<br>".join(errors) + "</div>"

    return f"""
<div class="meshcat-cases">
  <div class="meshcat-case-tabs">
    {"".join(buttons_html)}
  </div>
  {"".join(panels_html)}
</div>
{errors_html}
"""


__all__ = ["build_meshcat_section_html"]
