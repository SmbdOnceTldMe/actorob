"""Shared CSS and JavaScript fragments for dashboard HTML pages."""

from __future__ import annotations

PLOTLY_EMBED_STYLES = """
    .js-plotly-plot, .plotly-graph-div {
      width: 100% !important;
    }
    .plot-container, .svg-container {
      width: 100% !important;
      max-width: 100% !important;
    }
"""

MESHCAT_EMBED_STYLES = """
    .meshcat-note {
      border: 1px solid #d1d5db;
      border-radius: 10px;
      background: #f9fafb;
      color: #374151;
      padding: 10px 12px;
      font-size: 14px;
    }
    .meshcat-cases {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .meshcat-case-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .meshcat-case-button {
      border: 1px solid #d1d5db;
      background: #f9fafb;
      color: #1f2937;
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }
    .meshcat-case-button.active {
      background: #0f766e;
      border-color: #0f766e;
      color: #ffffff;
    }
    .meshcat-case-panel {
      display: none;
    }
    .meshcat-case-panel.active {
      display: block;
    }
"""


def plotly_resize_script() -> str:
    return """
    function resizePlotlyIn(container) {
      if (!container || !window.Plotly) {
        return;
      }
      const plots = container.querySelectorAll('.js-plotly-plot');
      plots.forEach((plot) => {
        try {
          window.Plotly.Plots.resize(plot);
        } catch (_err) {
        }
      });
    }
"""


def meshcat_case_tabs_script(*, lookup_expr: str) -> str:
    return f"""
    const meshcatButtons = document.querySelectorAll('.meshcat-case-button');
    const meshcatPanels = document.querySelectorAll('.meshcat-case-panel');
    meshcatButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const target = button.getAttribute('data-meshcat-case');
        meshcatButtons.forEach((item) => item.classList.remove('active'));
        meshcatPanels.forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        const panel = {lookup_expr};
        if (panel) {{
          panel.classList.add('active');
        }}
      }});
    }});
"""
