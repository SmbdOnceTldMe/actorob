from __future__ import annotations

from ._lazy_import_helpers import run_probe


def test_dashboard_module_defers_plotly_backend_until_access():
    result = run_probe(
        """
        import json
        import sys
        from importlib import import_module

        module = import_module("actorob.dashboard")
        before = {
            "dir_has": "build_trajectory_dashboard_html" in dir(module),
            "dashboard_loaded": "actorob.dashboard.trajectory.page" in sys.modules,
            "cached": "build_trajectory_dashboard_html" in module.__dict__,
        }

        builder = module.build_trajectory_dashboard_html
        after = {
            "dashboard_loaded": "actorob.dashboard.trajectory.page" in sys.modules,
            "cached": module.__dict__["build_trajectory_dashboard_html"] is builder,
            "callable": callable(builder),
        }

        print(json.dumps({"before": before, "after": after}))
        """
    )

    assert result["before"] == {
        "dir_has": True,
        "dashboard_loaded": False,
        "cached": False,
    }
    assert result["after"] == {
        "dashboard_loaded": True,
        "cached": True,
        "callable": True,
    }


def test_dashboard_module_defers_inverse_design_renderer_until_access():
    result = run_probe(
        """
        import json
        import sys
        from importlib import import_module

        module = import_module("actorob.dashboard")
        before = {
            "dir_has": "build_inverse_design_dashboard_html" in dir(module),
            "inverse_dashboard_loaded": "actorob.dashboard.invdes.page" in sys.modules,
            "cached": "build_inverse_design_dashboard_html" in module.__dict__,
        }

        builder = module.build_inverse_design_dashboard_html
        after = {
            "inverse_dashboard_loaded": "actorob.dashboard.invdes.page" in sys.modules,
            "cached": module.__dict__["build_inverse_design_dashboard_html"] is builder,
            "callable": callable(builder),
        }

        print(json.dumps({"before": before, "after": after}))
        """
    )

    assert result["before"] == {
        "dir_has": True,
        "inverse_dashboard_loaded": False,
        "cached": False,
    }
    assert result["after"] == {
        "inverse_dashboard_loaded": True,
        "cached": True,
        "callable": True,
    }
