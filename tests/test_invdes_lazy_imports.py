from __future__ import annotations

from ._lazy_import_helpers import run_probe


def test_invdes_module_defers_record_import_until_access():
    result = run_probe(
        """
        import json
        import sys
        from importlib import import_module

        module = import_module("actorob.invdes")
        before = {
            "dir_has": "InverseDesignRunRecord" in dir(module),
            "record_loaded": "actorob.invdes.record" in sys.modules,
            "inverse_dashboard_loaded": "actorob.dashboard.invdes.page" in sys.modules,
            "cached": "InverseDesignRunRecord" in module.__dict__,
        }

        record_cls = module.InverseDesignRunRecord
        after = {
            "record_loaded": "actorob.invdes.record" in sys.modules,
            "inverse_dashboard_loaded": "actorob.dashboard.invdes.page" in sys.modules,
            "cached": module.__dict__["InverseDesignRunRecord"] is record_cls,
            "name": record_cls.__name__,
        }

        print(json.dumps({"before": before, "after": after}))
        """
    )

    assert result["before"] == {
        "dir_has": True,
        "record_loaded": False,
        "inverse_dashboard_loaded": False,
        "cached": False,
    }
    assert result["after"] == {
        "record_loaded": True,
        "inverse_dashboard_loaded": False,
        "cached": True,
        "name": "InverseDesignRunRecord",
    }
