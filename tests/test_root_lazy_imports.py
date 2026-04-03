from __future__ import annotations

from ._lazy_import_helpers import run_probe


def test_actorob_root_import_is_pure_and_keeps_heavy_modules_lazy():
    result = run_probe(
        """
        import json
        import sys
        import numpy as np
        from importlib import import_module

        np.random.seed(123)
        module = import_module("actorob")
        payload = {
            "next_rand": f"{float(np.random.rand()):.17g}",
            "factory_loaded": "actorob.models.factory" in sys.modules,
            "inverse_dashboard_loaded": "actorob.dashboard.invdes.page" in sys.modules,
            "exports_dashboard": "dashboard" in module.__all__,
        }

        print(json.dumps(payload))
        """
    )

    import numpy as np

    np.random.seed(123)
    expected_next_rand = f"{float(np.random.rand()):.17g}"
    assert result == {
        "next_rand": expected_next_rand,
        "factory_loaded": False,
        "inverse_dashboard_loaded": False,
        "exports_dashboard": True,
    }
