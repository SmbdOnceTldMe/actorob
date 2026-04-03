from __future__ import annotations

from ._lazy_import_helpers import run_probe


def test_trajectories_module_loads_only_requested_submodules():
    result = run_probe(
        """
        import json
        import sys
        from importlib import import_module

        module = import_module("actorob.trajectories")
        before = {
            "dir_has": "TrajectoryRunRecord" in dir(module),
            "record_loaded": "actorob.trajectories.record" in sys.modules,
            "optimizer_loaded": "actorob.trajectories.optimizer" in sys.modules,
        }

        record_cls = module.TrajectoryRunRecord
        after = {
            "record_loaded": "actorob.trajectories.record" in sys.modules,
            "optimizer_loaded": "actorob.trajectories.optimizer" in sys.modules,
            "cached": module.__dict__["TrajectoryRunRecord"] is record_cls,
            "name": record_cls.__name__,
        }

        print(json.dumps({"before": before, "after": after}))
        """
    )

    assert result["before"] == {
        "dir_has": True,
        "record_loaded": False,
        "optimizer_loaded": False,
    }
    assert result["after"] == {
        "record_loaded": True,
        "optimizer_loaded": False,
        "cached": True,
        "name": "TrajectoryRunRecord",
    }


def test_models_module_can_expose_expand_config_without_loading_factory():
    result = run_probe(
        """
        import json
        import sys
        from importlib import import_module

        module = import_module("actorob.models")
        before = {
            "dir_has": "expand_config" in dir(module),
            "factory_loaded": "actorob.models.factory" in sys.modules,
            "utils_loaded": "actorob.models.utils" in sys.modules,
        }

        expand_config = module.expand_config
        after = {
            "factory_loaded": "actorob.models.factory" in sys.modules,
            "utils_loaded": "actorob.models.utils" in sys.modules,
            "cached": module.__dict__["expand_config"] is expand_config,
            "callable": callable(expand_config),
        }

        print(json.dumps({"before": before, "after": after}))
        """
    )

    assert result["before"] == {
        "dir_has": True,
        "factory_loaded": False,
        "utils_loaded": False,
    }
    assert result["after"] == {
        "factory_loaded": False,
        "utils_loaded": True,
        "cached": True,
        "callable": True,
    }


def test_models_module_can_expose_mjcf_asset_copy_without_loading_factory():
    result = run_probe(
        """
        import json
        import sys
        from importlib import import_module

        module = import_module("actorob.models")
        before = {
            "dir_has": "copy_mjcf_with_resolved_assets" in dir(module),
            "factory_loaded": "actorob.models.factory" in sys.modules,
            "mjcf_assets_loaded": "actorob.models.mjcf_assets" in sys.modules,
        }

        helper = module.copy_mjcf_with_resolved_assets
        after = {
            "factory_loaded": "actorob.models.factory" in sys.modules,
            "mjcf_assets_loaded": "actorob.models.mjcf_assets" in sys.modules,
            "cached": module.__dict__["copy_mjcf_with_resolved_assets"] is helper,
            "callable": callable(helper),
        }

        print(json.dumps({"before": before, "after": after}))
        """
    )

    assert result["before"] == {
        "dir_has": True,
        "factory_loaded": False,
        "mjcf_assets_loaded": False,
    }
    assert result["after"] == {
        "factory_loaded": False,
        "mjcf_assets_loaded": True,
        "cached": True,
        "callable": True,
    }
