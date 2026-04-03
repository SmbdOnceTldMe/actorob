from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

from actorob.invdes.optuna_adapter import OptunaCmaEsStudyFactory


class OptunaCmaEsStudyFactoryTest(unittest.TestCase):
    def test_factory_raises_clear_error_when_optuna_is_missing(self) -> None:
        factory = OptunaCmaEsStudyFactory(seed=123)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "optuna" or name.startswith("optuna."):
                raise ImportError("mocked missing optuna")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "Install 'optuna' and 'cmaes'"):
                factory.create()
