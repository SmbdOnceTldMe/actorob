from __future__ import annotations

import numpy as np

from actorob.models.utils import expand_config, random_rgba


def test_expand_config_without_explicit_target_dict_does_not_leak_state():
    first = expand_config({"hip_pitch": "hip"})
    second = expand_config({"knee_pitch": "knee"})

    assert first == {
        "left_hip_pitch_joint": "hip",
        "right_hip_pitch_joint": "hip",
    }
    assert second == {
        "left_knee_pitch_joint": "knee",
        "right_knee_pitch_joint": "knee",
    }


def test_random_rgba_does_not_advance_global_numpy_rng_state():
    np.random.seed(123)

    _ = random_rgba()
    observed = np.random.rand(3)

    np.random.seed(123)
    expected = np.random.rand(3)

    assert np.allclose(observed, expected)


def test_random_rgba_accepts_explicit_rng_for_reproducible_colors():
    rgba_one = random_rgba(rng=np.random.default_rng(7))
    rgba_two = random_rgba(rng=np.random.default_rng(7))

    assert rgba_one == rgba_two
    assert rgba_one[-1] == 1.0
