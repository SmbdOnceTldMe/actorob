import numpy as np


def rpm_to_radsec(rpm: float | np.ndarray) -> float | np.ndarray:
    """Convert angular velocity from rpm to rad/s."""
    return np.asarray(rpm) * (2 * np.pi) / 60.0


def radsec_to_rpm(omega: float | np.ndarray) -> float | np.ndarray:
    """Convert angular velocity from rad/s to rpm."""
    return np.asarray(omega) * 60.0 / (2 * np.pi)
