import numpy as np


def power_law(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """
    Compute a power law: y = a * x^b.

    Args:
        x: Input value(s).
        a: Coefficient (scale factor).
        b: Exponent.

    Returns:
        y: Computed power law value(s).
    """
    return a * np.power(x, b)
