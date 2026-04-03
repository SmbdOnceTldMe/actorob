from __future__ import annotations

import re


def parse_gear_ratio_from_name(name: str) -> float | None:
    """Extract the encoded gear ratio from an actuator name."""

    match = re.search(r"-P\d+-(\d+(?:\.\d+)?)", name)
    return None if match is None else float(match.group(1))


def infer_motor_mass_from_total_actuator_mass(total_mass: float, gear_ratio: float) -> float:
    """Infer motor mass that reproduces a measured total actuator mass."""

    from .actuator import ActuatorParameters
    from .gearbox import GearboxParameters
    from .motor import MotorParameters

    lo, hi = 0.01, max(total_mass, 0.02) * 4.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        predicted = ActuatorParameters(MotorParameters(mid), GearboxParameters(gear_ratio)).mass
        if predicted < total_mass:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
