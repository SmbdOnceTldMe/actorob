import pytest
from actorob.actuators import MotorParameters, GearboxParameters


@pytest.fixture
def real_motor():
    """Realistic motor parameters for tests."""
    motor = MotorParameters(
        mass=0.5,
    )
    return motor


@pytest.fixture
def real_gearbox():
    """Realistic gearbox parameters for tests."""
    gearbox = GearboxParameters(
        gear_ratio=10.0,
    )
    return gearbox
