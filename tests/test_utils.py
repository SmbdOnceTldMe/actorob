import pytest
from actorob.utils import radsec_to_rpm, rpm_to_radsec


def test_rad_sec_rpm_conversion():
    """Chceck conversion between rad/s and rpm."""
    omega = 100.0
    rpm = radsec_to_rpm(omega)
    assert rpm_to_radsec(rpm) == pytest.approx(omega)
