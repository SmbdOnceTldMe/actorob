from types import SimpleNamespace

import numpy as np

from actorob.actuators import compute_actuator_group_metrics, compute_actuator_metrics


def _metric_actuator(
    *,
    motor_constant: float,
    nominal_torque: float,
    gear_ratio: float,
    damping: float,
    gearbox_efficiency: float,
):
    return SimpleNamespace(
        motor_constant=motor_constant,
        nominal_torque=nominal_torque,
        gear_ratio=gear_ratio,
        damping=damping,
        gearbox_efficiency=gearbox_efficiency,
    )


def test_group_metrics_match_sum_of_individual_actuator_metrics():
    left = _metric_actuator(
        motor_constant=2.0,
        nominal_torque=3.0,
        gear_ratio=2.0,
        damping=0.1,
        gearbox_efficiency=0.8,
    )
    right = _metric_actuator(
        motor_constant=1.5,
        nominal_torque=2.0,
        gear_ratio=1.0,
        damping=0.05,
        gearbox_efficiency=0.9,
    )
    torque = np.array([[1.0, 0.5, 2.0], [1.5, 0.25, 1.0]], dtype=float)
    velocity = np.array([[0.2, 0.1, 0.4], [0.3, 0.2, 0.5]], dtype=float)

    combined = compute_actuator_group_metrics([left, None, right], torque, velocity, dt=0.1)
    left_metrics = compute_actuator_metrics(left, torque[:, 0], velocity[:, 0], dt=0.1)
    right_metrics = compute_actuator_metrics(right, torque[:, 2], velocity[:, 2], dt=0.1)

    np.testing.assert_allclose(combined["P_mech"], left_metrics["P_mech"] + right_metrics["P_mech"])
    np.testing.assert_allclose(
        combined["P_motor_losses"],
        left_metrics["P_motor_losses"] + right_metrics["P_motor_losses"],
    )
    np.testing.assert_allclose(
        combined["P_gearbox_losses"],
        left_metrics["P_gearbox_losses"] + right_metrics["P_gearbox_losses"],
    )
    np.testing.assert_allclose(combined["P_elec"], left_metrics["P_elec"] + right_metrics["P_elec"])
    assert combined["E_motor_losses"] == left_metrics["E_motor_losses"] + right_metrics["E_motor_losses"]
    assert combined["E_gearbox_losses"] == left_metrics["E_gearbox_losses"] + right_metrics["E_gearbox_losses"]
    assert combined["E_elec"] == left_metrics["E_elec"] + right_metrics["E_elec"]
