from typing import Optional, Protocol, Sequence

import numpy as np

from .constants import MOTOR_EXTRA_LOSS_RATIO
from .unit import ActuatorUnit


class ActuatorMetricLike(Protocol):
    """Minimal actuator interface required for metric evaluation."""

    motor_constant: float
    nominal_torque: float
    gear_ratio: float
    damping: float
    gearbox_efficiency: float


def _integrate_power(power: np.ndarray, time: Optional[np.ndarray] = None, dt: Optional[float] = None) -> float:
    """Integrate power over time to obtain energy [J]."""
    if time is not None:
        time = np.asarray(time)
        return np.trapezoid(power, x=time)
    elif dt is not None:
        return np.trapezoid(power, dx=dt)
    else:
        raise ValueError("Either 'time' or 'dt' must be provided for energy integration.")


def _evaluate_motor_losses(
    motor_constant: float,
    motor_nominal_torque: float,
    motor_torque_profile: np.ndarray,
    time: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Evaluate motor power and energy losses over time.

    Args:
        motor_constant (float): Motor constant [Nm/sqrt(W)].
        motor_nominal_torque (float): Nominal motor torque [Nm].
        motor_torque_profile (np.ndarray): Array of motor torque values [Nm].
        time (Optional[np.ndarray]): Time array corresponding to torque values [s].
        dt (Optional[float]): Time step between samples [s]. Used if time array is not provided.
    Returns:
        dict[str, np.ndarray]: Dictionary containing power and energy losses:
            - "P_motor_heat": Copper heating losses [W].
            - "P_motor_extra": Other losses [W].
            - "P_motor_total": Total losses [W].
            - "E_motor_heat": Copper heating energy losses [J].
            - "E_motor_extra": Other energy losses [J].
            - "E_motor_total": Total energy losses [J].
    """
    torque = np.asarray(motor_torque_profile)

    # Copper (Joule) losses [W]
    p_heat = torque**2 / motor_constant**2

    # Additional (magnetic, friction, eddy current) losses [W]
    p_heat_nom = motor_nominal_torque**2 / motor_constant**2
    p_extra = np.full_like(p_heat, p_heat_nom * MOTOR_EXTRA_LOSS_RATIO)

    # Total losses
    p_total = p_heat + p_extra

    return {
        "P_motor_heat": p_heat,
        "P_motor_extra": p_extra,
        "P_motor_total": p_total,
        "E_motor_heat": _integrate_power(p_heat, time, dt),
        "E_motor_extra": _integrate_power(p_extra, time, dt),
        "E_motor_total": _integrate_power(p_total, time, dt),
    }


def _evaluate_gearbox_losses(
    gearbox_damping: float,
    gearbox_efficiency: float,
    actuator_velocity_profile: np.ndarray,
    actuator_nominal_torque: float,
    time: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Evaluate gearbox friction losses.
    Args:
        gearbox_damping (float): Gearbox damping coefficient [Nm/(rad/s)].
        gearbox_efficiency (float): Gearbox efficiency [0-1].
        actuator_velocity_profile (np.ndarray): Angular velocity at gearbox output [rad/s].
        actuator_nominal_torque (float): Nominal torque at gearbox output [Nm].
        time (Optional[np.ndarray]): Time vector [s]. Used for energy calculation.
        dt (Optional[float]): Time step between samples [s]. Used if time array is not provided.
    Returns:
        dict[str, np.ndarray]: Dictionary containing power and energy losses:
            - "P_fric": Friction power losses [W].
            - "E_fric": Friction energy losses [J].
    """
    velocity = np.asarray(actuator_velocity_profile)

    # Torque losses [Nm]
    tau_fric = actuator_nominal_torque * ((1 - gearbox_efficiency) / gearbox_efficiency)

    # Dry + viscous friction
    tau_dry = tau_fric * np.sign(velocity)
    tau_viscous = gearbox_damping * velocity

    p_fric = (tau_dry + tau_viscous) * velocity

    return {
        "P_fric": p_fric,
        "E_fric": _integrate_power(p_fric, time, dt),
    }


def compute_actuator_metrics(
    actuator: ActuatorUnit,
    torque: np.ndarray,
    velocity: np.ndarray,
    time: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Compute actuator metrics including mechanical power, losses and electrical consumption.
    Args:
        actuator (ActuatorUnit): Actuator unit parameters.
        torque (np.ndarray): Actuator output torque [Nm].
        velocity (float): Actuator output angular velocity [rad/s].
        time (Optional[np.ndarray]): Time vector [s].
        dt (Optional[float]): Time step [s].

    Returns:
        dict[str, np.ndarray]: Dictionary containing power and energy metrics:
        - "P_mech": Mechanical power [W]
        - "P_motor_losses": Motor losses [W]
        - "P_gearbox_losses": Gearbox losses [W]
        - "P_elec": Electrical power consumption [W]
        - "E_motor_losses": Motor losses [J]
        - "E_gearbox_losses": Gearbox losses [J]
        - "E_elec": Electrical energy consumption [J]
        - "efficiency": Mechanical efficiency [0-1]
    """
    # Mechanical output power [W] (Only positive power)
    p_mech = np.maximum(0, torque * velocity)
    e_mech = _integrate_power(p_mech, time, dt)

    # Evaluate losses
    motor_losses = _evaluate_motor_losses(
        motor_constant=actuator.motor_constant,
        motor_nominal_torque=actuator.nominal_torque / actuator.gear_ratio,
        motor_torque_profile=torque / actuator.gear_ratio,
        time=time,
        dt=dt,
    )

    gearbox_losses = _evaluate_gearbox_losses(
        actuator.damping,
        actuator.gearbox_efficiency,
        actuator_velocity_profile=velocity,
        actuator_nominal_torque=actuator.nominal_torque,
        time=time,
        dt=dt,
    )

    # Total electrical input power [W]
    p_elec = p_mech + motor_losses["P_motor_total"] + gearbox_losses["P_fric"]

    # Integrate power
    e_elec = _integrate_power(p_elec, time, dt)

    # Instantaneous mechanical efficiency
    eta_mech = np.divide(p_mech, p_elec, out=np.zeros_like(p_mech), where=p_elec != 0)

    return {
        "P_mech": p_mech,
        "P_motor_losses": motor_losses["P_motor_total"],
        "P_gearbox_losses": gearbox_losses["P_fric"],
        "P_elec": p_elec,
        "E_mech": e_mech,
        "E_motor_losses": motor_losses["E_motor_total"],
        "E_gearbox_losses": gearbox_losses["E_fric"],
        "E_elec": e_elec,
        "efficiency": eta_mech,
    }


def compute_actuator_group_metrics(
    actuators: Sequence[ActuatorMetricLike | None],
    torque: np.ndarray,
    velocity: np.ndarray,
    time: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Aggregate actuator metrics across multiple joints over one trajectory."""

    torque_array = np.asarray(torque, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)

    if torque_array.ndim != 2:
        raise ValueError(f"Expected torque to be a 2D array, got shape {torque_array.shape}.")
    if velocity_array.ndim != 2:
        raise ValueError(f"Expected velocity to be a 2D array, got shape {velocity_array.shape}.")
    if velocity_array.shape[0] < torque_array.shape[0]:
        raise ValueError("Velocity profile must contain at least as many time steps as torque.")
    if torque_array.shape[1] != velocity_array.shape[1]:
        raise ValueError("Torque and velocity profiles must have the same joint dimension.")
    if len(actuators) < torque_array.shape[1]:
        raise ValueError("Actuator list must cover the joint dimension of the provided profiles.")

    step_count = torque_array.shape[0]
    velocity_array = velocity_array[:step_count]
    time_array = None if time is None else np.asarray(time, dtype=float)[:step_count]

    p_mech = np.zeros(step_count, dtype=float)
    p_motor_losses = np.zeros(step_count, dtype=float)
    p_gearbox_losses = np.zeros(step_count, dtype=float)
    p_elec = np.zeros(step_count, dtype=float)
    e_mech = 0.0
    e_motor_losses = 0.0
    e_gearbox_losses = 0.0
    e_elec = 0.0

    for joint_idx, actuator in enumerate(actuators[: torque_array.shape[1]]):
        if actuator is None:
            continue
        metrics = compute_actuator_metrics(
            actuator,
            torque_array[:, joint_idx],
            velocity_array[:, joint_idx],
            time=time_array,
            dt=dt,
        )
        p_mech += np.asarray(metrics["P_mech"], dtype=float)
        p_motor_losses += np.asarray(metrics["P_motor_losses"], dtype=float)
        p_gearbox_losses += np.asarray(metrics["P_gearbox_losses"], dtype=float)
        p_elec += np.asarray(metrics["P_elec"], dtype=float)
        e_mech += float(metrics["E_mech"])
        e_motor_losses += float(metrics["E_motor_losses"])
        e_gearbox_losses += float(metrics["E_gearbox_losses"])
        e_elec += float(metrics["E_elec"])

    efficiency = np.divide(p_mech, p_elec, out=np.zeros_like(p_mech), where=p_elec != 0)

    return {
        "P_mech": p_mech,
        "P_motor_losses": p_motor_losses,
        "P_gearbox_losses": p_gearbox_losses,
        "P_elec": p_elec,
        "E_mech": e_mech,
        "E_motor_losses": e_motor_losses,
        "E_gearbox_losses": e_gearbox_losses,
        "E_elec": e_elec,
        "efficiency": efficiency,
    }
