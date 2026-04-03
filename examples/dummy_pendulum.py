"""
Example: 0.5-meter pendulum with a base actuator ("housing").
The actuator is created from ActuatorParameters → Actuatoract_unit and inserted into an MJCF model.
"""

import numpy as np
from actorob.actuators import MotorParameters, GearboxParameters, ActuatorParameters, compute_actuator_metrics
from actorob.utils import render
from actorob.models import ModelFactory


def main():
    # Initialize motor and gearbox parameters
    motor = MotorParameters(mass=0.8)
    gearbox = GearboxParameters(gear_ratio=12.0)
    act_params = ActuatorParameters(motor=motor, gearbox=gearbox, name="pendulum_actuator")
    act_unit = act_params.actuator_unit

    # Build the full XML model and load it into MuJoCo
    ACT_SET = {"pendulum_joint": act_unit}
    model_factory = ModelFactory("robots/pendulum/pendulum.xml", ACT_SET)
    model, data = model_factory.build()

    print(f"Model loaded: {model.nbody} bodies, {model.nu} actuators")

    # Initial pendulum angle (radians)
    data.qpos[0] = -1.0

    # Logging containers
    trq_list = []
    vel_list = []
    time_list = []

    # PD controller parameters
    Kp = 15.0
    Kd = 10.0
    q_des = 1.0  # desired pendulum position [rad]

    def control_callback(model, data):
        q, qdot = data.qpos[0], data.qvel[0]
        tau = Kp * (q_des - q) - Kd * qdot
        tau = np.clip(tau, *model.actuator_ctrlrange[0])
        data.ctrl[0] = tau
        trq_list.append(tau)
        vel_list.append(qdot)
        time_list.append(data.time)

    render(
        model,
        data,
        duration=2.0,
        sleep_dt=0.01,
        make_step=True,
        control_callback=control_callback,
    )

    # Evaluate actuator power and energy consumption
    actuator_metrics = compute_actuator_metrics(
        actuator=act_unit, torque=np.array(trq_list), velocity=np.array(vel_list), time=np.array(time_list)
    )

    print(f"Total electrical energy consumption: {actuator_metrics['E_elec']:.4f} [J]")


if __name__ == "__main__":
    main()
