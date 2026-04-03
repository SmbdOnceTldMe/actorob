from dataclasses import dataclass
from actorob.actuators import MotorParameters, GearboxParameters
from .constants import (
    MARGIN_LENGTH,
    MARGIN_DIAMETER,
    MARGIN_MASS,
    RHO_STEEL,
    GEARBOX_VOLUME_COEFF_A,
    GEARBOX_VOLUME_COEFF_B,
    GEARBOX_LENGTH_COEFF_A,
    GEARBOX_LENGTH_COEFF_B,
)
from .unit import ActuatorUnit
from actorob.utils import power_law, radsec_to_rpm

DEFAULT_OUTPUT_DAMPING_COEFFICIENT = 0.01
DEFAULT_OUTPUT_FRICTION_COEFFICIENT = 0.01


@dataclass
class ActuatorParameters:
    """Combined motor and gearbox parameters for one synthesized actuator."""

    motor: MotorParameters
    gearbox: GearboxParameters
    name: str = "custom_actuator"
    model: str = "custom_model"
    vendor: str = "custom_vendor"

    def __post_init__(self):
        # 1. Compute gearbox volume by known motor's outer diameter
        self.gearbox.volume = power_law(self.motor.stator_diameter, GEARBOX_VOLUME_COEFF_A, GEARBOX_VOLUME_COEFF_B)

        # 2. Compute gearbox length by known motor's outer diameter
        self.gearbox.length = (
            power_law(self.motor.stator_diameter, GEARBOX_LENGTH_COEFF_A, GEARBOX_LENGTH_COEFF_B)
            * self.gearbox.n_stages
        )

        # 3. Compute gearbox mass by known volume and material density
        self.gearbox.mass = self.gearbox.volume * RHO_STEEL * self.gearbox.n_stages

    @property
    def mass(self) -> float:
        """Total actuator mass [kg] with margin."""
        return (self.motor.mass + self.gearbox.mass) * MARGIN_MASS

    @property
    def length(self) -> float:
        """Total actuator length [m] with margin."""
        return (self.motor.axial_length + self.gearbox.length) * MARGIN_LENGTH

    @property
    def diameter(self) -> float:
        """Total actuator diameter [m] with margin."""
        return self.motor.stator_diameter * MARGIN_DIAMETER

    @property
    def reflected_inertia(self) -> float:
        """Reflected inertia at the actuator output [kg*m^2]."""
        return self.motor.rotor_inertia * (self.gearbox.gear_ratio**2)

    @property
    def peak_torque(self) -> float:
        """Peak actuator torque at the output [Nm]."""
        return self.motor.peak_torque * self.gearbox.gear_ratio

    @property
    def nominal_torque(self) -> float:
        """Nominal actuator torque at the output [Nm]."""
        return self.motor.nominal_torque * self.gearbox.gear_ratio

    @property
    def no_load_velocity(self) -> float:
        """No-load actuator velocity at the output [rad/s]."""
        return self.motor.no_load_velocity / self.gearbox.gear_ratio

    @property
    def nominal_velocity(self) -> float:
        """Nominal actuator velocity at the output [rad/s]."""
        return self.motor.nominal_velocity / self.gearbox.gear_ratio

    @property
    def loss_torque(self) -> float:
        """Loss torque at the actuator output [Nm]."""
        eta = self.gearbox.efficiency
        return self.nominal_torque * ((1 - eta) / eta)

    @property
    def damping_coefficient(self) -> float:
        """Damping coefficient at the actuator output [Nm/(rad/s)].

        The open-source artifact currently ships with an explicit placeholder
        value until a fitted damping model is introduced.
        """
        return DEFAULT_OUTPUT_DAMPING_COEFFICIENT

    @property
    def friction_coefficient(self) -> float:
        """Coulomb friction coefficient for the MJCF model.

        The open-source artifact currently ships with an explicit placeholder
        value until a fitted friction model is introduced.
        """
        return DEFAULT_OUTPUT_FRICTION_COEFFICIENT

    @property
    def actuator_unit(self) -> ActuatorUnit:
        """Convert the synthesized actuator into an ``ActuatorUnit`` record."""

        return ActuatorUnit(
            name=self.name,
            gear_ratio=self.gearbox.gear_ratio,
            nominal_torque=self.nominal_torque,
            max_torque=self.peak_torque,
            nominal_velocity=radsec_to_rpm(self.nominal_velocity),
            max_velocity=radsec_to_rpm(self.no_load_velocity),
            # ActuatorUnit expects motor-side rotor inertia and derives the
            # reflected joint-side armature internally unless it is supplied.
            rotor_inertia=self.motor.rotor_inertia,
            armature=self.reflected_inertia,
            damping=self.damping_coefficient,
            friction_loss=self.friction_coefficient,
            gearbox_efficiency=self.gearbox.efficiency,
            torque_constant=self.motor.torque_constant,
            motor_constant=self.motor.motor_constant,
            resistance=self.motor.resistance,
            voltage=self.motor.voltage,
            diameter=self.diameter,
            length=self.length,
            mass=self.mass,
            vendor=self.vendor,
            model=self.model,
        )
