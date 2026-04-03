from dataclasses import dataclass
import numpy as np

from .constants import (
    ROTOR_INERTIA_COEFF_A,
    ROTOR_INERTIA_COEFF_B,
    PEAK_MOTOR_TORQUE_COEFF_A1,
    PEAK_MOTOR_TORQUE_COEFF_A2,
    PEAK_MOTOR_TORQUE_COEFF_B,
    NO_LOAD_VELOCITY_COEFF_A,
    NO_LOAD_VELOCITY_COEFF_B,
    MOTOR_CONSTANT_COEFF_A,
    MOTOR_CONSTANT_COEFF_B,
    MOTOR_S_DIAMETER_COEFF_A,
    MOTOR_S_DIAMETER_COEFF_B,
    RHO_EFF,
    ALPHA,
    PEAK_TO_NOMINAL_TORQUE_RATIO,
    NO_LOAD_TO_NOMINAL_VELOCITY_RATIO,
)


@dataclass
class MotorParameters:
    """Class for keeping track of a motor's parameters."""

    mass: float  # Motor mass [kg]
    voltage: float = 48.0  # Motor voltage [V]

    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError("Motor mass must be positive.")

    @staticmethod
    def power_law(x: float, a: float, b: float) -> float:
        """Compute a power law: y = a * x^b."""
        return a * np.power(x, b)

    @property
    def rotor_inertia(self) -> float:
        """Rotor inertia [kg*m^2]."""
        return self.power_law(self.mass, ROTOR_INERTIA_COEFF_A, ROTOR_INERTIA_COEFF_B)

    @property
    def peak_torque(self) -> float:
        """Peak motor torque [Nm]."""
        return (
            PEAK_MOTOR_TORQUE_COEFF_A1 * self.mass
            + PEAK_MOTOR_TORQUE_COEFF_A2 * self.mass**2
            + PEAK_MOTOR_TORQUE_COEFF_B
        )

    @property
    def nominal_torque(self) -> float:
        """Nominal motor torque [Nm]."""
        return self.peak_torque * PEAK_TO_NOMINAL_TORQUE_RATIO

    @property
    def no_load_velocity(self) -> float:
        """No-load motor velocity [rad/s]."""
        return self.power_law(self.mass, NO_LOAD_VELOCITY_COEFF_A, NO_LOAD_VELOCITY_COEFF_B)

    @property
    def nominal_velocity(self) -> float:
        """Nominal motor velocity [rad/s]."""
        return self.no_load_velocity * NO_LOAD_TO_NOMINAL_VELOCITY_RATIO

    @property
    def motor_constant(self) -> float:
        """Motor constant [Nm/sqrt(W)]."""
        return self.power_law(self.mass, MOTOR_CONSTANT_COEFF_A, MOTOR_CONSTANT_COEFF_B)

    @property
    def stator_diameter(self) -> float:
        """Motor stator diameter [m]."""
        return self.power_law(self.mass, MOTOR_S_DIAMETER_COEFF_A, MOTOR_S_DIAMETER_COEFF_B)

    @property
    def radius(self) -> float:
        """Stator outer radius [m]."""
        return self.stator_diameter / 2

    @property
    def torque_constant(self) -> float:
        """Torque constant [Nm/A]."""
        # Dummy implementation, to be replaced with proper regression if needed
        return 1

    @property
    def resistance(self) -> float:
        """Motor resistance [Ohm]."""
        # Dummy implementation, to be replaced with proper regression if needed
        return 1

    @property
    def axial_length(self) -> float:
        """Compute the axial length based on the motor's hollow-cylinder model."""
        volume_factor = np.pi * self.radius**2 * (1 - ALPHA**2)
        length = self.mass / (RHO_EFF * volume_factor)
        return length  # TODO: check either clipping is needed or not
