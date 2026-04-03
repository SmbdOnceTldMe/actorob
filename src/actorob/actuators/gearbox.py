from dataclasses import dataclass, field
import numpy as np
from .constants import EFFICIENCY_SLOPE, EFFICIENCY_INTERCEPT, ONE_STAGE_RATIO


@dataclass
class GearboxParameters:
    """Geometric and efficiency parameters for a gearbox model."""

    gear_ratio: float  # Gear ratio

    # Private attributes to store computed properties
    _mass: float = field(init=False, default=0.0)  # Gearbox mass [kg]
    _volume: float = field(init=False, default=0.0)  # Gearbox volume [m^3]
    _length: float = field(init=False, default=0.0)  # Gearbox length [m]

    def __post_init__(self):
        if self.gear_ratio <= 0:
            raise ValueError("Gear ratio must be greater than 0.")

    @property
    def efficiency(self) -> float:
        """Gearbox efficiency [0-1]."""
        efficiency = EFFICIENCY_SLOPE * self.n_stages + EFFICIENCY_INTERCEPT
        if efficiency < 0 or efficiency > 1:
            raise ValueError("Computed efficiency is out of bounds [0, 1].")
        return efficiency

    @property
    def n_stages(self) -> int:
        """Number of stages in the gearbox."""
        return max(1, int(np.ceil(np.log10(self.gear_ratio) / np.log10(ONE_STAGE_RATIO))))

    @property
    def mass(self) -> float:
        """Gearbox mass [kg]."""
        return self._mass

    @mass.setter
    def mass(self, value: float):
        """Store the inferred gearbox mass [kg]."""

        if value <= 0:
            raise ValueError("Mass must be positive.")
        self._mass = value

    @property
    def volume(self) -> float:
        """Gearbox volume [m^3]."""
        return self._volume

    @volume.setter
    def volume(self, value: float):
        """Store the inferred gearbox volume [m^3]."""

        if value <= 0:
            raise ValueError("Volume must be positive.")
        self._volume = value

    @property
    def length(self) -> float:
        """Gearbox length [m]."""
        return self._length

    @length.setter
    def length(self, value: float):
        """Store the inferred gearbox length [m]."""

        if value <= 0:
            raise ValueError("Length must be positive.")
        self._length = value

    @property
    def damping_coefficient(self) -> float:
        """Viscous damping coefficient [Nm/(rad/s)]."""
        # TODO: implement proper damping model
        return 0.0
