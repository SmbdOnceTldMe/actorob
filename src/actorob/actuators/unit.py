from typing import List, Optional
from actorob.utils import rpm_to_radsec
from math import isclose


class ActuatorPosition:
    """Pose parameters for placing an actuator (or its housing) in a model.

    This lightweight container holds a position vector and an optional
    orientation quaternion that describe where the actuator is mounted.
    These values are later consumed by the ``ModelFactory`` to attach the
    actuator to the specified body in the kinematic tree.

    Args:
        pos (List[float]):
            3D position of the actuator origin relative to the parent body
            frame, in meters. Expected shape: ``[x, y, z]``.
        quat (Optional[List[float]]):
            Orientation of the actuator frame as a unit quaternion.
            Expected order: ``[w, x, y, z]`` (right-handed, unit length).
            If ``None``, an identity rotation is assumed by downstream code.
        parent_body_name (Optional[str]):
            Name of the parent (mounting) body to which the actuator is
            attached. If ``None``, the factory may interpret this as parent of parent frame
            from joint location depending on the model configuration.
            For example hip_roll link consists hip_roll_joint and parent link for hip_roll will be hip_pitch.
            In this case the parent_body name will be hip_pitch.

    Attributes:
        pos (List[float]): Position vector in meters.
        quat (Optional[List[float]]): Orientation quaternion ``[w, x, y, z]``.
        body_name (Optional[str]): Name of the parent body (alias stored from
            ``parent_body_name``).

    Notes:
        * No validation is performed here (e.g., vector length, quaternion
          normalization). The ``ModelFactory`` or downstream utilities are
          expected to validate and/or normalize if required.
        * Coordinate-frame conventions (e.g., world vs. parent-local)
          are assumed to be consistent with the factory that consumes this
          object.
    """

    def __init__(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        parent_body_name: Optional[str] = None,
    ):
        self.pos = pos
        self.quat = quat
        self.body_name = parent_body_name


class ActuatorUnit:
    """Electrical and mechanical parameters of a single actuator.

    This class aggregates motor, gearing, and packaging data used by the
    ``ModelFactory`` to synthesize joint dynamics (inertia reflection,
    limits, viscous/friction losses) and to parameterize low-level drivers.
    It also computes simulation-friendly angular velocities from RPM.

    Args:
        name (str):
            Human-readable actuator identifier (unique within a model).
        gear_ratio (float):
            Gear ratio from motor shaft to output shaft (dimensionless).
            Use ``> 1`` when the motor spins faster than the output.
        nominal_torque (float):
            Continuous (rated) torque at the *output shaft*, in N·m.
        max_torque (float):
            Peak (instantaneous) torque at the *output shaft*, in N·m.
        nominal_velocity (float):
            Continuous (rated) speed at the *output shaft*, in RPM.
        max_velocity (float):
            Maximum permissible speed at the *output shaft*, in RPM.
        torque_constant (float):
            Motor torque constant, in N·m/A (at the *motor* shaft).
            If a driver model is used, this relates phase current to torque.
        mass (float):
            Actuator total mass (including gearbox and housing), in kg.
        diameter (float):
            Outer diameter of the actuator package, in meters.
        length (float):
            Overall axial length of the actuator package, in meters.
        motor_constant (Optional[float]):
            Motor constant ``K_m`` in SI (N·m/√W) or equivalently
            ``sqrt( torque_constant / resistance ) * ...`` depending on
            convention. If provided, used by power/thermal models.
        rotor_inertia (Optional[float]):
            Rotor inertia *at the motor shaft*, in kg·m². If given,
            the output-reflected armature inertia is computed internally as
            ``rotor_inertia * gear_ratio**2`` when ``armature`` is not set.
        armature (Optional[float]):
            Output-reflected rotational inertia (a.k.a. armature inertia) at
            the *output shaft*, in kg·m². If ``None`` while
            ``rotor_inertia`` is provided, it is derived as above.
        damping (Optional[float]):
            Viscous damping at the *output shaft*, in N·m·s/rad.
            Defaults to ``0.1``.
        friction_loss (Optional[float]):
            Coulomb/offset friction loss at the *output shaft*, in N·m.
            Defaults to ``0.1``.
        resistance (Optional[float]):
            Winding (phase) resistance, in ohms. Optional unless required by
            your driver/thermal model.
        voltage (Optional[float]):
            Nominal supply (bus) voltage, in volts.
        ipos (Optional[List[float]]):
            Optional initial pose hint for packaging (e.g., local placement),
            typically ``[x, y, z]`` in meters. Used by some factories/visuals.
        vendor (Optional[str]):
            Vendor/manufacturer string. Defaults to ``"custom_vendor"``.
        model (Optional[str]):
            Vendor model/PN string. Defaults to ``"custom_model"``.

    Raises:
        ValueError:
            If both ``armature`` and ``rotor_inertia`` are ``None``. At least
            one must be provided so that the output-reflected inertia can be
            determined.

    Attributes:
        name (str): Actuator identifier.
        gear_ratio (float): Gear ratio (motor → output).
        nominal_torque (float): Rated output torque [N·m].
        max_torque (float): Peak output torque [N·m].
        nominal_velocity (float): Rated output speed [RPM].
        max_velocity (float): Max output speed [RPM].
        torque_constant (float): Motor torque constant [N·m/A].
        motor_constant (Optional[float]): Motor constant (SI).
        rotor_inertia (Optional[float]): Rotor inertia at motor shaft [kg·m²].
        armature (float): Output-reflected inertia [kg·m²]. If not supplied,
            computed as ``rotor_inertia * gear_ratio**2``.
        damping (float): Viscous damping at output [N·m·s/rad].
        friction_loss (float): Coulomb/offset friction at output [N·m].
        gearbox_efficiency (float): Gearbox efficiency [0-1].
        resistance (Optional[float]): Winding resistance [Ω].
        voltage (Optional[float]): Bus voltage [V].
        diameter (float): Package diameter [m].
        length (float): Package length [m].
        ipos (Optional[List[float]]): Optional initial placement vector [m].
        mass (float): Total actuator mass [kg].
        vendor (str): Vendor/manufacturer label.
        model (str): Vendor model identifier.
        nominal_velocity_sim (float): Rated speed converted to rad/s
            (computed via ``rpm_to_radsec``).
        max_velocity_sim (float): Max speed converted to rad/s
            (computed via ``rpm_to_radsec``).

    Notes:
        * All torques and speeds are documented here at the **output shaft**,
          i.e., *after* gearing, unless explicitly marked as motor-side.
        * Provide **either** ``armature`` (output-side inertia) **or**
          ``rotor_inertia`` (motor-side inertia). If both are given,
          ``armature`` is used as provided and not overridden.
        * Parameter names and units are chosen to be directly consumable by
          typical rigid-body simulators and by the project's ``ModelFactory``.
    """

    def __init__(
        self,
        name: str,
        gear_ratio: float,
        nominal_torque: float,
        max_torque: float,
        nominal_velocity: float,
        max_velocity: float,
        torque_constant: float,
        mass: float,
        diameter: float,
        length: float,
        motor_constant: Optional[float] = None,
        rotor_inertia: Optional[float] = None,
        armature: Optional[float] = None,
        damping: Optional[float] = 0.1,
        friction_loss: Optional[float] = 0.1,
        gearbox_efficiency: Optional[float] = 0.9,
        resistance: Optional[float] = None,
        voltage: Optional[float] = None,
        ipos: Optional[List[float]] = None,
        vendor: Optional[str] = "custom_vendor",
        model: Optional[str] = "custom_model",
    ):
        if armature is None and rotor_inertia is None:
            raise ValueError(f"Set armature or rotor inertia in actuator unit {name}")

        self.name = name
        self.gear_ratio = gear_ratio
        self.nominal_torque = nominal_torque
        self.max_torque = max_torque
        self.nominal_velocity = nominal_velocity
        self.max_velocity = max_velocity
        self.torque_constant = torque_constant
        self.motor_constant = motor_constant
        self.rotor_inertia = rotor_inertia
        self.armature = armature or self.rotor_inertia * self.gear_ratio**2
        self.damping = damping
        self.friction_loss = friction_loss
        self.gearbox_efficiency = gearbox_efficiency
        self.resistance = resistance
        self.voltage = voltage
        self.diameter = diameter
        self.length = length
        self.ipos = ipos
        self.mass = mass
        self.vendor = vendor
        self.model = model

        self.nominal_velocity_sim = rpm_to_radsec(self.nominal_velocity)
        self.max_velocity_sim = rpm_to_radsec(self.max_velocity)

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, ActuatorUnit):
            return NotImplemented
        fields = (
            "nominal_torque",
            "max_torque",
            "nominal_velocity",
            "max_velocity",
            "torque_constant",
            "gear_ratio",
            "motor_constant",
            "rotor_inertia",
            "resistance",
            "voltage",
            "diameter",
            "length",
            "mass",
        )
        for f in fields:
            a, b = getattr(self, f), getattr(other, f)
            if a is None or b is None:
                if a != b:
                    return False
            else:
                if not isclose(a, b, rel_tol=0.1, abs_tol=0.1):
                    return False
        return True
