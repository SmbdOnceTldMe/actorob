"""Actuator design-space preparation for inverse design."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from actorob.invdes.evaluation.types import PreparedCandidate
from actorob.invdes.problem import FloatParameter


@dataclass(frozen=True)
class ActuatorDesignVariable:
    """Bounded inverse-design variables for one actuator joint family."""

    joint_family: str
    motor_mass_bounds: tuple[float, float]
    gear_ratio_bounds: tuple[float, float]
    initial_motor_mass: float
    initial_gear_ratio: float

    def __post_init__(self) -> None:
        if self.motor_mass_bounds[0] >= self.motor_mass_bounds[1]:
            raise ValueError("motor_mass_bounds must be an increasing pair.")
        if self.gear_ratio_bounds[0] >= self.gear_ratio_bounds[1]:
            raise ValueError("gear_ratio_bounds must be an increasing pair.")


class ActuatorPreparer:
    """Maps design variables to an internal MJCF with updated actuators."""

    def __init__(
        self,
        *,
        config: Any,
        design_variables: Sequence[ActuatorDesignVariable],
        model_factory_cls: type | None = None,
        actuator_builder: Callable[[str, float, float], Any] | None = None,
        expand_config_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._design_variables = tuple(design_variables)
        self._model_factory_cls = model_factory_cls or _load_model_factory()
        self._actuator_builder = actuator_builder or _default_actuator_builder
        self._expand_config_fn = expand_config_fn or _expand_actuator_config

    def build_parameters(self) -> tuple[FloatParameter, ...]:
        """Build the continuous optimization parameters for all design variables."""

        parameters: list[FloatParameter] = []
        for variable in self._design_variables:
            parameters.append(
                FloatParameter(
                    name=f"m_{variable.joint_family}",
                    low=float(variable.motor_mass_bounds[0]),
                    high=float(variable.motor_mass_bounds[1]),
                )
            )
            parameters.append(
                FloatParameter(
                    name=f"g_{variable.joint_family}",
                    low=float(variable.gear_ratio_bounds[0]),
                    high=float(variable.gear_ratio_bounds[1]),
                )
            )
        return tuple(parameters)

    def initial_guess_by_name(self) -> dict[str, float]:
        """Return the nominal initial guess keyed by optimization parameter name."""

        return {
            f"m_{variable.joint_family}": float(variable.initial_motor_mass) for variable in self._design_variables
        } | {f"g_{variable.joint_family}": float(variable.initial_gear_ratio) for variable in self._design_variables}

    def prepare(self, candidate: tuple[float, ...]) -> PreparedCandidate:
        """Generate an MJCF-backed runtime config for one candidate vector."""

        actuators = self._build_actuator_map(candidate)
        model_factory = self._model_factory_cls(self._config.base.mjcf_path, actuators)
        postfix = f"invdes_{uuid4().hex}"
        model_factory.postfix = postfix
        model_factory.save()

        generated_mjcf_path = Path(model_factory.xml_dir) / f"{model_factory.name}_{postfix}.xml"
        config = _replace_mjcf_path(self._config, str(generated_mjcf_path), actuators=actuators)
        return PreparedCandidate(
            config=config,
            solution=tuple(float(value) for value in candidate),
            metadata={
                "actuators": actuators,
                "generated_mjcf_path": str(generated_mjcf_path),
            },
        )

    def _build_actuator_map(self, candidate: tuple[float, ...]) -> dict[str, Any]:
        if len(candidate) != 2 * len(self._design_variables):
            raise ValueError(f"Expected {2 * len(self._design_variables)} design values, got {len(candidate)}.")

        grouped: dict[str, Any] = {}
        cursor = 0
        for variable in self._design_variables:
            mass = float(candidate[cursor])
            gear_ratio = float(candidate[cursor + 1])
            cursor += 2
            grouped[variable.joint_family] = self._actuator_builder(variable.joint_family, mass, gear_ratio)

        return self._expand_config_fn(grouped)


def default_actuator_design_variables() -> tuple[ActuatorDesignVariable, ...]:
    """Return the default actuator design space used by the inverse-design flow."""

    return (
        ActuatorDesignVariable("hip_roll", (0.15, 1.5), (10.0, 70.0), 1.0, 24.0),
        ActuatorDesignVariable("hip_pitch", (0.15, 1.5), (10.0, 70.0), 1.0, 24.0),
        ActuatorDesignVariable("knee_pitch", (0.15, 1.5), (10.0, 70.0), 1.0, 24.0),
    )


def _replace_mjcf_path(config: Any, mjcf_path: str, actuators: dict[str, Any] | None = None) -> Any:
    if is_dataclass(config) and is_dataclass(config.base):
        changes: dict[str, Any] = {"base": replace(config.base, mjcf_path=mjcf_path)}
        if any(field.name == "actuators" for field in fields(config)):
            changes["actuators"] = actuators
        return replace(config, **changes)

    cloned = deepcopy(config)
    cloned.base.mjcf_path = mjcf_path
    setattr(cloned, "actuators", actuators)
    return cloned


def _expand_actuator_config(grouped: dict[str, Any]) -> dict[str, Any]:
    from actorob.models import expand_config

    return expand_config(grouped, front_rear=True)


def _default_actuator_builder(joint_family: str, motor_mass: float, gear_ratio: float) -> Any:
    from actorob.actuators import ActuatorParameters, GearboxParameters, MotorParameters

    return ActuatorParameters(
        motor=MotorParameters(motor_mass),
        gearbox=GearboxParameters(gear_ratio),
        name=joint_family,
    ).actuator_unit


def _load_model_factory():
    from actorob.models import ModelFactory

    return ModelFactory


__all__ = [
    "ActuatorDesignVariable",
    "ActuatorPreparer",
    "default_actuator_design_variables",
]
