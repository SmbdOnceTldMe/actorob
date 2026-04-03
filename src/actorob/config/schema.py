from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class BaseSectionModel(_ConfigModel):
    """Pydantic model for the ``[base]`` configuration section."""

    robot: str = "dog"
    mjcf_path: str = "robots/dog/dog.xml"
    init_pose: tuple[float, ...] | None = None
    align_feet_to_ground: bool = True
    ground_z: float = 0.0


class TrajectorySectionModel(_ConfigModel):
    """Pydantic model for the ``[trajectory]`` configuration section."""

    dt: float = Field(default=0.02, gt=0)
    use_control_bounds: bool = Field(
        default=True,
        validation_alias=AliasChoices("use_control_bounds", "ubounded"),
    )
    use_kinematic_constraints: bool = Field(
        default=True,
        validation_alias=AliasChoices("use_kinematic_constraints", "kinematic_cstr"),
    )
    use_friction_cones: bool = True
    enforce_mechanical_characteristic: bool = True


class SolverSectionModel(_ConfigModel):
    """Pydantic model for the ``[solver]`` configuration section."""

    tol: float = 1e-4
    mu_init: float = 1e-4
    max_iter: int = Field(default=100, gt=0)
    verbose: Literal["QUIET", "VERBOSE"] = "QUIET"
    rollout_type: Literal["ROLLOUT_LINEAR", "ROLLOUT_NONLINEAR"] = "ROLLOUT_LINEAR"
    sa_strategy: Literal["SA_FILTER", "SA_LINESEARCH_ARMIJO", "SA_LINESEARCH_NONMONOTONE"] = "SA_FILTER"
    linear_solver_choice: Literal[
        "LQ_SOLVER_PARALLEL",
        "LQ_SOLVER_SERIAL",
        "LQ_SOLVER_STAGEDENSE",
    ] = "LQ_SOLVER_PARALLEL"
    filter_beta: float = Field(default=1e-5, gt=0)
    force_initial_condition: bool = True
    reg_min: float = Field(default=1e-6, ge=0)
    num_threads: int = Field(default=1, gt=0)

    @field_validator("verbose", "rollout_type", "sa_strategy", "linear_solver_choice", mode="before")
    @classmethod
    def _normalize_uppercase(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).upper()


class ContactSectionModel(_ConfigModel):
    """Pydantic model for the ``[contact]`` configuration section."""

    contact_frames_3d: tuple[str, ...] = Field(min_length=1)
    contact_mu: float = Field(default=0.8, gt=0)
    prox_accuracy: float = 1e-9
    prox_mu: float = 1e-9
    prox_max_iter: int = Field(default=10, gt=0)


class TrajectoryParamsModel(_ConfigModel):
    """Pydantic model for optional high-level gait parameters."""

    n_steps: int | None = Field(default=None, gt=0)
    step_time: float | None = Field(default=None, gt=0)
    ds_time: float | None = Field(default=None, ge=0)
    dx: float | None = None
    dy: float | None = None
    swing_apex: float | None = None
    stair_start: int | None = Field(default=None, ge=0)
    stair_end: int | None = Field(default=None, ge=0)
    stair_h: float | None = Field(default=None, ge=0)
    t_st: int | None = Field(default=None, ge=0)
    t_ft: int | None = Field(default=None, ge=0)
    jump_h: float | None = Field(default=None, ge=0)
    jump_l: float | None = None

    @model_validator(mode="after")
    def _validate_stair_range(self) -> "TrajectoryParamsModel":
        if self.stair_start is not None and self.stair_end is not None and self.stair_end < self.stair_start:
            raise ValueError("stair_end must be >= stair_start")
        return self


class StairSectionModel(_ConfigModel):
    """Pydantic model describing staircase geometry in the config file."""

    step_length: float = Field(gt=0)
    step_height: float = Field(gt=0)
    step_count: int = Field(gt=0)
    width: float = Field(default=0.5, gt=0)
    offset_x: float | None = None
    offset_y: float = 0.0
    offset_z: float = 0.0
    flat_length: float | None = Field(default=None, ge=0)


class TaskInputModel(_ConfigModel):
    """Raw per-task configuration before runtime normalization."""

    horizon: int | None = Field(default=None, gt=0)
    target_dx: float | None = None
    target_dy: float | None = None
    target_dz: float | None = None
    apex_dz: float | None = None
    touchdown_dx: float | None = None
    touchdown_dy: float | None = None
    touchdown_dz: float | None = None
    state_weight: float = 1.0
    control_weight: float = 1e-3
    terminal_weight: float = 10.0
    base_position_weight_scale: float = 40.0
    base_orientation_weight_scale: float = 5.0
    base_linear_velocity_weight_scale: float = 10.0
    base_angular_velocity_weight_scale: float = 2.0
    joint_targets: dict[str, float] = Field(default_factory=dict)
    contact_phases: tuple[str, ...] | None = None
    phase_durations: tuple[int, ...] | None = None
    stairs: StairSectionModel | None = None
    trajectory_params: TrajectoryParamsModel | None = None

    @field_validator("contact_phases", mode="before")
    @classmethod
    def _normalize_contact_phases(cls, value: Any) -> Any:
        if value is None:
            return value
        return tuple(str(phase).lower() for phase in value)

    @field_validator("phase_durations")
    @classmethod
    def _validate_phase_durations(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None and any(duration <= 0 for duration in value):
            raise ValueError("all phase_durations must be positive")
        return value

    @model_validator(mode="after")
    def _validate_contact_schedule_pair(self) -> "TaskInputModel":
        has_phases = self.contact_phases is not None
        has_durations = self.phase_durations is not None
        if has_phases != has_durations:
            raise ValueError("define both 'contact_phases' and 'phase_durations' or neither")
        if self.contact_phases is not None and self.phase_durations is not None:
            if len(self.contact_phases) != len(self.phase_durations):
                raise ValueError("contact_phases and phase_durations must have equal length")
        return self


class TrajectoryOptimizerFileModel(_ConfigModel):
    """Validated TOML schema for a trajectory optimizer config file."""

    base: BaseSectionModel
    trajectory: TrajectorySectionModel
    solver: SolverSectionModel
    contact: ContactSectionModel
    tasks: dict[str, TaskInputModel]

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_constraint_toggles(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        trajectory = value.get("trajectory")
        contact = value.get("contact")
        actuation = value.get("actuation")
        updated_trajectory = dict(trajectory) if isinstance(trajectory, dict) else {}
        changed = False

        if (
            isinstance(contact, dict)
            and "use_friction_cones" in contact
            and "use_friction_cones" not in updated_trajectory
        ):
            updated_trajectory["use_friction_cones"] = contact["use_friction_cones"]
            changed = True
        if (
            isinstance(actuation, dict)
            and "enforce_mechanical_characteristic" in actuation
            and "enforce_mechanical_characteristic" not in updated_trajectory
        ):
            updated_trajectory["enforce_mechanical_characteristic"] = actuation["enforce_mechanical_characteristic"]
            changed = True

        if not changed:
            return value

        updated = dict(value)
        updated["trajectory"] = updated_trajectory
        return updated

    @field_validator("tasks")
    @classmethod
    def _validate_tasks(cls, value: dict[str, TaskInputModel]) -> dict[str, TaskInputModel]:
        if not value:
            raise ValueError("At least one task must be defined in [tasks].")
        return value


def parse_trajectory_optimizer_file(raw: dict[str, Any]) -> TrajectoryOptimizerFileModel:
    """Parse and validate raw TOML data for the trajectory optimizer."""

    try:
        return TrajectoryOptimizerFileModel.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"]
        if location:
            lines.append(f"{location}: {message}")
        else:
            lines.append(message)
    return "\n".join(lines)


__all__ = [
    "BaseSectionModel",
    "ContactSectionModel",
    "SolverSectionModel",
    "StairSectionModel",
    "TaskInputModel",
    "TrajectoryOptimizerFileModel",
    "TrajectoryParamsModel",
    "TrajectorySectionModel",
    "parse_trajectory_optimizer_file",
]
