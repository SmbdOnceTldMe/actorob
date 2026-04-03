from __future__ import annotations

from typing import Any

import aligator
import numpy as np
import pinocchio as pin
from aligator import constraints

from .actuation_constraints import MechanicalCharacteristicResidual


class OptimizerConstraintBuilder:
    """Attach dynamics and stage constraints for trajectory optimization."""

    def __init__(self, optimizer: Any) -> None:
        self._optimizer = optimizer

    def build_dynamics_model(self, active_contacts: list[pin.RigidConstraintModel]):
        optimizer = self._optimizer
        if len(active_contacts) == 0:
            ode = aligator.dynamics.MultibodyFreeFwdDynamics(optimizer.space, optimizer.act_matrix)
        else:
            ode = aligator.dynamics.MultibodyConstraintFwdDynamics(
                optimizer.space,
                optimizer.act_matrix,
                active_contacts,
                optimizer.prox_settings,
            )
        return aligator.dynamics.IntegratorSemiImplEuler(ode, optimizer.config.trajectory.dt)

    def add_friction_cone_constraints(
        self,
        stage_model: aligator.StageModel,
        active_contacts: list[pin.RigidConstraintModel],
    ) -> None:
        optimizer = self._optimizer
        if not optimizer.config.trajectory.use_friction_cones:
            return

        for contact_model in active_contacts:
            stage_model.addConstraint(
                aligator.MultibodyFrictionConeResidual(
                    optimizer.space.ndx,
                    optimizer.rmodel,
                    optimizer.act_matrix,
                    active_contacts,
                    optimizer.prox_settings,
                    contact_model.name,
                    optimizer.config.contact.contact_mu,
                ),
                constraints.NegativeOrthant(),
            )

    def add_control_constraints(self, stage_model: aligator.StageModel) -> None:
        optimizer = self._optimizer
        if not optimizer.config.trajectory.use_control_bounds:
            return
        lower = -np.asarray(optimizer.rmodel.effortLimit[6:], dtype=float)
        upper = np.asarray(optimizer.rmodel.effortLimit[6:], dtype=float)
        stage_model.addConstraint(
            aligator.ControlErrorResidual(optimizer.space.ndx, np.zeros(optimizer.nu)),
            constraints.BoxConstraint(lower, upper),
        )

    def add_kinematic_constraints(self, stage_model: aligator.StageModel) -> None:
        optimizer = self._optimizer
        if not optimizer.config.trajectory.use_kinematic_constraints:
            return
        state_fn = aligator.StateErrorResidual(optimizer.space, optimizer.nu, optimizer.space.neutral())
        pos_fn = state_fn[6 : optimizer.nv]
        lower = np.asarray(optimizer.rmodel.lowerPositionLimit[7:], dtype=float)
        upper = np.asarray(optimizer.rmodel.upperPositionLimit[7:], dtype=float)
        stage_model.addConstraint(pos_fn, constraints.BoxConstraint(lower, upper))

    def add_hard_foot_constraints(
        self,
        stage_model: aligator.StageModel,
        active_contacts: list[pin.RigidConstraintModel],
        frame_names: set[str] | None = None,
        tol_xy: float = 1.5e-2,
        tol_z: float = 5.0e-3,
        as_equality: bool = False,
    ) -> None:
        optimizer = self._optimizer
        lower = np.array([-tol_xy, -tol_xy, -tol_z], dtype=float)
        upper = np.array([tol_xy, tol_xy, tol_z], dtype=float)
        for contact_model in active_contacts:
            if frame_names is not None and contact_model.name not in frame_names:
                continue
            frame_id = optimizer.contact_frame_ids.get(contact_model.name)
            if frame_id is None:
                continue
            target_translation = np.asarray(contact_model.joint2_placement.translation, dtype=float).reshape(3)
            residual = aligator.FrameTranslationResidual(
                optimizer.space.ndx,
                optimizer.nu,
                optimizer.rmodel,
                target_translation,
                frame_id,
            )
            if as_equality:
                stage_model.addConstraint(residual, constraints.EqualityConstraintSet())
            else:
                stage_model.addConstraint(residual, constraints.BoxConstraint(lower, upper))

    def add_mechanical_characteristic_constraint(self, stage_model: aligator.StageModel) -> None:
        optimizer = self._optimizer
        if not optimizer.config.trajectory.enforce_mechanical_characteristic:
            return
        if optimizer._mechanical_characteristic is None:
            return

        stage_model.addConstraint(
            MechanicalCharacteristicResidual(
                optimizer.rmodel,
                optimizer.space,
                optimizer._mechanical_characteristic.no_load_velocity,
                optimizer._mechanical_characteristic.slope,
            ),
            constraints.NegativeOrthant(),
        )
