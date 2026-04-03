from pathlib import Path

import numpy as np

from actorob.dashboard.invdes.actuation import _ActuationSpecs
from actorob.dashboard.invdes.candidate import _build_torque_speed_phase_portrait
from actorob.dashboard.invdes.page import build_inverse_design_dashboard_html
from actorob.invdes.evaluation.types import ObjectiveWeights
from actorob.invdes.evaluation.types import ScenarioEvaluation
from actorob.invdes.history import BatchHistoryEntry, OptimizationHistory, TrialHistoryEntry
from actorob.invdes.problem import OptimizationResult, OptimizationSettings
from actorob.invdes.record import InverseDesignRunRecord
from actorob.trajectories import JointTrajectoryData, TrajectoryRunRecord
from actorob.trajectories.actuation_constraints import MechanicalCharacteristicSpec


def _sample_trajectory_record() -> TrajectoryRunRecord:
    joint_names = (
        "front_left_hip_roll_joint",
        "front_left_hip_pitch_joint",
    )
    task = JointTrajectoryData(
        task_name="walk",
        converged=True,
        iterations=5,
        trajectory_cost=12.34,
        state_time=np.array([0.0, 0.02, 0.04]),
        control_time=np.array([0.0, 0.02]),
        floating_base_coordinates=np.array(
            [
                [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.41, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        floating_base_velocities=np.array(
            [
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        joint_positions=np.array([[0.0, 0.8], [0.1, 0.9], [0.2, 1.0]]),
        joint_velocities=np.array([[0.0, 0.0], [0.5, 0.4], [0.2, 0.1]]),
        joint_torques=np.array([[1.0, 1.2], [0.8, 0.9]]),
        electrical_power=np.array([10.0, 9.0]),
        friction_power=np.array([1.0, 0.8]),
        electrical_energy=0.38,
        friction_energy=0.04,
    )
    return TrajectoryRunRecord.now(
        robot="dog",
        mjcf_path="robots/dog/dog.xml",
        dt=0.02,
        contact_mu=0.8,
        joint_names=joint_names,
        joint_position_lower_limits=np.array([-2.0, -2.0]),
        joint_position_upper_limits=np.array([2.0, 2.0]),
        joint_velocity_lower_limits=np.array([-10.0, -10.0]),
        joint_velocity_upper_limits=np.array([10.0, 10.0]),
        joint_torque_lower_limits=np.array([-30.0, -30.0]),
        joint_torque_upper_limits=np.array([30.0, 30.0]),
        tasks=(task,),
    )


def _sample_run_record() -> InverseDesignRunRecord:
    history = OptimizationHistory(
        parameter_names=("m_hip_roll", "g_hip_roll"),
        batches=(
            BatchHistoryEntry(
                batch_index=1,
                total_batches=2,
                batch_size=2,
                evaluated_trials=2,
                completed_trials=2,
                failed_trials=0,
                duration_seconds=12.5,
                best_value=3.0,
                generation=0,
                sigma=0.25,
                trials=(
                    TrialHistoryEntry(
                        trial_number=0,
                        params=(1.0, 12.0),
                        status="complete",
                        value=4.0,
                        error=None,
                        generation=0,
                        sigma=0.25,
                    ),
                    TrialHistoryEntry(
                        trial_number=1,
                        params=(0.9, 11.0),
                        status="complete",
                        value=3.0,
                        error=None,
                        generation=0,
                        sigma=0.25,
                    ),
                ),
            ),
            BatchHistoryEntry(
                batch_index=2,
                total_batches=2,
                batch_size=2,
                evaluated_trials=4,
                completed_trials=3,
                failed_trials=1,
                duration_seconds=11.0,
                best_value=2.5,
                generation=1,
                sigma=0.2,
                trials=(
                    TrialHistoryEntry(
                        trial_number=2,
                        params=(0.8, 10.0),
                        status="complete",
                        value=2.5,
                        error=None,
                        generation=1,
                        sigma=0.2,
                    ),
                    TrialHistoryEntry(
                        trial_number=3,
                        params=(1.1, 13.0),
                        status="fail",
                        value=None,
                        error="boom",
                        generation=1,
                        sigma=0.2,
                    ),
                ),
            ),
        ),
    )
    return InverseDesignRunRecord.now(
        config_path="configs/dog_aligator_minimal.toml",
        task_names=("walk", "upstairs", "jump_forward"),
        settings=OptimizationSettings(max_iterations=2, parallelism=2, population_size=2),
        seed=12345,
        sigma0=0.25,
        weights=ObjectiveWeights(traj_cost=1.0, energy=1.0, friction=1.0),
        result=OptimizationResult(
            best_params=(0.8, 10.0),
            best_value=2.5,
            completed_trials=(),
            failed_trials=(),
            history=history,
        ),
        best_trajectory_record=_sample_trajectory_record(),
        normalization_stats={
            "WALK": {
                "traj_cost": {"mean": 10.0, "std": 2.0},
                "electrical_energy": {"mean": 0.3, "std": 0.1},
                "friction_loss": {"mean": 0.05, "std": 0.01},
            }
        },
        best_scenarios=(
            ScenarioEvaluation(
                case_name="walk",
                mode="WALK",
                traj_cost=12.34,
                electrical_energy=0.38,
                friction_loss=0.04,
                converged=True,
                iterations=5,
            ),
        ),
    )


def test_inverse_design_dashboard_roundtrip(tmp_path: Path):
    run = _sample_run_record()
    out = tmp_path / "invdes.pkl"
    saved = run.save(out)

    loaded = InverseDesignRunRecord.load(saved)

    assert loaded.best_value == 2.5
    assert loaded.history.batches[1].failed_trials == 1


def test_inverse_design_dashboard_html_generation(tmp_path: Path):
    run = _sample_run_record()
    out = tmp_path / "invdes.html"
    html_path = build_inverse_design_dashboard_html(run, out, include_meshcat=False)
    content = html_path.read_text(encoding="utf-8")

    assert "Inverse Design Dashboard" in content
    assert "Optimization History" in content
    assert "Iteration Explorer" in content
    assert "Best Candidate" in content
    assert "Meshcat simulation disabled by flag." in content
    assert "Fitness Landscape by Trial" in content
    assert "Sigma by Iteration" in content
    assert "Iteration 1 / 2" in content
    assert "weighted sum of z-scores" in content
    assert "Tasks / Offspring" in content
    assert "Total Optimization" in content
    assert "23.50s" in content
    assert "boom" in content
    assert "Error" in content
    assert "Phase Portraits" in content
    assert "Torque-Speed Phase Portrait with OCP Actuation Limits" in content
    assert "Electrical Power and Friction Losses by Task" in content
    assert "traj_cost" in content
    assert "energy_z" in content
    assert content.count("configs/dog_aligator_minimal.toml") == 1


def test_phase_portrait_shows_mechanical_constraint_boundary():
    record = _sample_trajectory_record()
    fig = _build_torque_speed_phase_portrait(
        record,
        _ActuationSpecs(
            mechanical=MechanicalCharacteristicSpec(
                no_load_velocity=np.array([10.0, 10.0]),
                slope=np.array([0.1, 0.1]),
            ),
        ),
    )

    trace_names = {trace.name for trace in fig.data}
    assert "Mechanical characteristic" in trace_names
    assert "Power limit" not in trace_names
    assert "Regen power limit" not in trace_names
