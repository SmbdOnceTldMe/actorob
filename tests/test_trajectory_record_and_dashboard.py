from pathlib import Path

import numpy as np

from actorob.dashboard import build_trajectory_dashboard_html
from actorob.trajectories import JointTrajectoryData, TrajectoryRunRecord


def _sample_record() -> TrajectoryRunRecord:
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
        foot_target_refs={
            "front_left_foot": np.array(
                [
                    [0.1, 0.2, 0.0],
                    [0.2, 0.2, 0.05],
                    [0.3, 0.2, 0.0],
                ]
            )
        },
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


def test_trajectory_record_save_load_roundtrip(tmp_path: Path):
    record = _sample_record()
    out = tmp_path / "record.pkl"
    saved = record.save(out)

    loaded = TrajectoryRunRecord.load(saved)
    assert loaded.robot == record.robot
    assert loaded.dt == record.dt
    assert loaded.joint_names == record.joint_names
    assert loaded.tasks[0].task_name == "walk"
    assert loaded.tasks[0].joint_positions.shape == (3, 2)
    assert "front_left_foot" in loaded.tasks[0].foot_target_refs
    assert loaded.tasks[0].foot_target_refs["front_left_foot"].shape == (3, 3)


def test_dashboard_html_generation(tmp_path: Path):
    record = _sample_record()
    out = tmp_path / "dashboard.html"
    html_path = build_trajectory_dashboard_html(record, out, include_meshcat=False)
    content = html_path.read_text(encoding="utf-8")

    assert "Trajectory Dashboard" in content
    assert "Floating Base" in content
    assert "Positions" in content
    assert "Velocities" in content
    assert "Torques" in content
    assert "Ground Reaction Forces" in content
    assert "Meshcat Simulation" in content
