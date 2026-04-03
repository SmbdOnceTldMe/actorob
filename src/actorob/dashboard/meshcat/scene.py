"""Meshcat scene construction for dashboard task views."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from actorob.dashboard.meshcat.geometry import (
    StairRenderSpec,
    foot_target_color,
    sampled_foot_target_refs,
    stair_flat_center,
    stair_step_center,
)
from actorob.mjcf import resolve_mjcf_path
from actorob.trajectories import TrajectoryRunRecord


def build_meshcat_task_html(
    record: TrajectoryRunRecord,
    task_idx: int,
    frame_stride: int,
    fps: int,
) -> str:
    if frame_stride <= 0:
        raise ValueError(f"meshcat frame_stride must be > 0, got {frame_stride}.")

    import pinocchio as pin
    from meshcat.animation import Animation
    from pinocchio.visualize import MeshcatVisualizer

    task = record.tasks[task_idx]
    q_trajectory = np.hstack([task.floating_base_coordinates, task.joint_positions])
    if q_trajectory.shape[1] == 0:
        raise ValueError("Empty q trajectory.")

    rmodel, cmodel, vmodel = pin.buildModelsFromMJCF(str(resolve_mjcf_path(record.mjcf_path)))
    visualizer = MeshcatVisualizer(rmodel, cmodel, vmodel)
    root_name = f"{record.robot}_{task.task_name}"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        visualizer.initViewer(open=False)
        visualizer.loadViewerModel(rootNodeName=root_name)

    try:
        stair_spec = _infer_stair_spec(record, task, q_trajectory, rmodel, pin)
        if stair_spec is not None:
            _add_stairs_to_scene(visualizer, root_name=root_name, spec=stair_spec)

        animation = Animation(default_framerate=fps)
        sampled_indices = list(range(0, q_trajectory.shape[0], frame_stride))
        if sampled_indices[-1] != q_trajectory.shape[0] - 1:
            sampled_indices.append(q_trajectory.shape[0] - 1)
        sampled_target_refs = sampled_foot_target_refs(getattr(task, "foot_target_refs", {}), sampled_indices)
        _add_foot_target_trails(visualizer, root_name=root_name, sampled_target_refs=sampled_target_refs)
        _add_foot_target_markers(visualizer, root_name=root_name, sampled_target_refs=sampled_target_refs)

        scene_viewer = visualizer.viewer
        try:
            for frame_number, sample_idx in enumerate(sampled_indices):
                visualizer.viewer = animation.at_frame(scene_viewer, frame_number)
                visualizer.display(q_trajectory[sample_idx])
                _animate_foot_target_markers(
                    viewer=visualizer.viewer,
                    root_name=root_name,
                    sampled_target_refs=sampled_target_refs,
                    sample_idx=frame_number,
                )
        finally:
            visualizer.viewer = scene_viewer

        visualizer.display(q_trajectory[sampled_indices[0]])
        _animate_foot_target_markers(
            viewer=scene_viewer,
            root_name=root_name,
            sampled_target_refs=sampled_target_refs,
            sample_idx=0,
        )
        scene_viewer.set_animation(animation, play=False, repetitions=1)
        return visualizer.viewer.static_html()
    finally:
        _shutdown_meshcat_server(visualizer)


def _infer_stair_spec(
    record: TrajectoryRunRecord,
    task: Any,
    q_trajectory: np.ndarray,
    rmodel: Any,
    pin: Any,
) -> StairRenderSpec | None:
    stair_spec = _stair_spec_from_task_config(record, task, q_trajectory, rmodel, pin)
    if stair_spec is not None:
        return stair_spec
    stair_spec = _stair_spec_from_foot_targets(task)
    if stair_spec is not None:
        return stair_spec
    return _infer_stair_spec_heuristic(task, q_trajectory, rmodel, pin)


def _is_stair_task_name(task_name: str) -> bool:
    lname = task_name.lower()
    return ("upstairs" in lname) or ("stair" in lname) or ("pedestal" in lname)


def _infer_contact_offsets(
    task: Any,
    rmodel: Any,
    q_trajectory: np.ndarray,
    pin: Any,
    default_x: float,
    default_y: float,
    default_z: float,
) -> tuple[float, float, float]:
    contact_frame_names = tuple(getattr(task, "contact_frame_names", ()))
    if len(contact_frame_names) == 0:
        return default_x, default_y, default_z

    data = rmodel.createData()
    q0 = np.asarray(q_trajectory[0], dtype=float)
    pin.forwardKinematics(rmodel, data, q0)
    pin.updateFramePlacements(rmodel, data)

    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    for frame_name in contact_frame_names:
        if not rmodel.existFrame(frame_name):
            continue
        frame_id = rmodel.getFrameId(frame_name)
        tr = data.oMf[frame_id].translation
        x_values.append(float(tr[0]))
        y_values.append(float(tr[1]))
        z_values.append(float(tr[2]))
    if len(x_values) == 0:
        return default_x, default_y, default_z
    return max(x_values) + 0.02, float(np.mean(y_values)), float(min(z_values))


def _front_contact_origin(
    task: Any, rmodel: Any, q_trajectory: np.ndarray, pin: Any
) -> tuple[float, float, float] | None:
    contact_frame_names = tuple(getattr(task, "contact_frame_names", ()))
    if len(contact_frame_names) == 0:
        return None

    preferred_names = [name for name in contact_frame_names if "front" in name.lower()]
    if len(preferred_names) == 0:
        preferred_names = list(contact_frame_names)

    data = rmodel.createData()
    q0 = np.asarray(q_trajectory[0], dtype=float)
    pin.forwardKinematics(rmodel, data, q0)
    pin.updateFramePlacements(rmodel, data)

    points: list[np.ndarray] = []
    for frame_name in preferred_names:
        if not rmodel.existFrame(frame_name):
            continue
        frame_id = rmodel.getFrameId(frame_name)
        points.append(np.asarray(data.oMf[frame_id].translation, dtype=float))
    if len(points) == 0:
        return None

    first = points[0]
    y_mean = float(np.mean([point[1] for point in points]))
    z_min = float(np.min([point[2] for point in points]))
    return float(first[0]), y_mean, z_min


def _stair_spec_from_task_config(
    record: TrajectoryRunRecord,
    task: Any,
    q_trajectory: np.ndarray,
    rmodel: Any,
    pin: Any,
) -> StairRenderSpec | None:
    stairs_cfg = getattr(task, "stairs", None)
    if stairs_cfg is None:
        return None
    if stairs_cfg.step_length <= 0 or stairs_cfg.step_height <= 0 or stairs_cfg.step_count <= 0:
        return None

    offset_x = stairs_cfg.offset_x
    offset_y = float(stairs_cfg.offset_y)
    offset_z = float(stairs_cfg.offset_z)
    if offset_x is None:
        front_contact_origin = _front_contact_origin(task, rmodel, q_trajectory, pin)
        if front_contact_origin is not None and stairs_cfg.start_step is not None:
            front_x, _, _ = front_contact_origin
            foot_offset_x = 0.15 if record.robot.lower() == "dog" else 0.02
            offset_x = front_x + foot_offset_x + float(stairs_cfg.start_step) * float(stairs_cfg.step_length)
        else:
            default_x = float(q_trajectory[0, 0] + 0.1)
            offset_x, offset_y, offset_z = _infer_contact_offsets(
                task,
                rmodel,
                q_trajectory,
                pin,
                default_x,
                offset_y,
                offset_z,
            )

    flat_length = stairs_cfg.flat_length
    if flat_length is None:
        if stairs_cfg.start_step is not None and stairs_cfg.total_steps is not None:
            flat_steps = max(int(stairs_cfg.total_steps) - int(stairs_cfg.start_step) - int(stairs_cfg.step_count), 0)
            flat_length = float(flat_steps) * float(stairs_cfg.step_length)
        else:
            flat_length = stairs_cfg.step_length

    return StairRenderSpec(
        step_length=float(stairs_cfg.step_length),
        step_height=float(stairs_cfg.step_height),
        step_count=int(stairs_cfg.step_count),
        width=float(stairs_cfg.width),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
        offset_z=float(offset_z),
        flat_length=float(flat_length),
    )


def _stair_spec_from_foot_targets(task: Any) -> StairRenderSpec | None:
    if not _is_stair_task_name(task.task_name):
        return None

    foot_target_refs = getattr(task, "foot_target_refs", {})
    if not foot_target_refs:
        return None

    preferred_order = (
        "front_left_foot",
        "front_right_foot",
        "rear_left_foot",
        "rear_right_foot",
    )
    ordered_frame_names = [name for name in preferred_order if name in foot_target_refs]
    if len(ordered_frame_names) == 0:
        ordered_frame_names = list(foot_target_refs)

    front_frame_names = [name for name in ordered_frame_names if "front" in name.lower()]
    stair_frame_names = front_frame_names[:2] if len(front_frame_names) >= 2 else ordered_frame_names[:2]

    rounded_points: list[np.ndarray] = []
    for frame_name in stair_frame_names:
        arr = np.asarray(foot_target_refs[frame_name], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            continue
        rounded_points.append(np.round(arr, 3))
    if not rounded_points:
        return None

    all_points = np.vstack(rounded_points)
    ground_z = float(np.min(all_points[:, 2]))
    positive = all_points[all_points[:, 2] > ground_z + 1e-6]
    if positive.shape[0] == 0:
        return None

    z_levels: list[float] = []
    for z in sorted(np.unique(positive[:, 2]).tolist()):
        if np.sum(np.isclose(positive[:, 2], z, atol=1e-6)) >= 3:
            z_levels.append(float(z))
    if len(z_levels) == 0:
        return None

    centers_x: list[float] = []
    for z in z_levels:
        x_candidates = positive[np.isclose(positive[:, 2], z, atol=1e-6), 0]
        if x_candidates.size == 0:
            continue
        unique_x, counts = np.unique(np.round(x_candidates, 3), return_counts=True)
        centers_x.append(float(unique_x[int(np.argmax(counts))]))
    if len(centers_x) != len(z_levels):
        return None

    step_count = len(z_levels)
    if step_count >= 2:
        step_length = float(np.median(np.diff(np.asarray(centers_x, dtype=float))))
        step_height = float(np.median(np.diff(np.asarray([ground_z] + z_levels, dtype=float))))
    else:
        stairs_cfg = getattr(task, "stairs", None)
        step_length = float(getattr(stairs_cfg, "step_length", 0.2))
        step_height = float(z_levels[0] - ground_z)

    if step_length <= 0 or step_height <= 0:
        return None

    stairs_cfg = getattr(task, "stairs", None)
    width = float(getattr(stairs_cfg, "width", 0.5))
    offset_y = float(getattr(stairs_cfg, "offset_y", 0.0))
    flat_length_cfg = getattr(getattr(task, "stairs", None), "flat_length", None)
    flat_length = step_length if flat_length_cfg is None else float(flat_length_cfg)

    return StairRenderSpec(
        step_length=step_length,
        step_height=step_height,
        step_count=step_count,
        width=width,
        offset_x=float(centers_x[0] - step_length / 2.0),
        offset_y=offset_y,
        offset_z=ground_z,
        flat_length=flat_length,
    )


def _infer_stair_spec_heuristic(task: Any, q_trajectory: np.ndarray, rmodel: Any, pin: Any) -> StairRenderSpec | None:
    if not _is_stair_task_name(task.task_name):
        return None
    if q_trajectory.shape[0] < 2:
        return None

    dx_total = float(q_trajectory[-1, 0] - q_trajectory[0, 0])
    dz_total = float(q_trajectory[-1, 2] - q_trajectory[0, 2])
    if dx_total <= 1e-5 or dz_total <= 1e-5:
        return None

    step_count = int(np.clip(np.round(dz_total / 0.04), 1, 12))
    step_height = dz_total / float(step_count)
    step_length = max(dx_total / float(step_count), 0.08)

    default_x = float(q_trajectory[0, 0] + 0.1)
    offset_x, offset_y, offset_z = _infer_contact_offsets(task, rmodel, q_trajectory, pin, default_x, 0.0, 0.0)

    return StairRenderSpec(
        step_length=step_length,
        step_height=step_height,
        step_count=step_count,
        width=0.50,
        offset_x=offset_x,
        offset_y=offset_y,
        offset_z=offset_z,
        flat_length=step_length,
    )


def _add_stairs_to_scene(visualizer: Any, *, root_name: str, spec: StairRenderSpec) -> None:
    import meshcat.geometry as g
    import meshcat.transformations as tf

    material = g.MeshLambertMaterial(color=int(0x8F8F8F), opacity=0.95)
    stairs_root = visualizer.viewer[f"{root_name}/stairs"]

    for step_idx in range(spec.step_count):
        node = stairs_root[f"step_{step_idx}"]
        node.set_object(g.Box([spec.step_length, spec.width, spec.step_height]), material)
        node.set_transform(tf.translation_matrix(stair_step_center(spec, step_idx)))

    if spec.flat_length > 1e-6:
        flat = stairs_root["flat"]
        flat.set_object(g.Box([spec.flat_length, spec.width, spec.step_height]), material)
        flat.set_transform(tf.translation_matrix(stair_flat_center(spec)))


def _add_foot_target_trails(
    visualizer: Any,
    *,
    root_name: str,
    sampled_target_refs: dict[str, np.ndarray],
) -> None:
    import meshcat.geometry as g

    if len(sampled_target_refs) == 0:
        return

    targets_root = visualizer.viewer[f"{root_name}/foot_targets"]
    for frame_name, points in sampled_target_refs.items():
        if points.shape[0] < 2:
            continue
        node = targets_root[f"{frame_name}/trail"]
        node.set_object(
            g.Line(
                g.PointsGeometry(points.T),
                g.LineBasicMaterial(color=foot_target_color(frame_name), linewidth=2.0),
            )
        )


def _add_foot_target_markers(
    visualizer: Any,
    *,
    root_name: str,
    sampled_target_refs: dict[str, np.ndarray],
) -> None:
    import meshcat.geometry as g
    import meshcat.transformations as tf

    if len(sampled_target_refs) == 0:
        return

    targets_root = visualizer.viewer[f"{root_name}/foot_targets"]
    for frame_name, points in sampled_target_refs.items():
        node = targets_root[f"{frame_name}/marker"]
        node.set_object(
            g.Sphere(0.018),
            g.MeshLambertMaterial(color=foot_target_color(frame_name), opacity=0.85),
        )
        node.set_transform(tf.translation_matrix(points[0].tolist()))


def _animate_foot_target_markers(
    viewer: Any,
    *,
    root_name: str,
    sampled_target_refs: dict[str, np.ndarray],
    sample_idx: int,
) -> None:
    import meshcat.transformations as tf

    targets_root = viewer[f"{root_name}/foot_targets"]
    for frame_name, points in sampled_target_refs.items():
        point = points[sample_idx]
        targets_root[f"{frame_name}/marker"].set_transform(tf.translation_matrix(point.tolist()))


def _shutdown_meshcat_server(visualizer: Any) -> None:
    server_proc = getattr(visualizer.viewer.window, "server_proc", None)
    if server_proc is not None and server_proc.poll() is None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except Exception:
            server_proc.kill()


__all__ = ["build_meshcat_task_html"]
