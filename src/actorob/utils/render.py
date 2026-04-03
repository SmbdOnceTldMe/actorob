import mujoco
import mujoco.viewer
import time
import sys
import os
from pathlib import Path


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData = None,
    initial_height: float = 1.105,
    fix: bool = True,
    duration: float = 5,
    sleep_dt: float = 0.01,
    show_ui: bool = False,
    make_step=False,
    control_callback=None,
):
    """Visualize a MuJoCo model in a passive viewer window.

    Opens a MuJoCo viewer for the provided model and optionally runs a
    lightweight simulation loop. The function can fix the base height and
    orientation of the model (useful for free-floating robots), or step
    the simulation in real time if ``make_step`` is enabled.

    Args:
        model (mujoco.MjModel):
            Compiled MuJoCo model to visualize.
        data (Optional[mujoco.MjData]):
            Simulation data structure. If ``None``, a new one is created for the
            given model.
        initial_height (float, optional):
            Starting z-position for a free joint body (e.g., floating robot base).
            Defaults to ``1.105`` meters.
        fix (bool, optional):
            Whether to fix the robot's position and orientation throughout the
            render loop (applies only to models with a free joint). Defaults to ``True``.
        duration (float, optional):
            Duration of the visualization loop in seconds. Defaults to ``5``.
        sleep_dt (float, optional):
            Time (in seconds) to sleep between render frames. Controls visual
            update rate. Defaults to ``0.01``.
        show_ui (bool, optional):
            Whether to display MuJoCo's left and right UI panels in the viewer.
            Defaults to ``False``.
        make_step (bool, optional):
            If ``True``, advances the simulation using :func:`mujoco.mj_step`
            each frame. If ``False``, only renders the current static pose.
            Defaults to ``False``.
        control_callback (Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]]):
            Optional callback function to apply custom control logic at each
            simulation step. The function should accept the model and data as
            arguments. Only used if ``make_step`` is ``True``.

    Notes:
        * If the model includes a free joint (``mjJNT_FREE``), its initial
          height and orientation are adjusted for better visibility.
        * The viewer camera is automatically positioned to give a clear
          front-side view of the robot with moderate zoom and tilt.
        * The function is blocking — it runs until the window is closed or
          ``duration`` seconds have elapsed.
        * This function is intended for debugging and visualization, not for
          high-performance simulation or control loops.
    """
    if data is None:
        data = mujoco.MjData(model)

    # On macOS passive MuJoCo viewer requires the script to run under mjpython.
    if sys.platform == "darwin":
        executable_name = Path(sys.executable).name.lower()
        is_mjpython = (
            executable_name == "mjpython" or "MJPYTHON_BIN" in os.environ or "MJPYTHON_LIBPYTHON" in os.environ
        )
        if not is_mjpython:
            raise RuntimeError(
                "MuJoCo passive viewer on macOS must be started with `mjpython`.\n"
                "Run your script as: `pixi run mjpython <script.py>`."
            )

    free_jnt_idx = None
    for idx, joint_type in enumerate(model.jnt_type):
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            free_jnt_idx = idx
            break

    if free_jnt_idx is not None:
        data.joint(free_jnt_idx).qpos[2] = initial_height  # Set initial height

    with mujoco.viewer.launch_passive(model, data, show_left_ui=show_ui, show_right_ui=show_ui) as viewer:
        if free_jnt_idx is not None:
            # Set camera to front view
            viewer.cam.lookat[0] = data.joint(free_jnt_idx).qpos[0]  # Look ahead of robot
            viewer.cam.lookat[1] = data.joint(free_jnt_idx).qpos[1]  # y position
            viewer.cam.lookat[2] = data.joint(free_jnt_idx).qpos[2]  # fixed height
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 150.0
        viewer.cam.elevation = -30.0

        start = time.time()
        while viewer.is_running() and time.time() - start < duration:
            if fix and free_jnt_idx is not None:
                # Keep position and orientation fixed
                data.joint(free_jnt_idx).qpos[2] = initial_height  # Fixed height
                data.joint(free_jnt_idx).qpos[3:7] = [
                    1,
                    0,
                    0,
                    0,
                ]  # Fixed orientation (quaternion)
            if make_step:
                if control_callback is not None:
                    control_callback(model, data)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(sleep_dt)
