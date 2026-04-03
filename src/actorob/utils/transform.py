import mujoco
import numpy as np


def quat2mat(quat: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Parameters:
        quat: A numpy array [w, x, y, z]

    Returns:
        R: A 3x3 numpy array representing a rotation matrix.
    """
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat)
    rot_mat = rot_mat / np.max(rot_mat)
    return rot_mat.reshape((3, 3))


def mat2quat(rot_mat: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a quaternion (w, x, y, z).

    Parameters:
        R: A 3x3 numpy array representing a rotation matrix.

    Returns:
        A numpy array [w, x, y, z]
    """
    quat = np.zeros(4)
    rot_mat = rot_mat.reshape((9,))
    mujoco.mju_mat2Quat(quat, rot_mat)
    return quat


def get_fullinertia(body: mujoco.MjsBody) -> np.ndarray:
    """
    Returns the inertia matrix of the body in the global coordinate frame.

    - If the `inertia` field is all zeros, `fullinertia` is used (already in global coordinates).
    - Otherwise, `inertia` is treated as a diagonal matrix in the body's local frame
        and is rotated to the global frame using the body's quaternion.

    Parameters:
        body: The MjsBody object.

    Returns:
        A 3x3 numpy array — the inertia matrix in global coordinates.
    """
    if np.any(body.inertia == 0):
        inertia = np.array(
            [
                [body.fullinertia[0], body.fullinertia[3], body.fullinertia[4]],
                [body.fullinertia[3], body.fullinertia[1], body.fullinertia[5]],
                [body.fullinertia[4], body.fullinertia[5], body.fullinertia[2]],
            ]
        ).astype(np.float64)
    else:
        R_i = quat2mat(body.iquat)
        inertia = R_i @ np.diag(body.inertia).astype(np.float64) @ R_i.T
    return inertia
