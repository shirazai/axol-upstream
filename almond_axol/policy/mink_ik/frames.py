"""Convert current Axol wire poses to the pinned XR-1 checkpoint frame."""

from __future__ import annotations

import numpy as np

from .ik import pose6_to_pos_rot_np

# The current URDF added +pi/2 yaw at this existing root-joint origin.
# Match inference.xr1.robot.pose_to_current_axol_root at the wire boundary.
_ROOT_ORIGIN = np.array([-2.77556e-17, -6.93889e-18, 0.86], dtype=np.float64)
_CURRENT_TO_LEGACY_R = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def pose6_current_to_legacy(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a current-world pose6 into the legacy solver's ``(pos, rot)``.

    Cartesian actions on the unified interface use the current Axol URDF
    world frame. The pinned Mink model uses the checkpoint's earlier frame.
    Invert the root yaw about its translated origin for positions and apply
    the same inverse rotation to orientations. Results retain the legacy
    solver's float32 inputs; no JAX or stateful solver calls are involved.
    """
    position, rotation = pose6_to_pos_rot_np(pose6)
    legacy_position = _ROOT_ORIGIN + _CURRENT_TO_LEGACY_R @ (position - _ROOT_ORIGIN)
    legacy_rotation = _CURRENT_TO_LEGACY_R @ rotation
    return legacy_position.astype(np.float32), legacy_rotation.astype(np.float32)
