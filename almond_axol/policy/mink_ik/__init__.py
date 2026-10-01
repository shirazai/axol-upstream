"""The legacy XR-1 Mink controller, independent of JAX and hardware access.

The solver owns mutable MuJoCo data and must be used by one control thread.
Its poses are in the checkpoint's yaw-zero frame; adapters using the current
Axol world frame must convert at the boundary. See ``PROVENANCE.md`` for the
source revision, asset provenance, and numerical runtime requirements.
"""

from .frames import pose6_current_to_legacy
from .ik import MinkIK, pose6_to_pos_rot_np
from .ik_config import PINNED_URDF, MinkIKConfig

__all__ = [
    "PINNED_URDF",
    "MinkIK",
    "MinkIKConfig",
    "pose6_current_to_legacy",
    "pose6_to_pos_rot_np",
]
