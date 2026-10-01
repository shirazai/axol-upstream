"""MuJoCo model loading for the Axol URDF.  # shiraz (shirazai/shiraz#210)

MuJoCo's URDF importer needs compiler overrides before the bundled
``axol.urdf`` loads usefully for kinematics work:

- ``strippath``/``meshdir``: mesh references are ``package://`` URIs; MuJoCo
  must resolve just the basenames against the bundled ``meshes`` directory.
- ``fusestatic="false"``: the IK end-effector frames (``left/right_gripper``)
  and the TCP frames (``left/right_hand_tcp``) attach by fixed URDF joints;
  the default import fuses them into ``*_w2`` / deletes them (see the note
  in ``almond_axol/constants.py``), which would leave the differential-IK
  frame tasks nothing to target.

The gravity compensator keeps its own stripped-geometry loader
(``robot/gravity.py``, visual/collision blocks removed); this loader
preserves collision geometry so constraint-based consumers (mink's
``CollisionAvoidanceLimit``) can build geom pairs from it.

xr1-rustcore (shiraz #550, K34): this is the fork-main (80e7a8c)
``almond_axol/kinematics/mujoco_model.py`` vendored into the XR-1 package
with ONE change — the default URDF is the pinned fork asset
(``ik_config.PINNED_URDF``) instead of the vendor tree's ``URDF_PATH``. The
chemical-speak URDF carries a +90 deg root yaw the checkpoint FK contract
does not know about (design 5.2); a silently-defaulted load of the vendor
file would pass every self-consistency test and steer 90 deg off.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from .ik_config import PINNED_URDF  # xr1-rustcore: K34 (was almond_axol.constants.URDF_PATH)

# Injected as the first child of <robot>: MuJoCo reads an embedded <mujoco>
# extension element from URDF files for compiler settings.
_COMPILER_TMPL = (
    '<mujoco><compiler strippath="true" meshdir="{meshdir}" '
    'fusestatic="false" discardvisual="true"/></mujoco>'
)


def load_mj_model(urdf_path: Path = PINNED_URDF) -> mujoco.MjModel:  # xr1-rustcore: K34 default = pinned asset
    """Load the Axol URDF as a :class:`mujoco.MjModel` with all frames kept.

    Args:
        urdf_path: URDF file to load; meshes are resolved from the sibling
            ``meshes`` directory. Defaults to the pinned fork-main asset.

    Returns:
        Compiled model with every URDF link preserved as a named body
        (including the fixed-jointed EE/TCP frames) and collision geoms
        loaded. Visual-only duplicates are discarded.
    """
    text = urdf_path.read_text(encoding="utf-8")
    inject = _COMPILER_TMPL.format(meshdir=str(urdf_path.parent / "meshes"))
    patched, n = re.subn(r"(<robot\b[^>]*>)", r"\1" + inject, text, count=1)
    if n != 1:
        raise ValueError(f"{urdf_path} has no <robot> element to patch")
    return mujoco.MjModel.from_xml_string(patched)


def qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """qpos addresses of the named (1-DOF) joints, in the given order."""
    return np.array(
        [model.jnt_qposadr[_joint_id(model, n)] for n in joint_names], dtype=int
    )


def dof_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """Velocity-space (dof) addresses of the named joints, in the given order."""
    return np.array(
        [model.jnt_dofadr[_joint_id(model, n)] for n in joint_names], dtype=int
    )


def body_geom_ids(model: mujoco.MjModel, body_names: list[str]) -> list[int]:
    """All geom ids attached to the named bodies (skips geom-less bodies)."""
    ids: list[int] = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in model")
        start = model.body_geomadr[bid]
        ids.extend(range(start, start + model.body_geomnum[bid]))
    return ids


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint {name!r} not in model")
    return jid
