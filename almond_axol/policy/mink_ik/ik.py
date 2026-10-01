"""Vendored mink IK for the XR-1 Rust-core serving path (shiraz #550).

RUSTCORE_DESIGN.md 5.1-5.3. :class:`MinkIK` reproduces the ``backend="mink"``
branch of fork-main ``almond_axol/kinematics/solver.py:KinematicsSolver.ik``
(80e7a8c) — reach clamp per arm, ``MinkTracker.solve``, componentwise
backstop clamp — on a 14-joint ``q`` (the arm joints, left then right, in
``urdf_arm_joint_names`` order) without the pyroki/jaxls solver around it.

Why not the fork ``KinematicsSolver`` itself: on the chemical-speak base
``almond_axol.kinematics.__init__`` imports the JAX solver, ``KinematicsConfig``
has no ``mink_*`` fields, and the vendor URDF carries a +90 deg root yaw the
checkpoint FK contract does not know about (K01/K02). This module therefore
imports only ``almond_axol.constants`` (name tables) and the vendored
``ik_mink_backend`` / ``ik_mujoco_model`` on the pinned fork URDF. No jax,
jaxlie, pyroki or ``almond_axol.kinematics`` anywhere in ``ik_*.py``
(``test_xr1_rt_ik_verbatim.py`` asserts it statically).

Equivalence to the fork glue is pinned by ``test_xr1_rt_minkik_equiv.py``
(bit-for-bit over a seeded 30 Hz stream against the fork-main venv) and by
``test_xr1_rt_fk_parity.py`` (``fk()`` vs ``fk.py`` v1 at 1e-5 m / 1e-4).

Threading: one :class:`MinkIK` belongs to one thread (the control thread);
``solve``/``fk`` share one ``MjData``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from almond_axol.constants import Joint, urdf_arm_joint_names, urdf_body_name

from .ik_config import PINNED_URDF, MinkIKConfig
from .ik_mink_backend import MinkTracker
from .ik_mujoco_model import qpos_indices
from .vendor_io import aa2rotm

_logger = logging.getLogger(__name__)

N_ARM_JOINTS = 14


def _clamp_reach(pos: np.ndarray, center: np.ndarray, max_reach: float) -> np.ndarray:
    """Clamp EE target position to within max_reach of center (shoulder position).

    Verbatim fork ``solver.py:_clamp_reach`` (K04): a hard radial clamp, so
    an out-of-reach target becomes the nearest reachable-sphere point instead
    of stretching the QP against the joint limits.
    """
    d = pos - center
    dist = np.linalg.norm(d)
    if dist > max_reach:
        return (center + d * (max_reach / dist)).astype(np.float32)
    return pos


def pose6_to_pos_rot_np(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a 6-axis pose ``[x, y, z, rx, ry, rz]`` into ``(pos, rot_3x3)``.

    Numpy re-implementation of chemical-speak ``kinematics/fk.py:
    pose6_to_pos_rot`` (K05/K34): that function is ``jaxlie.SO3.exp`` and its
    module import drags jax+jaxlie+pyroki into the process (2.1 s on the box,
    305 ms first-call JIT > the 150 ms watchdog). The Rodrigues map here is
    the vendor's own ``aa2rotm`` (``vendor_io.py``), which the adapter's
    rotation round-trip already certifies against SO3.exp;
    ``test_xr1_rt_pose6_numpy.py`` pins the parity at 1e-6 (measured 4.8e-7).

    Returns float32 ``pos`` (3,) and float32 ``rot`` (3, 3), like the original.
    """
    pose6 = np.asarray(pose6, dtype=np.float32)
    if pose6.shape != (6,):
        raise ValueError(f"pose6 must have shape (6,), got {pose6.shape}")
    pos = pose6[:3].copy()
    rot = np.asarray(aa2rotm(pose6[3:6]), dtype=np.float32)
    return pos, rot


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or a.samefile(b)
    except OSError:
        return False


class MinkIK:
    """Fork ``KinematicsSolver.ik`` (mink branch) on 14 arm joints.

    Args:
        config: Solver parameters; defaults are the fork-main serving values.
        urdf_path: Accepted for interface symmetry with the fork loader but
            MUST be the pinned asset — the vendored backend loads
            ``PINNED_URDF`` unconditionally (design 5.1, edit 4/4), so any
            other URDF here would silently disagree with the tracker's model.
        warmup: Run one zero-error solve at construction (pages in mink/daqp
            code paths off the control thread) and flush the gate memory.

    ``q`` everywhere is the 14-vector ``joint_names`` order (left s1..w2,
    then right s1..w2) — the dataset/action layout minus the two grippers.
    """

    def __init__(
        self,
        config: MinkIKConfig = MinkIKConfig(),
        urdf_path: Path | str = PINNED_URDF,
        *,
        warmup: bool = True,
    ) -> None:
        urdf_path = Path(urdf_path)
        if not _same_file(urdf_path, PINNED_URDF):
            raise ValueError(
                f"MinkIK runs on the pinned fork URDF only ({PINNED_URDF}); "
                f"got {urdf_path}. The vendored mink backend loads the pinned "
                "asset unconditionally (RUSTCORE_DESIGN.md 5.1)."
            )
        self._config = config
        names = list(urdf_arm_joint_names(is_left=True)) + list(
            urdf_arm_joint_names(is_left=False)
        )
        assert len(names) == N_ARM_JOINTS, names
        self._joint_names = names
        ee_bodies = (
            urdf_body_name(Joint.GRIPPER, is_left=True),
            urdf_body_name(Joint.GRIPPER, is_left=False),
        )
        elbow_bodies = (
            urdf_body_name(Joint.ELBOW, is_left=True),
            urdf_body_name(Joint.ELBOW, is_left=False),
        )
        # 14-joint tracker: the fork builds it on pyroki's 18-name order
        # (arms + 4 finger sliders) and freezes the fingers; with q = the 14
        # arm joints the same sliders are frozen at 0 by DofFreezingTask and
        # mink's name-mapped _to_qpos/solve are order-agnostic (K03).
        self._tracker = MinkTracker(
            config,
            joint_names=list(names),
            arm_joint_names=list(names),
            ee_bodies=ee_bodies,
            elbow_bodies=elbow_bodies,
        )
        # The tracker's own MjModel: FK, reach-clamp centres and the QP must
        # agree on ONE model (design 5.1). Private attribute by necessity —
        # the verbatim backend exposes no accessor.
        self._model: mujoco.MjModel = self._tracker._model
        self._qadr = qpos_indices(self._model, names)
        self._data = mujoco.MjData(self._model)
        self._ee_bid = {
            side: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body)
            for side, body in zip(("left", "right"), ee_bodies, strict=True)
        }
        # Shoulder centres for the reach clamp: the fork takes pyroki FK at
        # zero; the MuJoCo body position at qpos=0 agrees to 1.4e-8 m (K04).
        # float32 like the fork's, so the clamped branch rounds identically.
        self._data.qpos[:] = 0.0
        mujoco.mj_kinematics(self._model, self._data)
        self._shoulder: dict[str, np.ndarray] = {}
        for side, is_left in (("left", True), ("right", False)):
            sid = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                urdf_body_name(Joint.SHOULDER_1, is_left=is_left),
            )
            self._shoulder[side] = np.asarray(self._data.xpos[sid], dtype=np.float32).copy()
        self._last_solve_ms = 0.0
        if warmup:
            self.warmup()

    # -- Properties ----------------------------------------------------------

    @property
    def config(self) -> MinkIKConfig:
        return self._config

    @property
    def joint_names(self) -> list[str]:
        """The 14 arm joint names (left then right) — the ``q`` order."""
        return list(self._joint_names)

    @property
    def model(self) -> mujoco.MjModel:
        """The tracker's MuJoCo model (pinned fork URDF, all frames kept)."""
        return self._model

    @property
    def tracker(self) -> MinkTracker:
        return self._tracker

    @property
    def shoulder_positions(self) -> dict[str, np.ndarray]:
        """Reach-clamp centres (``left``/``right`` shoulder-1 body at q=0)."""
        return {k: v.copy() for k, v in self._shoulder.items()}

    @property
    def per_call_step_bound(self) -> float:
        """Per-call joint displacement bound (rad) = ``mink_max_joint_delta``
        (fork ``KinematicsSolver.per_call_step_bound`` on the mink branch)."""
        return float(self._config.mink_max_joint_delta)

    @property
    def fail_count(self) -> int:
        """QP solves that fell through the failure ladder to a HOLD (#523)."""
        return self._tracker.fail_count

    @property
    def last_solve_ms(self) -> float:
        """Wall time of the last ``tracker.solve`` (ms) — the ``ik_ms`` source."""
        return self._last_solve_ms

    # -- Posture / state -----------------------------------------------------

    def set_rest_posture(self, q14: np.ndarray) -> None:
        """Pin the posture attractor to the (settled) rest configuration.

        The serving path pins it ONCE to REST (design 5.3) — never
        ``set_posture`` (a no-op under ``mink_posture_target="rest"``).
        """
        self._tracker.set_rest_posture(self._as_q14(q14))

    def reset_tracking_state(self) -> None:
        """Clear the speed-gate memory at a control discontinuity (#523)."""
        self._tracker.reset_tracking_state()

    def warmup(self) -> None:
        """One zero-error solve at the range midpoints, then flush gate memory.

        Pages in the mink/daqp code paths before the first real tick. A HOLD
        here means the QP stack itself is broken (a solver/mujoco mismatch) —
        better to die at construction than to freeze the arms at TICK0.
        """
        model = self._model
        jids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self._joint_names
        ]
        q_mid = ((model.jnt_range[jids, 0] + model.jnt_range[jids, 1]) / 2.0).astype(
            np.float32
        )
        left, right = self.fk(q_mid)
        fails_before = self.fail_count
        self.solve(q_mid, left, right)
        if self.fail_count != fails_before:
            raise RuntimeError(
                "mink warm-up solve HELD (QP infeasible on a zero-error target); "
                "the mink/daqp/mujoco stack is not usable"
            )
        self.reset_tracking_state()

    # -- Kinematics ----------------------------------------------------------

    def fk(
        self, q14: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """EE poses ``((pos_l, R_l), (pos_r, R_r))`` of the gripper bodies.

        Same model and frame as the solve targets (float64 copies). Used by
        the startup FK-parity gate against ``fk.py`` v1 and by tests; not on
        the per-tick path.
        """
        q = self._as_q14(q14)
        self._data.qpos[:] = 0.0
        self._data.qpos[self._qadr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self._model, self._data)
        out = []
        for side in ("left", "right"):
            bid = self._ee_bid[side]
            out.append(
                (self._data.xpos[bid].copy(), self._data.xmat[bid].reshape(3, 3).copy())
            )
        return out[0], out[1]

    def solve(
        self,
        q14: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None,
        right_pose: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """Fork ``KinematicsSolver.ik`` mink branch, 14 joints, no elbow hints.

        Args:
            q14: Seed (and clamp anchor) — the caller applies the fork seed
                rule (previous solution unless >= 0.35 rad from measured).
            left_pose / right_pose: ``(pos3, R3x3)`` in the robot root frame
                (fk.py v1 frame == the pinned URDF's world frame), or None.

        Returns:
            float32 (14,). Reach-clamped targets, one ``tracker.solve``, then
            the componentwise clamp at ``mink_max_joint_delta`` — a
            never-binding backstop in normal operation (the QP budgets 0.98
            of it), kept because it is the fork's contract.
        """
        q_current = self._as_q14(q14)
        if left_pose is None and right_pose is None:
            return q_current

        cfg = self._config

        lp = lr = rp = rr = None
        if left_pose is not None:
            lp, lr = left_pose
            lp = _clamp_reach(
                np.asarray(lp, dtype=np.float32), self._shoulder["left"], cfg.max_reach
            )
        if right_pose is not None:
            rp, rr = right_pose
            rp = _clamp_reach(
                np.asarray(rp, dtype=np.float32),
                self._shoulder["right"],
                cfg.max_reach,
            )

        t0 = time.perf_counter()
        q_result_np = self._tracker.solve(
            np.asarray(q_current, dtype=np.float32),
            None if lp is None else (lp, np.asarray(lr, dtype=np.float32)),
            None if rp is None else (rp, np.asarray(rr, dtype=np.float32)),
            None,
            None,
        )
        self._last_solve_ms = (time.perf_counter() - t0) * 1e3
        max_delta = cfg.mink_max_joint_delta
        delta = np.clip(q_result_np - q_current, -max_delta, max_delta)
        q_out = (q_current + delta).astype(np.float32)
        return q_out

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _as_q14(q: np.ndarray) -> np.ndarray:
        """Shape guard only — dtype is left to the caller (the fork glue
        does float32 arithmetic on the caller's array; coercing here would
        change the rounding the equivalence test pins)."""
        arr = np.asarray(q)
        if arr.shape != (N_ARM_JOINTS,):
            raise ValueError(
                f"q must be the 14 arm joints (left s1..w2, right s1..w2), got shape {arr.shape}"
            )
        return arr
