"""Velocity-QP tracking backend (mink + daqp).  # shiraz (shirazai/shiraz#210)

Per-tick differential IK in the OpenArm/mink formulation: joint limits and
per-tick velocity as **hard QP constraints** (direction-preserving, unlike a
componentwise output clamp), global Tikhonov damping for singularity
robustness, a fixed rest-pose posture attractor for the null space, and
optional hard self-collision rows on the vendor torso<->arm pair scope.

This replaces only the per-tick tracking solve inside
:class:`~almond_axol.kinematics.solver.KinematicsSolver` (``backend="mink"``);
FK, reset-trajectory planning, and the engage/snap machinery stay on pyroki.

Design notes:

- The QP works in MuJoCo qpos space; ``q`` vectors at the interface stay in
  the pyroki actuated-joint order and are mapped by joint NAME, so the
  backend is agnostic to whether the finger joints are part of ``q``.
- Non-arm dofs (the gripper finger sliders) are frozen with an equality
  constraint and spliced back from the input, so they never move here.
- Per-call displacement is bounded to ``config.mink_max_joint_delta`` by
  construction (per-iteration ``VelocityLimit``); the caller's outer clamp
  becomes a never-binding backstop instead of a direction-distorting box.
- Failure ladder per iteration: constrained solve -> retry without the
  collision rows -> hold (return the seed unchanged). Drop-and-warn, never
  raise: an IK hiccup must not kill the dispatch thread.
"""

from __future__ import annotations

import logging

import mink
import mujoco
import numpy as np

from almond_axol.constants import Joint, urdf_body_name  # xr1-rustcore: K34 edit 1/4
from .ik_config import PINNED_URDF, MinkIKConfig as KinematicsConfig  # xr1-rustcore: K34 edit 2/4
from .ik_mujoco_model import body_geom_ids, load_mj_model, qpos_indices  # xr1-rustcore: K34 edit 3/4

_logger = logging.getLogger(__name__)

# Arm bodies guarded against the torso, per side (the vendor torso<->arm
# collision scope: fingers and TCP frames excluded). The s2 shoulder-yoke
# body is ALSO excluded: it rotates captive inside the torso mount, so its
# mesh legitimately rides at 0-10 mm clearance through the normal workspace
# — that interference is constrained by joint limits, not by IK (the vendor
# capsule model reaches the same outcome via its home-penetration pass).
_ARM_BODY_SUFFIXES = ("s3", "e1", "e2", "w0", "w1", "w2", "gripper")
_TORSO_BODIES = ("base", "s1")

# Fraction of mink_max_joint_delta the QP may use per call; the remaining
# margin guarantees the caller's componentwise clamp stays a no-op backstop.
_DELTA_BUDGET = 0.98


class MinkTracker:
    """Velocity-QP tracker with the :meth:`solve` contract of the NLLS path.

    Args:
        config: Kinematics configuration (``mink_*`` fields).
        joint_names: Actuated joint names in the caller's ``q`` order.
        arm_joint_names: The 14 arm joint names (left then right); these are
            the dofs the QP may move — everything else is frozen.
        ee_bodies: ``(left, right)`` end-effector body names (the IK target
            frame — the gripper bodies, not the TCPs).
        elbow_bodies: ``(left, right)`` elbow body names for the hint tasks.
    """

    def __init__(
        self,
        config: KinematicsConfig,
        joint_names: list[str],
        arm_joint_names: list[str],
        ee_bodies: tuple[str, str],
        elbow_bodies: tuple[str, str],
    ) -> None:
        self._config = config
        self._model = load_mj_model(PINNED_URDF)  # xr1-rustcore: K34 edit 4/4
        self._configuration = mink.Configuration(self._model)

        # Name-mapped views between the caller's q order and MuJoCo qpos.
        self._q_names = list(joint_names)
        self._qadr = qpos_indices(self._model, self._q_names)
        # Per-mapped-joint ranges: the seed must be clamped into them before
        # the QP sees it. An out-of-range seed makes ConfigurationLimit's
        # retreat row (0.95×violation per iteration) contradict the velocity
        # rows for violations over ~1°, turning every solve infeasible — a
        # permanent hold. Clamping recovers at the outer clamp's walk-back
        # rate instead (the jaxls backend recovers softly via its limit
        # cost; this is the hard-constraint equivalent).
        jids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self._q_names
        ]
        limited = self._model.jnt_limited[jids].astype(bool)
        self._q_lo = np.where(
            limited, self._model.jnt_range[jids, 0], -np.inf
        )
        self._q_hi = np.where(limited, self._model.jnt_range[jids, 1], np.inf)
        arm_set = set(arm_joint_names)
        self._frozen_dofs = [
            int(self._model.jnt_dofadr[j])
            for j in range(self._model.njnt)
            if mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
            not in arm_set
        ]

        lm = config.mink_lm_damping
        self._ee_tasks = {
            side: mink.FrameTask(
                frame_name=name,
                frame_type="body",
                position_cost=config.mink_position_cost,
                orientation_cost=config.mink_orientation_cost,
                lm_damping=lm,
            )
            for side, name in zip(("left", "right"), ee_bodies, strict=True)
        }
        self._elbow_tasks = {
            side: mink.FrameTask(
                frame_name=name,
                frame_type="body",
                position_cost=config.mink_elbow_cost,
                orientation_cost=0.0,
                lm_damping=lm,
            )
            for side, name in zip(("left", "right"), elbow_bodies, strict=True)
        }
        self._posture_task = mink.PostureTask(
            self._model, cost=config.mink_posture_cost
        )
        self._posture_task.set_target(np.zeros(self._model.nq))
        # Speed-gated posture: remember the last EE targets to estimate the
        # demanded speed; the attractor's effective cost rolls off
        # quadratically above mink_posture_speed_gate (see config). The
        # applied scale is additionally rate-limited through a one-pole
        # low-pass (mink_posture_gate_tau) so a speed-gate crossing shifts
        # the QP optimum over ~tau instead of snapping it in one solve —
        # ep4 of motion_benchmarks_5 measured a 0.46 rad null-space snap in
        # 0.25 s when the hand re-crossed the gate after a slow phase.
        self._gate_scale: float | None = None
        self._last_ee_target: dict[str, np.ndarray | None] = {
            "left": None, "right": None,
        }

        # Elbow-hint projection: with position_multiplier > 1 the scaled
        # human elbow hint orbits well outside the robot elbow's reachable
        # sphere (~0.55 m demanded vs ~0.40 m reachable on the recorded
        # sessions) — an unreachable point target tilts the whole arm (the
        # "elbow way too high" symptom) instead of shaping its direction.
        # Project every hint onto the sphere about the (fixed) shoulder
        # center so the task carries direction only.
        data = mujoco.MjData(self._model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(self._model, data)
        self._shoulder_pos = {}
        self._elbow_bid = {}
        self._elbow_radius = {}  # qpos=0 fallback; solve() uses per-pose
        for side, is_left, elbow_body in zip(
            ("left", "right"), (True, False), elbow_bodies, strict=True
        ):
            sid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY,
                urdf_body_name(Joint.SHOULDER_1, is_left=is_left),
            )
            eid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, elbow_body
            )
            self._shoulder_pos[side] = data.xpos[sid].copy()
            self._elbow_bid[side] = eid
            self._elbow_radius[side] = float(
                np.linalg.norm(data.xpos[eid] - data.xpos[sid])
            )

        self._constraints = (
            [mink.DofFreezingTask(self._model, dof_indices=self._frozen_dofs)]
            if self._frozen_dofs
            else []
        )

        # Per-iteration velocity cap sized so the whole call's displacement
        # stays inside mink_max_joint_delta (the outer clamp never binds).
        per_iter_vmax = (config.mink_max_joint_delta * _DELTA_BUDGET) / (
            config.mink_iterations * config.mink_dt
        )
        base_limits: list[object] = [
            mink.ConfigurationLimit(self._model),
            mink.VelocityLimit(
                self._model, dict.fromkeys(arm_joint_names, per_iter_vmax)
            ),
        ]
        self._limits = list(base_limits)
        self._limits_no_collision = list(base_limits)
        # Kept torso<->arm geom-id pairs (introspection/diagnostics).
        self.collision_pairs: list[tuple[int, int]] = []
        if config.mink_collision:
            pairs = self._home_feasible_collision_pairs()
            self.collision_pairs = [(a[0], b[0]) for a, b in pairs]
            if pairs:
                self._limits.append(
                    mink.CollisionAvoidanceLimit(
                        self._model,
                        geom_pairs=pairs,
                        minimum_distance_from_collisions=(
                            config.mink_collision_min_distance
                        ),
                        collision_detection_distance=(
                            config.mink_collision_detection_distance
                        ),
                    )
                )
        self._fail_count = 0

    def _home_feasible_collision_pairs(
        self,
    ) -> list[tuple[list[int], list[int]]]:
        """Torso<->arm geom pairs, minus any already too close at home.

        Mirrors the vendor collision model: pairs that violate the clearance
        at the home pose are over-conservative fits the IK could never
        separate — keeping them would make the home configuration infeasible
        for the hard rows.
        """
        torso = body_geom_ids(self._model, list(_TORSO_BODIES))
        arms = body_geom_ids(
            self._model,
            [
                f"{side}_{sfx}"
                for side in ("left", "right")
                for sfx in _ARM_BODY_SUFFIXES
            ],
        )
        data = mujoco.MjData(self._model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(self._model, data)
        fromto = np.empty(6)
        pairs: list[tuple[list[int], list[int]]] = []
        dropped = 0
        for tg in torso:
            for ag in arms:
                d = mujoco.mj_geomDistance(self._model, data, tg, ag, 1.0, fromto)
                if d <= self._config.mink_collision_min_distance:
                    dropped += 1
                    continue
                pairs.append(([tg], [ag]))
        if dropped:
            _logger.info(
                "mink collision rows: %d torso<->arm geom pairs (%d dropped "
                "as infeasible at home)",
                len(pairs),
                dropped,
            )
        return pairs

    def set_posture(self, q: np.ndarray) -> None:
        """Engage-edge posture update; a no-op in ``"rest"`` target mode."""
        if self._config.mink_posture_target == "engage":
            self._posture_task.set_target(self._to_qpos(q))

    def reset_tracking_state(self) -> None:
        """Clear cross-call gate memory at a control discontinuity.

        shiraz (shirazai/shiraz#523): ``_last_ee_target``/``_gate_scale``
        persist across calls by design (the speed finite-difference), but a
        hand-back / episode seam moves the EE target discontinuously — the
        stale memory would read as a huge one-call speed spike and then
        recover only over ``mink_posture_gate_tau``. Callers that re-anchor
        the seed at a discontinuity (``reset_cartesian_seed``) clear this
        too; the warmup dummy solve is also flushed this way.
        """
        self._last_ee_target = {"left": None, "right": None}
        self._gate_scale = None

    @property
    def fail_count(self) -> int:
        """QP solves that fell through the failure ladder to a HOLD.

        shiraz (shirazai/shiraz#523): a failed solve returns the seed
        unchanged — a frozen command the guards cannot see. Consumers export
        this so a session verdict can complain instead of reading clean.
        """
        return self._fail_count

    def set_rest_posture(self, q: np.ndarray) -> None:
        """Pin the posture attractor to the (settled) rest configuration."""
        self._posture_task.set_target(self._to_qpos(q))

    def solve(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None,
        right_pose: tuple[np.ndarray, np.ndarray] | None,
        left_elbow_pos: np.ndarray | None,
        right_elbow_pos: np.ndarray | None,
    ) -> np.ndarray:
        """One tracking step; same target semantics as the NLLS ``ik()``.

        Returns the next joint vector in the caller's ``q`` order. Frozen
        (non-arm) entries pass through from ``q_current`` unchanged.
        """
        cfg = self._config
        self._configuration.update(self._to_qpos(q_current))

        # Posture speed gate: estimate the fastest EE-target speed this call
        # (per-call target delta x the configured solve cadence,
        # cfg.mink_solve_hz) and roll the attractor off.
        if cfg.mink_posture_speed_gate > 0.0:
            speed = 0.0
            for side, pose in (("left", left_pose), ("right", right_pose)):
                if pose is None:
                    continue
                pos = np.asarray(pose[0], dtype=np.float64)
                last = self._last_ee_target[side]
                if last is not None:
                    speed = max(speed, float(np.linalg.norm(pos - last))
                                * cfg.mink_solve_hz)
                self._last_ee_target[side] = pos
            scale = 1.0 / (1.0 + (speed / cfg.mink_posture_speed_gate) ** 2)
            if cfg.mink_posture_gate_tau > 0.0:
                if self._gate_scale is None:
                    self._gate_scale = scale
                else:
                    alpha = min((1.0 / cfg.mink_solve_hz)
                                / cfg.mink_posture_gate_tau, 1.0)
                    self._gate_scale += alpha * (scale - self._gate_scale)
                scale = self._gate_scale
            self._posture_task.set_cost(cfg.mink_posture_cost * scale)

        tasks: list[object] = [self._posture_task]
        for side, pose in (("left", left_pose), ("right", right_pose)):
            if pose is None:
                continue
            pos, rot = pose
            task = self._ee_tasks[side]
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(np.asarray(rot, dtype=np.float64)),
                    np.asarray(pos, dtype=np.float64),
                )
            )
            tasks.append(task)
        if cfg.mink_elbow_cost > 0.0:
            for side, hint in (
                ("left", left_elbow_pos),
                ("right", right_elbow_pos),
            ):
                if hint is None:
                    continue
                center = self._shoulder_pos[side]
                d = np.asarray(hint, dtype=np.float64) - center
                norm = float(np.linalg.norm(d))
                if norm < 0.05:
                    # A hint within 5 cm of the shoulder center carries no
                    # usable direction — a few mm of tracking noise would
                    # swing the projected target across the whole sphere.
                    continue
                # The shoulder joint cluster is not concentric (s2/s3
                # anchors sit ~7 cm off the s1 axis): the true elbow radius
                # varies ±5 cm with pose, so measure it from the CURRENT
                # configuration rather than the qpos=0 constant.
                elbow_now = self._configuration.data.xpos[
                    self._elbow_bid[side]
                ]
                radius = float(np.linalg.norm(elbow_now - center))
                if radius < 0.05:
                    radius = self._elbow_radius[side]
                projected = center + d * (radius / norm)
                task = self._elbow_tasks[side]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3.identity(), projected
                    )
                )
                tasks.append(task)

        for _ in range(cfg.mink_iterations):
            vel = self._solve_step(tasks)
            if vel is None:
                return np.asarray(q_current, dtype=np.float32).copy()
            self._configuration.integrate_inplace(vel, cfg.mink_dt)

        out = np.asarray(q_current, dtype=np.float64).copy()
        out[: len(self._qadr)] = self._configuration.q[self._qadr]
        return out.astype(np.float32)

    def _solve_step(self, tasks: list[object]) -> np.ndarray | None:
        """Failure ladder: full -> no collision rows -> hold (None)."""
        cfg = self._config
        last_exc: Exception | None = None
        for limits in (self._limits, self._limits_no_collision):
            try:
                return mink.solve_ik(
                    self._configuration,
                    tasks,
                    cfg.mink_dt,
                    "daqp",
                    damping=cfg.mink_damping,
                    safety_break=False,
                    limits=limits,
                    constraints=self._constraints,
                )
            except Exception as exc:  # noqa: BLE001 - drop-and-warn semantics
                last_exc = exc
                continue
        self._fail_count += 1
        if self._fail_count == 1 or self._fail_count % 100 == 0:
            _logger.warning(
                "mink solve failed (%d so far), holding last q: %s",
                self._fail_count,
                last_exc,
            )
        return None

    def _to_qpos(self, q: np.ndarray) -> np.ndarray:
        """Caller-order joint vector -> full MuJoCo qpos (unmapped dofs 0).

        Values are clamped into the joint ranges — see the seed-clamp note
        in ``__init__``.
        """
        qpos = np.zeros(self._model.nq)
        qpos[self._qadr] = np.clip(
            np.asarray(q, dtype=np.float64)[: len(self._qadr)],
            self._q_lo,
            self._q_hi,
        )
        return qpos
