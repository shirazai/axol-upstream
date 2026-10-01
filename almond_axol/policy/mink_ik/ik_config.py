"""Configuration of the vendored mink IK (shiraz #550, RUSTCORE_DESIGN.md 5.1).

Why a separate dataclass instead of the vendor's ``KinematicsConfig``: on
the Rust-core (chemical-speak) branch the vendor config has none of the
``mink_*`` fields — the fork's velocity-QP backend was never upstreamed —
and importing ``almond_axol.kinematics`` at all drags jax/jaxls/pyroki into
the process (its ``__init__`` imports the JAX solver). The XR-1 serving path
must stay JAX-free on the control thread (K05/K34), so the mink stack gets
its own config with the fork-main (80e7a8c) values frozen in.

Every default below is the value the hardware sessions of #456/#483/#523
were validated with; the vendor's teleop defaults (``mink_solve_hz`` 120,
``mink_max_joint_delta`` 2*pi/120) are deliberately NOT used — see
``fork almond_axol/lerobot/robot/robot_axol.py:_ensure_ik`` for the cadence
argument and ``KinematicsConfig`` (fork ``kinematics/config.py``) for the
per-field tuning rationale, which this module does not repeat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# xr1-rustcore: K01/K02 — the IK must invert the CHECKPOINT's FK (fk.py v1,
# yaw-0 root, EE = the ``*_gripper`` link origin), not track the vendor's
# latest URDF: chemical-speak rotated the root by +90 deg, which would put
# every solved pose up to 1.119 m off. The fork-main URDF and its 22 meshes
# are pinned under ``assets/`` so the loaded model never depends on which
# vendor tree is checked out.
PINNED_URDF: Path = Path(__file__).resolve().parent / "assets" / "axol_fork_80e7a8c.urdf"

# The serving per-call joint displacement budget (rad): the jaxls
# anti-teleport clamp of fork ``KinematicsConfig.max_joint_delta``
# (0.0055 * 2 * pi = 0.0346 rad/call = 1.04 rad/s at 30 Hz). Dispatch parity
# with everything #456/#483 validated; raising it is #511's lever and needs
# its co-requisites (iterations, collision shell) — the adapter refuses
# overrides above 0.052 rad (design 5.3).
SERVING_STEP_BUDGET_RAD: float = 0.0055 * 2 * math.pi


@dataclass
class MinkIKConfig:
    """Cost weights and solver parameters of :class:`~shiraz_axol.xr1.ik.MinkIK`.

    Field names match the fork ``KinematicsConfig`` ``mink_*`` fields one for
    one so ``ik_mink_backend.py`` (fork ``mink_backend.py`` verbatim) reads
    them unchanged. Defaults are the fork-main serving values.

    Attributes:
        mink_position_cost: QP weight on end-effector position error.
        mink_orientation_cost: QP weight on end-effector orientation error.
        mink_elbow_cost: QP weight on the operator-elbow hint (0 = off; the
            serving path never passes elbow hints).
        mink_posture_cost: QP weight of the null-space posture attractor.
        mink_posture_speed_gate: EE-target speed (m/s) at which the posture
            attractor's effective weight halves (0 disables gating).
        mink_posture_gate_tau: Time constant (s) of the one-pole low-pass on
            the speed-gate scale (0 = raw gate).
        mink_posture_target: ``"rest"`` pins the attractor to the settled
            rest pose (the serving mode); ``"engage"`` re-pins at each
            :meth:`MinkTracker.set_posture` call (teleop semantics).
        mink_lm_damping: Per-task Levenberg-Marquardt damping.
        mink_damping: Global Tikhonov damping on joint velocity.
        mink_iterations: solve/integrate iterations per call.
        mink_dt: Integration step per iteration (keep 1.0 — mink 1.2.0's QP
            variable is a displacement; see the fork config rationale).
        mink_collision: Hard torso<->arm self-collision rows on/off.
        mink_collision_min_distance: Minimum allowed geom clearance (m).
        mink_collision_detection_distance: Distance (m) at which collision
            rows activate.
        mink_solve_hz: The ``solve()`` cadence the per-second gate quantities
            assume — 30 Hz = the policy tick (one solve per plan row).
        max_joint_delta: Kept for the fork's ``per_call_step_bound`` naming
            parity; equal to ``mink_max_joint_delta`` by construction.
        mink_max_joint_delta: Per-call joint displacement budget (rad) of the
            QP velocity rows AND of the componentwise backstop clamp.
        max_reach: Shoulder-to-EE reach clamp (m) applied to the targets
            before the solve. 0.8 = the fork value (NOT chemical-speak's
            0.82 "asymptotic cap").
    """

    mink_position_cost: float = 1.0
    mink_orientation_cost: float = 0.25
    mink_elbow_cost: float = 0.0
    mink_posture_cost: float = 0.05
    mink_posture_speed_gate: float = 0.15
    mink_posture_gate_tau: float = 1.0
    mink_posture_target: str = "rest"
    mink_lm_damping: float = 0.01
    mink_damping: float = 0.8
    mink_iterations: int = 4
    mink_dt: float = 1.0
    mink_collision: bool = True
    mink_collision_min_distance: float = 0.01
    mink_collision_detection_distance: float = 0.06
    mink_solve_hz: float = 30.0
    max_joint_delta: float = SERVING_STEP_BUDGET_RAD
    mink_max_joint_delta: float = SERVING_STEP_BUDGET_RAD
    max_reach: float = 0.8
