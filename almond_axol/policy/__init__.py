"""Run your own model on Axol — no LeRobot checkpoint required.

Write a :class:`Policy` (or a plain ``obs -> chunk`` function), start it with
:func:`serve`, and run ``axol run-policy --policy_type custom`` (control
panel: **Run Policy**, policy type ``custom``). The robot sends each
:class:`Observation` — joint state, RGB camera frames, task — and executes the
action chunks you return. See :mod:`almond_axol.policy.protocol` for the wire
format if your model lives outside Python.

For compressed images and accepted-plan continuation, use :class:`PlanPolicy`
with :class:`PlanPolicyServer` and ``custom_protocol=2``. Its instruction and
model-specific conditioning remain on the desktop; the robot sends measured
observations and a reference to its accepted plan. Version 1 remains available.

This package only needs Axol's base install (numpy, websockets, and OpenCV for
the v2 PNG codec).
"""

from .client import PolicyClient, policy_url
from .plan_client import PlanPolicyClient
from .plan_protocol import (
    PLAN_PROTOCOL_VERSION,
    Continuation,
    PlanActions,
    PlanObservation,
    PlanSpec,
)
from .plan_server import PlanPolicy, PlanPolicyServer, PredictionCache
from .protocol import (
    PROTOCOL_VERSION,
    CameraSpec,
    Observation,
    PolicyProtocolError,
    PolicyRemoteError,
    PolicySpec,
    ReadyInfo,
)
from .server import Policy, PolicyServer, serve

__all__ = [
    "PLAN_PROTOCOL_VERSION",
    "PROTOCOL_VERSION",
    "CameraSpec",
    "Continuation",
    "Observation",
    "PlanActions",
    "PlanObservation",
    "PlanPolicy",
    "PlanPolicyClient",
    "PlanPolicyServer",
    "PlanSpec",
    "Policy",
    "PolicyClient",
    "PolicyProtocolError",
    "PolicyRemoteError",
    "PolicyServer",
    "PolicySpec",
    "PredictionCache",
    "ReadyInfo",
    "policy_url",
    "serve",
]
