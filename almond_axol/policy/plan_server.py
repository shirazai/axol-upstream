"""Desktop SDK for cached-plan continuation, without robot/model coupling."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from .plan_protocol import (
    Continuation,
    PlanActions,
    PlanObservation,
    PlanSpec,
    decode_hello,
    decode_observation,
    decode_reset,
    encode_actions,
    encode_ready,
    encode_reset,
    validate_actions,
)
from .protocol import (
    MAX_MESSAGE_BYTES,
    PolicyProtocolError,
    decode_message,
    encode_error,
)

_logger = logging.getLogger(__name__)


class PredictionCache:
    """Bounded exact float32 predictions, retaining the last referenced plan.

    Call ``remaining(obs.continuation)`` before ``put`` on every inference,
    including bootstrap (``None``). A generated but rejected candidate then
    cannot evict the robot's currently accepted prediction. Clear on reset.
    No copies are changed in place; the returned suffix is read-only.
    """

    def __init__(self, capacity: int = 4) -> None:
        if type(capacity) is not int or capacity < 2:
            raise ValueError("Prediction cache capacity must be at least two.")
        self.capacity = capacity
        self._chunks: OrderedDict[str, np.ndarray] = OrderedDict()
        self._referenced: str | None = None

    def clear(self) -> None:
        self._chunks.clear()
        self._referenced = None

    def remaining(self, continuation: Continuation | None) -> np.ndarray | None:
        if continuation is None:
            self._referenced = None
            return None
        chunk = self._chunks.get(continuation.prediction_id)
        if chunk is None:
            raise PolicyProtocolError(
                f"Unknown continuation prediction {continuation.prediction_id!r}; reset the episode."
            )
        row = continuation.from_row
        if type(row) is not int or not 0 <= row < len(chunk):
            raise PolicyProtocolError(
                "Continuation from_row is outside its prediction."
            )
        self._referenced = continuation.prediction_id
        return chunk[row:]

    def put(self, prediction_id: str, actions: np.ndarray) -> None:
        if prediction_id in self._chunks:
            raise PolicyProtocolError("Prediction IDs cannot be reused.")
        chunk = np.array(actions, dtype=np.float32, copy=True)
        if chunk.ndim != 2 or not len(chunk) or not np.isfinite(chunk).all():
            raise PolicyProtocolError(
                "Cached predictions must be finite action matrices."
            )
        chunk.flags.writeable = False
        self._chunks[prediction_id] = chunk
        while len(self._chunks) > self.capacity:
            victim = next(key for key in self._chunks if key != self._referenced)
            del self._chunks[victim]


class PlanPolicy:
    """Model adapters own conditioning, desktop task selection, and transforms.

    ``setup`` validates the complete session specification. ``reset`` clears
    every model cache. ``infer`` must use the accepted continuation (never
    assume the most recent prediction was executed), and returns absolute
    robot-space targets in the negotiated layout. A response row retains its
    original time slot even if inference finishes late or early.
    """

    def setup(self, spec: PlanSpec) -> None:
        """Validate the requested layouts, camera preprocessing, and timing."""

    def reset(self) -> None:
        """Clear model state and cached predictions at an episode boundary."""

    def infer(self, obs: PlanObservation) -> np.ndarray | PlanActions:
        raise NotImplementedError


class PlanPolicyServer:
    """One active robot connection; strict v2 protocol, one request in flight."""

    def __init__(
        self, policy: PlanPolicy, host: str = "0.0.0.0", port: int = 8765
    ) -> None:
        from websockets.sync.server import serve

        if not isinstance(policy, PlanPolicy):
            raise TypeError("policy must implement PlanPolicy")
        self.policy = policy
        self._session = threading.Lock()
        self._server = serve(
            self._handle, host, port, max_size=MAX_MESSAGE_BYTES, compression=None
        )

    @property
    def port(self) -> int:
        return self._server.socket.getsockname()[1]

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()

    def _handle(self, ws: Any) -> None:
        from websockets.exceptions import ConnectionClosed

        if not self._session.acquire(blocking=False):
            ws.send(encode_error("Policy server already has a robot connected."))
            return
        spec: PlanSpec | None = None
        episode: int | None = None
        seen: set[str] = set()
        cache = PredictionCache()
        try:
            for message in ws:
                try:
                    header, payload = decode_message(message)
                    kind = header["type"]
                    if spec is None:
                        spec = decode_hello(header, payload)
                        self.policy.setup(spec)
                        ws.send(encode_ready(spec))
                    elif kind == "reset":
                        episode = decode_reset(header, payload)
                        self.policy.reset()
                        cache.clear()
                        seen.clear()
                        ws.send(encode_reset(episode, reply=True))
                    elif kind == "infer":
                        if episode is None:
                            raise PolicyProtocolError(
                                "Reset the episode before inference."
                            )
                        obs = decode_observation(header, payload, spec)
                        if obs.request_id in seen:
                            raise PolicyProtocolError(
                                "Duplicate request_id within an episode."
                            )
                        if len(seen) >= 1_000_000:
                            raise PolicyProtocolError(
                                "Episode request limit reached; reset required."
                            )
                        cache.remaining(obs.continuation)
                        seen.add(obs.request_id)
                        result = self.policy.infer(obs)
                        reply = (
                            result
                            if isinstance(result, PlanActions)
                            else PlanActions(obs.request_id, result)
                        )
                        reply = validate_actions(reply, spec)
                        if reply.request_id != obs.request_id:
                            raise PolicyProtocolError(
                                "Adapter returned the wrong request_id."
                            )
                        cache.put(obs.request_id, reply.actions)
                        ws.send(encode_actions(reply, spec))
                    else:
                        raise PolicyProtocolError(
                            f"Unexpected session message {kind!r}."
                        )
                except ConnectionClosed:
                    raise
                except Exception as exc:  # fail closed, visible to caller
                    _logger.exception("Plan policy request failed")
                    ws.send(encode_error(f"{type(exc).__name__}: {exc}"))
                    return
        except ConnectionClosed:
            pass
        finally:
            self._session.release()
