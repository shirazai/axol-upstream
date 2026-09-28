"""Hardware-free protocol conformance and real WebSocket session tests."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from almond_axol.policy.plan_client import PlanPolicyClient
from almond_axol.policy.plan_protocol import (
    Continuation,
    PlanActions,
    PlanObservation,
    PlanSpec,
    decode_actions,
    decode_hello,
    decode_observation,
    decode_ready,
    decode_reset,
    encode_actions,
    encode_hello,
    encode_observation,
    encode_ready,
    encode_reset,
)
from almond_axol.policy.plan_server import PlanPolicy, PlanPolicyServer, PredictionCache
from almond_axol.policy.protocol import (
    CameraSpec,
    PolicyProtocolError,
    PolicyRemoteError,
    decode_message,
    encode_error,
    encode_message,
)


@pytest.fixture
def spec():
    return PlanSpec(
        state_names=("joint.left", "joint.right"),
        action_names=("x", "y", "grip"),
        cameras=(CameraSpec("overhead", (12, 18, 3)), CameraSpec("wrist", (8, 9, 3))),
        fps=30,
        actions_per_chunk=12,
        request_interval=4,
        max_adoption_offset_steps=6,
    )


def observation(spec, request_id="request-1", continuation=None, delay_steps=None):
    rng = np.random.default_rng(17)
    return PlanObservation(
        request_id=request_id,
        state=np.array([0.1, -0.2], dtype=np.float32),
        images={
            camera.name: rng.integers(0, 256, camera.shape, dtype=np.uint8)
            for camera in spec.cameras
        },
        state_sample_time_ns=5_000_000_001,
        image_capture_time_ns={
            camera.name: 4_999_999_000 + i for i, camera in enumerate(spec.cameras)
        },
        continuation=continuation,
        delay_steps=delay_steps,
    )


class TestCodec:
    def test_negotiated_spec_is_frozen_and_has_no_task(self, spec):
        hello = encode_hello(spec)
        assert decode_hello(*decode_message(hello)) == spec
        assert decode_ready(*decode_message(encode_ready(spec))) == spec
        header, _ = decode_message(hello)
        assert header["version"] == 2
        assert "task" not in header
        assert header["cameras"][0]["codec"] == "png"
        assert len(spec.state_names) != len(spec.action_names)

    def test_rgb_png_roundtrip_and_minimal_request(self, spec):
        obs = observation(spec, continuation=Continuation("accepted-0", 3))
        header, payload = decode_message(encode_observation(obs, spec))
        assert set(header) == {"type", "request_id", "observation", "continuation"}
        assert bytes(payload[:8]) == b"\x89PNG\r\n\x1a\n"
        decoded = decode_observation(header, payload, spec)
        np.testing.assert_array_equal(decoded.state, obs.state)
        assert decoded.request_id == obs.request_id
        assert decoded.continuation == obs.continuation
        assert decoded.image_capture_time_ns == obs.image_capture_time_ns
        assert decoded.state_sample_time_ns == obs.state_sample_time_ns
        for name in spec.camera_names:
            np.testing.assert_array_equal(decoded.images[name], obs.images[name])
        assert decoded.delay_steps is None

    def test_adaptive_delay_is_explicit_and_optional(self, spec):
        obs = observation(spec, delay_steps=0)
        packet = encode_observation(obs, spec)
        assert decode_message(packet)[0]["delay_steps"] == 0
        assert decode_observation(*decode_message(packet), spec).delay_steps == 0

    def test_action_reply_is_exact_float32(self, spec):
        actions = np.linspace(-0.8, 0.9, 36, dtype=np.float32).reshape(12, 3)
        reply = PlanActions("request-1", actions)
        header, payload = decode_message(encode_actions(reply, spec))
        assert set(header) == {"type", "request_id", "shape"}
        decoded = decode_actions(header, payload, spec)
        np.testing.assert_array_equal(decoded.actions, actions)
        assert decoded.actions.dtype == np.float32
        assert decoded.request_id == reply.request_id

    def test_reply_can_tighten_but_not_relax_adoption_bound(self, spec):
        reply = PlanActions("request-1", np.zeros((12, 3)), 4)
        assert (
            decode_actions(
                *decode_message(encode_actions(reply, spec)), spec
            ).max_adoption_offset_steps
            == 4
        )
        with pytest.raises(PolicyProtocolError, match="relax"):
            encode_actions(replace(reply, max_adoption_offset_steps=7), spec)

    @pytest.mark.parametrize(
        "update",
        [
            {"version": 1},
            {"version": True},
            {"task": "desktop only"},
            {"fps": True},
            {"fps": 0},
            {"state_names": ["same", "same"]},
            {"actions_per_chunk": 0},
            {"request_interval": 13},
            {"max_adoption_offset_steps": 12},
        ],
    )
    def test_invalid_hello(self, spec, update):
        header, payload = decode_message(encode_hello(spec))
        header.update(update)
        with pytest.raises(PolicyProtocolError):
            decode_hello(header, payload)

    @pytest.mark.parametrize(
        "shape,codec",
        [([12, 18, 1], "png"), ([12, 18, 3], "raw"), ([8193, 18, 3], "png")],
    )
    def test_camera_codec_and_shape_are_restricted(self, spec, shape, codec):
        header, payload = decode_message(encode_hello(spec))
        header["cameras"][0].update(shape=shape, codec=codec)
        with pytest.raises(PolicyProtocolError):
            decode_hello(header, payload)

    def test_total_decoded_image_budget(self, spec):
        large = replace(spec, cameras=(CameraSpec("big", (8192, 8192, 3)),))
        with pytest.raises(PolicyProtocolError, match="decoded image size"):
            encode_hello(large)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("state", [float("nan"), 0]),
            ("state", [True, 0]),
            ("state", [float("inf"), 0]),
            ("state", [1e20, 0]),
            ("state", [0]),
            ("state_sample_time_ns", 1.5),
            ("state_sample_time_ns", -1),
            ("state_sample_time_ns", True),
            ("task", "not here"),
        ],
    )
    def test_invalid_observation(self, spec, key, value):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        header["observation"][key] = value
        with pytest.raises(PolicyProtocolError):
            decode_observation(header, payload, spec)

    @pytest.mark.parametrize(
        "value",
        [
            {"prediction_id": "previous", "from_row": -1},
            {"prediction_id": "previous", "from_row": True},
            {"prediction_id": "", "from_row": 0},
            {"prediction_id": "previous", "from_row": 0, "actions": []},
        ],
    )
    def test_invalid_continuation(self, spec, value):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        header["continuation"] = value
        with pytest.raises(PolicyProtocolError):
            decode_observation(header, payload, spec)

    def test_unknown_request_fields_and_null_delay_rejected(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        with pytest.raises(PolicyProtocolError):
            decode_observation({**header, "delay_steps": None}, payload, spec)
        with pytest.raises(PolicyProtocolError):
            decode_observation({**header, "task": "no"}, payload, spec)

    def test_truncated_and_trailing_images(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        for bad in (payload[:-1], memoryview(bytes(payload) + b"x")):
            with pytest.raises(PolicyProtocolError):
                decode_observation(header, bad, spec)

    def test_png_size_checked_before_decoder_allocation(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        corrupt = bytearray(payload)
        corrupt[16:20] = (1_000_000).to_bytes(4, "big")
        with patch("cv2.imdecode") as decoder:
            with pytest.raises(PolicyProtocolError, match="dimensions/format"):
                decode_observation(header, memoryview(corrupt), spec)
            decoder.assert_not_called()

    def test_png_signature_and_camera_order_checked(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        with pytest.raises(PolicyProtocolError, match="PNG"):
            decode_observation(header, memoryview(b"x" + bytes(payload[1:])), spec)
        header["observation"]["images"].reverse()
        with pytest.raises(PolicyProtocolError, match="camera order"):
            decode_observation(header, payload, spec)

    def test_control_messages_reject_payload(self, spec):
        for decode, encoded in (
            (decode_hello, encode_hello(spec)),
            (decode_ready, encode_ready(spec)),
            (decode_reset, encode_reset(2)),
        ):
            header, _ = decode_message(encoded)
            with pytest.raises(PolicyProtocolError, match="payload"):
                decode(header, memoryview(b"unexpected"))

    @pytest.mark.parametrize(
        "chunk", [np.full((12, 3), np.nan), np.zeros((13, 3)), np.zeros((12, 4))]
    )
    def test_invalid_model_output(self, spec, chunk):
        with pytest.raises(PolicyProtocolError):
            encode_actions(PlanActions("r", chunk), spec)

    def test_invalid_action_payload_and_nonfinite_wire(self, spec):
        header = {"type": "actions", "request_id": "r", "shape": [12, 3]}
        with pytest.raises(PolicyProtocolError, match="byte length"):
            decode_actions(header, memoryview(b""), spec)
        payload = memoryview(np.full((12, 3), np.inf, dtype="<f4").tobytes())
        with pytest.raises(PolicyProtocolError, match="non-finite"):
            decode_actions(header, payload, spec)


class EchoPolicy(PlanPolicy):
    def __init__(self):
        self.cache = PredictionCache()
        self.resets = 0
        self.received = []

    def setup(self, spec):
        self.spec = spec

    def reset(self):
        self.cache.clear()
        self.resets += 1

    def infer(self, obs):
        remaining = self.cache.remaining(obs.continuation)
        self.received.append(obs)
        chunk = np.full(
            (self.spec.actions_per_chunk, len(self.spec.action_names)),
            len(self.received),
            dtype=np.float32,
        )
        if remaining is not None:
            chunk[: len(remaining)] = remaining
        self.cache.put(obs.request_id, chunk)
        return chunk


@contextmanager
def running_server(policy):
    server = PlanPolicyServer(policy, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.port}"
    finally:
        server.shutdown()
        thread.join(timeout=3)
        assert not thread.is_alive()


class TestSessions:
    def test_png_request_cache_continuation_and_reset_roundtrip(self, spec):
        policy = EchoPolicy()
        with running_server(policy) as url, PlanPolicyClient(url) as client:
            assert client.connect(spec) == spec
            client.reset(1)
            obs = observation(spec)
            first = client.infer(obs)
            np.testing.assert_array_equal(
                policy.received[0].images["overhead"], obs.images["overhead"]
            )
            second = client.infer(
                observation(spec, "request-2", Continuation(first.request_id, 8))
            )
            np.testing.assert_array_equal(second.actions[:4], first.actions[8:])
            np.testing.assert_array_equal(second.actions[4:], 2)
            client.reset(2)
            assert policy.resets == 2
            # IDs may be reused after an explicit reset; previous cache cannot.
            fresh = client.infer(observation(spec))
            np.testing.assert_array_equal(fresh.actions, 3)

    def test_requires_connect_and_episode_reset(self, spec):
        with (
            PlanPolicyClient("ws://127.0.0.1:1") as client,
            pytest.raises(PolicyProtocolError, match="Connect and reset"),
        ):
            client.infer(observation(spec))
        with running_server(EchoPolicy()) as url, PlanPolicyClient(url) as client:
            client.connect(spec)
            with pytest.raises(PolicyProtocolError, match="Connect and reset"):
                client.infer(observation(spec))

    @pytest.mark.parametrize(
        "continuation", [Continuation("missing", 0), Continuation("request-1", 12)]
    )
    def test_unknown_or_exhausted_reference_fails_closed(self, spec, continuation):
        policy = EchoPolicy()
        with running_server(policy) as url, PlanPolicyClient(url) as client:
            client.connect(spec)
            client.reset(1)
            client.infer(observation(spec))
            with pytest.raises(PolicyRemoteError, match="continuation|Continuation"):
                client.infer(observation(spec, "r2", continuation))
            assert len(policy.received) == 1
            assert client.ready is None

    def test_old_episode_reference_cannot_survive_reset(self, spec):
        with running_server(EchoPolicy()) as url, PlanPolicyClient(url) as client:
            client.connect(spec)
            client.reset(1)
            client.infer(observation(spec))
            client.reset(2)
            with pytest.raises(PolicyRemoteError, match="Unknown continuation"):
                client.infer(observation(spec, "r2", Continuation("request-1", 1)))

    def test_duplicate_request_rejected_before_model(self, spec):
        policy = EchoPolicy()
        with running_server(policy) as url, PlanPolicyClient(url) as client:
            client.connect(spec)
            client.reset(1)
            client.infer(observation(spec))
            with pytest.raises(PolicyRemoteError, match="Duplicate request_id"):
                client.infer(observation(spec))
            assert len(policy.received) == 1

    def test_another_robot_is_rejected(self, spec):
        with (
            running_server(EchoPolicy()) as url,
            PlanPolicyClient(url) as first,
            PlanPolicyClient(url) as second,
        ):
            first.connect(spec)
            with pytest.raises(PolicyRemoteError, match="already has a robot"):
                second.connect(spec)
            assert second.ready is None

    def test_refusal_before_hello_send_is_reported_and_closed(self, spec):
        from websockets.exceptions import ConnectionClosedOK
        from websockets.frames import Close

        ws = Mock()
        ws.send.side_effect = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
        ws.recv.return_value = encode_error(
            "Policy server already has a robot connected."
        )
        with (
            patch("websockets.sync.client.connect", return_value=ws),
            PlanPolicyClient("ws://unused") as client,
        ):
            with pytest.raises(PolicyRemoteError, match="already has a robot"):
                client.connect(spec)
            assert client.ready is None
            ws.close.assert_called_once()

    @pytest.mark.parametrize("queued_success", [True, False])
    def test_send_failure_never_consumes_queued_success(self, spec, queued_success):
        from websockets.exceptions import ConnectionClosedOK
        from websockets.frames import Close

        closed = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
        ws = Mock()
        ws.send.side_effect = closed
        if queued_success:
            ws.recv.return_value = encode_ready(spec)
        else:
            ws.recv.side_effect = closed
        with (
            patch("websockets.sync.client.connect", return_value=ws),
            PlanPolicyClient("ws://unused") as client,
        ):
            with pytest.raises(ConnectionClosedOK) as error:
                client.connect(spec)
            assert error.value is closed
            assert client.ready is None

    def test_adapter_cannot_return_another_request_id(self, spec):
        class WrongId(EchoPolicy):
            def infer(self, obs):
                return PlanActions("other", super().infer(obs))

        with running_server(WrongId()) as url, PlanPolicyClient(url) as client:
            client.connect(spec)
            client.reset(1)
            with pytest.raises(PolicyRemoteError, match="wrong request_id"):
                client.infer(observation(spec))

    def test_v1_hello_is_not_silently_accepted(self, spec):
        from websockets.sync.client import connect

        with running_server(EchoPolicy()) as url, connect(url) as ws:
            header, _ = decode_message(encode_hello(spec))
            ws.send(encode_message({**header, "version": 1}))
            error, _ = decode_message(ws.recv(timeout=2))
            assert error["type"] == "error"
            assert "version 2" in error["message"]


class TestPredictionCache:
    def test_rejected_candidates_do_not_evict_accepted_prediction(self):
        cache = PredictionCache(capacity=2)
        original = np.arange(12, dtype=np.float32).reshape(4, 3)
        cache.put("accepted", original)
        original[:] = -100
        for number in range(8):
            prefix = cache.remaining(Continuation("accepted", 2))
            np.testing.assert_array_equal(prefix, np.arange(6, 12).reshape(2, 3))
            assert not prefix.flags.writeable
            cache.put(str(number), np.ones((4, 3)))
        cache.clear()
        with pytest.raises(PolicyProtocolError, match="Unknown"):
            cache.remaining(Continuation("accepted", 0))

    def test_no_reference_is_an_explicit_bootstrap(self):
        cache = PredictionCache(capacity=2)
        cache.put("old", np.ones((4, 3)))
        cache.remaining(Continuation("old", 0))
        assert cache.remaining(None) is None
        cache.put("new", np.ones((4, 3)))
        cache.put("newer", np.ones((4, 3)))
        with pytest.raises(PolicyProtocolError):
            cache.remaining(Continuation("old", 0))
