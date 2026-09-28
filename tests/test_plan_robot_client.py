"""Plan client integration with mocked hardware and real loopback WebSockets."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.constants import Joint
from almond_axol.policy.plan_protocol import PlanActions, PlanObservation
from almond_axol.policy.plan_scheduler import PlanRuntimeConfig, PlanSchedulingError
from almond_axol.policy.plan_server import PlanPolicy, PlanPolicyServer

STATE = tuple(
    f"{side}_{joint.value}.pos" for side in ("left", "right") for joint in Joint
)
ACTIONS = tuple(
    name
    for side in ("left", "right")
    for name in (
        *(f"{side}_ee.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")),
        f"{side}_gripper.pos",
    )
)


class _Policy(PlanPolicy):
    def __init__(self) -> None:
        self.observations: list[PlanObservation] = []
        self.resets = 0

    def setup(self, spec) -> None:
        self.spec = spec

    def reset(self) -> None:
        self.resets += 1

    def infer(self, obs):
        self.observations.append(obs)
        result = np.zeros((30, 14), dtype=np.float32)
        result[:, 0] = np.arange(30) * 0.001
        return result


class PlanRobotClientTest(unittest.TestCase):
    def make_client(self, url: str, **options):
        from almond_axol.cli.run_policy import _build_axol_robot_client
        from almond_axol.lerobot.inference_patch import (
            import_robot_client_preserving_logging,
        )

        import_robot_client_preserving_logging()
        self.sent: list[tuple[float, dict]] = []
        observations = dict.fromkeys(STATE, float)
        observations["overhead"] = (6, 10, 3)

        def capture():
            now = time.perf_counter_ns()
            raw = dict.fromkeys(STATE, 0.0)
            raw["overhead"] = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
            return raw, now - 1_000_000, {"overhead": now - 2_000_000}

        def send(action):
            self.sent.append((time.perf_counter(), action))
            return action

        robot = SimpleNamespace(
            action_features=dict.fromkeys(ACTIONS, float),
            observation_features=observations,
            config=SimpleNamespace(observe_cartesian=False, action_space="cartesian"),
            cartesian_actions=True,
            positions=(np.zeros(8), np.zeros(8)),
            _joints_to_cartesian=lambda *args: dict.fromkeys(ACTIONS, 0.0),
            get_observation_with_sensor_timestamps=capture,
            connect=mock.Mock(
                side_effect=AssertionError("hardware connect is forbidden")
            ),
            send_action=send,
            torque_residuals=lambda: (np.zeros(7), np.zeros(7)),
        )
        config = SimpleNamespace(
            fps=30,
            environment_dt=1 / 30,
            server_address="unused",
            policy_type="custom",
            pretrained_name_or_path="custom",
            actions_per_chunk=30,
            policy_device="cpu",
            client_device="cpu",
            task="local recording label",
            aggregate_fn=None,
        )
        return _build_axol_robot_client(
            config=config,
            robot=robot,
            publisher=None,
            custom_policy_url=url,
            custom_protocol=2,
            plan_config=PlanRuntimeConfig(**options),
        )

    def test_real_transport_drives_independent_layout_without_task_or_blending(self):
        policy = _Policy()
        server = PlanPolicyServer(policy, host="127.0.0.1", port=0)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        client = self.make_client(
            f"ws://127.0.0.1:{server.port}", output_width=5, output_height=3
        )
        workers = []
        try:
            self.assertTrue(client.start())
            client.reset_episode_state()
            for target, args in [
                (client.receive_actions, ()),
                (client.control_loop, ("ignored",)),
                (client.observation_loop, ("ignored",)),
            ]:
                worker = threading.Thread(target=target, args=args, daemon=True)
                workers.append(worker)
                worker.start()
            deadline = time.perf_counter() + 5
            while (
                len(self.sent) < 22
                and client.fatal_error is None
                and time.perf_counter() < deadline
            ):
                time.sleep(0.01)
            self.assertIsNone(client.fatal_error)
            self.assertGreaterEqual(len(self.sent), 22)
            self.assertEqual(policy.spec.state_names, STATE)
            self.assertEqual(policy.spec.action_names, ACTIONS)
            self.assertEqual(policy.spec.cameras[0].shape, (3, 5, 3))
            self.assertEqual(policy.resets, 1)
            self.assertIsNone(policy.observations[0].continuation)
            continuation = policy.observations[1].continuation
            self.assertEqual(
                continuation.prediction_id, policy.observations[0].request_id
            )
            self.assertGreaterEqual(continuation.from_row, 10)
            self.assertIsNone(policy.observations[1].delay_steps)
            self.assertFalse(hasattr(policy.observations[1], "task"))
            # The first targets pass unchanged; inherited temporal ensemble and
            # arrival alignment would alter this trace.
            np.testing.assert_allclose(
                [a["left_ee.x"] for _, a in self.sent[:10]],
                np.arange(10) * 0.001,
                atol=1e-8,
            )
            intervals = np.diff([stamp for stamp, _ in self.sent[:10]])
            self.assertTrue(np.all(intervals > 0.010), intervals)  # no catch-up burst
            client.robot.connect.assert_not_called()
        finally:
            client.stop()
            for worker in workers:
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
            server.shutdown()
            serving.join(timeout=3)

    def test_cartesian_jump_is_checked_against_dispatched_predecessor(self):
        client = self.make_client("ws://unused")
        client._plan_last_target = np.zeros(14)
        row = np.zeros(14)
        row[0] = 0.051
        with self.assertRaises(PlanSchedulingError):
            client._check_plan_step(row)
        self.assertEqual(self.sent, [])

    def test_stop_invalidates_pending_without_waiting_for_network(self):
        client = self.make_client("ws://unused")
        request = client._scheduler.begin_request(time.perf_counter_ns())
        checked = []

        def close():
            checked.append(client.shutdown_event.is_set())
            checked.append(client._scheduler.pending is None)

        client._policy_client.close = close
        client.stop()
        self.assertEqual(checked, [True, True])
        self.assertFalse(
            client._scheduler.adopt(request.request_id, np.zeros((30, 14)), 0)
        )

    def test_missed_dispatch_slot_recovers_instead_of_silently_retiming_plan(self):
        client = self.make_client("ws://unused", late_policy="abort")
        client._action_schema_confirmed = True
        client.start_barrier = threading.Barrier(1)
        request = client._scheduler.begin_request(time.perf_counter_ns())
        client._scheduler.adopt(
            request.request_id, np.zeros((30, 14)), time.perf_counter_ns()
        )
        original_send = client.robot.send_action

        def delayed_send(action):
            time.sleep(0.045)  # misses the next 30 Hz slot, less than two ticks
            return original_send(action)

        client.robot.send_action = delayed_send
        client.control_loop("ignored")
        self.assertEqual(len(self.sent), 1)
        self.assertIsInstance(client.fatal_error, PlanSchedulingError)
        self.assertIn("dispatch exceeded", str(client.fatal_error))
        self.assertIsNone(client._scheduler.actions)

    def queue_request(self, client, *, state_age_s=0):
        now = time.perf_counter_ns()
        with client._plan_ready:
            pending = client._scheduler.begin_request(now)
            client._plan_slot = PlanObservation(
                request_id=pending.request_id,
                state=np.zeros(16, dtype=np.float32),
                images={"overhead": np.zeros((6, 10, 3), dtype=np.uint8)},
                state_sample_time_ns=now - round(state_age_s * 1e9),
                image_capture_time_ns={"overhead": now},
            )
            client._plan_ready.notify_all()
        return pending

    def test_receiver_failure_invalidates_targets_and_stops_dispatch(self):
        for error in (TimeoutError("transport timed out"), ValueError("bad reply")):
            with self.subTest(error=error):
                client = self.make_client("ws://unused")
                client.start_barrier = threading.Barrier(1)
                client._wire_generation = client._scheduler.generation
                pending = self.queue_request(client)
                client._policy_client.infer = mock.Mock(side_effect=error)
                client.receive_actions()
                self.assertIs(client.fatal_error, error)
                self.assertTrue(client.shutdown_event.is_set())
                self.assertIsNone(client._scheduler.actions)
                self.assertIsNone(client._scheduler.pending)
                self.assertFalse(client._network_busy)
                self.assertFalse(
                    client._scheduler.adopt(
                        pending.request_id, np.zeros((30, 14)), time.perf_counter_ns()
                    )
                )
                self.assertEqual(self.sent, [])

    def test_stale_unsent_observation_is_discarded_before_transport(self):
        client = self.make_client("ws://unused")
        client.start_barrier = threading.Barrier(1)
        client._wire_generation = client._scheduler.generation
        self.queue_request(client, state_age_s=1)
        client._policy_client.infer = mock.Mock(
            side_effect=AssertionError("stale observation reached transport")
        )
        worker = threading.Thread(target=client.receive_actions, daemon=True)
        worker.start()
        try:
            with client._plan_ready:
                self.assertTrue(
                    client._plan_ready.wait_for(
                        lambda: client._scheduler.pending is None, timeout=2
                    )
                )
            self.assertIsNone(client.fatal_error)
            client._policy_client.infer.assert_not_called()
            self.assertTrue(client._scheduler.request_due)
        finally:
            client.stop()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def test_recovery_drains_old_reply_before_reset_and_fresh_inference(self):
        client = self.make_client("ws://unused")
        client.start_barrier = threading.Barrier(1)
        client._wire_generation = client._scheduler.generation
        old_request = self.queue_request(client)
        entered = threading.Event()
        release = threading.Event()
        events = []

        def infer(observation):
            events.append(("infer", observation.request_id))
            if observation.request_id == old_request.request_id:
                entered.set()
                if not release.wait(timeout=2):
                    raise AssertionError("test did not release old inference")
            events.append(("reply", observation.request_id))
            return PlanActions(observation.request_id, np.zeros((30, 14)))

        client._policy_client.infer = infer
        client._policy_client.reset = lambda episode: events.append(("reset", episode))
        worker = threading.Thread(target=client.receive_actions, daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            with client._plan_ready:
                client._scheduler.recover("test hold during inference")
                self.assertTrue(client._network_busy)
            self.assertFalse(any(kind == "reset" for kind, _ in events))
            release.set()
            with client._plan_ready:
                self.assertTrue(
                    client._plan_ready.wait_for(
                        lambda: not client._network_busy, timeout=2
                    )
                )
                self.assertIsNone(client._scheduler.actions)
            fresh = self.queue_request(client)
            self.assertTrue(fresh.bootstrap)
            self.assertIsNone(fresh.prediction_id)
            with client._plan_ready:
                self.assertTrue(
                    client._plan_ready.wait_for(
                        lambda: client._scheduler.prediction_id == fresh.request_id,
                        timeout=2,
                    )
                )
            self.assertEqual(
                events,
                [
                    ("infer", old_request.request_id),
                    ("reply", old_request.request_id),
                    ("reset", 1),
                    ("infer", fresh.request_id),
                    ("reply", fresh.request_id),
                ],
            )
            self.assertIsNone(client.fatal_error)
            self.assertEqual(self.sent, [])
        finally:
            release.set()
            client.stop()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())


class RobotLayoutAndTimestampTest(unittest.TestCase):
    def robot(self, **kwargs):
        from almond_axol.lerobot.robot.config_axol import AxolRobotConfig
        from almond_axol.lerobot.robot.robot_axol import AxolRobot

        return AxolRobot(AxolRobotConfig(cameras={}, **kwargs))

    def test_independent_observation_and_action_layout_matrix(self):
        for observation_cartesian in (False, True):
            for action_space in (None, "joint", "cartesian"):
                robot = self.robot(
                    observe_cartesian=observation_cartesian, action_space=action_space
                )
                expected_action_cartesian = (
                    observation_cartesian
                    if action_space is None
                    else action_space == "cartesian"
                )
                self.assertEqual(
                    tuple(robot.observation_features),
                    ACTIONS if observation_cartesian else STATE,
                )
                self.assertEqual(
                    tuple(robot.action_features),
                    ACTIONS if expected_action_cartesian else STATE,
                )

    def test_recorded_actions_follow_action_space_not_observation_space(self):
        robot = self.robot(observe_cartesian=False, action_space="cartesian")
        values = dict.fromkeys(ACTIONS, 0.5)
        robot._joints_to_cartesian = mock.Mock(return_value=values)
        self.assertEqual(robot.action_to_dataset(dict.fromkeys(STATE, 0.0)), values)
        robot = self.robot(observe_cartesian=True, action_space="joint")
        joints = dict.fromkeys(STATE, 0.0)
        self.assertIs(robot.action_to_dataset(joints), joints)

    def test_sensor_timestamps_survive_observation_cache_reserve(self):
        robot = self.robot()
        now = time.perf_counter()
        state_ts = now - 0.010
        camera_times = {"a": now - 0.012, "b": now - 0.008}
        for name, stamp in camera_times.items():
            frame = np.full((2, 3, 3), 7, dtype=np.uint8)
            robot.cameras[name] = SimpleNamespace(
                fps=60,
                latest_capture_ts=lambda s=stamp: (s, now),
                read_latest_with_ts=lambda s=stamp, f=frame: (f, s, now),
            )
        robot._axol = SimpleNamespace(
            state_nearest=mock.Mock(
                return_value=(
                    np.zeros(8),
                    np.zeros(8),
                    np.zeros(8),
                    np.zeros(8),
                    state_ts,
                )
            )
        )
        first, first_state, first_cameras = (
            robot.get_observation_with_sensor_timestamps()
        )
        second, second_state, second_cameras = (
            robot.get_observation_with_sensor_timestamps()
        )
        self.assertEqual(first_state, round(state_ts * 1e9))
        self.assertEqual(
            first_cameras,
            {name: round(stamp * 1e9) for name, stamp in camera_times.items()},
        )
        self.assertEqual((second_state, second_cameras), (first_state, first_cameras))
        self.assertEqual(first.keys(), second.keys())
        robot._axol.state_nearest.assert_called_once()


if __name__ == "__main__":
    unittest.main()
