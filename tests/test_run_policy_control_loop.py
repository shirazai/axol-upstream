"""Unit tests for the run-policy hot control loop.

A trajectory replayed slower than its configured fps is a correctness bug,
not a performance nit: action chunks are timestep-indexed with no time base
of their own, so a loop that falls behind executes the plan in slow motion.
These tests pin the fixed behavior with a fake robot + real event loop and
no hardware:

- the loop consumes actions at the configured fps (event-loop-native sends),
- a transient stall is absorbed by skipping at most MAX_CATCHUP_TICKS stale
  actions (a small, bounded hop — never slow-motion replay),
- a single long stall, or a steady drip of skips past
  EPISODE_SKIP_FAULT_BUDGET, is a FAULT: the loop aborts the episode through
  fatal_error instead of executing a degraded plan,
- starvation lag (empty queue; server still computing) never skips or
  faults, even when a fresh chunk lands mid-stall,
- a send that never returns faults within the per-send timeout instead of
  parking the loop and its watchdog,
- an instance-level ``robot.send_action`` override (test stubs, dry-run
  wrappers) is still honored — the async path must never bypass it.

Self-contained; no test-framework dependency (also collects under pytest)::

    python tests/test_run_policy_control_loop.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import ModuleType, SimpleNamespace


def _stub_pyzed() -> None:
    """The ZED SDK only exists on the robot host; the client logic under test
    never touches it, but ``almond_axol.cli.run_policy`` imports the camera
    module at load time."""
    if "pyzed" in sys.modules:
        return
    pyzed = ModuleType("pyzed")
    pyzed.sl = ModuleType("pyzed.sl")
    sys.modules["pyzed"] = pyzed
    sys.modules["pyzed.sl"] = pyzed.sl


class _FakeRobot:
    """Just enough of ``AxolRobot`` for the client hot loop: an event loop,
    both send paths, and the feature layout."""

    def __init__(self, joint_keys: list[str]) -> None:
        self.action_features = list(joint_keys)
        self.observation_features = {k: float for k in joint_keys}
        self.async_sent: list[tuple[float, dict]] = []
        self.sync_sent: list[tuple[float, dict]] = []
        self.send_delay_s: float = 0.0
        # One-shot extra delays: send index -> seconds (consumed on use).
        self.stall_at: dict[int, float] = {}
        # Send indices that hang forever (a wedged CAN write).
        self.hang_at: set[int] = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    async def send_action_async(self, action: dict) -> dict:
        if len(self.async_sent) in self.hang_at:
            await asyncio.sleep(3600.0)
        extra = self.stall_at.pop(len(self.async_sent), 0.0)
        if self.send_delay_s or extra:
            await asyncio.sleep(self.send_delay_s + extra)
        self.async_sent.append((time.perf_counter(), action))
        return action

    def send_action(self, action: dict) -> dict:
        self.sync_sent.append((time.perf_counter(), action))
        return action

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)


def _build_client(fps: float, robot: _FakeRobot):
    _stub_pyzed()
    from almond_axol.cli.run_policy import _build_axol_robot_client

    config = SimpleNamespace(
        policy_type="pi05",
        pretrained_name_or_path="stub",
        actions_per_chunk=50,
        policy_device="cpu",
        server_address="localhost:1",  # never dialed: gRPC channels are lazy
        environment_dt=1.0 / fps,
        fps=fps,
        chunk_size_threshold=0.9,
        task="test",
    )
    client = _build_axol_robot_client(
        config=config,
        robot=robot,
        publisher=None,
        aggregate_strategy="latest_only",
        temporal_ensemble_coeff=0.01,
    )
    # The real barrier syncs 3 session threads; the tests drive only this one.
    client.start_barrier = threading.Barrier(1)
    return client


def _fill_queue(client, n: int, dim: int) -> None:
    import torch
    from lerobot.async_inference.helpers import TimedAction

    for i in range(n):
        client.action_queue.put(
            TimedAction(timestamp=i * 0.001, timestep=i, action=torch.zeros(dim))
        )


def _run_episode(
    client, robot: _FakeRobot, drained, timeout_s: float, expect_fault: bool = False
) -> float:
    """Run control_loop on a thread until ``drained()`` or timeout; return wall time."""
    thread = threading.Thread(target=client.control_loop, args=("test",), daemon=True)
    start = time.perf_counter()
    thread.start()
    while not drained() and time.perf_counter() - start < timeout_s:
        time.sleep(0.005)
    elapsed = time.perf_counter() - start
    client.shutdown_event.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "control loop failed to shut down"
    if expect_fault:
        assert client.fatal_error is not None, "expected the loop to fault"
    else:
        assert client.fatal_error is None, f"control loop died: {client.fatal_error!r}"
    return elapsed


_JOINT_KEYS = [f"left_j{i}.pos" for i in range(8)] + [
    f"right_j{i}.pos" for i in range(8)
]


def test_hot_loop_consumes_at_fps() -> None:
    """40 queued actions at 200 fps must take ~0.2 s — paced, not burst."""
    fps, n = 200.0, 40
    robot = _FakeRobot(_JOINT_KEYS)
    try:
        client = _build_client(fps, robot)
        _fill_queue(client, n, len(_JOINT_KEYS))
        elapsed = _run_episode(
            client, robot, drained=lambda: len(robot.async_sent) >= n, timeout_s=5.0
        )
        assert len(robot.async_sent) == n, f"sent {len(robot.async_sent)}/{n}"
        assert not robot.sync_sent, "hot loop must use send_action_async"
        nominal = n / fps
        assert nominal * 0.8 <= elapsed <= nominal * 2.5, (
            f"{n} actions @ {fps} fps took {elapsed:.3f}s (nominal {nominal:.3f}s)"
        )
        assert client._episode_skipped_actions == 0, "healthy loop must not skip"
    finally:
        robot.close()


def test_transient_stall_skips_within_bound_and_continues() -> None:
    """One ~4.5-tick stall: a bounded skip (<= MAX_CATCHUP_TICKS), no fault.

    The deadline schedule absorbs ~2 ticks of a one-shot stall for free
    (wakeups run one tick ahead of the deadline, and the first late send
    re-grids immediately), so a 4.5-tick stall leaves ~2 overdue actions to
    skip — inside the bound. The episode finishes near nominal wall time:
    the transient degrades to a small hop on the planned trajectory, not
    slow motion and not an abort.
    """
    fps, n = 100.0, 40
    robot = _FakeRobot(_JOINT_KEYS)
    robot.stall_at[10] = 4.5 / fps
    try:
        client = _build_client(fps, robot)
        _fill_queue(client, n, len(_JOINT_KEYS))

        def drained() -> bool:
            return len(robot.async_sent) + client._episode_skipped_actions >= n

        elapsed = _run_episode(client, robot, drained, timeout_s=5.0)
        skipped = client._episode_skipped_actions
        assert 1 <= skipped <= client.MAX_CATCHUP_TICKS, f"skipped {skipped}"
        assert len(robot.async_sent) + skipped == n
        nominal = n / fps
        assert elapsed <= nominal * 1.5, (
            f"transient stall stretched the episode to {elapsed:.3f}s "
            f"(nominal {nominal:.3f}s)"
        )
    finally:
        robot.close()


def test_single_long_stall_faults() -> None:
    """A stall past the catch-up bound is a fault, not a big skip.

    After a ~6-tick stall the loop must abort via fatal_error rather than
    hop several actions forward on a physical robot.
    """
    fps, n = 100.0, 40
    robot = _FakeRobot(_JOINT_KEYS)
    robot.stall_at[5] = 6.0 / fps
    try:
        client = _build_client(fps, robot)
        _fill_queue(client, n, len(_JOINT_KEYS))
        _run_episode(
            client,
            robot,
            drained=lambda: client.fatal_error is not None,
            timeout_s=5.0,
            expect_fault=True,
        )
        assert "cannot hold" in str(client.fatal_error)
        assert len(robot.async_sent) < n, "loop must abort, not drain the queue"
    finally:
        robot.close()


def test_sustained_deficit_faults_via_budget() -> None:
    """A send that always costs ~3 ticks must exhaust the skip budget and
    fault — a systematic deficit never silently degrades the whole episode."""
    fps, n = 100.0, 200
    robot = _FakeRobot(_JOINT_KEYS)
    robot.send_delay_s = 3.0 / fps
    try:
        client = _build_client(fps, robot)
        _fill_queue(client, n, len(_JOINT_KEYS))
        _run_episode(
            client,
            robot,
            drained=lambda: client.fatal_error is not None,
            timeout_s=10.0,
            expect_fault=True,
        )
        assert "budget" in str(client.fatal_error)
        assert client._episode_skipped_actions > client.EPISODE_SKIP_FAULT_BUDGET
        assert len(robot.async_sent) < n // 2, (
            "loop must fault early, not grind through the episode"
        )
    finally:
        robot.close()


def test_exact_bound_skip_is_allowed_not_faulted() -> None:
    """n_stale == MAX_CATCHUP_TICKS is the largest PERMITTED skip.

    A ~5.5-tick stall leaves exactly 3 overdue actions after the schedule's
    ~2-tick slack: the loop must skip 3 and continue — faulting here would
    be an off-by-one on the bound.
    """
    fps, n = 50.0, 40
    robot = _FakeRobot(_JOINT_KEYS)
    robot.stall_at[10] = 5.5 / fps
    try:
        client = _build_client(fps, robot)
        _fill_queue(client, n, len(_JOINT_KEYS))

        def drained() -> bool:
            return len(robot.async_sent) + client._episode_skipped_actions >= n

        _run_episode(client, robot, drained, timeout_s=5.0)
        assert client._episode_skipped_actions == client.MAX_CATCHUP_TICKS, (
            f"expected exactly {client.MAX_CATCHUP_TICKS} skips, got "
            f"{client._episode_skipped_actions}"
        )
        assert client.fatal_error is None
    finally:
        robot.close()


def test_starvation_stall_never_skips_or_faults() -> None:
    """Lag accrued on an EMPTY queue is starvation, not lateness.

    The loop idles waiting for the first chunk; the robot's event loop then
    stalls for ~8 ticks and the chunk lands MID-stall (the receiver-thread
    race). On wake the queue is non-empty and the wall clock is far behind —
    but nothing was ever due, so the loop must re-anchor and execute the
    whole chunk: no skips, no fault.
    """
    fps, n = 50.0, 20
    robot = _FakeRobot(_JOINT_KEYS)
    try:
        client = _build_client(fps, robot)  # queue deliberately empty
        thread = threading.Thread(
            target=client.control_loop, args=("test",), daemon=True
        )
        thread.start()
        time.sleep(5.0 / fps)  # let the loop idle on schedule
        robot.event_loop.call_soon_threadsafe(time.sleep, 8.0 / fps)
        time.sleep(2.0 / fps)  # mid-stall: the chunk arrives
        _fill_queue(client, n, len(_JOINT_KEYS))
        start = time.perf_counter()
        while len(robot.async_sent) < n and time.perf_counter() - start < 5.0:
            time.sleep(0.005)
        client.shutdown_event.set()
        thread.join(timeout=5.0)
        assert client.fatal_error is None, f"starvation faulted: {client.fatal_error!r}"
        assert client._episode_skipped_actions == 0, (
            f"starvation skipped {client._episode_skipped_actions} actions"
        )
        assert len(robot.async_sent) == n
    finally:
        robot.close()


def test_wedged_send_faults_within_timeout() -> None:
    """A send that never returns must fault via the per-send timeout —
    never park the loop (and with it the rate watchdog) indefinitely."""
    fps, n = 100.0, 10
    robot = _FakeRobot(_JOINT_KEYS)
    robot.hang_at.add(3)
    try:
        client = _build_client(fps, robot)
        _fill_queue(client, n, len(_JOINT_KEYS))
        elapsed = _run_episode(
            client,
            robot,
            drained=lambda: client.fatal_error is not None,
            timeout_s=10.0,
            expect_fault=True,
        )
        assert "wedged" in str(client.fatal_error)
        assert elapsed < client._send_timeout_s + 2.0, (
            f"fault took {elapsed:.1f}s — timeout not enforced"
        )
    finally:
        robot.close()


def test_instance_send_action_override_is_honored() -> None:
    """An instance-level ``robot.send_action`` override (test stub, dry-run
    wrapper) must be routed through by the async hot path instead of
    actuating via ``send_action_async``."""
    fps, n = 200.0, 10
    robot = _FakeRobot(_JOINT_KEYS)
    try:
        client = _build_client(fps, robot)
        stub_calls: list[dict] = []
        robot.send_action = lambda action: (stub_calls.append(action), action)[1]
        _fill_queue(client, n, len(_JOINT_KEYS))
        _run_episode(client, robot, drained=lambda: len(stub_calls) >= n, timeout_s=5.0)
        assert len(stub_calls) == n, f"stub saw {len(stub_calls)}/{n} actions"
        assert not robot.async_sent, "instance override bypassed — robot would actuate"
    finally:
        robot.close()


def main() -> int:
    tests = [
        (name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")
    ]
    failures = 0
    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc!r}")
        else:
            print(f"ok   {name} ({time.perf_counter() - start:.2f}s)")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
