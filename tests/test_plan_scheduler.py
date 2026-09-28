"""Deterministic traces for the model-independent cached-suffix dispatcher."""

from __future__ import annotations

import unittest

import numpy as np

from almond_axol.policy.plan_scheduler import (
    PlanRuntimeConfig,
    PlanScheduler,
    PlanSchedulingError,
    prepare_plan_images,
    validate_sensor_times,
)


def chunk(base: int = 0) -> np.ndarray:
    return np.arange(base, base + 30, dtype=np.float32).reshape(30, 1)


class SchedulerTest(unittest.TestCase):
    def scheduler(self, **options) -> PlanScheduler:
        return PlanScheduler(
            fps=30, horizon=30, width=1, config=PlanRuntimeConfig(**options)
        )

    def bootstrap(self, scheduler: PlanScheduler) -> str:
        pending = scheduler.begin_request(0)
        self.assertIsNone(pending.prediction_id)
        self.assertTrue(scheduler.adopt(pending.request_id, chunk(), 2_000_000_000))
        return pending.request_id

    def test_bootstrap_holds_cursor_and_preserves_row_zero(self):
        scheduler = self.scheduler()
        pending = scheduler.begin_request(0)
        for _ in range(30):
            self.assertIsNone(scheduler.pop())
        self.assertEqual(scheduler.next_tick, 0)
        self.assertFalse(scheduler.request_due)
        scheduler.adopt(pending.request_id, chunk(), 9_000_000_000)
        tick, row = scheduler.pop()
        self.assertEqual((tick, row.item()), (0, 0))

    def test_slow_bootstrap_adaptive_delay_hint_saturates_at_wire_horizon(self):
        scheduler = self.scheduler(advertise_delay=True)
        pending = scheduler.begin_request(0)
        scheduler.adopt(pending.request_id, chunk(), 9_000_000_000)
        self.assertEqual(scheduler.delay_steps, scheduler.horizon)
        self.assertEqual(scheduler.pop()[1].item(), 0)
        for _ in range(9):
            scheduler.pop()
        pending = scheduler.begin_request(10_000_000_000)
        scheduler.pop()
        scheduler.adopt(pending.request_id, chunk(), 10_050_000_000)
        self.assertEqual(scheduler.delay_steps, 2)

    def test_early_adoption_skips_elapsed_rows_and_keeps_k10_schedule(self):
        scheduler = self.scheduler()
        accepted = self.bootstrap(scheduler)
        for expected in range(10):
            self.assertEqual(scheduler.pop()[1].item(), expected)
        request = scheduler.begin_request(333_333_333)
        self.assertEqual(
            (request.prediction_id, request.from_row, request.origin_tick),
            (accepted, 10, 10),
        )
        with self.assertRaises(PlanSchedulingError):
            scheduler.begin_request(333_333_334)
        for expected in range(10, 14):
            self.assertEqual(scheduler.pop()[1].item(), expected)
        rows = chunk(100)
        scheduler.adopt(request.request_id, rows, 470_000_000)
        rows[:] = -1  # adoption owns its copy
        self.assertEqual(scheduler.pop()[1].item(), 104)
        self.assertFalse(scheduler.request_due)
        for _ in range(5):
            scheduler.pop()
        self.assertTrue(scheduler.request_due)  # tick 20, not adoption tick + k
        following = scheduler.begin_request(666_666_666)
        self.assertEqual(
            (following.prediction_id, following.from_row), (request.request_id, 10)
        )

    def test_late_reply_enters_explicit_unprefixed_blocking_refresh(self):
        scheduler = self.scheduler()
        self.bootstrap(scheduler)
        for _ in range(10):
            scheduler.pop()
        request = scheduler.begin_request(333_333_333)
        for _ in range(7):
            scheduler.pop()
        self.assertFalse(scheduler.adopt(request.request_id, chunk(100), 600_000_000))
        self.assertTrue(scheduler.blocking)
        self.assertIsNone(scheduler.pop())
        refresh = scheduler.begin_request(700_000_000)
        self.assertIsNone(refresh.prediction_id)
        self.assertEqual(refresh.origin_tick, 17)
        scheduler.adopt(refresh.request_id, chunk(200), 2_000_000_000)
        self.assertEqual(
            [scheduler.pop()[1].item() for _ in range(10)], list(range(200, 210))
        )
        self.assertIsNone(scheduler.pop())
        self.assertIsNone(scheduler.begin_request(3_000_000_000).prediction_id)

    def test_inclusive_bound_and_strict_mode(self):
        scheduler = self.scheduler(late_policy="abort")
        self.bootstrap(scheduler)
        for _ in range(10):
            scheduler.pop()
        request = scheduler.begin_request(0)
        for _ in range(6):
            scheduler.pop()
        self.assertTrue(scheduler.adopt(request.request_id, chunk(), 0))
        for _ in range(4):
            scheduler.pop()
        request = scheduler.begin_request(0)
        for _ in range(7):
            scheduler.pop()
        with self.assertRaises(PlanSchedulingError):
            scheduler.adopt(request.request_id, chunk(), 0)
        self.assertIsNone(scheduler.actions)

    def test_reset_and_hold_invalidate_pending_and_never_reuse_request_ids(self):
        scheduler = self.scheduler()
        request = scheduler.begin_request(0)
        scheduler.invalidate()
        self.assertFalse(scheduler.adopt(request.request_id, chunk(), 0))
        scheduler.reset()
        new_request = scheduler.begin_request(1)
        self.assertNotEqual(request.request_id, new_request.request_id)
        self.assertFalse(scheduler.adopt(request.request_id, chunk(), 2))
        self.assertTrue(scheduler.adopt(new_request.request_id, chunk(), 3))
        self.assertFalse(scheduler.adopt(new_request.request_id, chunk(), 4))

    def test_horizon_exhaustion_cannot_adopt_old_pending_work(self):
        scheduler = self.scheduler()
        self.bootstrap(scheduler)
        for _ in range(10):
            scheduler.pop()
        request = scheduler.begin_request(0)
        for _ in range(20):
            scheduler.pop()
        self.assertIsNone(scheduler.pop())
        self.assertFalse(scheduler.adopt(request.request_id, chunk(), 0))
        self.assertTrue(scheduler.blocking)

    def test_reply_cannot_relax_negotiated_adoption_bound(self):
        scheduler = self.scheduler(late_policy="abort")
        self.bootstrap(scheduler)
        for _ in range(10):
            scheduler.pop()
        request = scheduler.begin_request(0)
        for _ in range(7):
            scheduler.pop()
        with self.assertRaises(PlanSchedulingError):
            scheduler.adopt(
                request.request_id, chunk(), 0, max_adoption_offset_steps=29
            )

    def test_nonfinite_or_wrong_width_reply_is_never_installed(self):
        for bad in [np.zeros((30, 2)), np.full((30, 1), np.nan)]:
            scheduler = self.scheduler()
            request = scheduler.begin_request(0)
            with self.assertRaises(PlanSchedulingError):
                scheduler.adopt(request.request_id, bad, 0)
            self.assertIsNone(scheduler.pop())

    def test_generic_preparation_matches_legacy_pillow_rgb_default(self):
        from PIL import Image

        frame = np.random.default_rng(41).integers(
            0, 256, (600, 960, 3), dtype=np.uint8
        )
        config = PlanRuntimeConfig(output_width=480, output_height=288)
        actual = prepare_plan_images({"camera": frame}, config)["camera"]
        # The legacy vendor resize_image uses Image.resize without resample;
        # Pillow selects BICUBIC for RGB. This checks every prepared pixel.
        expected = np.asarray(Image.fromarray(frame).resize((480, 288)))
        np.testing.assert_array_equal(actual, expected)

    def test_actual_sensor_age_and_skew_are_enforced(self):
        config = PlanRuntimeConfig()
        validate_sensor_times(
            950_000_000, {"a": 950_000_000, "b": 940_000_000}, 1_000_000_000, config
        )
        for state, cameras in [
            (800_000_000, {"a": 990_000_000}),
            (990_000_000, {"a": 700_000_000}),
            (990_000_000, {"a": 990_000_000, "b": 900_000_000}),
            (1_000_000_001, {"a": 990_000_000}),
        ]:
            with self.assertRaises(PlanSchedulingError):
                validate_sensor_times(state, cameras, 1_000_000_000, config)


if __name__ == "__main__":
    unittest.main()
