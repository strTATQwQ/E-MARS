import unittest

from internvla_ros2.protocol import (
    DualClockWindow,
    GenerationBarrier,
    IdempotencyCache,
    ProtocolError,
    RequestIdentity,
    STATUS_INVALID_REQUEST,
    STATUS_RESET_MISMATCH,
    STATUS_STALE,
    STATUS_TIMEOUT,
    TimeWindow,
)
from internvla_ros2.observation_guard import (
    ObservationStampGate,
    describe_request_identity,
)


class ProtocolTests(unittest.TestCase):
    def test_generation_sequence_and_reset_barrier(self):
        barrier = GenerationBarrier()
        barrier.initialize("episode-a")
        first = RequestIdentity("episode-a", 0, 0, "request-0")
        barrier.commit(first)
        with self.assertRaises(ProtocolError) as stale:
            barrier.commit(RequestIdentity("episode-a", 0, 2, "request-2"))
        self.assertEqual(stale.exception.status_code, STATUS_STALE)
        with self.assertRaises(ProtocolError) as mismatch:
            barrier.reset("episode-b", 1, 0)
        self.assertEqual(mismatch.exception.status_code, STATUS_RESET_MISMATCH)
        self.assertEqual(barrier.reset("episode-b", 0, 0), 1)
        barrier.commit(RequestIdentity("episode-b", 1, 0, "request-b0"))

    def test_idempotency_replays_identical_and_rejects_collision(self):
        cache = IdempotencyCache[str](maximum=2)
        identity = RequestIdentity("episode-a", 0, 0, "request-0")
        cache.put(identity, "digest-a", "result")
        self.assertEqual(cache.get(identity, "digest-a"), "result")
        with self.assertRaises(ProtocolError) as collision:
            cache.get(identity, "digest-b")
        self.assertEqual(collision.exception.status_code, STATUS_INVALID_REQUEST)

    def test_deadline_and_validity(self):
        window = TimeWindow(deadline_ns=20, valid_until_ns=30)
        window.validate(10)
        self.assertFalse(window.response_is_stale(30))
        self.assertTrue(window.response_is_stale(31))

    def test_dual_clock_uses_duration_not_cross_host_monotonic_epoch(self):
        window = DualClockWindow(
            sim_stamp_ns=100,
            client_wall_monotonic_ns=999_999_999_999,
            deadline_ns=200,
            valid_until_ns=250,
        )
        local_deadline, local_validity = window.local_monotonic_limits(120, 10_000)
        self.assertEqual(local_deadline, 10_080)
        self.assertEqual(local_validity, 10_130)
        self.assertFalse(window.deadline_expired(150, 10_050, local_deadline))
        self.assertTrue(window.deadline_expired(150, 10_081, local_deadline))
        self.assertTrue(window.response_is_stale(251, 10_100, local_validity))

    def test_sim_time_only_ignores_wall_for_semantic_expiry(self):
        window = DualClockWindow(
            sim_stamp_ns=100,
            client_wall_monotonic_ns=1,
            deadline_ns=200,
            valid_until_ns=250,
        )

        window.validate(150, sim_time_only=True, previous_sim_ns=140)
        self.assertFalse(
            window.deadline_expired(
                150, 10_000, 1, sim_time_only=True
            )
        )
        self.assertFalse(
            window.response_is_stale(
                150, 10_000, 1, sim_time_only=True
            )
        )
        self.assertTrue(
            window.deadline_expired(
                201, 0, 10_000, sim_time_only=True
            )
        )
        self.assertTrue(
            window.response_is_stale(
                251, 0, 10_000, sim_time_only=True
            )
        )

    def test_sim_time_only_fails_closed_on_zero_or_regression(self):
        window = DualClockWindow(
            sim_stamp_ns=100,
            client_wall_monotonic_ns=1,
            deadline_ns=200,
            valid_until_ns=250,
        )

        with self.assertRaises(ProtocolError) as zero:
            window.validate(0, sim_time_only=True)
        self.assertEqual(zero.exception.status_code, STATUS_TIMEOUT)
        with self.assertRaises(ProtocolError) as regressed:
            window.validate(149, sim_time_only=True, previous_sim_ns=150)
        self.assertEqual(regressed.exception.status_code, STATUS_STALE)

    def test_observation_stamp_gate_requires_advance_for_new_identity(self):
        first = RequestIdentity("b::121", 4, 0, "b::121:4:0")
        second = RequestIdentity("b::121", 4, 1, "b::121:4:1")
        gate = ObservationStampGate()

        self.assertTrue(gate.accepts(first, 100))
        gate.commit(first, 100)
        self.assertFalse(gate.accepts(second, 99))
        self.assertFalse(gate.accepts(second, 100))
        self.assertTrue(gate.accepts(second, 101))
        gate.commit(second, 101)
        self.assertEqual(gate.last_identity, second)
        self.assertEqual(gate.last_stamp_ns, 101)

    def test_observation_stamp_gate_allows_idempotent_retry_without_regression(self):
        identity = RequestIdentity("episode-a", 0, 0, "request-0")
        gate = ObservationStampGate(identity, 100)

        self.assertFalse(gate.accepts(identity, 99))
        self.assertTrue(gate.accepts(identity, 100))
        self.assertTrue(gate.accepts(identity, 101))
        with self.assertRaises(ValueError):
            gate.commit(RequestIdentity("episode-a", 0, 1, "request-1"), 100)

    def test_identity_diagnostic_includes_expected_and_observed_fields(self):
        description = describe_request_identity(
            RequestIdentity("b::121", 4, 1, "b::121:4:1")
        )
        self.assertIn("episode_id='b::121'", description)
        self.assertIn("reset_generation=4", description)
        self.assertIn("sequence_id=1", description)
        self.assertIn("request_id='b::121:4:1'", description)


if __name__ == "__main__":
    unittest.main()
