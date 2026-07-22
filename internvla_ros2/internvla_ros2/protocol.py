"""Pure protocol state and validation shared by ROS-facing nodes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


PROTOCOL_VERSION = 1

STATUS_OK = 0
STATUS_NOT_INITIALIZED = 1
STATUS_INVALID_REQUEST = 2
STATUS_STALE = 3
STATUS_TIMEOUT = 4
STATUS_CANCELED = 5
STATUS_RESET_MISMATCH = 6
STATUS_OBSERVATION_MISSING = 7
STATUS_INTERNAL_ERROR = 8
STATUS_SHUTTING_DOWN = 9

ACTION_STAND_STILL = -1
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
VALID_ACTIONS = frozenset(
    {ACTION_STAND_STILL, ACTION_STOP, ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}
)


class ProtocolError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = int(status_code)


@dataclass(frozen=True, order=True)
class RequestIdentity:
    episode_id: str
    reset_generation: int
    sequence_id: int
    request_id: str

    def validate(self) -> None:
        if not self.episode_id or len(self.episode_id) > 256:
            raise ProtocolError(STATUS_INVALID_REQUEST, "episode_id is required and bounded")
        if self.reset_generation < 0 or self.sequence_id < 0:
            raise ProtocolError(STATUS_INVALID_REQUEST, "generation and sequence must be nonnegative")
        if not self.request_id or len(self.request_id) > 256:
            raise ProtocolError(STATUS_INVALID_REQUEST, "request_id is required and bounded")


@dataclass(frozen=True)
class TimeWindow:
    deadline_ns: int
    valid_until_ns: int

    def validate(self, now_ns: int) -> None:
        if self.deadline_ns <= 0 or self.valid_until_ns <= 0:
            raise ProtocolError(STATUS_INVALID_REQUEST, "deadline and valid_until are required")
        if self.valid_until_ns < self.deadline_ns:
            raise ProtocolError(STATUS_INVALID_REQUEST, "valid_until precedes deadline")
        if now_ns > self.deadline_ns:
            raise ProtocolError(STATUS_TIMEOUT, "request deadline expired")

    def response_is_stale(self, now_ns: int) -> bool:
        return now_ns > self.valid_until_ns


@dataclass(frozen=True)
class DualClockWindow:
    """A simulator-time contract with an optional monotonic fail-safe budget.

    Monotonic clock values are intentionally not compared between machines.
    The remote value is an identity/audit field; each receiver derives its own
    monotonic deadline from the remaining simulator-time duration in the
    legacy dual-clock mode.  ``sim_time_only`` callers use ROS simulation time
    for semantic deadline/validity decisions and keep wall time for separate
    liveness watchdogs.
    """

    sim_stamp_ns: int
    client_wall_monotonic_ns: int
    deadline_ns: int
    valid_until_ns: int

    def validate(
        self,
        now_sim_ns: int,
        *,
        sim_time_only: bool = False,
        previous_sim_ns: int | None = None,
    ) -> None:
        if self.sim_stamp_ns <= 0:
            raise ProtocolError(STATUS_INVALID_REQUEST, "sim_stamp is required")
        if self.client_wall_monotonic_ns <= 0:
            raise ProtocolError(
                STATUS_INVALID_REQUEST, "client_wall_monotonic_ns is required"
            )
        if self.deadline_ns < self.sim_stamp_ns:
            raise ProtocolError(STATUS_INVALID_REQUEST, "deadline precedes sim_stamp")
        if sim_time_only:
            self.validate_sim_progress(now_sim_ns, previous_sim_ns)
        TimeWindow(self.deadline_ns, self.valid_until_ns).validate(now_sim_ns)

    def validate_sim_progress(
        self, now_sim_ns: int, previous_sim_ns: int | None = None
    ) -> None:
        now_sim_ns = int(now_sim_ns)
        if now_sim_ns <= 0:
            raise ProtocolError(STATUS_TIMEOUT, "ROS simulation clock is zero")
        comparison_ns = self.sim_stamp_ns
        if previous_sim_ns is not None:
            comparison_ns = max(comparison_ns, int(previous_sim_ns))
        if now_sim_ns < comparison_ns:
            raise ProtocolError(
                STATUS_STALE,
                "ROS simulation clock regressed: "
                f"previous_ns={comparison_ns} observed_ns={now_sim_ns}",
            )

    def local_monotonic_limits(
        self, now_sim_ns: int, now_monotonic_ns: int
    ) -> tuple[int, int]:
        self.validate(now_sim_ns)
        effective_sim_now = max(self.sim_stamp_ns, int(now_sim_ns))
        deadline_remaining = max(0, self.deadline_ns - effective_sim_now)
        validity_remaining = max(0, self.valid_until_ns - effective_sim_now)
        return (
            int(now_monotonic_ns) + deadline_remaining,
            int(now_monotonic_ns) + validity_remaining,
        )

    def deadline_expired(
        self,
        now_sim_ns: int,
        now_monotonic_ns: int,
        local_deadline_ns: int,
        *,
        sim_time_only: bool = False,
    ) -> bool:
        return now_sim_ns > self.deadline_ns or (
            not sim_time_only and now_monotonic_ns > local_deadline_ns
        )

    def response_is_stale(
        self,
        now_sim_ns: int,
        now_monotonic_ns: int,
        local_valid_until_ns: int,
        *,
        sim_time_only: bool = False,
    ) -> bool:
        return (
            now_sim_ns > self.valid_until_ns
            or not sim_time_only
            and now_monotonic_ns > local_valid_until_ns
        )


@dataclass
class GenerationBarrier:
    initialized: bool = False
    episode_id: str = ""
    reset_generation: int = 0
    last_sequence_id: int = -1

    def initialize(self, episode_id: str) -> None:
        if not episode_id:
            raise ProtocolError(STATUS_INVALID_REQUEST, "initial episode_id is required")
        self.initialized = True
        self.episode_id = episode_id
        self.reset_generation = 0
        self.last_sequence_id = -1

    def restore(
        self, episode_id: str, reset_generation: int, last_sequence_id: int
    ) -> None:
        """Restore an externally quiesced generation after a process restart."""

        if self.initialized:
            raise ProtocolError(
                STATUS_INVALID_REQUEST, "cannot restore an initialized barrier"
            )
        if (
            not episode_id
            or len(episode_id) > 256
            or isinstance(reset_generation, bool)
            or not isinstance(reset_generation, int)
            or reset_generation < 0
            or isinstance(last_sequence_id, bool)
            or not isinstance(last_sequence_id, int)
            or last_sequence_id < -1
        ):
            raise ProtocolError(STATUS_INVALID_REQUEST, "invalid restored generation")
        self.initialized = True
        self.episode_id = episode_id
        self.reset_generation = int(reset_generation)
        self.last_sequence_id = int(last_sequence_id)

    def validate_current(self, identity: RequestIdentity) -> None:
        identity.validate()
        if not self.initialized:
            raise ProtocolError(STATUS_NOT_INITIALIZED, "model is not initialized")
        if identity.episode_id != self.episode_id:
            raise ProtocolError(STATUS_STALE, "episode_id does not match current episode")
        if identity.reset_generation != self.reset_generation:
            raise ProtocolError(STATUS_RESET_MISMATCH, "reset_generation does not match barrier")

    def validate_new_sequence(self, identity: RequestIdentity) -> None:
        self.validate_current(identity)
        expected = self.last_sequence_id + 1
        if identity.sequence_id != expected:
            raise ProtocolError(
                STATUS_STALE,
                f"new sequence_id must be {expected}, observed {identity.sequence_id}",
            )

    def commit(self, identity: RequestIdentity) -> None:
        self.validate_new_sequence(identity)
        self.last_sequence_id = identity.sequence_id

    def validate_reset(self, next_episode_id: str, expected_generation: int, barrier_sequence: int) -> None:
        if not self.initialized:
            raise ProtocolError(STATUS_NOT_INITIALIZED, "model is not initialized")
        if expected_generation != self.reset_generation:
            raise ProtocolError(STATUS_RESET_MISMATCH, "reset expected_generation mismatch")
        if barrier_sequence < self.last_sequence_id:
            raise ProtocolError(STATUS_STALE, "reset barrier is behind the committed sequence")
        if not next_episode_id:
            raise ProtocolError(STATUS_INVALID_REQUEST, "next_episode_id is required")

    def reset(self, next_episode_id: str, expected_generation: int, barrier_sequence: int) -> int:
        self.validate_reset(next_episode_id, expected_generation, barrier_sequence)
        self.episode_id = next_episode_id
        self.reset_generation += 1
        self.last_sequence_id = -1
        return self.reset_generation

    def abort_generation(self) -> int:
        """Invalidate every in-flight/current-generation response after model mutation."""
        if not self.initialized:
            raise ProtocolError(STATUS_NOT_INITIALIZED, "model is not initialized")
        self.reset_generation += 1
        self.last_sequence_id = -1
        return self.reset_generation


T = TypeVar("T")


class IdempotencyCache(Generic[T]):
    """Bounded identity cache that rejects same-ID/different-observation reuse."""

    def __init__(self, maximum: int = 512):
        if maximum < 1:
            raise ValueError("maximum must be positive")
        self.maximum = int(maximum)
        self._items: OrderedDict[RequestIdentity, tuple[str, T]] = OrderedDict()

    def get(self, identity: RequestIdentity, observation_digest: str) -> T | None:
        item = self._items.get(identity)
        if item is None:
            return None
        cached_digest, result = item
        if cached_digest != observation_digest:
            raise ProtocolError(
                STATUS_INVALID_REQUEST,
                "duplicate identity reused with a different observation",
            )
        self._items.move_to_end(identity)
        return result

    def put(self, identity: RequestIdentity, observation_digest: str, result: T) -> None:
        existing = self.get(identity, observation_digest)
        if existing is not None:
            return
        self._items[identity] = (observation_digest, result)
        self._items.move_to_end(identity)
        while len(self._items) > self.maximum:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def validate_protocol_version(value: int) -> None:
    if int(value) != PROTOCOL_VERSION:
        raise ProtocolError(
            STATUS_INVALID_REQUEST,
            f"protocol_version must be {PROTOCOL_VERSION}, observed {value}",
        )


def validate_action(value: int) -> int:
    action = int(value)
    if action not in VALID_ACTIONS:
        raise ProtocolError(STATUS_INTERNAL_ERROR, f"model returned invalid action {action}")
    return action
