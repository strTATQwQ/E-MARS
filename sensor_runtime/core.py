"""Dependency-free latest-only buffering and model-free stepping primitives."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .contract import REQUIRED_STREAMS


class SensorFault(RuntimeError):
    """A fail-closed violation of the sensor generation/time contract."""


def assert_unpaired_generation_event_fresh(
    events: Any,
    now_monotonic_ns: int,
    timeout_sec: float = 0.35,
) -> None:
    """Fail when the oldest unstamped generation event lacks an identity."""

    if events and (int(now_monotonic_ns) - int(events[0][1])) / 1e9 >= timeout_sec:
        raise SensorFault("generation_topic_event_unpaired_timeout")


@dataclass(frozen=True, slots=True)
class SensorBatch:
    stamp_ns: int
    generation: int
    sequence: int
    payloads: Mapping[str, Any]
    stream_stamps_ns: Mapping[str, int]
    safe_stop: Mapping[str, Any]
    reset_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stamp_ns, bool) or not isinstance(self.stamp_ns, int) or self.stamp_ns <= 0:
            raise ValueError("stamp_ns must be a positive integer")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        payloads = dict(self.payloads)
        if set(payloads) != set(REQUIRED_STREAMS):
            raise ValueError("batch must contain exactly the frozen required streams")
        stamps = {str(name): int(value) for name, value in self.stream_stamps_ns.items()}
        if set(stamps) != set(REQUIRED_STREAMS) or any(value != self.stamp_ns for value in stamps.values()):
            raise ValueError("D435i/LiDAR/pose/TF must share one sim timestamp")
        stop = dict(self.safe_stop)
        expected_identity = f"sensor-soak:{self.generation}:{self.sequence}"
        if (
            stop.get("identity") != expected_identity
            or float(stop.get("linear_x", 1.0)) != 0.0
            or float(stop.get("angular_z", 1.0)) != 0.0
            or stop.get("emergency_stop") is not True
            or int(stop.get("applied_step_count", 0)) <= 0
        ):
            raise ValueError("batch lacks identity-bound continuous safe-stop evidence")
        render_id = int(stop.get("render_id", 0))
        if render_id <= 0 or stop.get("render_generation") != self.generation:
            raise ValueError("batch lacks a current-generation render identity")
        for stream in ("d435i_rgb", "d435i_depth"):
            payload = payloads[stream]
            if (
                not isinstance(payload, Mapping)
                or int(payload.get("render_id", 0)) != render_id
                or int(payload.get("render_generation", -1)) != self.generation
            ):
                raise ValueError("camera payload was not captured from the current render")
        if self.sequence == 0 and self.reset_reason not in {"initial", "active_periodic"}:
            raise ValueError("first generation batch requires reset evidence")
        if self.sequence == 0 and stop.get("reset_kind") != "continuous_world_articulation_state":
            raise ValueError("generation barrier lacks the frozen continuous-world reset kind")
        if self.sequence > 0 and self.reset_reason is not None:
            raise ValueError("reset evidence is only valid on sequence zero")
        object.__setattr__(self, "payloads", MappingProxyType(payloads))
        object.__setattr__(self, "stream_stamps_ns", MappingProxyType(stamps))
        object.__setattr__(self, "safe_stop", MappingProxyType(stop))


class LatestOnlySlot:
    """Capacity-one handoff with an explicit, fail-closed reset barrier."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._batch: SensorBatch | None = None
        self._generation = -1
        self._last_sequence: int | None = None
        self._last_stamp_ns: int | None = None
        self._closed = False
        self._fault: str | None = None
        self.accepted_count = 0
        self.overwrite_count = 0
        self.reset_clear_count = 0
        self.barrier_drop_count = 0

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @property
    def fault(self) -> str | None:
        with self._condition:
            return self._fault

    def _trip(self, reason: str) -> None:
        self._batch = None
        self._fault = reason
        self._condition.notify_all()

    def reset(self, generation: int) -> None:
        with self._condition:
            if self._closed:
                raise SensorFault("slot_closed")
            if generation < 0 or generation <= self._generation:
                self._trip("generation_not_strictly_increasing")
                raise SensorFault(self._fault)
            if self._batch is not None:
                self.reset_clear_count += 1
            self._batch = None
            self._generation = generation
            self._last_sequence = None
            self._last_stamp_ns = None
            self._fault = None
            self._condition.notify_all()

    def offer(self, batch: SensorBatch) -> str:
        with self._condition:
            if self._closed:
                raise SensorFault("slot_closed")
            if self._fault:
                raise SensorFault(self._fault)
            if batch.generation != self._generation:
                self._trip("generation_contamination")
                raise SensorFault(self._fault)
            if self._last_sequence is None:
                if batch.sequence != 0:
                    self._trip("generation_did_not_begin_at_sequence_zero")
                    raise SensorFault(self._fault)
            elif batch.sequence <= self._last_sequence:
                self._trip("duplicate_or_replayed_sequence")
                raise SensorFault(self._fault)
            if self._last_stamp_ns is not None and batch.stamp_ns <= self._last_stamp_ns:
                self._trip("duplicate_or_rollback_sim_time")
                raise SensorFault(self._fault)
            outcome = "stored"
            if (
                self._batch is not None
                and self._batch.generation == batch.generation
                and self._batch.sequence == 0
                and batch.sequence > 0
            ):
                # A generation barrier is never replaceable.  We still commit
                # replay/time state so the producer remains non-blocking and a
                # later sequence cannot replay a barrier-dropped identity.
                self.barrier_drop_count += 1
                outcome = "barrier_dropped"
            elif self._batch is not None:
                self.overwrite_count += 1
                self._batch = batch
                outcome = "replaced"
            else:
                self._batch = batch
            self._last_sequence = batch.sequence
            self._last_stamp_ns = batch.stamp_ns
            self.accepted_count += 1
            self._condition.notify_all()
            return outcome

    def take(self, timeout_sec: float = 0.0) -> SensorBatch | None:
        with self._condition:
            if self._fault:
                raise SensorFault(self._fault)
            if self._batch is None and not self._closed and timeout_sec > 0:
                self._condition.wait(timeout_sec)
            if self._fault:
                raise SensorFault(self._fault)
            batch = self._batch
            self._batch = None
            return batch

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._batch = None
            self._condition.notify_all()
