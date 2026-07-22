"""Independent monotonic workload; no evaluator, dataset, model, or navigation state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .contract import PHYSICS_HZ, REQUIRED_STREAMS, SensorProfile
from .core import SensorBatch


@dataclass(frozen=True, slots=True)
class SafeStep:
    sim_stamp_ns: int
    physics_step: int
    applied_step_count: int
    rendered: bool
    render_id: int
    render_generation: int


class ModelFreeBackend(Protocol):
    def reset(self, generation: int) -> None: ...

    def step_safe_stop(self) -> SafeStep: ...

    def capture(self, step: SafeStep) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class CaptureNotReady(RuntimeError):
    """Current-generation render is real but its sensor content is not ready."""


class BatchSink(Protocol):
    def reset(self, generation: int) -> None: ...

    def submit(self, batch: SensorBatch) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkloadSummary:
    generation: int
    active_reset_count: int
    capture_count: int
    safe_stop_step_count: int
    started_monotonic_ns: int
    bounded_end_monotonic_ns: int
    elapsed_sec: float


class BoundedModelFreeWorkload:
    """Drive real stepping/capture from one outer monotonic clock."""

    def __init__(
        self,
        backend: ModelFreeBackend,
        sink: BatchSink,
        profile: SensorProfile,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.backend = backend
        self.sink = sink
        self.profile = profile
        self.monotonic_ns = monotonic_ns
        self.generation = 0
        self.sequence = 0
        self.capture_count = 0
        self.active_reset_count = 0
        self._last_physics_stamp_ns: int | None = None
        self._last_safe_count = 0
        self._last_capture_safe_count = 0
        self._next_capture_ns = 1
        self._pending_reset_reason: str | None = "initial"

    def initialize(self) -> None:
        self.backend.reset(0)
        self.sink.reset(0)

    def _active_reset(self) -> None:
        self.generation += 1
        self.backend.reset(self.generation)
        self.sink.reset(self.generation)
        self.sequence = 0
        # Active reset is a state reset inside one continuously stepping world;
        # retaining this value enforces monotonic sim time across generations.
        self._next_capture_ns = (self._last_physics_stamp_ns or 0) + 1
        self._pending_reset_reason = "active_periodic"
        self.active_reset_count += 1

    def _step_and_maybe_capture(self) -> None:
        step = self.backend.step_safe_stop()
        if step.sim_stamp_ns <= 0:
            raise RuntimeError("Isaac physics step returned non-positive simulation time")
        if self._last_physics_stamp_ns is not None and step.sim_stamp_ns <= self._last_physics_stamp_ns:
            raise RuntimeError("Isaac simulation time duplicated or rolled back without reset")
        if step.applied_step_count != self._last_safe_count + 1:
            raise RuntimeError("safe-stop was not applied exactly once on every physics step")
        if step.physics_step <= 0:
            raise RuntimeError("invalid physics-step identity")
        self._last_physics_stamp_ns = step.sim_stamp_ns
        self._last_safe_count = step.applied_step_count
        if step.sim_stamp_ns < self._next_capture_ns or not step.rendered:
            return

        if step.render_id <= 0 or step.render_generation != self.generation:
            raise RuntimeError("capture attempted without a current-generation render")

        try:
            payloads = dict(self.backend.capture(step))
        except CaptureNotReady:
            # Do not emit or advance sequence/reset evidence. Continue exact
            # safe-stop stepping until a later real render becomes usable.
            return
        if set(payloads) != set(REQUIRED_STREAMS):
            raise RuntimeError("Isaac capture omitted a frozen sensor stream")
        safe_steps_since_capture = step.applied_step_count - self._last_capture_safe_count
        if safe_steps_since_capture <= 0:
            raise RuntimeError("safe-stop coverage did not advance between captures")
        batch = SensorBatch(
            stamp_ns=step.sim_stamp_ns,
            generation=self.generation,
            sequence=self.sequence,
            payloads=payloads,
            stream_stamps_ns={name: step.sim_stamp_ns for name in REQUIRED_STREAMS},
            safe_stop={
                "identity": f"sensor-soak:{self.generation}:{self.sequence}",
                "linear_x": 0.0,
                "angular_z": 0.0,
                "emergency_stop": True,
                "physics_step": step.physics_step,
                "applied_step_count": step.applied_step_count,
                "steps_since_previous_capture": safe_steps_since_capture,
                "render_id": step.render_id,
                "render_generation": step.render_generation,
                "reset_kind": "continuous_world_articulation_state",
            },
            reset_reason=self._pending_reset_reason,
        )
        self.sink.submit(batch)
        self._pending_reset_reason = None
        self._last_capture_safe_count = step.applied_step_count
        self.sequence += 1
        self.capture_count += 1
        capture_period_ns = int(round(1_000_000_000 / self.profile.capture_hz))
        while self._next_capture_ns <= step.sim_stamp_ns:
            self._next_capture_ns += capture_period_ns

    def run_bounded(self, cancelled: Callable[[], bool] = lambda: False) -> WorkloadSummary:
        self.initialize()
        started = self.monotonic_ns()
        deadline = started + int(self.profile.duration_sec * 1_000_000_000)
        next_reset = started + int(self.profile.reset_interval_sec * 1_000_000_000)
        while self.monotonic_ns() < deadline:
            if cancelled():
                raise InterruptedError("bounded workload interrupted by an external signal")
            if self.monotonic_ns() >= next_reset:
                self._active_reset()
                next_reset += int(self.profile.reset_interval_sec * 1_000_000_000)
            self._step_and_maybe_capture()
        ended = self.monotonic_ns()
        return WorkloadSummary(
            generation=self.generation,
            active_reset_count=self.active_reset_count,
            capture_count=self.capture_count,
            safe_stop_step_count=self._last_safe_count,
            started_monotonic_ns=started,
            bounded_end_monotonic_ns=ended,
            elapsed_sec=(ended - started) / 1_000_000_000,
        )

    def hold_until(self, stopped: Callable[[], bool]) -> None:
        """Remain required and healthy while the outer supervisor validates."""

        while not stopped():
            self._step_and_maybe_capture()

    def hold_until_freeze(
        self,
        stopped: Callable[[], bool],
        freeze_requested: Callable[[], bool],
    ) -> None:
        while not stopped() and not freeze_requested():
            self._step_and_maybe_capture()
        if stopped():
            raise InterruptedError("post-duration workload interrupted before snapshot")

    def safe_stop_until(self, stopped: Callable[[], bool]) -> None:
        """Keep applying zero motion after the terminal evidence snapshot."""

        while not stopped():
            step = self.backend.step_safe_stop()
            if step.applied_step_count != self._last_safe_count + 1:
                raise RuntimeError("safe-stop coverage broke during snapshot/validation hold")
            self._last_safe_count = step.applied_step_count


def expected_physics_steps_per_capture() -> int:
    return int(round(PHYSICS_HZ / 20.0))
