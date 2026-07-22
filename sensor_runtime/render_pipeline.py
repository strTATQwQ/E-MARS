"""Dependency-free one-render timestamp and generation binding."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .contract import PHYSICS_HZ


@dataclass(frozen=True, slots=True)
class RenderSnapshot:
    """Physical state frozen at the simulation time requested for one render."""

    generation: int
    stamp_ns: int
    world_time: float
    state: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RenderClockCalibration:
    """Frozen integer-step mapping to camera ReferenceTime."""

    origin_time: float
    reference_lag_sec: float
    physics_step_ns: int
    warm_stamps_ns: tuple[int, ...]
    seed_stamp_ns: int

    def reference_time(self, physical_time: float) -> float:
        if not math.isfinite(physical_time):
            raise RuntimeError("physical time is not finite")
        return physical_time - self.reference_lag_sec

    def stamp_ns(self, reference_time: float) -> int:
        if not math.isfinite(reference_time):
            raise RuntimeError("camera reference time is not finite")
        stamp_ns = int(round((reference_time - self.origin_time) * 1_000_000_000))
        if stamp_ns <= 0:
            raise RuntimeError("camera reference timestamp is not positive")
        return stamp_ns

    def stamp_for_physical_step(
        self,
        physical_step: int,
        seed_physical_step: int,
    ) -> int:
        """Advance the calibrated seed by exact integer physics intervals."""

        if (
            type(physical_step) is not int
            or type(seed_physical_step) is not int
            or seed_physical_step <= 0
            or physical_step < seed_physical_step
            or self.physics_step_ns <= 0
        ):
            raise RuntimeError("integer physics clock step is invalid")
        stamp_ns = self.seed_stamp_ns + (
            physical_step - seed_physical_step
        ) * self.physics_step_ns
        if stamp_ns <= 0:
            raise RuntimeError("integer physics clock timestamp is not positive")
        return stamp_ns

    def reference_time_for_physical_step(
        self,
        physical_step: int,
        seed_physical_step: int,
    ) -> float:
        stamp_ns = self.stamp_for_physical_step(
            physical_step,
            seed_physical_step,
        )
        reference_time = self.origin_time + stamp_ns / 1_000_000_000
        if not math.isfinite(reference_time):
            raise RuntimeError("integer physics clock reference time is not finite")
        return reference_time


def bounded_zero_frame_resync(
    first_metadata: dict[str, tuple[int, int, float]],
    *,
    render_once: Callable[[], dict[str, tuple[int, int, float]]],
    target_time: float,
    limit: int,
) -> tuple[dict[str, tuple[int, int, float]] | None, int]:
    """Return current zero-frame metadata or a bounded drop decision."""

    if type(limit) is not int or not 1 <= limit <= 4 or not math.isfinite(target_time):
        raise ValueError("render resync inputs are outside the frozen bound")
    metadata = first_metadata
    for drain_count in range(1, limit + 1):
        color = metadata.get("color")
        if (
            not isinstance(color, tuple)
            or len(color) != 3
            or not math.isfinite(float(color[2]))
        ):
            raise RuntimeError("render resync metadata is malformed")
        if abs(float(color[2]) - target_time) <= 1.0 / PHYSICS_HZ + 1e-9:
            return metadata, drain_count
        if drain_count < limit:
            metadata = render_once()
    return None, limit


def calibrate_render_clock(
    *,
    origin_time: float,
    physical_times: tuple[float, ...],
    render_time: float,
    lag_steps: int,
) -> RenderClockCalibration:
    """Calibrate the fixed Isaac camera clock convention without tolerating drift."""

    physics_dt = 1.0 / PHYSICS_HZ
    if (
        lag_steps <= 0
        or len(physical_times) != lag_steps + 1
        or not math.isfinite(origin_time)
        or not math.isfinite(render_time)
    ):
        raise RuntimeError("render clock calibration inputs are invalid")
    for previous, current in zip(physical_times, physical_times[1:]):
        if (
            not math.isfinite(previous)
            or not math.isfinite(current)
            or not math.isclose(
                current - previous,
                physics_dt,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise RuntimeError("render clock physical prewarm steps are not exact")
    observed_lag_sec = physical_times[-1] - render_time
    if abs(observed_lag_sec) > physics_dt + 1e-9:
        raise RuntimeError(
            "camera ReferenceTime offset is outside one physics step: "
            f"physical={physical_times[-1]:.12f}, render={render_time:.12f}, "
            f"physical_minus_render={observed_lag_sec:.12f}"
        )
    reference_lag_ns = int(round(observed_lag_sec * 1_000_000_000))
    physics_step_ns = int(round(physics_dt * 1_000_000_000))
    if not math.isclose(
        physics_step_ns / 1_000_000_000,
        physics_dt,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("physics step is not exactly representable in nanoseconds")
    if not -physics_step_ns <= reference_lag_ns <= physics_step_ns:
        raise RuntimeError("camera ReferenceTime offset rounded outside one physics step")
    reference_lag_sec = reference_lag_ns / 1_000_000_000
    if abs(observed_lag_sec - reference_lag_sec) > 1e-9:
        raise RuntimeError("camera ReferenceTime offset is not nanosecond representable")
    seed_stamp_ns = int(round((render_time - origin_time) * 1_000_000_000))
    if seed_stamp_ns <= 0:
        raise RuntimeError("camera reference timestamp is not positive")
    warm_stamps_ns = tuple(
        seed_stamp_ns - (lag_steps - 1 - index) * physics_step_ns
        for index in range(lag_steps)
    )
    calibration = RenderClockCalibration(
        origin_time=origin_time,
        reference_lag_sec=reference_lag_sec,
        physics_step_ns=physics_step_ns,
        warm_stamps_ns=warm_stamps_ns,
        seed_stamp_ns=seed_stamp_ns,
    )
    if (
        warm_stamps_ns[0] <= 0
        or any(
            current - previous != physics_step_ns
            for previous, current in zip(warm_stamps_ns, warm_stamps_ns[1:])
        )
    ):
        raise RuntimeError("calibrated camera reference timestamps did not advance")
    if warm_stamps_ns[-1] != seed_stamp_ns:
        raise RuntimeError("prewarm state and camera ReferenceTime do not share one stamp")
    return calibration


class SingleRenderPipeline:
    """Negotiate and freeze zero/one-frame render latency with exact stamps.

    Isaac may expose the current render or the prior render after startup
    draining.  The first bind accepts only those two exact identities, freezes
    the observed mode, and rejects every later skip, replay, or mode switch.
    """

    ZERO_FRAME = "zero_frame"
    ONE_FRAME = "one_frame"

    def __init__(self, lag_steps: int, *, max_stamp_deviation_ns: int = 0) -> None:
        if lag_steps <= 0:
            raise ValueError("render pipeline lag must be positive")
        if type(max_stamp_deviation_ns) is not int or max_stamp_deviation_ns < 0:
            raise ValueError("render stamp deviation tolerance must be a nonnegative integer")
        self.lag_steps = lag_steps
        self.max_stamp_deviation_ns = max_stamp_deviation_ns
        self._stamp_queue: deque[int] = deque()
        self._pending: deque[RenderSnapshot] = deque()
        self._seed_identity: tuple[int, float] | None = None
        self.render_latency_mode: str | None = None
        self.last_physical_stamp_ns = 0
        self.last_output_stamp_ns = 0
        self.last_render_id = 0
        self.last_render_time = -1.0
        self.last_render_generation = -1
        self.stamp_deviation_count = 0
        self.maximum_stamp_deviation_ns = 0
        self.last_stamp_deviation_ns = 0

    @staticmethod
    def _valid_stamp(value: int) -> bool:
        return not isinstance(value, bool) and isinstance(value, int) and value > 0

    def warm_stamp(self, physical_stamp_ns: int) -> None:
        if self._seed_identity is not None or len(self._stamp_queue) >= self.lag_steps:
            raise RuntimeError("render timestamp prewarm was reused or overfilled")
        if (
            not self._valid_stamp(physical_stamp_ns)
            or physical_stamp_ns <= self.last_physical_stamp_ns
        ):
            raise RuntimeError("physical simulation timestamp did not advance during prewarm")
        self._stamp_queue.append(physical_stamp_ns)
        self.last_physical_stamp_ns = physical_stamp_ns

    def seed(
        self,
        snapshot: RenderSnapshot,
        render_id: int,
        render_time: float,
    ) -> None:
        if (
            self._seed_identity is not None
            or self._pending
            or len(self._stamp_queue) != self.lag_steps
            or snapshot.stamp_ns != self.last_physical_stamp_ns
            or render_id <= 0
            or not math.isfinite(render_time)
            or abs(snapshot.world_time - render_time) > 1e-9
        ):
            raise RuntimeError("initial render pipeline snapshot is inconsistent")
        self._pending.append(snapshot)
        self._seed_identity = (render_id, render_time)
        self.last_render_id = render_id
        self.last_render_time = render_time
        self.last_render_generation = snapshot.generation

    def begin_step(self, physical_stamp_ns: int) -> int:
        if self._seed_identity is None:
            raise RuntimeError("render pipeline was not seeded")
        if (
            not self._valid_stamp(physical_stamp_ns)
            or physical_stamp_ns <= self.last_physical_stamp_ns
        ):
            raise RuntimeError("physical simulation timestamp did not advance")
        self.last_physical_stamp_ns = physical_stamp_ns
        if self.render_latency_mode == self.ZERO_FRAME:
            if self._stamp_queue:
                raise RuntimeError("zero-frame render mode retained a delayed timestamp")
            output_stamp_ns = physical_stamp_ns
        else:
            self._stamp_queue.append(physical_stamp_ns)
            if len(self._stamp_queue) != self.lag_steps + 1:
                raise RuntimeError("render timestamp delay queue lost its frozen depth")
            output_stamp_ns = self._stamp_queue.popleft()
        if output_stamp_ns <= self.last_output_stamp_ns:
            raise RuntimeError("render-aligned simulation timestamp duplicated or rolled back")
        self.last_output_stamp_ns = output_stamp_ns
        return output_stamp_ns

    def _matches_render(
        self,
        snapshot: RenderSnapshot,
        render_time: float,
        render_stamp_ns: int,
    ) -> bool:
        return (
            abs(snapshot.stamp_ns - render_stamp_ns)
            <= self.max_stamp_deviation_ns
            and math.isfinite(render_time)
            and abs(snapshot.world_time - render_time) <= 1e-9
        )

    @staticmethod
    def _identity_failure(
        returned: RenderSnapshot,
        requested: RenderSnapshot,
        render_id: int,
        render_time: float,
        render_stamp_ns: int,
        pending_depth: int,
    ) -> RuntimeError:
        return RuntimeError(
            "single render returned a frame for an unbuffered physical state: "
            f"returned_world={returned.world_time:.12f}, "
            f"render_world={render_time:.12f}, "
            f"requested_world={requested.world_time:.12f}, "
            f"returned_stamp={returned.stamp_ns}, "
            f"render_stamp={render_stamp_ns}, "
            f"requested_stamp={requested.stamp_ns}, "
            f"render_id={render_id}, pending_depth={pending_depth}"
        )

    def bind(
        self,
        requested: RenderSnapshot,
        render_id: int,
        render_time: float,
        render_stamp_ns: int,
        current_generation: int,
    ) -> RenderSnapshot | None:
        pending_contract_valid = (
            not self._pending
            if self.render_latency_mode == self.ZERO_FRAME
            else bool(self._pending)
            and requested.stamp_ns > self._pending[-1].stamp_ns
        )
        if (
            self._seed_identity is None
            or requested.stamp_ns != self.last_physical_stamp_ns
            or not pending_contract_valid
            or not self._valid_stamp(render_stamp_ns)
            or isinstance(render_id, bool)
            or not isinstance(render_id, int)
            or render_id <= 0
            or not math.isfinite(render_time)
        ):
            raise RuntimeError("render request is not bound to the current physical step")
        self._pending.append(requested)
        prior_mode = self.render_latency_mode
        if prior_mode is None:
            seed = self._pending[0]
            if self._matches_render(seed, render_time, render_stamp_ns):
                self.render_latency_mode = self.ONE_FRAME
                returned = self._pending.popleft()
                if (render_id, render_time) != self._seed_identity:
                    raise RuntimeError("one-frame negotiation did not return the seeded frame")
            elif self._matches_render(requested, render_time, render_stamp_ns):
                self.render_latency_mode = self.ZERO_FRAME
                returned = requested
                self._pending.clear()
                self._stamp_queue.clear()
                if returned.stamp_ns <= self.last_output_stamp_ns:
                    raise RuntimeError("zero-frame negotiation rolled simulation time back")
                self.last_output_stamp_ns = returned.stamp_ns
                if render_id <= self.last_render_id or render_time <= self.last_render_time:
                    raise RuntimeError("zero-frame negotiation replayed seeded render metadata")
            else:
                raise self._identity_failure(
                    seed,
                    requested,
                    render_id,
                    render_time,
                    render_stamp_ns,
                    len(self._pending),
                )
        else:
            returned = self._pending.popleft()
            if abs(returned.world_time - render_time) > 1e-9:
                raise self._identity_failure(
                    returned,
                    requested,
                    render_id,
                    render_time,
                    render_stamp_ns,
                    len(self._pending),
                )
        if prior_mode is not None and (
            render_id <= self.last_render_id or render_time <= self.last_render_time
        ):
            raise RuntimeError("camera render metadata did not advance across one-render ticks")
        self.last_render_id = render_id
        self.last_render_time = render_time
        self.last_render_generation = returned.generation
        if returned.generation != current_generation:
            return None
        camera_deviation_ns = render_stamp_ns - returned.stamp_ns
        safe_step_deviation_ns = self.last_output_stamp_ns - returned.stamp_ns
        maximum_deviation_ns = max(
            abs(camera_deviation_ns), abs(safe_step_deviation_ns)
        )
        if maximum_deviation_ns > self.max_stamp_deviation_ns:
            raise RuntimeError(
                "camera frame and delayed SafeStep timestamp are not identical: "
                f"snapshot={returned.stamp_ns}, camera={render_stamp_ns}, "
                f"safe_step={self.last_output_stamp_ns}"
            )
        self.last_stamp_deviation_ns = camera_deviation_ns
        if maximum_deviation_ns:
            self.stamp_deviation_count += 1
            self.maximum_stamp_deviation_ns = max(
                self.maximum_stamp_deviation_ns, maximum_deviation_ns
            )
        return returned
