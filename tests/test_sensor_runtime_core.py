from __future__ import annotations

from dataclasses import replace

import pytest

from sensor_runtime.contract import REQUIRED_STREAMS, SensorProfile
from sensor_runtime.core import (
    LatestOnlySlot,
    SensorBatch,
    SensorFault,
    assert_unpaired_generation_event_fresh,
)
from sensor_runtime.workload import BoundedModelFreeWorkload, CaptureNotReady, SafeStep


def _batch(generation: int, sequence: int, stamp: int = 1) -> SensorBatch:
    payloads = {name: {"stamp_ns": stamp} for name in REQUIRED_STREAMS}
    for name in ("d435i_rgb", "d435i_depth"):
        payloads[name].update({"render_id": 1, "render_generation": generation})
    return SensorBatch(
        stamp_ns=stamp,
        generation=generation,
        sequence=sequence,
        payloads=payloads,
        stream_stamps_ns={name: stamp for name in REQUIRED_STREAMS},
        safe_stop={
            "identity": f"sensor-soak:{generation}:{sequence}",
            "linear_x": 0.0,
            "angular_z": 0.0,
            "emergency_stop": True,
            "physics_step": 1,
            "applied_step_count": 1,
            "steps_since_previous_capture": 1,
            "render_id": 1,
            "render_generation": generation,
            "reset_kind": "continuous_world_articulation_state",
        },
        reset_reason=("initial" if generation == 0 else "active_periodic") if sequence == 0 else None,
    )


def test_sensor_batch_rejects_mixed_stamp_and_render_generation() -> None:
    with pytest.raises(ValueError, match="share one sim timestamp"):
        replace(_batch(0, 0), stream_stamps_ns={**{name: 1 for name in REQUIRED_STREAMS}, "tf": 2})
    with pytest.raises(ValueError, match="current-generation render"):
        replace(_batch(0, 0), safe_stop={**_batch(0, 0).safe_stop, "render_generation": 1})


def test_latest_only_pins_sequence_zero_and_reconciles_all_capacity_one_drops() -> None:
    slot = LatestOnlySlot()
    slot.reset(0)
    assert slot.offer(_batch(0, 0, 1)) == "stored"
    assert slot.offer(_batch(0, 1, 2)) == "barrier_dropped"
    assert slot.offer(_batch(0, 2, 3)) == "barrier_dropped"
    assert slot.take(0.0).sequence == 0
    assert slot.offer(_batch(0, 3, 4)) == "stored"
    assert slot.offer(_batch(0, 4, 5)) == "replaced"
    assert slot.overwrite_count == 1
    assert slot.barrier_drop_count == 2
    assert slot.take(0.0).sequence == 4
    slot.offer(_batch(0, 5, 6))
    slot.reset(1)
    assert slot.reset_clear_count == 1
    assert slot.take(0.0) is None
    assert slot.accepted_count == 6
    assert slot.accepted_count == 2 + slot.overwrite_count + slot.reset_clear_count + slot.barrier_drop_count
    with pytest.raises(SensorFault, match="generation_contamination"):
        slot.offer(_batch(0, 6, 7))


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns


class FakeBackend:
    def __init__(self, clock: FakeClock, render_every: int = 10) -> None:
        self.clock = clock
        self.render_every = render_every
        self.generation = -1
        self.physics = 0
        self.safe = 0
        self.render = 0
        self.reset_safe_applications = 0
        self.close_safe_applications = 0

    def reset(self, generation: int) -> None:
        self.generation = generation
        self.physics = 0
        self.reset_safe_applications += 1

    def step_safe_stop(self) -> SafeStep:
        self.clock.ns += 5_000_000
        self.physics += 1
        self.safe += 1
        rendered = self.physics % self.render_every == 0
        if rendered:
            self.render += 1
        return SafeStep(
            sim_stamp_ns=self.clock.ns,
            physics_step=self.physics,
            applied_step_count=self.safe,
            rendered=rendered,
            render_id=self.render,
            render_generation=self.generation if rendered else self.generation,
        )

    def capture(self, step: SafeStep):
        assert step.rendered and step.render_generation == self.generation
        payloads = {name: {"render_id": step.render_id} for name in REQUIRED_STREAMS}
        for name in ("d435i_rgb", "d435i_depth"):
            payloads[name]["render_generation"] = self.generation
        return payloads

    def close(self) -> None:
        self.close_safe_applications += 1


class Sink:
    def __init__(self) -> None:
        self.batches: list[SensorBatch] = []
        self.resets: list[int] = []

    def reset(self, generation: int) -> None:
        self.resets.append(generation)

    def submit(self, batch: SensorBatch) -> None:
        self.batches.append(batch)


@pytest.mark.parametrize("render_every", [10, 13, 20])
def test_independent_bounded_workload_survives_active_reset_and_render_cadence(render_every: int) -> None:
    clock, sink = FakeClock(), Sink()
    backend = FakeBackend(clock, render_every)
    profile = SensorProfile("fake", 0.55, 0.3, 20.0, 10.0, 20.0, 0.2, 2, 1)
    summary = BoundedModelFreeWorkload(backend, sink, profile, monotonic_ns=clock).run_bounded()
    assert summary.elapsed_sec >= profile.duration_sec
    assert summary.active_reset_count == 2
    assert sink.resets == [0, 1, 2]
    assert [(item.generation, item.sequence) for item in sink.batches if item.sequence == 0] == [(0, 0), (1, 0), (2, 0)]
    assert all(item.safe_stop["reset_kind"] == "continuous_world_articulation_state" for item in sink.batches)
    assert all(later.stamp_ns > earlier.stamp_ns for earlier, later in zip(sink.batches, sink.batches[1:]))
    assert all(item.safe_stop["identity"] == f"sensor-soak:{item.generation}:{item.sequence}" for item in sink.batches)
    assert backend.reset_safe_applications == 3
    # Reset safe-stop applications are deliberately separate from the exact
    # once-per-physics-step coverage counter.
    assert summary.safe_stop_step_count == int(round(summary.elapsed_sec * 200))


def test_generation_zero_and_post_reset_first_physics_step_advance_exactly_one() -> None:
    clock, sink, backend = FakeClock(), Sink(), None
    backend = FakeBackend(clock)
    workload = BoundedModelFreeWorkload(
        backend,
        sink,
        SensorProfile("fake", 1.0, 0.0, 20.0, 1.0, 20.0, 1.0, 0, 0),
        monotonic_ns=clock,
    )
    workload.initialize()
    workload._step_and_maybe_capture()
    assert backend.safe == 1
    workload._active_reset()
    workload._step_and_maybe_capture()
    assert backend.safe == 2
    backend.close()
    assert backend.safe == 2 and backend.close_safe_applications == 1


def test_unpaired_generation_event_times_out_at_035_seconds() -> None:
    events = [(0, 1_000_000_000)]
    assert_unpaired_generation_event_fresh(events, 1_349_999_999)
    with pytest.raises(SensorFault, match="unpaired_timeout"):
        assert_unpaired_generation_event_fresh(events, 1_350_000_000)


def test_transient_invalid_current_render_does_not_emit_or_consume_sequence_zero() -> None:
    clock, sink = FakeClock(), Sink()

    class DelayedBackend(FakeBackend):
        def __init__(self, value: FakeClock) -> None:
            super().__init__(value, render_every=2)
            self.attempts = 0

        def capture(self, step: SafeStep):
            self.attempts += 1
            if self.attempts == 1:
                raise CaptureNotReady("placeholder depth")
            return super().capture(step)

    backend = DelayedBackend(clock)
    workload = BoundedModelFreeWorkload(
        backend,
        sink,
        SensorProfile("fake", 0.03, 0.0, 20.0, 1.0, 20.0, 1.0, 0, 0),
        monotonic_ns=clock,
    )
    workload.run_bounded()
    assert backend.attempts >= 2
    assert sink.batches[0].sequence == 0 and sink.batches[0].reset_reason == "initial"
