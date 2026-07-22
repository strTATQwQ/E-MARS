from __future__ import annotations

import pytest

from sensor_runtime.render_pipeline import (
    RenderSnapshot,
    SingleRenderPipeline,
    bounded_zero_frame_resync,
    calibrate_render_clock,
)


STEP_NS = 5_000_000


def _snapshot(generation: int, stamp_ns: int) -> RenderSnapshot:
    return RenderSnapshot(
        generation=generation,
        stamp_ns=stamp_ns,
        world_time=stamp_ns / 1e9,
        state={"identity": (generation, stamp_ns)},
    )


def _seeded_pipeline() -> SingleRenderPipeline:
    pipeline = SingleRenderPipeline(lag_steps=10)
    for index in range(1, 11):
        pipeline.warm_stamp(index * STEP_NS)
    pipeline.seed(_snapshot(-1, 50_000_000), 50_000_000, 0.05)
    return pipeline


def test_render_clock_calibrates_one_step_reference_lag_with_positive_prewarm() -> None:
    physical_times = tuple(index * STEP_NS / 1e9 for index in range(1, 12))
    calibration = calibrate_render_clock(
        origin_time=0.0,
        physical_times=physical_times,
        render_time=0.05,
        lag_steps=10,
    )
    assert calibration.reference_lag_sec == pytest.approx(0.005, abs=1e-12)
    assert calibration.warm_stamps_ns == tuple(
        index * STEP_NS for index in range(1, 11)
    )
    assert calibration.seed_stamp_ns == 50_000_000
    assert calibration.stamp_ns(calibration.reference_time(0.105)) == 100_000_000


def test_render_clock_accepts_zero_and_signed_one_step_offsets_and_rejects_drift() -> None:
    physical_times = tuple(index * STEP_NS / 1e9 for index in range(1, 12))
    zero = calibrate_render_clock(
        origin_time=0.0,
        physical_times=physical_times,
        render_time=0.055,
        lag_steps=10,
    )
    assert zero.reference_lag_sec == 0.0
    assert zero.warm_stamps_ns[0] == 10_000_000
    assert zero.seed_stamp_ns == 55_000_000

    ahead = calibrate_render_clock(
        origin_time=0.0,
        physical_times=physical_times,
        render_time=0.060,
        lag_steps=10,
    )
    assert ahead.reference_lag_sec == pytest.approx(-0.005, abs=1e-12)
    assert ahead.warm_stamps_ns[0] == 15_000_000
    assert ahead.seed_stamp_ns == 60_000_000

    with pytest.raises(RuntimeError, match="offset is outside one physics step"):
        calibrate_render_clock(
            origin_time=0.0,
            physical_times=physical_times,
            render_time=0.049,
            lag_steps=10,
        )
    with pytest.raises(RuntimeError, match="offset is outside one physics step"):
        calibrate_render_clock(
            origin_time=0.0,
            physical_times=physical_times,
            render_time=0.061,
            lag_steps=10,
        )


def test_render_clock_uses_integer_step_count_instead_of_accumulated_world_float() -> None:
    physical_times = tuple(index * STEP_NS / 1e9 for index in range(1, 12))
    calibration = calibrate_render_clock(
        origin_time=0.0,
        physical_times=physical_times,
        render_time=0.05,
        lag_steps=10,
    )

    # The attempt-19 failure mode rounds one nanosecond low when the accumulated
    # world float is converted again. The calibrated seed plus the global step
    # count is authoritative and advances by exact 5 ms integer intervals.
    assert calibration.stamp_ns(
        calibration.reference_time(0.10999999943)
    ) == 104_999_999
    assert calibration.stamp_for_physical_step(22, 11) == 105_000_000
    assert calibration.reference_time_for_physical_step(22, 11) == pytest.approx(
        0.105,
        abs=1e-15,
    )
    assert calibration.physics_step_ns == STEP_NS

    with pytest.raises(RuntimeError, match="integer physics clock step is invalid"):
        calibration.stamp_for_physical_step(10, 11)


def _advance(
    pipeline: SingleRenderPipeline,
    first_index: int,
    last_index: int,
) -> list[int]:
    return [
        pipeline.begin_step(index * STEP_NS)
        for index in range(first_index, last_index + 1)
    ]


def test_single_render_pipeline_delays_all_step_stamps_and_skips_seed_generation() -> None:
    pipeline = _seeded_pipeline()
    assert _advance(pipeline, 11, 20) == [index * STEP_NS for index in range(1, 11)]
    assert (
        pipeline.bind(
            _snapshot(0, 100_000_000),
            50_000_000,
            0.05,
            50_000_000,
            0,
        )
        is None
    )
    assert pipeline.render_latency_mode == SingleRenderPipeline.ONE_FRAME

    assert _advance(pipeline, 21, 30)[-1] == 100_000_000
    returned = pipeline.bind(
        _snapshot(0, 150_000_000),
        100_000_000,
        0.10,
        100_000_000,
        0,
    )
    assert returned is not None
    assert (returned.generation, returned.stamp_ns) == (0, 100_000_000)


def test_single_render_pipeline_negotiates_zero_frame_before_workload() -> None:
    pipeline = _seeded_pipeline()
    assert _advance(pipeline, 11, 20)[-1] == 50_000_000
    returned = pipeline.bind(
        _snapshot(0, 100_000_000),
        100_000_000,
        0.10,
        100_000_000,
        0,
    )
    assert returned is not None
    assert returned.stamp_ns == 100_000_000
    assert pipeline.render_latency_mode == SingleRenderPipeline.ZERO_FRAME
    assert pipeline.last_output_stamp_ns == 100_000_000

    assert _advance(pipeline, 21, 30) == [
        index * STEP_NS for index in range(21, 31)
    ]
    returned = pipeline.bind(
        _snapshot(0, 150_000_000),
        150_000_000,
        0.15,
        150_000_000,
        0,
    )
    assert returned is not None and returned.stamp_ns == 150_000_000


def test_single_render_pipeline_discards_old_generation_then_binds_new_generation() -> None:
    pipeline = _seeded_pipeline()
    _advance(pipeline, 11, 20)
    pipeline.bind(_snapshot(0, 100_000_000), 50_000_000, 0.05, 50_000_000, 0)
    _advance(pipeline, 21, 30)
    pipeline.bind(_snapshot(0, 150_000_000), 100_000_000, 0.10, 100_000_000, 0)

    # Reset happens three physics-only steps after the prior render.  The first
    # new-generation render still returns the pending old-generation frame.
    _advance(pipeline, 31, 33)
    assert _advance(pipeline, 34, 43)[-1] == 165_000_000
    assert (
        pipeline.bind(
            _snapshot(1, 215_000_000),
            150_000_000,
            0.15,
            150_000_000,
            1,
        )
        is None
    )

    assert _advance(pipeline, 44, 53)[-1] == 215_000_000
    returned = pipeline.bind(
        _snapshot(1, 265_000_000),
        215_000_000,
        0.215,
        215_000_000,
        1,
    )
    assert returned is not None
    assert (returned.generation, returned.stamp_ns) == (1, 215_000_000)


def test_single_render_pipeline_rejects_skipped_or_misstamped_camera_frames() -> None:
    skipped = _seeded_pipeline()
    _advance(skipped, 11, 20)
    with pytest.raises(RuntimeError, match="returned_world=.*render_world=.*requested_world"):
        skipped.bind(
            _snapshot(0, 100_000_000),
            75_000_000,
            0.075,
            75_000_000,
            0,
        )

    misstamped = _seeded_pipeline()
    _advance(misstamped, 11, 20)
    misstamped.bind(
        _snapshot(0, 100_000_000),
        50_000_000,
        0.05,
        50_000_000,
        0,
    )
    _advance(misstamped, 21, 30)
    with pytest.raises(RuntimeError, match="timestamp are not identical"):
        misstamped.bind(
            _snapshot(0, 150_000_000),
            100_000_000,
            0.10,
            99_000_000,
            0,
        )

    switched = _seeded_pipeline()
    _advance(switched, 11, 20)
    switched.bind(
        _snapshot(0, 100_000_000),
        50_000_000,
        0.05,
        50_000_000,
        0,
    )
    _advance(switched, 21, 30)
    with pytest.raises(RuntimeError, match="unbuffered physical state"):
        switched.bind(
            _snapshot(0, 150_000_000),
            150_000_000,
            0.15,
            150_000_000,
            0,
        )


def test_completion_pipeline_records_bounded_render_stamp_deviation() -> None:
    pipeline = SingleRenderPipeline(lag_steps=10, max_stamp_deviation_ns=1_000_000)
    for index in range(1, 11):
        pipeline.warm_stamp(index * STEP_NS)
    pipeline.seed(_snapshot(-1, 50_000_000), 50_000_000, 0.05)
    _advance(pipeline, 11, 20)
    pipeline.bind(
        _snapshot(0, 100_000_000),
        50_000_000,
        0.05,
        49_999_999,
        -1,
    )
    _advance(pipeline, 21, 30)
    returned = pipeline.bind(
        _snapshot(0, 150_000_000),
        100_000_000,
        0.10,
        99_999_999,
        0,
    )
    assert returned is not None and returned.stamp_ns == 100_000_000
    assert pipeline.stamp_deviation_count == 2
    assert pipeline.maximum_stamp_deviation_ns == 1
    assert pipeline.last_stamp_deviation_ns == -1

    _advance(pipeline, 31, 40)
    with pytest.raises(RuntimeError, match="timestamp are not identical"):
        pipeline.bind(
            _snapshot(0, 200_000_000),
            150_000_000,
            0.15,
            148_999_999,
            0,
        )


def test_single_render_pipeline_fails_closed_on_queue_or_render_replay() -> None:
    pipeline = _seeded_pipeline()
    with pytest.raises(RuntimeError, match="did not advance"):
        pipeline.begin_step(50_000_000)
    _advance(pipeline, 11, 20)
    pipeline.bind(_snapshot(0, 100_000_000), 50_000_000, 0.05, 50_000_000, 0)
    _advance(pipeline, 21, 30)
    with pytest.raises(RuntimeError, match="unbuffered physical state"):
        pipeline.bind(
            _snapshot(0, 150_000_000),
            50_000_000,
            0.05,
            50_000_000,
            0,
        )


def _render_metadata(render_time: float) -> dict[str, tuple[int, int, float]]:
    identity = (int(round(render_time * 1e9)), 1_000_000_000, render_time)
    return {"color": identity, "depth": identity, "front": identity}


def test_completion_zero_frame_resync_drains_without_advancing_physics() -> None:
    observed = iter([0.165])
    metadata, count = bounded_zero_frame_resync(
        _render_metadata(0.115),
        render_once=lambda: _render_metadata(next(observed)),
        target_time=0.165,
        limit=4,
    )
    assert metadata is not None and metadata["color"][2] == 0.165
    assert count == 2


def test_completion_zero_frame_resync_drops_after_frozen_bound() -> None:
    observed = iter([0.115, 0.115, 0.115])
    metadata, count = bounded_zero_frame_resync(
        _render_metadata(0.115),
        render_once=lambda: _render_metadata(next(observed)),
        target_time=0.165,
        limit=4,
    )
    assert metadata is None
    assert count == 4


def test_render_resync_rejects_unbounded_or_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="frozen bound"):
        bounded_zero_frame_resync(
            _render_metadata(0.115),
            render_once=lambda: _render_metadata(0.165),
            target_time=0.165,
            limit=5,
        )
    with pytest.raises(RuntimeError, match="malformed"):
        bounded_zero_frame_resync(
            {"color": (1, 2)},  # type: ignore[dict-item]
            render_once=lambda: _render_metadata(0.165),
            target_time=0.165,
            limit=4,
        )
