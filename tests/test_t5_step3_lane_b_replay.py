from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts import probe_t5_step3_lane_b_replay as replay
from slow_planner.base import PlannerDecision, PlannerMetrics
from slow_planner.lane_b import LaneBPlannerMode


def _images(root: Path) -> list[Path]:
    paths = []
    for index in range(4):
        path = root / f"{index}.jpg"
        Image.new("RGB", replay.IMAGE_SIZE, (index * 32, 16, 64)).save(path, "JPEG")
        paths.append(path)
    return paths


def test_replay_uses_fixed_deadline_order_and_redacted_output(tmp_path, monkeypatch) -> None:
    class Client:
        def __init__(self, endpoint, *, timeout_ms):
            assert endpoint == "tcp://127.0.0.1:8200"
            assert timeout_ms == 12_000

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decide(self, request):
            assert tuple(image.view_id for image in request.ordered_images) == replay.REV_C_VIEW_ORDER
            return (
                PlannerDecision(
                    episode_id=request.episode_id,
                    snapshot_id=request.snapshot_id,
                    decision="select_frontier",
                    frontier_id=1,
                    target_relative_xz=None,
                    confidence=0.8,
                    raw_text="private reasoning must not escape",
                ),
                PlannerMetrics(end_to_end_ms=10.0),
            )

    monkeypatch.setattr(replay, "SlowPlannerClient", Client)
    value = replay.run_replay(
        endpoint="tcp://127.0.0.1:8200",
        paths=_images(tmp_path),
        mode=LaneBPlannerMode.BOUNDED_ADVISOR,
        episode_id="episode-1",
        sequence_id=0,
        deadline_ms=replay.DEADLINE_MS,
    )
    assert value["status"] == "PASS"
    assert value["deadline_met"] is True
    assert value["decision"]["intent"] == "frontier_advice"
    assert "raw_text" not in str(value)


def test_replay_failure_is_deterministic_internvla_fallback(tmp_path, monkeypatch) -> None:
    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decide(self, request):
            raise RuntimeError("private remote error")

    monkeypatch.setattr(replay, "SlowPlannerClient", Client)
    value = replay.run_replay(
        endpoint="tcp://127.0.0.1:8200",
        paths=_images(tmp_path),
        mode=LaneBPlannerMode.BOUNDED_ADVISOR,
        episode_id="episode-1",
        sequence_id=1,
    )
    assert value["status"] == "PASS"
    assert value["model_request_status"] == "ERROR"
    assert value["decision"]["requires_internvla_fallback"] is True
    assert value["decision"]["fallback_reason"] == "step3_service_failure"
    assert "private remote error" not in str(value)


def test_direct_replay_failure_requires_safe_stop_without_internvla(
    tmp_path, monkeypatch
) -> None:
    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decide(self, request):
            raise RuntimeError("private remote error")

    monkeypatch.setattr(replay, "SlowPlannerClient", Client)
    value = replay.run_replay(
        endpoint="tcp://127.0.0.1:8200",
        paths=_images(tmp_path),
        mode=LaneBPlannerMode.DIRECT_HIGH_LEVEL,
        episode_id="episode-1",
        sequence_id=1,
    )
    assert value["status"] == "PASS"
    assert value["model_request_status"] == "ERROR"
    assert value["decision"]["requires_internvla_fallback"] is False
    assert value["decision"]["requires_safe_stop"] is True
    assert value["decision"]["safe_stop_reason"] == "step3_service_failure"
    assert "private remote error" not in str(value)


def test_replay_accepts_only_fixed_production_or_warmup_deadline(tmp_path) -> None:
    try:
        replay.run_replay(
            endpoint="tcp://127.0.0.1:8200",
            paths=_images(tmp_path),
            mode=LaneBPlannerMode.BOUNDED_ADVISOR,
            episode_id="episode-1",
            sequence_id=0,
            deadline_ms=12_001,
        )
    except ValueError as error:
        assert "fixed warmup timeout" in str(error)
    else:
        raise AssertionError("arbitrary replay timeout was accepted")
