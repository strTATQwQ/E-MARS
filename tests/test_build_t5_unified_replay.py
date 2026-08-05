from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_t5_unified_replay import ReplayBuildError, build_replay, main


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, *values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def _artifact(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def _timeline(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "replay" / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_builds_stable_cross_host_timeline_and_references_private_trace(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _json(root / "input_binding.json", {"run_id": "paired-10-a", "lane": "a"})

    d435 = "remote/x86/evaluator/d435_rgb_5hz"
    _artifact(root, f"{d435}/frames/00000000.png")
    _jsonl(
        root / d435 / "frames.jsonl",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "source_sequence": 10,
            "sim_stamp_ns": 100,
            "path": "frames/00000000.png",
            "height": 480,
            "width": 640,
        },
    )
    model = "remote/x86/evaluator/full_rgb_5hz"
    _artifact(root, f"{model}/frames/00000000.png")
    _jsonl(
        root / model / "frames.jsonl",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "sequence_id": 5,
            "sim_stamp_ns": 50,
            "path": "frames/00000000.png",
        },
    )

    snapshot_root = "remote/x86/evaluator/revc_snapshots/snap"
    camera_rows = []
    for ordinal, view in enumerate(("front_left", "front", "front_right", "rear")):
        relative = f"remote/x86/evaluator/revc_snapshots/snap/{ordinal:02d}_{view}.png"
        _artifact(root, relative)
        camera_rows.append(
            {
                "identity": view,
                "path": f"revc_snapshots/snap/{ordinal:02d}_{view}.png",
                "encoding": "png_rgb8",
            }
        )
    observer = "remote/x86/evaluator/revc_snapshots/snap/04_observer.png"
    _artifact(root, observer)
    _json(
        root / snapshot_root / "snapshot.json",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "sequence_id": 7,
            "request_id": "req-snapshot",
            "sim_stamp_before_ns": 75,
            "wall_time_unix": 10.25,
            "cameras": camera_rows,
            "observer": {
                "identity": "third_person",
                "path": "revc_snapshots/snap/04_observer.png",
            },
        },
    )

    _jsonl(
        root / "remote/x86/evaluator/step3_timeout_advice.jsonl",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "trigger_sequence_id": 8,
            "trigger_request_id": "req-client",
            "status": "ACCEPTED",
            "service_wall_latency_sec": 0.25,
            "wall_time_unix": 10.5,
        },
    )
    _jsonl(
        root / "remote/x86/evaluator/task_state/events.jsonl",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "trigger_sequence_id": 9,
            "snapshot_sim_stamp_ns": 90,
            "event": "task_state_updated",
        },
    )
    _jsonl(
        root / "remote/dgx/step3_timeout_service/step3/step3_timeout_service.jsonl",
        {
            "request": {
                "episode_id": "a::7",
                "snapshot_id": "a::a::7::2::7",
                "instruction": "turn left",
                "env": {"OPENAI_API_KEY": "do-not-copy"},
            },
            "decision": {"decision": "select_frontier", "frontier_id": 2},
            "metrics": {
                "server_total_ms": 12.0,
                "end_to_end_ms": 10.0,
                "model_generate_ms": 8.0,
                "prefill_ttft_ms": 3.0,
                "decode_ms": 5.0,
            },
            "argv": ["--secret"],
            "password": "do-not-copy",
        },
    )
    _jsonl(
        root / "remote/dgx/step3_timeout_service/private_trace/private_trace.jsonl",
        {
            "episode_id": "a::7",
            "snapshot_id": "a::a::7::2::7",
            "raw_text": "private reasoning must stay in source",
            "token": "do-not-copy",
            "request_received_wall_monotonic_ns": 1_000_000_000,
            "response_completed_wall_monotonic_ns": 1_020_000_000,
        },
    )
    _jsonl(
        root / "remote/dgx/client/client_records.jsonl",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "sequence_id": 8,
            "request_id": "req-client",
            "sim_stamp_ns": 80,
            "client_wall_monotonic_ns": 123,
            "model_discrete_action": 1,
            "inference_latency_sec": 0.010,
            "action_round_trip_latency_sec": 0.020,
            "network_ros_residual_latency_sec": 0.004,
            "observation_encode_latency_sec": 0.002,
            "nav2_resolution_latency_sec": 0.003,
        },
    )
    _jsonl(
        root / "remote/dgx/onboard/motion_gate_records.jsonl",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "sequence_id": 8,
            "sim_stamp_ns": 85,
            "decision": "safe_stop_complete",
        },
    )
    _jsonl(
        root / "remote/dgx/onboard/controller_records.jsonl",
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "sequence_id": 8,
            "wall_time_unix": 10.75,
            "desired_angular_z": 0.2,
            "collision_pairs": [["large", "repeated", "payload"]],
        },
    )

    index = build_replay(root)
    rows = _timeline(root)
    assert index["event_count"] == 10
    assert [row["sim_stamp_ns"] for row in rows] == [50, 75, 75, 75, 80, 80, 85, 90, 100, None]
    assert [row["event_type"] for row in rows[1:4]] == [
        "revc_snapshot",
        "step3_public_trace",
        "step3_private_trace",
    ]
    assert rows[4]["event_type"] == "step3_timeout_advice"
    assert rows[4]["payload"]["timeline_alignment"]["method"] == "unique_request_id"
    private = rows[3]
    assert private["payload"]["private_trace_reference_only"] is True
    assert set(private["payload"]) == {
        "private_trace_reference_only",
        "source_line",
        "record_sha256",
        "private_wall_timing",
        "wall_latencies_ms",
        "timeline_alignment",
    }
    assert private["payload"]["private_wall_timing"]["duration_ns"] == 20_000_000
    snapshot = rows[1]
    assert snapshot["snapshot_id"] == "a::a::7::2::7"
    assert [item["view_id"] for item in snapshot["artifacts"]] == [
        "front_left",
        "front",
        "front_right",
        "rear",
        "third_person",
    ]
    assert all(item["present"] for item in snapshot["artifacts"])
    timeline_text = (root / "replay/timeline.jsonl").read_text(encoding="utf-8")
    assert "do-not-copy" not in timeline_text
    assert "private reasoning" not in timeline_text
    assert '"argv"' not in timeline_text
    assert index["alignment"] == {
        "explicit_sim_stamp_count": 6,
        "matched_sim_stamp_count": 3,
        "no_sim_stamp_count": 1,
        "unmatched_identity_count": 0,
        "unmatched_artifact_count": 0,
    }
    controller = rows[-1]
    assert controller["event_type"] == "controller_record"
    assert controller["payload"]["desired_angular_z"] == 0.2
    assert "collision_pairs" not in controller["payload"]
    assert index["timeline_sha256"] == hashlib.sha256(
        (root / "replay/timeline.jsonl").read_bytes()
    ).hexdigest()
    components = index["wall_latency_summary"]["components"]
    assert components["step3.advisor_service"] == {
        "count": 1,
        "p50_ms": 250.0,
        "p95_ms": 250.0,
        "max_ms": 250.0,
    }
    assert components["step3.private_trace_response"]["p95_ms"] == 20.0
    assert components["step3.server_total"]["max_ms"] == 12.0
    assert components["step3.end_to_end"]["max_ms"] == 10.0
    assert components["step3.model_generate"]["max_ms"] == 8.0
    assert components["step3.prefill_ttft"]["max_ms"] == 3.0
    assert components["step3.decode"]["max_ms"] == 5.0
    assert components["internvla.inference"]["max_ms"] == 10.0
    assert components["internvla.action_round_trip"]["max_ms"] == 20.0
    assert components["internvla.network_ros_residual"]["max_ms"] == 4.0
    assert components["internvla.observation_encode"]["max_ms"] == 2.0
    assert components["internvla.nav2_resolution"]["max_ms"] == 3.0
    assert index["per_episode_ranges"] == [
        {
            "episode_id": "a::7",
            "reset_generation": 2,
            "event_count": 10,
            "timeline_start_index": 0,
            "timeline_end_index": 9,
            "first_event_id": "t5-replay-000000000",
            "last_event_id": "t5-replay-000000009",
            "first_sim_stamp_ns": 50,
            "last_sim_stamp_ns": 100,
        }
    ]

    first_timeline = (root / "replay/timeline.jsonl").read_bytes()
    first_index = (root / "replay/timeline_index.json").read_bytes()
    assert build_replay(root) == index
    assert (root / "replay/timeline.jsonl").read_bytes() == first_timeline
    assert (root / "replay/timeline_index.json").read_bytes() == first_index


def test_missing_sources_produce_valid_empty_replay(tmp_path: Path) -> None:
    root = tmp_path / "empty-run"
    root.mkdir()
    index = build_replay(root)
    assert index["run_id"] == "empty-run"
    assert index["event_count"] == 0
    assert index["source_inventory"] == []
    assert index["per_episode_ranges"] == []
    assert (root / "replay/timeline.jsonl").read_bytes() == b""


@pytest.mark.parametrize(
    "bad_line",
    ["not-json\n", "[]\n", '{"sim_stamp_ns": NaN}\n'],
)
def test_invalid_jsonl_fails_closed(tmp_path: Path, bad_line: str) -> None:
    root = tmp_path / "run"
    path = root / "remote/x86/evaluator/full_rgb_5hz/frames.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(bad_line, encoding="utf-8")
    with pytest.raises(ReplayBuildError, match="JSONL|JSON object|non-finite"):
        build_replay(root)


@pytest.mark.parametrize("artifact", ["../escape.png", "/tmp/escape.png", "C:/escape.png"])
def test_artifact_path_escape_fails_closed(tmp_path: Path, artifact: str) -> None:
    root = tmp_path / "run"
    _jsonl(
        root / "remote/x86/evaluator/full_rgb_5hz/frames.jsonl",
        {"sim_stamp_ns": 1, "source_sequence": 1, "path": artifact},
    )
    with pytest.raises(ReplayBuildError, match="artifact path"):
        build_replay(root)


def test_cli_reports_malformed_run_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    root = tmp_path / "run"
    path = root / "remote/dgx/client/client_records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{\n", encoding="utf-8")
    assert main([str(root)]) == 2
    assert "ERROR:" in capsys.readouterr().err
