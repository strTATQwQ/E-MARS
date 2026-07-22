from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from slow_planner_frontend.contracts import snapshot_content_sha256
from slow_planner_frontend.state import (
    FilesystemStateSource,
    FrontendStateStore,
    FrontendConfig,
    sanitize_public,
)


VIEWS = ("front_left", "front", "front_right", "rear")


def snapshot_record(camera_dir):
    record = {
        "schema_version": 1,
        "kind": "lane_b_rev_c_snapshot",
        "lane_id": "b",
        "episode_id": "episode-3",
        "reset_id": 1,
        "sequence_id": 9,
        "snapshot_id": "b::episode-3::1::9",
        "snapshot_sim_stamp_s": 12.0,
        "config_sha256": "a" * 64,
        "view_order": list(VIEWS),
        "images": [
            {
                "view_id": view,
                "source_frame_id": f"lane_b/{view}",
                "sim_stamp_s": 11.9,
                "age_s": 0.1,
                "width": 640,
                "height": 480,
                "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "extrinsic_sha256": "b" * 64,
                "jpeg_sha256": hashlib.sha256(
                    (camera_dir / f"{view}.jpg").read_bytes()
                ).hexdigest(),
                "jpeg_path": str(camera_dir / f"{view}.jpg"),
            }
            for view in VIEWS
        ],
    }
    record["snapshot_content_sha256"] = snapshot_content_sha256(record)
    return record


def test_sanitizer_recursively_removes_raw_generation_and_reasoning() -> None:
    public = sanitize_public(
        {
            "decision": "select_frontier",
            "raw_text": "secret",
            "nested": {
                "chain-of-thought": "secret",
                "reasoning": "secret",
                "fallback_reason": "timeout",
            },
        }
    )
    assert public == {
        "decision": "select_frontier",
        "nested": {"fallback_reason": "timeout"},
    }


def test_state_exposes_metadata_but_never_jpeg_or_private_text(tmp_path) -> None:
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    for view in VIEWS:
        (camera_dir / f"{view}.jpg").write_bytes(f"jpeg-{view}".encode())
    store = FrontendStateStore()
    store.publish_snapshot_record(snapshot_record(camera_dir), camera_dir)
    store.publish_decision(
        {
            "decision": {
                "episode_id": "episode-3",
                "snapshot_id": "b::episode-3::1::9",
                "intent": "frontier_advice",
                "confidence": 0.7,
                "scene_summary": "open hallway ahead",
                "target_evidence": ["white rug visible"],
                "blocked_directions": ["rear"],
                "recommended_frontier": 7,
                "target_found": False,
                "abstain": False,
                "raw_text": "private",
                "not_in_schema": "private-extra",
            },
            "metrics": {
                "end_to_end_ms": 123.0,
                "reasoning": "private",
                "not_in_schema": "private-extra",
            },
            "not_in_schema": "private-extra",
        }
    )
    store.publish_gpu(
        {
            "unified_memory_total_mib": 120_000.0,
            "unified_memory_available_mib": 80_000.0,
            "system_swap_used_mib": 0.0,
            "step3_process_rss_mib": 900.0,
            "step3_process_swap_mib": 0.0,
        }
    )
    payload = store.state_payload()
    serialized = json.dumps(payload)
    assert [row["view_id"] for row in payload["cameras"]] == list(VIEWS)
    assert all(row["available"] for row in payload["cameras"])
    assert "private" not in serialized
    assert "private-extra" not in serialized
    assert "raw_text" not in serialized
    assert "jpeg-front" not in serialized
    assert "jpeg_path" not in serialized
    assert str(camera_dir.resolve()) not in serialized
    assert store.camera("front").jpeg == b"jpeg-front"
    assert store.camera("../../secret") is None
    assert payload["decision"]["decision"]["scene_summary"] == "open hallway ahead"
    assert payload["decision"]["decision"]["recommended_frontier"] == 7
    assert payload["gpu"]["unified_memory_available_mib"] == 80_000.0


def test_state_hides_decision_until_snapshot_identity_matches(tmp_path) -> None:
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    for view in VIEWS:
        (camera_dir / f"{view}.jpg").write_bytes(f"jpeg-{view}".encode())
    store = FrontendStateStore()
    store.publish_snapshot_record(snapshot_record(camera_dir), camera_dir)
    store.publish_decision(
        {
            "decision": {
                "episode_id": "other-episode",
                "snapshot_id": "b::other-episode::0::1",
                "intent": "frontier_advice",
            },
            "metrics": {"end_to_end_ms": 10.0},
        }
    )
    mismatched = store.state_payload()
    assert mismatched["decision"] == {}
    assert mismatched["latency"] == {}
    assert "decision_snapshot_identity_mismatch" in mismatched["ingest_warnings"]

    store.publish_decision(
        {
            "decision": {
                "episode_id": "episode-3",
                "snapshot_id": "b::episode-3::1::9",
                "intent": "frontier_advice",
            },
            "metrics": {"end_to_end_ms": 11.0},
        }
    )
    matched = store.state_payload()
    assert matched["decision"]["decision"]["snapshot_id"] == "b::episode-3::1::9"
    assert matched["latency"]["end_to_end_ms"] == 11.0
    assert "decision_snapshot_identity_mismatch" not in matched["ingest_warnings"]


def test_snapshot_record_rejects_camera_path_outside_configured_root(tmp_path) -> None:
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    for view in VIEWS:
        (camera_dir / f"{view}.jpg").write_bytes(b"jpeg")
    record = snapshot_record(camera_dir)
    record["images"][0]["jpeg_path"] = str(tmp_path / "secret.jpg")
    store = FrontendStateStore()
    with pytest.raises(ValueError, match="escapes"):
        store.publish_snapshot_record(record, camera_dir)


def test_snapshot_record_rejects_wrong_kind_identity_dimensions_and_hashes(
    tmp_path,
) -> None:
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    for view in VIEWS:
        (camera_dir / f"{view}.jpg").write_bytes(f"jpeg-{view}".encode())
    mutations = (
        (lambda record: record.update(kind="wrong"), "record kind"),
        (lambda record: record.update(reset_id=2), "identity fields"),
        (lambda record: record["images"][0].update(width=1), "must be 640x480"),
        (
            lambda record: record["images"][0].update(extrinsic_sha256="not-a-hash"),
            "SHA-256",
        ),
    )
    for mutate, message in mutations:
        record = snapshot_record(camera_dir)
        mutate(record)
        with pytest.raises(ValueError, match=message):
            FrontendStateStore().publish_snapshot_record(record, camera_dir)


def test_snapshot_record_rejects_tampered_jpeg_and_content_hash(tmp_path) -> None:
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    for view in VIEWS:
        (camera_dir / f"{view}.jpg").write_bytes(f"jpeg-{view}".encode())
    record = snapshot_record(camera_dir)
    (camera_dir / "front.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="JPEG hash mismatch"):
        FrontendStateStore().publish_snapshot_record(record, camera_dir)

    (camera_dir / "front.jpg").write_bytes(b"jpeg-front")
    record = snapshot_record(camera_dir)
    record["snapshot_content_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="content hash"):
        FrontendStateStore().publish_snapshot_record(record, camera_dir)


def test_filesystem_source_tails_read_only_artifacts(tmp_path, monkeypatch) -> None:
    root = tmp_path / "runtime"
    camera_dir = root / "cameras"
    camera_dir.mkdir(parents=True)
    for view in VIEWS:
        (camera_dir / f"{view}.jpg").write_bytes(b"jpeg")
    sidecar = root / "snapshots.jsonl"
    sidecar.write_text(json.dumps(snapshot_record(camera_dir)) + "\n", encoding="utf-8")
    decision = root / "decisions.jsonl"
    decision.write_text(
        json.dumps(
            {
                "decision": {
                    "episode_id": "episode-3",
                    "snapshot_id": "b::episode-3::1::9",
                    "intent": "INTERNVLA_FALLBACK_REQUIRED",
                    "fallback_used": True,
                    "fallback_reason": "step3_internal_fallback",
                    "requires_internvla_fallback": True,
                    "fallback_owner": "coordinator_frozen_internvla_candidate",
                    "motion_authority": "none",
                    "raw_text": "private",
                },
                "metrics": {"end_to_end_ms": 7.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    health = root / "health.json"
    health.write_text(
        json.dumps({"ready": True, "model_variant": "step3_vl_10b_bf16"}),
        encoding="utf-8",
    )
    gpu = root / "gpu.json"
    gpu.write_text(json.dumps({"utilization_percent": 55}), encoding="utf-8")
    monkeypatch.setenv("LANE_B_RUNTIME", str(root))
    config = FrontendConfig.from_mapping(
        {
            "frontend": {
                "host": "127.0.0.1",
                "port": 8300,
                "poll_interval_s": 0.25,
                "paths": {
                    "snapshot_sidecar_path": "${LANE_B_RUNTIME}/snapshots.jsonl",
                    "decision_log_path": "${LANE_B_RUNTIME}/decisions.jsonl",
                    "health_path": "${LANE_B_RUNTIME}/health.json",
                    "gpu_telemetry_path": "${LANE_B_RUNTIME}/gpu.json",
                    "camera_dir": "${LANE_B_RUNTIME}/cameras",
                },
            }
        }
    )
    store = FrontendStateStore()
    FilesystemStateSource(config).refresh(store)
    payload = store.state_payload()
    assert payload["health"]["ready"] is True
    assert payload["gpu"]["utilization_percent"] == 55
    assert payload["decision"]["decision"]["intent"] == "INTERNVLA_FALLBACK_REQUIRED"
    assert payload["latency"]["end_to_end_ms"] == 7.0
    assert not payload["ingest_warnings"]


def test_store_exposes_allowlisted_ros_state_and_cameras(tmp_path) -> None:
    camera_dir = tmp_path / "ros" / "cameras"
    camera_dir.mkdir(parents=True)
    jpeg = b"\xff\xd8panel-preview\xff\xd9"
    jpeg_path = camera_dir / "d435_color.jpg"
    jpeg_path.write_bytes(jpeg)
    usb_jpeg = b"\xff\xd8usb-preview\xff\xd9"
    usb_jpeg_path = camera_dir / "front_left.jpg"
    usb_jpeg_path.write_bytes(usb_jpeg)
    store = FrontendStateStore()
    store.publish_ros_state(
        {
            "schema_version": 1,
            "kind": "vla_nav_panel_ros_state",
            "ready": True,
            "status": "live",
            "updated_wall_time_s": 123.0,
            "node": {"name": "panel", "domain_id": 0, "secret": "drop"},
            "topics": {
                "low_state": {
                    "topic": "/lowstate",
                    "received": True,
                    "count": 3,
                },
                "unknown": {"topic": "/private"},
            },
            "robot": {
                "battery": {"soc": 91, "raw_text": "private"},
                "motion": {"velocity": [0.0, 0.0, 0.0]},
            },
        }
    )
    store.publish_ros_camera_manifest(
        {
            "schema_version": 1,
            "kind": "vla_nav_panel_ros_cameras",
            "cameras": [
                {
                    "view_id": "front_left",
                    "source_topic": "/dev/v4l/by-path/camera-front-left",
                    "stamp_s": 9.5,
                    "received_wall_time_s": 123.0,
                    "width": 640,
                    "height": 480,
                    "encoding": "v4l2-mjpeg",
                    "jpeg_path": str(usb_jpeg_path),
                    "jpeg_sha256": hashlib.sha256(usb_jpeg).hexdigest(),
                },
                {
                    "view_id": "d435_color",
                    "source_topic": "/check/d435/color/image_raw",
                    "stamp_s": 10.0,
                    "received_wall_time_s": 123.0,
                    "width": 640,
                    "height": 480,
                    "encoding": "rgb8",
                    "jpeg_path": str(jpeg_path),
                    "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                }
            ],
        },
        camera_dir,
    )
    payload = store.state_payload()
    assert payload["ros"]["ready"] is True
    assert payload["robot"]["battery"] == {"soc": 91}
    assert "unknown" not in payload["ros"]["topics"]
    assert "secret" not in payload["ros"]["node"]
    assert store.camera("d435_color").jpeg == jpeg
    assert store.camera("front_left").jpeg == usb_jpeg
    assert next(
        row for row in payload["cameras"] if row["view_id"] == "front_left"
    )["source_topic"].startswith("/dev/v4l/")
    assert next(
        row for row in payload["live_cameras"] if row["view_id"] == "d435_color"
    )["available"] is True


def test_ros_camera_manifest_rejects_escape_and_write_authority(tmp_path) -> None:
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    outside = tmp_path / "d435_color.jpg"
    outside.write_bytes(b"jpeg")
    manifest = {
        "schema_version": 1,
        "kind": "vla_nav_panel_ros_cameras",
        "cameras": [
            {
                "view_id": "d435_color",
                "source_topic": "/camera",
                "stamp_s": 1.0,
                "received_wall_time_s": 1.0,
                "width": 1,
                "height": 1,
                "encoding": "rgb8",
                "jpeg_path": str(outside),
                "jpeg_sha256": hashlib.sha256(b"jpeg").hexdigest(),
            }
        ],
    }
    with pytest.raises(ValueError, match="escapes"):
        FrontendStateStore().publish_ros_camera_manifest(manifest, camera_dir)
    with pytest.raises(ValueError, match="unknown ROS state kind"):
        FrontendStateStore().publish_ros_state(
            {
                "schema_version": 1,
                "kind": "unknown_ros_state",
            }
        )


def test_decision_ingestion_rejects_unknown_schema_and_unstructured_reason() -> None:
    store = FrontendStateStore()
    with pytest.raises(ValueError, match="unknown planner intent"):
        store.publish_decision({"decision": {"intent": "safe_hold"}})
    with pytest.raises(ValueError, match="unstructured fallback reason"):
        store.publish_decision(
            {
                "decision": {
                    "intent": "INTERNVLA_FALLBACK_REQUIRED",
                    "fallback_reason": "private chain of thought",
                }
            }
        )


def test_standalone_config_selects_lan_8300_and_ros_paths(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("T5_LANE_B_RESULTS", str(tmp_path / "results"))
    value = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "configs/frontend.yaml"
        ).read_text(encoding="utf-8")
    )
    config = FrontendConfig.from_mapping(value)
    assert config.host == "0.0.0.0"
    assert config.port == 8300
    assert config.snapshot_sidecar_path.name == "snapshots.jsonl"
    assert config.camera_dir.name == "cameras"
    assert config.ros_state_path.name == "ros_state.json"
    assert config.ros_camera_manifest_path.name == "ros_cameras.json"
    assert config.ros_camera_dir.name == "cameras"
    assert value["frontend"]["ros_adapter"]["preview_hz"] == 1.0
    assert value["frontend"]["ros_adapter"]["go2_camera_source"] == "videohub_rpc"
    assert value["frontend"]["ros_adapter"]["usb_cameras"]["enabled"] is True
    assert value["frontend"]["forbidden_http_methods"] == [
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    ]
    assert tuple(value["contract"]["rev_c_view_order"]) == VIEWS
    assert value["contract"]["motion_authority"] == "none"
