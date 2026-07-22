from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import run_t5_step3_frozen_fixed5 as replay
from slow_planner.base import PlannerMetrics, StructuredPlannerDecision
from slow_planner.lane_b import LaneBPlannerMode, snapshot_content_sha256
from slow_planner_frontend.state import FrontendStateStore


VIEWS = ("front_left", "front", "front_right", "rear")
CONFIG_SHA = "a" * 64
EXTRINSICS = {view: hashlib.sha256(view.encode()).hexdigest() for view in VIEWS}


def _write_bundle(root: Path) -> tuple[Path, str]:
    cameras = root / "cameras"
    records = []
    cases = []
    for index in range(5):
        episode = f"episode-{index}"
        snapshot_id = f"b::{episode}::0::0"
        camera_dir = cameras / episode / "reset-0" / "sequence-0"
        camera_dir.mkdir(parents=True)
        rows = []
        for view_index, view in enumerate(VIEWS):
            path = camera_dir / f"{view}.jpg"
            Image.new(
                "RGB", (640, 480), (index * 20, view_index * 20, 32)
            ).save(path, "JPEG")
            payload = path.read_bytes()
            rows.append(
                {
                    "view_id": view,
                    "source_frame_id": f"lane_b/{view}",
                    "sim_stamp_s": 10.0,
                    "age_s": 0.0,
                    "width": 640,
                    "height": 480,
                    "pose": [0.0, 0.0, 0.2, 0.0, 0.0],
                    "extrinsic_sha256": EXTRINSICS[view],
                    "jpeg_sha256": hashlib.sha256(payload).hexdigest(),
                    "jpeg_path": f"ignored-source/{view}.jpg",
                }
            )
        record = {
            "schema_version": 1,
            "kind": "lane_b_rev_c_snapshot",
            "lane_id": "b",
            "episode_id": episode,
            "reset_id": 0,
            "sequence_id": 0,
            "snapshot_id": snapshot_id,
            "snapshot_sim_stamp_s": 10.0,
            "written_wall_time_s": 20.0,
            "config_sha256": CONFIG_SHA,
            "view_order": list(VIEWS),
            "inter_camera_skew_s": 0.0,
            "max_frame_age_s": 0.5,
            "max_inter_camera_skew_s": 0.2,
            "images": rows,
        }
        record["snapshot_content_sha256"] = snapshot_content_sha256(record)
        records.append(record)
        cases.append(
            {
                "case_id": f"case-{index}",
                "episode_key": f"key-{index}",
                "snapshot_id": snapshot_id,
                "instruction": "Go to the visible doorway.",
                "candidate_frontiers": [
                    {
                        "frontier_id": 0,
                        "relative_xz": [0.0, 1.0],
                        "distance_m": 1.0,
                        "bearing_deg": 0.0,
                    }
                ],
                "agent_pose": [0.0, 0.0, 0.0, 0.0],
                "visited_frontiers": [],
                "compact_history": [],
            }
        )
    sidecar = root / "snapshots.jsonl"
    sidecar.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "kind": "t5_lane_b_step3_frozen_fixed5",
        "frozen": True,
        "evaluation_scope": "interface_screening_only",
        "candidate_frontier_source":
            "deterministic_frozen_replay_triplet_not_live_nav2",
        "agent_pose_encoding":
            "dataset_start_position_xyz_plus_rotation_xyzw",
        "episode_count": 5,
        "snapshot_sidecar": "snapshots.jsonl",
        "snapshot_sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "camera_root": "cameras",
        "expected_config_sha256": CONFIG_SHA,
        "expected_extrinsic_sha256": EXTRINSICS,
        "cases": cases,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mode,expected_intent",
    [
        (LaneBPlannerMode.BOUNDED_ADVISOR, "frontier_advice"),
        (LaneBPlannerMode.DIRECT_HIGH_LEVEL, "frontier_goal_candidate"),
    ],
)
def test_fixed5_replay_publishes_frontend_sidecars_without_control(
    tmp_path, monkeypatch, mode, expected_intent
) -> None:
    manifest, manifest_sha = _write_bundle(tmp_path / "bundle")

    class Client:
        def __init__(self, endpoint, *, timeout_ms):
            assert endpoint == "tcp://127.0.0.1:8200"
            assert timeout_ms == 12_000

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decide(self, request):
            return (
                StructuredPlannerDecision(
                    episode_id=request.episode_id,
                    snapshot_id=request.snapshot_id,
                    decision="select_frontier",
                    frontier_id=0,
                    target_relative_xz=None,
                    confidence=0.8,
                    raw_text="private chain of thought",
                    scene_summary="open doorway ahead",
                    target_evidence=("doorway visible",),
                    blocked_directions=("rear",),
                    recommended_frontier=0,
                    target_found=False,
                    abstain=False,
                ),
                PlannerMetrics(end_to_end_ms=4.0, peak_memory_mib=20_000.0),
            )

    monkeypatch.setattr(replay, "SlowPlannerClient", Client)
    runtime = tmp_path / "runtime" / "step3"
    output = tmp_path / "fixed5"
    summary = replay.run_fixed5(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        endpoint="tcp://127.0.0.1:8200",
        mode=mode,
        runtime_step3_dir=runtime,
        output_dir=output,
    )
    assert summary["status"] == "PASS"
    assert summary["promotion_eligible"] is True
    assert summary["model_interface_eligible"] is True
    assert summary["navigation_effect_claim_eligible"] is False
    assert summary["evaluation_scope"] == "interface_screening_only"
    assert summary["case_count"] == 5
    assert summary["offline_acceptance_gate_pass"] is True
    assert summary["structured_json_parse_rate"] == 1.0
    assert summary["legal_frontier_or_explicit_abstain_rate"] == 1.0
    assert summary["nonfallback_frontier_selection_count"] == 5
    result_text = (output / "results.jsonl").read_text(encoding="utf-8")
    assert len(result_text.splitlines()) == 5
    assert "raw_text" not in result_text
    assert "chain of thought" not in result_text
    decision = json.loads(
        (runtime / "frontend_decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert decision["decision"]["intent"] == expected_intent
    assert decision["decision"]["motion_authority"] == "none"
    assert decision["decision"]["scene_summary"] == "open doorway ahead"
    assert decision["decision"]["recommended_frontier"] == 0

    snapshot = json.loads(
        (runtime / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    store = FrontendStateStore()
    store.publish_snapshot_record(snapshot, runtime / "cameras")
    store.publish_decision(decision)
    state = store.state_payload()
    assert all(camera["available"] for camera in state["cameras"])
    assert state["decision"]["decision"]["intent"] == expected_intent


def test_fixed5_replay_records_deterministic_fallback_without_private_error(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_sha = _write_bundle(tmp_path / "bundle")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decide(self, request):
            raise RuntimeError("secret remote traceback")

    monkeypatch.setattr(replay, "SlowPlannerClient", Client)
    summary = replay.run_fixed5(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        endpoint="tcp://127.0.0.1:8200",
        mode=LaneBPlannerMode.BOUNDED_ADVISOR,
        runtime_step3_dir=tmp_path / "runtime" / "step3",
        output_dir=tmp_path / "fixed5",
    )
    assert summary["status"] == "FAIL"
    assert summary["promotion_eligible"] is False
    assert summary["offline_acceptance_gate_pass"] is False
    assert summary["internvla_fallback_required_count"] == 5
    public = (tmp_path / "fixed5" / "results.jsonl").read_text(encoding="utf-8")
    assert "secret remote traceback" not in public
    assert "INTERNVLA_FALLBACK_REQUIRED" in public


def test_direct_fixed5_failure_requires_safe_stop_without_internvla_fallback(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_sha = _write_bundle(tmp_path / "bundle")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decide(self, request):
            raise RuntimeError("private failure")

    monkeypatch.setattr(replay, "SlowPlannerClient", Client)
    summary = replay.run_fixed5(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        endpoint="tcp://127.0.0.1:8200",
        mode=LaneBPlannerMode.DIRECT_HIGH_LEVEL,
        runtime_step3_dir=tmp_path / "runtime/step3",
        output_dir=tmp_path / "fixed5",
    )
    assert summary["status"] == "FAIL"
    assert summary["direct_safe_stop_required_count"] == 5
    assert summary["internvla_fallback_required_count"] == 0
    public = (tmp_path / "fixed5/results.jsonl").read_text(encoding="utf-8")
    assert "DIRECT_SAFE_STOP_REQUIRED" in public
    assert "INTERNVLA_FALLBACK_REQUIRED" not in public
    assert "private failure" not in public


def test_fixed5_replay_fails_closed_on_manifest_or_jpeg_drift(tmp_path) -> None:
    manifest, manifest_sha = _write_bundle(tmp_path / "bundle")
    with pytest.raises(replay.FrozenReplayError, match="manifest SHA"):
        replay.run_fixed5(
            manifest_path=manifest,
            manifest_sha256="0" * 64,
            endpoint="tcp://127.0.0.1:8200",
            mode=LaneBPlannerMode.BOUNDED_ADVISOR,
            runtime_step3_dir=tmp_path / "runtime-a",
            output_dir=tmp_path / "output-a",
        )

    camera = tmp_path / "bundle" / "cameras" / "episode-0" / "reset-0" / "sequence-0" / "front.jpg"
    camera.write_bytes(b"tampered")
    with pytest.raises(replay.FrozenReplayError, match="JPEG hash mismatch"):
        replay.run_fixed5(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            endpoint="tcp://127.0.0.1:8200",
            mode=LaneBPlannerMode.BOUNDED_ADVISOR,
            runtime_step3_dir=tmp_path / "runtime-b",
            output_dir=tmp_path / "output-b",
        )


def test_frozen_fixed5_shell_entry_reuses_shadow_and_has_no_control_surface() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_t5_step3_frozen_fixed5.sh"
    ).read_text(encoding="utf-8")
    assert "run_t5_step3_shadow.sh" in text
    assert "tcp://127.0.0.1:8200" in text
    assert "http://127.0.0.1:8300/api/v1/health" in text
    assert "http://127.0.0.1:8300/api/v1/state" in text
    assert "INTERNNAV_T5_RESOURCE_LEASE_ACK" in text
    assert "cmd_vel" not in text
    assert "terminal STOP" not in text
    assert '"motion_authority":"none"' in text
