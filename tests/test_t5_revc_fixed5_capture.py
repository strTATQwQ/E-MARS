from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import probe_t5_revc_fixed5_capture as fixed5


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_t5_distributed_isaac.sh"
COORDINATOR = ROOT / "coordination" / "run_t5_fast_lane_online.sh"
CONTRACT = ROOT / "configs" / "internnav_t5" / "revc_four_camera_snapshot.json"
EPISODES = ["628", "259", "676", "1339", "121"]


def _ack(status: str, episode: str, reset: int = 0, sequence: int = 0) -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "request_id": "b::request",
        "episode_id": f"b::{episode}",
        "reset_generation": reset,
        "sequence_id": sequence,
    }


def test_retry_classification_waits_syncs_and_fails_if_target_was_missed() -> None:
    request = {
        "episode_id": "b::676",
        "reset_generation": 0,
        "sequence_id": 0,
    }
    assert fixed5.classify_retry(
        request, _ack("IDENTITY_MISMATCH", "259", 0, 4), EPISODES, 2
    ) == ("wait_for_episode", 0, 0)
    assert fixed5.classify_retry(
        request, _ack("IDENTITY_MISMATCH", "676", 1, 7), EPISODES, 2
    ) == ("sync_identity", 1, 7)
    synced = {**request, "reset_generation": 1, "sequence_id": 7}
    assert fixed5.classify_retry(
        synced, _ack("RATE_LIMITED", "676", 1, 7), EPISODES, 2
    ) == ("rate_limited", 1, 7)
    render_realign = _ack("WARN_CAPTURE_FAILED", "676", 1, 7)
    render_realign["error"] = (
        "Rev-C sensor t5_revc_front_left ReferenceTime changed at paused "
        "render barrier"
    )
    assert fixed5.classify_retry(
        synced, render_realign, EPISODES, 2
    ) == ("render_realign", 1, 7)
    render_realign["error"] = "invalid Rev-C frame shape"
    with pytest.raises(fixed5.Fixed5CaptureError, match="not CAPTURED"):
        fixed5.classify_retry(synced, render_realign, EPISODES, 2)
    with pytest.raises(fixed5.Fixed5CaptureError, match="advanced past"):
        fixed5.classify_retry(
            request, _ack("IDENTITY_MISMATCH", "1339"), EPISODES, 2
        )
    with pytest.raises(fixed5.Fixed5CaptureError, match="outside"):
        fixed5.classify_retry(
            request, _ack("IDENTITY_MISMATCH", "999"), EPISODES, 2
        )


def test_run_captures_five_natural_unique_episode_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = tmp_path / "order.json"
    order.write_text(json.dumps({
        "schema_version": 1,
        "status": "PASS",
        "dataset_episode_count": 5,
        "ordered_episode_ids": EPISODES,
    }), encoding="utf-8")
    request_path = tmp_path / "revc_snapshot.request.json"
    ack_path = tmp_path / "revc_snapshot.ack.json"
    realigned = False

    def fake_wait(path: Path, request_id: str, timeout_sec: float) -> dict:
        nonlocal realigned
        assert path == ack_path and timeout_sec > 0
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["request_id"] == request_id
        request_path.unlink()
        if request["episode_id"] == "b::259" and not realigned:
            realigned = True
            ack = {
                "schema_version": 1,
                "status": "WARN_CAPTURE_FAILED",
                **{key: request[key] for key in (
                    "request_id", "episode_id", "reset_generation", "sequence_id"
                )},
                "error": (
                    "Rev-C sensor t5_revc_front_left ReferenceTime changed at "
                    "paused render barrier"
                ),
            }
            ack_path.write_text(json.dumps(ack), encoding="utf-8")
            return ack
        snapshot_root = tmp_path / "revc_snapshots"
        count = len(list(snapshot_root.glob("*"))) if snapshot_root.exists() else 0
        snapshot_dir = snapshot_root / f"snapshot-{count}"
        snapshot_dir.mkdir(parents=True)
        sidecar = snapshot_dir / "snapshot.json"
        sidecar.write_text("{}", encoding="utf-8")
        ack = {
            "schema_version": 1,
            "status": "CAPTURED",
            **{key: request[key] for key in (
                "request_id", "episode_id", "reset_generation", "sequence_id"
            )},
            "sidecar": sidecar.relative_to(tmp_path).as_posix(),
        }
        ack_path.write_text(json.dumps(ack), encoding="utf-8")
        return ack

    def fake_validate(
        result_root: Path, contract: Path, request: dict, ack: dict, *,
        expected_sidecar_count: int, profile: str,
    ) -> dict:
        assert result_root == tmp_path.resolve()
        assert contract == CONTRACT
        assert expected_sidecar_count == len(list(
            (tmp_path / "revc_snapshots").glob("*/snapshot.json")
        ))
        assert profile == fixed5.PROFILE
        return {
            "schema_version": 1,
            "status": "PASS",
            "profile": profile,
            "lane": "b",
            "request": request,
            "sidecar": ack["sidecar"],
            "contract_sha256": "a" * 64,
            "same_render_tick": True,
            "camera_order": list(fixed5.smoke.CAMERA_ORDER),
            "images": [{"identity": item} for item in fixed5.smoke.CAMERA_ORDER],
        }

    monkeypatch.setattr(fixed5.smoke, "wait_for_matching_ack", fake_wait)
    monkeypatch.setattr(fixed5.smoke, "validate_capture", fake_validate)
    result = fixed5.run(argparse.Namespace(
        result_root=tmp_path,
        order_manifest=order,
        contract=CONTRACT,
        request=request_path,
        ack=ack_path,
        output=tmp_path / "revc_fixed5_capture.json",
        run_token="t5-fixed5-test",
        timeout_sec=30.0,
        episode_poll_sec=0.25,
        retry_interval_sec=1.0,
    ))
    assert result["status"] == "PASS"
    assert result["capture_count"] == 5
    assert result["ordered_episode_ids"] == EPISODES
    assert result["unique_execution_identities"] is True
    assert realigned is True
    assert result["request_attempt_count"] == 6
    assert any(
        item.get("retry_disposition") == "render_realign"
        for item in result["request_attempts"]
    )
    assert [item["request"]["episode_id"] for item in result["snapshots"]] == [
        f"b::{item}" for item in EPISODES
    ]
    assert len(list((tmp_path / "revc_snapshots").glob("*/snapshot.json"))) == 5


def test_fixed5_profile_is_lane_b_gpu1_odd_cpu_and_does_not_replace_smoke() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    assert "lane_b_revc_smoke" in runner
    assert "lane_b_revc_fixed5_capture" in runner
    assert 'test "$execution_profile" = fixed_dataset' in runner
    assert "probe_t5_revc_fixed5_capture.py" in runner
    assert "gpu=1" in runner
    assert 'test "$cpuset" = 1,3,5,7,9,11,13,15,17' in runner
    assert "lane_b_revc_fixed5_capture" in coordinator
    assert 'test "$profile" = fixed5 || usage' in coordinator
    assert "remote/x86/evaluator/revc_fixed5_capture.json" in coordinator
    assert "collect_revc_fixed5_snapshots()" in coordinator
    assert 'test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture || return 0' in coordinator
    assert "-name snapshot.json -o -name '*.png'" in coordinator
    assert '"snapshot_sidecar_count_five": len(sidecars) == 5' in coordinator
    assert '"png_count_twenty": len(pngs) == 20' in coordinator
    assert '"sidecar_references_exact_png_set": set(referenced) == set(pngs)' in coordinator
    assert '"capture_summary_exact_binding"' in coordinator
    assert 'capture_row.get("sidecar_sha256")' in coordinator
    assert 'capture_row.get("images") != image_bindings' in coordinator
    assert "image.is_relative_to(root)" not in coordinator
    assert "image.relative_to(root)" in coordinator
    assert "collect_revc_fixed5_snapshots || incoming=1" in coordinator
    assert '"revc_fixed5_local_materialization_ready"' in coordinator
    assert 'ordered_episode_manifest=load("remote/x86/ordered_episode_manifest.json")' in coordinator
    assert '"natural_episode_order_binding"' in coordinator
    ready = runner.index("write_health_state READY")
    natural_wait = runner.rindex('wait "$gate_pid"')
    fixed5_wait = runner.rindex("  wait_revc_fixed5_after_evaluator")
    assert ready < natural_wait < fixed5_wait
    smoke_guard = runner.index(
        'if test "$isaac_sensor_profile" = lane_b_revc_smoke; then',
        runner.index('setsid python3 -u "$root/scripts/probe_t5_revc_snapshot_smoke.py"'),
    )
    assert smoke_guard < ready
    assert 'INTERNNAV_T5_REVC_FIXED5_TIMEOUT_SEC:-24000' in runner
    assert 'Rev-C fixed-five probe failed before the evaluator completed' in runner
