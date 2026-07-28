from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import zlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_t5_revc_snapshot_smoke.py"
RUNNER = ROOT / "scripts" / "run_t5_distributed_isaac.sh"
CONTRACT = ROOT / "configs" / "internnav_t5" / "revc_four_camera_snapshot.json"
ORDER = ("front_left", "front", "front_right", "rear")

spec = importlib.util.spec_from_file_location("probe_t5_revc_snapshot_smoke", SCRIPT)
assert spec and spec.loader
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png() -> bytes:
    scanlines = b"".join(b"\0" + b"\0" * (640 * 3) for _ in range(480))
    return (
        smoke.PNG_SIGNATURE
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(scanlines, 9))
        + _chunk(b"IEND", b"")
    )


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    request = {
        "schema_version": 1,
        "request_id": "b::revc-smoke::fixture::1",
        "episode_id": "b::628",
        "reset_generation": 0,
        "sequence_id": 0,
    }
    snapshot_dir = tmp_path / "revc_snapshots" / "snapshot-1"
    snapshot_dir.mkdir(parents=True)
    render_identity = {
        "referenceTimeNumerator": 31,
        "referenceTimeDenominator": 60,
        "rendering_time": 31 / 60,
    }
    cameras = []
    for index, identity in enumerate(ORDER):
        expected_camera = contract["cameras"][index]
        path = snapshot_dir / f"{index:02d}_{identity}.png"
        path.write_bytes(_png())
        cameras.append(
            {
                "identity": identity,
                "order_index": index,
                "sensor_name": expected_camera["sensor_name"],
                "frame_id": expected_camera["frame_id"],
                "prim_path": expected_camera["prim_path"],
                "position_F_M_mm": expected_camera["position_F_M_mm"],
                "yaw_deg": expected_camera["yaw_deg"],
                "pitch_down_deg": contract["optics"]["pitch_down_deg"],
                "hfov_deg": contract["optics"]["hfov_deg"],
                "resolution": [640, 480],
                "encoding": "png_rgb8",
                "render_identity": render_identity,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    sidecar = {
        "schema_version": 1,
        "contract_id": "internnav-t5-revc-four-camera-v1",
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "scope": "completion_sim_only",
        **{key: request[key] for key in (
            "request_id", "episode_id", "reset_generation", "sequence_id"
        )},
        "render_barrier_id": "31000000:31/60",
        "render_identity_source": (
            "replicator_annotator:ReferenceTime+SimulationManager"
        ),
        "render_identity_value": render_identity,
        "sim_stamp_before_ns": 31_000_000,
        "sim_stamp_after_ns": 31_000_000,
        "same_render_tick": True,
        "camera_order": list(ORDER),
        "external_preview_max_hz": 1.0,
        "cuvslam_stereo_is_separate": True,
        "cameras": cameras,
    }
    sidecar_path = snapshot_dir / "snapshot.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    ack = {
        "schema_version": 1,
        "status": "CAPTURED",
        **{key: request[key] for key in (
            "request_id", "episode_id", "reset_generation", "sequence_id"
        )},
        "render_barrier_id": sidecar["render_barrier_id"],
        "render_identity_source": sidecar["render_identity_source"],
        "render_identity_value": render_identity,
        "sidecar": sidecar_path.relative_to(tmp_path).as_posix(),
    }
    return request, ack


def test_capture_validator_binds_same_tick_four_images_and_preview_rate(
    tmp_path: Path,
) -> None:
    request, ack = _fixture(tmp_path)
    result = smoke.validate_capture(tmp_path, CONTRACT, request, ack)
    assert result["status"] == "PASS"
    assert result["same_render_tick"] is True
    assert result["camera_order"] == list(ORDER)
    assert result["external_preview_max_hz"] == 1.0
    assert len(result["images"]) == 4


@pytest.mark.parametrize("mutation", ("preview", "render_identity", "cross_root"))
def test_capture_validator_fails_closed_on_scope_or_render_drift(
    tmp_path: Path, mutation: str,
) -> None:
    request, ack = _fixture(tmp_path)
    sidecar_path = tmp_path / ack["sidecar"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if mutation == "preview":
        sidecar["external_preview_max_hz"] = 1.01
    elif mutation == "render_identity":
        sidecar["cameras"][3]["render_identity"]["referenceTimeNumerator"] = 32
    else:
        outside = tmp_path.parent / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        ack["sidecar"] = "../outside.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(smoke.SmokeContractError):
        smoke.validate_capture(tmp_path, CONTRACT, request, ack)


def test_request_uses_first_materialized_episode_and_lane_b_identity() -> None:
    request = smoke.make_request(
        {
            "ordered_episode_ids": ["628", "259", "676", "1339", "121"],
        },
        "run-1",
    )
    assert request["request_id"].startswith("b::revc-smoke::run-1::")
    assert request["episode_id"] == "b::628"
    assert request["reset_generation"] == 0
    assert request["sequence_id"] == 0


def test_identity_mismatch_allows_only_bounded_same_episode_sequence_sync() -> None:
    request = {
        "episode_id": "b::628",
        "reset_generation": 0,
        "sequence_id": 0,
    }
    assert smoke._next_sequence_after_identity_mismatch(
        request,
        {
            "status": "IDENTITY_MISMATCH",
            "mismatched_field": "sequence_id",
            "episode_id": "b::628",
            "reset_generation": 0,
            "sequence_id": 2,
        },
    ) == 2
    for mutation in (
        {"episode_id": "b::259"},
        {"reset_generation": 1},
        {"sequence_id": 0},
        {"mismatched_field": "episode_id"},
    ):
        ack = {
            "status": "IDENTITY_MISMATCH",
            "mismatched_field": "sequence_id",
            "episode_id": "b::628",
            "reset_generation": 0,
            "sequence_id": 2,
            **mutation,
        }
        with pytest.raises(smoke.SmokeContractError):
            smoke._next_sequence_after_identity_mismatch(request, ack)


def test_distributed_runner_requires_explicit_lane_b_canary_profile() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    profile = text.index("lane_b_revc_smoke)")
    probe = text.index('setsid python3 -u "$root/scripts/probe_t5_revc_snapshot_smoke.py"')
    gate = text.index('setsid bash "$root/scripts/run_t4_sensor_gate.sh"')
    ready = text.index("write_health_state READY")
    assert 'isaac_sensor_profile="${INTERNNAV_T5_ISAAC_SENSOR_PROFILE:-baseline}"' in text
    assert 'test "${INTERNVLA_T5_REVC_ENABLE:-0}" = 0' in text
    profile_block = text[profile : profile + 420]
    assert 'test "$lane" = b' in profile_block
    assert 'test "$mode" = model' in profile_block
    assert 'test "$engineering_canary_sec" = 60' in profile_block
    assert 'export INTERNVLA_T5_REVC_ENABLE=1' in profile_block
    assert 'case "$rtf_ablation_profile" in off|navigation_fast)' in text
    assert probe < gate
    assert '--run-token "$run_token"' in text
    assert 'if test "$isaac_sensor_profile" = lane_b_revc_smoke; then' in text
    assert text.index("wait_revc_snapshot_probe", probe) < ready
    assert gate < ready
    assert 'gpu=1' in text and 'test "$cpuset" = 1,3,5,7,9,11,13,15,17' in text


def test_fast_coordinator_propagates_and_seals_only_explicit_revc_profile() -> None:
    text = (
        ROOT / "coordination" / "run_t5_fast_lane_online.sh"
    ).read_text(encoding="utf-8")
    assert (
        'isaac_sensor_profile="${INTERNNAV_T5_ISAAC_SENSOR_PROFILE:-baseline}"'
        in text
    )
    assert 'test "$lane" = b || usage' in text
    assert 'test "$profile" = canary60 || usage' in text
    assert 'case "$rtf_ablation_profile" in navigation_fast|off)' in text
    assert 'INTERNNAV_T5_ISAAC_SENSOR_PROFILE="$isaac_sensor_profile"' in text
    assert 'isaac_sensor_profile="${16}"' in text
    assert "remote/x86/evaluator/revc_snapshot_smoke.json" in text
    assert '"revc_snapshot_smoke":revc_snapshot_seal' in text
