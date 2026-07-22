from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sensor_runtime.camera_contract import matrices_from_fov
from sensor_runtime.contract import DIAGNOSTIC_GEOMETRY, DIAGNOSTIC_LIGHT, FRAMES
from sensor_runtime.graph_handshake import (
    INNER_HANDSHAKE_ARTIFACTS,
    ROLE_OWNED_QOS_CONTRACT,
)
from sensor_runtime.isaac_eula import (
    FROZEN_ISAAC_EULA_POLICY,
    FROZEN_ISAAC_RUNTIME_PREFLIGHT,
    ISAAC_RUNTIME_PREFLIGHT_FILENAME,
)
from sensor_runtime.isaac_experience import (
    ISAAC_STARTUP_READY_FILENAME,
    build_startup_ready,
    frozen_policy_evidence,
)
from sensor_runtime.pythonpath_policy import FROZEN_SETUP_PYTHONPATH_SHA256
from sensor_runtime.validate import validate_result


PARTS = {
    "clock", "tf", "identity", "generation_topic", "rgb", "rgb_info", "depth",
    "depth_info", "depth_points", "lidar", "lidar_base", "front", "front_info",
    "imu", "safety", "odom",
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def _camera(width: int, height: int, frame: str, hfov: float, vfov: float) -> dict:
    return {
        "width": width, "height": height, "distortion_model": "plumb_bob",
        "frame_id": frame,
        **matrices_from_fov(width, height, hfov, vfov),
    }


def _build_result(root: Path) -> dict[str, list[dict]]:
    count, boundary = 1001, 600
    sensor: list[dict] = []
    controller: list[dict] = []
    bridge: list[dict] = []
    downstream: list[dict] = []
    for index in range(count):
        generation = 0 if index < boundary else 1
        sequence = index if generation == 0 else index - boundary
        stamp = (index + 1) * 50_000_000
        wall = 1_000_000_000 + index * 50_000_000
        sensor.append(
            {
                "wall_monotonic_ns": wall, "stamp_ns": stamp, "generation": generation,
                "sequence": sequence, "stream_stamps_ns": {name: stamp for name in ("d435i_rgb", "d435i_depth", "lidar", "pose", "tf")},
                "dynamic_link_count": 13, "self_filter_input_points": 640 * 480,
                "self_filter_output_points": 200000,
                "depth_cloud_reduction": "4x4_nearest_valid_after_full_resolution_filter",
                "depth_cloud_tile_stride": 4,
                "depth_cloud_width": 160,
                "depth_cloud_height": 120,
                "depth_cloud_point_count": 19200,
                "depth_cloud_output_points": 18000,
                "depth_valid_in_range_input_points": 250000,
                "depth_valid_in_range_ratio": 250000 / (640 * 480),
                "lidar_finite_input_points": 1200,
                "diagnostic_geometry_ids": [item["name"] for item in DIAGNOSTIC_GEOMETRY],
                "diagnostic_light_evidence": {**DIAGNOSTIC_LIGHT, "temperature_enabled": True},
                "d435i_rgb_content": {"finite": True, "valid": True, "minimum": 0, "maximum": 220, "dynamic_range": 220, "nonzero_count": 1000},
                "front_rgb_content": {"finite": True, "valid": True, "minimum": 1, "maximum": 180, "dynamic_range": 179, "nonzero_count": 900},
                "capture_timing": {"clock": "monotonic_perf_counter", "camera_read_content_sec": 0.01, "lidar_query_sec": 0.02, "total_capture_sec": 0.04, "lidar_raycast_count": 1440},
            }
        )
        controller.append(
            {
                "wall_monotonic_ns": wall, "generation": generation, "sequence": sequence,
                "identity": f"sensor-soak:{generation}:{sequence}", "linear_x": 0.0,
                "angular_z": 0.0, "emergency_stop": True,
                "applied_step_count": (index + 1) * 10, "steps_since_previous_capture": 10,
            }
        )
        bridge.append(
            {
                "wall_monotonic_ns": wall, "stamp_ns": stamp, "generation": generation,
                "sequence": sequence, "matched_parts": sorted({"clock", "tf", "identity", "centers", "lidar", "depth", "front", "front_info"}),
                "link_center_count": 13, "lidar_finite_input_points": 1200,
                "lidar_output_points": 1000, "safety_point_count": 201000,
            }
        )
        observed = set(PARTS)
        if sequence == 0:
            observed.add("tf_static")
        downstream.append(
            {
                "wall_monotonic_ns": wall, "stamp_ns": stamp, "generation": generation,
                "sequence": sequence, "render_id": index + 1, "observed_parts": sorted(observed),
                "same_stamp_observed": True, "static_tf_same_stamp_observed": sequence == 0,
                "tf_lookup": {"stamp_ns": stamp, "lookup_success_count": 6, "lookup_exception_count": 0},
                "generation_topic": {"observed_generation": generation},
                "camera_contracts": {
                    "rgb": _camera(640, 480, FRAMES["d435i_color"], 69.4, 42.5),
                    "depth": _camera(640, 480, FRAMES["d435i_depth"], 87.0, 58.0),
                    "front": _camera(160, 120, FRAMES["front_rgb"], 120.0, 75.0),
                },
                "image_contracts": {
                    "rgb": {"width": 640, "height": 480, "encoding": "rgb8"},
                    "depth": {"width": 640, "height": 480, "encoding": "32FC1"},
                    "front": {"width": 160, "height": 120, "encoding": "rgb8"},
                },
                "cloud_contracts": {
                    "depth_points": {"width": 160, "height": 120, "point_count": 19200, "finite_point_count": 18000},
                    **{
                        name: {"width": 10, "height": 10, "point_count": 100, "finite_point_count": 90}
                        for name in ("lidar", "lidar_base", "safety")
                    },
                },
            }
        )
    resets = [
        {"generation": 0, "sequence": 0, "reason": "initial", "reset_kind": "continuous_world_articulation_state", "atomic_latest_clear": True},
        {"generation": 1, "sequence": 0, "reason": "active_periodic", "reset_kind": "continuous_world_articulation_state", "atomic_latest_clear": True},
    ]
    identity = (1, count - boundary - 1)
    snapshot_id = "snapshot-1"
    owner_liveness = {
        "status": "HELD",
        "mechanism": "flock_exclusive_nonblocking",
        "path": str(root / "outer_alive.lock"),
        "owner_pid": 10,
        "device": 20,
        "inode": 30,
    }
    inner_liveness = {
        "status": "OUTER_ALIVE",
        "mechanism": "flock_exclusive_nonblocking",
        "path": str(root / "outer_alive.lock"),
        "device": 20,
        "inode": 30,
    }
    isaac_startup_ready = build_startup_ready(
        policy=frozen_policy_evidence(),
        simulation_app_elapsed_sec=11.0,
        extensions_ready_elapsed_sec=11.1,
    )
    _write_json(
        root / "session_manifest.json",
        {
            "outer_liveness": owner_liveness,
            "bind_mount_flock_verified": True,
            "inner_outer_liveness_probe": inner_liveness,
            "setup_pythonpath_sha256": FROZEN_SETUP_PYTHONPATH_SHA256,
            "setup_pythonpath_verified": True,
            "isaac_eula_policy": dict(FROZEN_ISAAC_EULA_POLICY),
            "isaac_runtime_preflight": dict(FROZEN_ISAAC_RUNTIME_PREFLIGHT),
            "isaac_runtime_preflight_verified": True,
            "isaac_startup_ready": isaac_startup_ready,
            "isaac_startup_ready_verified": True,
        },
    )
    _write_json(
        root / ISAAC_RUNTIME_PREFLIGHT_FILENAME,
        dict(FROZEN_ISAAC_RUNTIME_PREFLIGHT),
    )
    _write_json(root / ISAAC_STARTUP_READY_FILENAME, isaac_startup_ready)
    _write_json(
        root / "inner_supervisor_identity.json",
        {
            "outer_liveness": inner_liveness,
            "setup_pythonpath_sha256": FROZEN_SETUP_PYTHONPATH_SHA256,
        },
    )
    for name, (status, role) in INNER_HANDSHAKE_ARTIFACTS.items():
        handshake = {
            "schema_version": 2 if status == "ROLE_READY" else 1,
            "status": status,
            "role": role,
        }
        if status == "ROLE_READY":
            handshake["owned_qos"] = ROLE_OWNED_QOS_CONTRACT[role]
        else:
            handshake.update(
                {
                    "ready": True,
                    "owned_qos_proof_required": True,
                    "rmw_history_depth_unavailable_count": 1,
                }
            )
        _write_json(root / name, handshake)
    _write_json(root / "session_ready.json", {"status": "READY", "session_ready_elapsed_sec": 10.0, "sensor_record_count": 3, "controller_record_count": 3, "downstream_record_count": 3})
    _write_json(root / "workload_completion.json", {
        "status": "BOUNDED_DURATION_REACHED", "profile": "bootstrap", "snapshot_id": snapshot_id,
        "sensor_record_count": count, "controller_record_count": count, "reset_record_count": 2,
        "bridge_record_count": count, "downstream_record_count": count, "required_roles_alive": True,
        "ready_post_growth_count": count - 3, "frozen_prefix_reconciled": True,
    })
    _write_json(root / "producer_completion.json", {"navigation_evaluator_used": False, "elapsed_sec": 60.0, "active_reset_count": 1, "generation": identity[0]})
    _write_json(root / "diagnostic_geometry.json", {"status": "FROZEN", "objects": [dict(item) for item in DIAGNOSTIC_GEOMETRY], "light": dict(DIAGNOSTIC_LIGHT)})
    _write_json(root / "producer_snapshot_ack.json", {
        "status": "PASS", "generation": identity[0], "sequence": identity[1],
        "capture_count": count,
        "emitter": {
            "accepted_count": count, "sent_count": count, "overwrite_count": 0,
            "reset_clear_count": 0, "barrier_drop_count": 0,
            "last_submitted": list(identity), "last_sent": list(identity),
            "thread_alive": True, "fault": None,
        },
    })
    _write_json(root / "sidecar_snapshot_ack.json", {"status": "PASS", "snapshot_id": snapshot_id, "generation": identity[0], "sequence": identity[1], "server_received_count": count, "receive_overwrite_count": 0, "receive_reset_clear_count": 0, "receive_barrier_drop_count": 0, "writer_counts": {"sensor": count, "controller": count, "reset": 2}, "writer_thread_alive": True, "writer_fault": None})
    _write_json(root / "bridge_snapshot_ack.json", {"status": "PASS", "snapshot_id": snapshot_id, "generation": identity[0], "sequence": identity[1], "frame_count": count, "writer_thread_alive": True, "writer_fault": None})
    _write_json(root / "downstream_snapshot_ack.json", {"status": "PASS", "snapshot_id": snapshot_id, "generation": identity[0], "sequence": identity[1], "frame_count": count, "pending_count": 0, "generation_event_queue_count": 0, "tf_lookup_exception_count": 0, "recorder_fault": None})
    _write_jsonl(root / "sensor_producer_audit.jsonl", sensor)
    _write_jsonl(root / "controller_stop_audit.jsonl", controller)
    _write_jsonl(root / "reset_audit.jsonl", resets)
    _write_jsonl(root / "bridge/go2_sensor_bridge_frames.jsonl", bridge)
    _write_jsonl(root / "downstream/downstream_frames.jsonl", downstream)
    return {"sensor": sensor, "controller": controller, "bridge": bridge, "downstream": downstream}


def test_validator_accepts_reconciled_actual_downstream_prefix(tmp_path: Path) -> None:
    _build_result(tmp_path)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "PASS", payload["errors"]
    assert payload["observed_rate_hz"] == pytest.approx(20.0)
    assert payload["generation_boundary_wall_gap_count"] == 1


@pytest.mark.parametrize(
    "target",
    ["policy", "manifest_preflight", "verified", "runtime_preflight"],
)
def test_validator_rejects_isaac_eula_evidence_drift(
    tmp_path: Path, target: str
) -> None:
    _build_result(tmp_path)
    manifest_path = tmp_path / "session_manifest.json"
    runtime_path = tmp_path / ISAAC_RUNTIME_PREFLIGHT_FILENAME
    manifest = json.loads(manifest_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    if target == "policy":
        manifest["isaac_eula_policy"]["effective_value"] = "YES"
    elif target == "manifest_preflight":
        manifest["isaac_runtime_preflight"]["stdin_target"] = "pipe:[1]"
    elif target == "verified":
        manifest["isaac_runtime_preflight_verified"] = False
    else:
        runtime["before_isaacsim_import"] = False
    _write_json(manifest_path, manifest)
    _write_json(runtime_path, runtime)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert any("Isaac" in error for error in payload["errors"])


@pytest.mark.parametrize(
    "target", ["artifact_policy", "artifact_timing", "manifest", "verified"]
)
def test_validator_rejects_isaac_startup_evidence_drift(
    tmp_path: Path, target: str
) -> None:
    _build_result(tmp_path)
    manifest_path = tmp_path / "session_manifest.json"
    startup_path = tmp_path / ISAAC_STARTUP_READY_FILENAME
    manifest = json.loads(manifest_path.read_text())
    startup = json.loads(startup_path.read_text())
    if target == "artifact_policy":
        startup["policy"]["launch_config"]["hide_ui"] = False
    elif target == "artifact_timing":
        startup["extensions_ready_elapsed_sec"] = 20.0
    elif target == "manifest":
        manifest["isaac_startup_ready"]["status"] = "STALE"
    else:
        manifest["isaac_startup_ready_verified"] = False
    _write_json(manifest_path, manifest)
    _write_json(startup_path, startup)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert any("startup" in error for error in payload["errors"])


@pytest.mark.parametrize("target", ["identity", "manifest", "verified"])
def test_validator_rejects_ros_setup_pythonpath_evidence_drift(
    tmp_path: Path, target: str
) -> None:
    _build_result(tmp_path)
    identity_path = tmp_path / "inner_supervisor_identity.json"
    manifest_path = tmp_path / "session_manifest.json"
    identity = json.loads(identity_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if target == "identity":
        identity["setup_pythonpath_sha256"] = "0" * 64
    elif target == "manifest":
        manifest["setup_pythonpath_sha256"] = "0" * 64
    else:
        manifest["setup_pythonpath_verified"] = False
    _write_json(identity_path, identity)
    _write_json(manifest_path, manifest)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert any("setup PYTHONPATH" in error for error in payload["errors"])


def test_validator_rejects_mutated_owned_qos_handshake(tmp_path: Path) -> None:
    _build_result(tmp_path)
    role_path = tmp_path / "downstream_role_ready.json"
    role = json.loads(role_path.read_text())
    role["owned_qos"]["dynamic"]["depth"] = 1
    _write_json(role_path, role)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert any("role/graph handshake" in error for error in payload["errors"])


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "not_verified",
        "wrong_status",
        "wrong_mechanism",
        "wrong_device",
        "wrong_inode",
    ],
)
def test_validator_rejects_invalid_bind_mount_flock_evidence(
    tmp_path: Path, fault: str
) -> None:
    _build_result(tmp_path)
    manifest_path = tmp_path / "session_manifest.json"
    identity_path = tmp_path / "inner_supervisor_identity.json"
    manifest = json.loads(manifest_path.read_text())
    identity = json.loads(identity_path.read_text())
    if fault == "missing":
        manifest.pop("inner_outer_liveness_probe")
    elif fault == "not_verified":
        manifest["bind_mount_flock_verified"] = False
    elif fault == "wrong_status":
        identity["outer_liveness"]["status"] = "OUTER_GONE"
    elif fault == "wrong_mechanism":
        identity["outer_liveness"]["mechanism"] = "timestamp_guess"
    elif fault == "wrong_device":
        identity["outer_liveness"]["device"] += 1
    else:
        identity["outer_liveness"]["inode"] += 1
    if fault not in ("missing", "not_verified"):
        manifest["inner_outer_liveness_probe"] = identity["outer_liveness"]
    _write_json(manifest_path, manifest)
    _write_json(identity_path, identity)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert any("flock liveness" in error for error in payload["errors"])


@pytest.mark.parametrize(
    "fault", ["gap", "tf", "generation", "missing_part", "camera", "rgb_content", "light", "timing"]
)
def test_validator_rejects_bad_actual_downstream_even_when_sidecar_self_reports_pass(tmp_path: Path, fault: str) -> None:
    evidence = _build_result(tmp_path)
    downstream = evidence["downstream"]
    if fault == "gap":
        for index in range(501, len(downstream)):
            downstream[index]["wall_monotonic_ns"] += 400_000_000
    elif fault == "tf":
        downstream[500]["tf_lookup"]["lookup_exception_count"] = 1
    elif fault == "generation":
        downstream[500]["generation_topic"]["observed_generation"] = 9
    elif fault == "missing_part":
        downstream[500]["observed_parts"].remove("imu")
    elif fault == "camera":
        downstream[500]["camera_contracts"]["rgb"]["k"][2] += 0.25
    elif fault == "rgb_content":
        evidence["sensor"][500]["d435i_rgb_content"] = {
            "finite": True, "valid": False, "minimum": 0, "maximum": 0,
            "dynamic_range": 0, "nonzero_count": 0,
        }
    elif fault == "light":
        evidence["sensor"][500]["diagnostic_light_evidence"]["temperature_enabled"] = False
    else:
        evidence["sensor"][500]["capture_timing"]["lidar_raycast_count"] = 1439
    _write_jsonl(tmp_path / "downstream/downstream_frames.jsonl", downstream)
    _write_jsonl(tmp_path / "sensor_producer_audit.jsonl", evidence["sensor"])
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    # Sidecar producer evidence was deliberately untouched and still looks healthy.
    assert not any("sidecar input" in error for error in payload["errors"])


def test_validator_includes_reset_boundary_in_actual_downstream_gap(tmp_path: Path) -> None:
    evidence = _build_result(tmp_path)
    downstream = evidence["downstream"]
    for index in range(600, len(downstream)):
        downstream[index]["wall_monotonic_ns"] += 360_000_000
    _write_jsonl(tmp_path / "downstream/downstream_frames.jsonl", downstream)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert payload["max_wall_gap_sec"] >= 0.35


@pytest.mark.parametrize("stage", ["sensor", "bridge"])
def test_validator_rejects_intermediate_cross_stage_identity_mismatch(
    tmp_path: Path, stage: str
) -> None:
    evidence = _build_result(tmp_path)
    evidence[stage][500]["sequence"] += 7
    path = (
        tmp_path / "sensor_producer_audit.jsonl"
        if stage == "sensor"
        else tmp_path / "bridge/go2_sensor_bridge_frames.jsonl"
    )
    _write_jsonl(path, evidence[stage])
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert any("exact frozen identity/stamp" in error for error in payload["errors"])


@pytest.mark.parametrize("empty", ["depth", "lidar"])
def test_validator_rejects_empty_or_all_nan_sensor_streams(tmp_path: Path, empty: str) -> None:
    evidence = _build_result(tmp_path)
    if empty == "depth":
        for record in evidence["sensor"]:
            record["depth_valid_in_range_input_points"] = 0
            record["self_filter_output_points"] = 0
        _write_jsonl(tmp_path / "sensor_producer_audit.jsonl", evidence["sensor"])
    else:
        for record in evidence["bridge"]:
            record["lidar_finite_input_points"] = 0
            record["lidar_output_points"] = 0
        _write_jsonl(tmp_path / "bridge/go2_sensor_bridge_frames.jsonl", evidence["bridge"])
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"


@pytest.mark.parametrize("stage", ["sidecar", "downstream"])
def test_validator_rejects_depth_cloud_transport_contract_drift(
    tmp_path: Path, stage: str
) -> None:
    evidence = _build_result(tmp_path)
    if stage == "sidecar":
        evidence["sensor"][500]["depth_cloud_tile_stride"] = 8
        _write_jsonl(tmp_path / "sensor_producer_audit.jsonl", evidence["sensor"])
    else:
        evidence["downstream"][500]["cloud_contracts"]["depth_points"]["width"] = 80
        _write_jsonl(
            tmp_path / "downstream/downstream_frames.jsonl",
            evidence["downstream"],
        )
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    assert any("depth" in error and "contract" in error or "transport" in error for error in payload["errors"])


@pytest.mark.parametrize(
    "artifact,path,value",
    [
        ("producer_snapshot_ack.json", ("emitter", "accepted_count"), 1002),
        ("producer_snapshot_ack.json", ("emitter", "thread_alive"), False),
        ("producer_snapshot_ack.json", ("emitter", "fault"), "writer died"),
        ("sidecar_snapshot_ack.json", ("writer_counts", "controller"), 1000),
        ("sidecar_snapshot_ack.json", ("writer_counts", "reset"), 1),
        ("downstream_snapshot_ack.json", ("generation_event_queue_count",), 1),
        ("downstream_snapshot_ack.json", ("recorder_fault",), "disk fault"),
    ],
)
def test_standalone_validator_mirrors_strict_snapshot_health_fields(
    tmp_path: Path, artifact: str, path: tuple[str, ...], value: object
) -> None:
    _build_result(tmp_path)
    artifact_path = tmp_path / artifact
    payload = json.loads(artifact_path.read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write_json(artifact_path, payload)
    assert validate_result(tmp_path, "bootstrap")["status"] == "FAIL"


def _strict_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def test_validator_always_writes_finite_fail_json_for_empty_gap_and_timing_samples(tmp_path: Path) -> None:
    evidence = _build_result(tmp_path)
    for record in evidence["sensor"]:
        record.pop("capture_timing", None)
    _write_jsonl(tmp_path / "sensor_producer_audit.jsonl", evidence["sensor"])
    _write_jsonl(tmp_path / "downstream/downstream_frames.jsonl", evidence["downstream"][:1])
    completion = json.loads((tmp_path / "workload_completion.json").read_text())
    completion["downstream_record_count"] = 1
    _write_json(tmp_path / "workload_completion.json", completion)
    ack = json.loads((tmp_path / "downstream_snapshot_ack.json").read_text())
    ack["frame_count"] = 1
    _write_json(tmp_path / "downstream_snapshot_ack.json", ack)
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    persisted = _strict_json(tmp_path / "session_validation.json")
    assert persisted["status"] == "FAIL"
    assert persisted["p95_wall_gap_sec"] is None
    assert persisted["capture_stage_timing"]["lidar_query_sec"]["sample_count"] == 0


@pytest.mark.parametrize("mutation", ["missing_stamp", "wrong_type"])
def test_validator_converts_semantic_exceptions_to_atomic_fail_json(tmp_path: Path, mutation: str) -> None:
    evidence = _build_result(tmp_path)
    if mutation == "missing_stamp":
        evidence["downstream"][0].pop("stamp_ns")
    else:
        evidence["downstream"][0]["wall_monotonic_ns"] = {"bad": True}
    _write_jsonl(tmp_path / "downstream/downstream_frames.jsonl", evidence["downstream"])
    payload = validate_result(tmp_path, "bootstrap")
    assert payload["status"] == "FAIL"
    persisted = _strict_json(tmp_path / "session_validation.json")
    assert persisted["status"] == "FAIL"
    assert persisted["errors"][0].startswith("validation_exception:")
