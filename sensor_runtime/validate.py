#!/usr/bin/env python3
"""Fail-closed validation of one four-party frozen sensor evidence prefix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json
from .camera_contract import matrices_from_fov, require_camera_matrices
from .contract import (
    DIAGNOSTIC_GEOMETRY,
    DIAGNOSTIC_LIGHT,
    FRAMES,
    MAX_GAP_SEC,
    P95_GAP_SEC,
    PROFILES,
    REQUIRED_STREAMS,
)
from .graph_handshake import (
    INNER_HANDSHAKE_ARTIFACTS,
    validate_inner_handshake_documents,
)
from .isaac_eula import (
    FROZEN_ISAAC_EULA_POLICY,
    ISAAC_RUNTIME_PREFLIGHT_FILENAME,
    require_frozen_runtime_preflight,
)
from .isaac_experience import (
    ISAAC_STARTUP_READY_FILENAME,
    require_frozen_startup_ready,
)
from .pythonpath_policy import FROZEN_SETUP_PYTHONPATH_SHA256


class ValidationFailed(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationFailed(f"{path.name} is not a JSON object")
    return value


def _read_snapshot(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValidationFailed(f"invalid evidence snapshot count for {path.name}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != count:
        raise ValidationFailed(f"{path.name} differs from its exact frozen count")
    records = [json.loads(line) for line in lines]
    if any(not isinstance(item, dict) for item in records):
        raise ValidationFailed(f"{path.name} contains a non-object record")
    return records


def _percentile95(values: list[float]) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _camera_ok(
    value: Any,
    width: int,
    height: int,
    frame: str,
    hfov: float,
    vfov: float,
) -> bool:
    if not isinstance(value, dict):
        return False
    header_ok = (
        value.get("width") == width
        and value.get("height") == height
        and value.get("frame_id") == frame
        and value.get("distortion_model") == "plumb_bob"
    )
    if not header_ok:
        return False
    try:
        require_camera_matrices(value, matrices_from_fov(width, height, hfov, vfov))
    except (TypeError, ValueError):
        return False
    return True


def _validate_result_impl(result_dir: Path, profile_name: str) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    errors: list[str] = []
    try:
        ready = _read_json(result_dir / "session_ready.json")
        session_manifest = _read_json(result_dir / "session_manifest.json")
        isaac_runtime_preflight = _read_json(
            result_dir / ISAAC_RUNTIME_PREFLIGHT_FILENAME
        )
        isaac_startup_ready = _read_json(
            result_dir / ISAAC_STARTUP_READY_FILENAME
        )
        inner_identity = _read_json(result_dir / "inner_supervisor_identity.json")
        inner_handshake_documents = {
            name: _read_json(result_dir / name) for name in INNER_HANDSHAKE_ARTIFACTS
        }
        completion = _read_json(result_dir / "workload_completion.json")
        producer = _read_json(result_dir / "producer_completion.json")
        producer_ack = _read_json(result_dir / "producer_snapshot_ack.json")
        sidecar_ack = _read_json(result_dir / "sidecar_snapshot_ack.json")
        bridge_ack = _read_json(result_dir / "bridge_snapshot_ack.json")
        downstream_ack = _read_json(result_dir / "downstream_snapshot_ack.json")
        diagnostic = _read_json(result_dir / "diagnostic_geometry.json")
        counts = {
            "sensor": int(completion["sensor_record_count"]),
            "controller": int(completion["controller_record_count"]),
            "reset": int(completion["reset_record_count"]),
            "bridge": int(completion["bridge_record_count"]),
            "downstream": int(completion["downstream_record_count"]),
        }
        sensor = _read_snapshot(result_dir / "sensor_producer_audit.jsonl", counts["sensor"])
        controller = _read_snapshot(result_dir / "controller_stop_audit.jsonl", counts["controller"])
        resets = _read_snapshot(result_dir / "reset_audit.jsonl", counts["reset"])
        bridge = _read_snapshot(result_dir / "bridge/go2_sensor_bridge_frames.jsonl", counts["bridge"])
        downstream = _read_snapshot(result_dir / "downstream/downstream_frames.jsonl", counts["downstream"])
    except BaseException as exc:
        payload = {
            "schema_version": 2,
            "status": "FAIL",
            "profile": profile_name,
            "errors": [f"evidence_load: {type(exc).__name__}: {exc}"],
        }
        atomic_write_json(result_dir / "session_validation.json", payload)
        return payload

    snapshot_id = completion.get("snapshot_id")
    owner_liveness = session_manifest.get("outer_liveness")
    inner_liveness = inner_identity.get("outer_liveness")
    manifest_inner_liveness = session_manifest.get("inner_outer_liveness_probe")
    try:
        frozen_isaac_runtime_preflight = require_frozen_runtime_preflight(
            isaac_runtime_preflight
        )
    except RuntimeError as exc:
        frozen_isaac_runtime_preflight = None
        errors.append(str(exc))
    if (
        session_manifest.get("isaac_eula_policy") != dict(FROZEN_ISAAC_EULA_POLICY)
        or session_manifest.get("isaac_runtime_preflight_verified") is not True
        or frozen_isaac_runtime_preflight is None
        or session_manifest.get("isaac_runtime_preflight")
        != frozen_isaac_runtime_preflight
    ):
        errors.append("Isaac non-interactive EULA evidence is invalid")
    try:
        frozen_isaac_startup_ready = require_frozen_startup_ready(
            isaac_startup_ready
        )
    except RuntimeError as exc:
        frozen_isaac_startup_ready = None
        errors.append(str(exc))
    if (
        frozen_isaac_startup_ready is None
        or session_manifest.get("isaac_startup_ready_verified") is not True
        or session_manifest.get("isaac_startup_ready")
        != frozen_isaac_startup_ready
    ):
        errors.append("Isaac frozen startup experience evidence is invalid")
    if (
        session_manifest.get("bind_mount_flock_verified") is not True
        or not isinstance(owner_liveness, dict)
        or owner_liveness.get("status") != "HELD"
        or owner_liveness.get("mechanism") != "flock_exclusive_nonblocking"
        or not isinstance(inner_liveness, dict)
        or inner_liveness.get("status") != "OUTER_ALIVE"
        or inner_liveness.get("mechanism") != "flock_exclusive_nonblocking"
        or manifest_inner_liveness != inner_liveness
        or int(owner_liveness.get("device", -1)) <= 0
        or int(owner_liveness.get("inode", -1)) <= 0
        or int(inner_liveness.get("device", -2)) != int(owner_liveness.get("device", -1))
        or int(inner_liveness.get("inode", -2)) != int(owner_liveness.get("inode", -1))
    ):
        errors.append("outer/inner bind-mounted flock liveness evidence is invalid")
    inner_handshake = validate_inner_handshake_documents(inner_handshake_documents)
    if inner_handshake.get("ready") is not True:
        errors.append(
            f"inner ROS role/graph handshake evidence is invalid: {inner_handshake['errors']}"
        )
    if (
        inner_identity.get("setup_pythonpath_sha256")
        != FROZEN_SETUP_PYTHONPATH_SHA256
        or session_manifest.get("setup_pythonpath_sha256")
        != FROZEN_SETUP_PYTHONPATH_SHA256
        or session_manifest.get("setup_pythonpath_verified") is not True
    ):
        errors.append("ROS setup PYTHONPATH evidence differs from frozen release")
    if (
        diagnostic.get("status") != "FROZEN"
        or diagnostic.get("objects") != [dict(item) for item in DIAGNOSTIC_GEOMETRY]
        or diagnostic.get("light") != dict(DIAGNOSTIC_LIGHT)
    ):
        errors.append("diagnostic collision geometry evidence differs from frozen contract")
    producer_identity = (int(producer_ack.get("generation", -1)), int(producer_ack.get("sequence", -1)))
    if producer_ack.get("status") != "PASS":
        errors.append("producer snapshot acknowledgement is not PASS")
    emitter = producer_ack.get("emitter", {})
    accepted = int(emitter.get("accepted_count", -1))
    sent = int(emitter.get("sent_count", -1))
    overwrite = int(emitter.get("overwrite_count", -1))
    reset_clear = int(emitter.get("reset_clear_count", -1))
    barrier_drop = int(emitter.get("barrier_drop_count", -1))
    if accepted != sent + overwrite + reset_clear + barrier_drop:
        errors.append(
            "producer emitter accepted count does not reconcile sent/overwrite/reset/barrier"
        )
    if emitter.get("thread_alive") is not True or emitter.get("fault") not in (None, ""):
        errors.append("producer emitter was unhealthy at frozen snapshot")
    if int(producer_ack.get("capture_count", -2)) != accepted:
        errors.append("producer capture_count differs from emitter accepted_count")
    for name in ("last_submitted", "last_sent"):
        value = emitter.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != 2 or (int(value[0]), int(value[1])) != producer_identity:
            errors.append(f"producer emitter {name} differs from frozen target")
    if completion.get("frozen_prefix_reconciled") is not True:
        errors.append("workload completion lacks frozen-prefix reconciliation")
    for name, ack in (("sidecar", sidecar_ack), ("bridge", bridge_ack), ("downstream", downstream_ack)):
        if ack.get("status") != "PASS" or ack.get("snapshot_id") != snapshot_id:
            errors.append(f"{name} snapshot acknowledgement is not the selected frozen prefix")
        if (int(ack.get("generation", -2)), int(ack.get("sequence", -2))) != producer_identity:
            errors.append(f"{name} snapshot target identity differs from producer")
    if len({counts["sensor"], counts["controller"], counts["bridge"], counts["downstream"]}) != 1:
        errors.append("sidecar/controller/bridge/downstream frozen frame counts differ")
    sensor_sequence = [
        (int(item["generation"]), int(item["sequence"]), int(item["stamp_ns"]))
        for item in sensor
    ]
    bridge_sequence = [
        (int(item["generation"]), int(item["sequence"]), int(item["stamp_ns"]))
        for item in bridge
    ]
    downstream_sequence = [
        (int(item["generation"]), int(item["sequence"]), int(item["stamp_ns"]))
        for item in downstream
    ]
    controller_sequence = [
        (int(item["generation"]), int(item["sequence"])) for item in controller
    ]
    if not (
        sensor_sequence == bridge_sequence == downstream_sequence
        and controller_sequence
        == [(generation, sequence) for generation, sequence, _stamp in downstream_sequence]
    ):
        errors.append(
            "sensor/controller/bridge/downstream exact frozen identity/stamp sequences differ"
        )
    if (
        int(bridge_ack.get("frame_count", -1)) != counts["bridge"]
        or int(downstream_ack.get("frame_count", -1)) != counts["downstream"]
        or int(sidecar_ack.get("writer_counts", {}).get("sensor", -1)) != counts["sensor"]
    ):
        errors.append("frozen file counts differ from acknowledgement counts")
    server_received = int(sidecar_ack.get("server_received_count", -1))
    if server_received != int(producer_ack.get("emitter", {}).get("sent_count", -2)):
        errors.append("producer sent and sidecar received counts differ")
    if server_received != (
        counts["sensor"]
        + int(sidecar_ack.get("receive_overwrite_count", -1))
        + int(sidecar_ack.get("receive_reset_clear_count", -1))
        + int(sidecar_ack.get("receive_barrier_drop_count", -1))
    ):
        errors.append("sidecar receive latest-only counts do not reconcile")
    writer_counts = sidecar_ack.get("writer_counts", {})
    if any(int(writer_counts.get(name, -1)) != counts[name] for name in ("sensor", "controller", "reset")):
        errors.append("sidecar writer counts differ from exact frozen evidence counts")
    if sidecar_ack.get("writer_thread_alive") is not True or sidecar_ack.get("writer_fault") not in (None, ""):
        errors.append("sidecar writer was unhealthy at freeze")
    if bridge_ack.get("writer_thread_alive") is not True or bridge_ack.get("writer_fault") not in (None, ""):
        errors.append("bridge writer was unhealthy at freeze")
    if (
        int(downstream_ack.get("pending_count", -1)) != 0
        or int(downstream_ack.get("generation_event_queue_count", -1)) != 0
        or int(downstream_ack.get("tf_lookup_exception_count", -1)) != 0
        or downstream_ack.get("recorder_fault") not in (None, "")
    ):
        errors.append("downstream freeze retained pending data or TF lookup errors")

    if ready.get("status") != "READY" or float(ready.get("session_ready_elapsed_sec", math.inf)) > profile.ready_deadline_sec:
        errors.append("machine ready evidence missed the session-to-ready 20 second deadline")
    if int(ready.get("sensor_record_count", 0)) < 3 or int(ready.get("controller_record_count", 0)) < 3 or int(ready.get("downstream_record_count", 0)) < 1:
        errors.append("ready was declared before sustained safe-stop/downstream evidence")
    post_ready = len(downstream) - int(ready.get("downstream_record_count", len(downstream)))
    if post_ready < profile.minimum_post_ready_records or int(completion.get("ready_post_growth_count", -1)) != post_ready:
        errors.append("actual downstream audit did not grow sufficiently after ready")
    if completion.get("profile") != profile_name or completion.get("status") != "BOUNDED_DURATION_REACHED":
        errors.append("bounded workload completion is invalid")
    if completion.get("required_roles_alive") is not True:
        errors.append("a required role was not alive at validation boundary")
    if producer.get("navigation_evaluator_used") is not False:
        errors.append("producer depended on navigation evaluator lifecycle")
    if float(producer.get("elapsed_sec", 0.0)) < profile.duration_sec:
        errors.append("independent monotonic workload ended early")
    if int(producer.get("active_reset_count", 0)) < profile.minimum_active_resets:
        errors.append("producer did not perform required active resets")

    wall_gaps = [
        (int(current["wall_monotonic_ns"]) - int(previous["wall_monotonic_ns"])) / 1e9
        for previous, current in zip(downstream, downstream[1:])
    ]
    if any(gap <= 0.0 for gap in wall_gaps):
        errors.append("actual downstream wall clock duplicated or rolled back")
    maximum_gap = max(wall_gaps, default=math.inf)
    p95_gap = _percentile95(wall_gaps)
    boundary_gap_count = sum(
        int(current["generation"]) != int(previous["generation"])
        for previous, current in zip(downstream, downstream[1:])
    )
    if p95_gap >= P95_GAP_SEC:
        errors.append("actual downstream p95 wall gap is not below 0.2 seconds")
    if maximum_gap >= MAX_GAP_SEC:
        errors.append("actual downstream maximum wall gap is not below 0.35 seconds")
    effective_sec = (
        (int(downstream[-1]["wall_monotonic_ns"]) - int(downstream[0]["wall_monotonic_ns"])) / 1e9
        if len(downstream) > 1
        else 0.0
    )
    observed_rate_hz = (len(downstream) - 1) / effective_sec if effective_sec > 0 else 0.0
    if effective_sec < profile.minimum_effective_sec:
        errors.append("actual downstream evidence duration is too short")
    if observed_rate_hz < profile.minimum_rate_hz:
        errors.append("actual downstream publication rate is below 10 Hz")

    current_generation = -1
    last_sequence = -1
    last_stamp = -1
    generation_transitions = 0
    required_parts = {
        "clock", "tf", "identity", "generation_topic", "rgb", "rgb_info", "depth",
        "depth_info", "depth_points", "lidar", "lidar_base", "front", "front_info",
        "imu", "safety", "odom",
    }
    for index, record in enumerate(downstream):
        generation, sequence, stamp = int(record["generation"]), int(record["sequence"]), int(record["stamp_ns"])
        observed_parts = set(record.get("observed_parts", []))
        expected_parts = required_parts | ({"tf_static"} if sequence == 0 else set())
        if observed_parts != expected_parts or record.get("same_stamp_observed") is not True:
            errors.append(f"actual downstream frame {index} omits a frozen same-stamp output")
        lookup = record.get("tf_lookup", {})
        if int(lookup.get("lookup_exception_count", 1)) != 0 or int(lookup.get("lookup_success_count", 0)) != 6 or int(lookup.get("stamp_ns", -1)) != stamp:
            errors.append(f"actual TF buffer lookup failed at frame {index}")
        generation_topic = record.get("generation_topic", {})
        if int(generation_topic.get("observed_generation", -1)) != generation:
            errors.append(f"unstamped generation topic disagrees at frame {index}")
        cameras = record.get("camera_contracts", {})
        if (
            not _camera_ok(cameras.get("rgb"), 640, 480, FRAMES["d435i_color"], 69.4, 42.5)
            or not _camera_ok(cameras.get("depth"), 640, 480, FRAMES["d435i_depth"], 87.0, 58.0)
            or not _camera_ok(cameras.get("front"), 160, 120, FRAMES["front_rgb"], 120.0, 75.0)
        ):
            errors.append(f"complete CameraInfo contract failed at frame {index}")
        images = record.get("image_contracts", {})
        expected_images = {
            "rgb": (640, 480, "rgb8"),
            "depth": (640, 480, "32FC1"),
            "front": (160, 120, "rgb8"),
        }
        if any(
            not isinstance(images.get(name), dict)
            or (
                int(images[name].get("width", -1)),
                int(images[name].get("height", -1)),
                images[name].get("encoding"),
            )
            != expected
            for name, expected in expected_images.items()
        ):
            errors.append(f"image wire contract failed at frame {index}")
        clouds = record.get("cloud_contracts", {})
        if any(
            not isinstance(clouds.get(name), dict)
            or int(clouds[name].get("width", 0)) <= 0
            or int(clouds[name].get("height", 0)) <= 0
            or int(clouds[name].get("point_count", 0))
            != int(clouds[name].get("width", 0)) * int(clouds[name].get("height", 0))
            or int(clouds[name].get("finite_point_count", 0)) <= 0
            for name in ("depth_points", "lidar", "lidar_base", "safety")
        ):
            errors.append(f"PointCloud2 wire contract failed at frame {index}")
        depth_cloud = clouds.get("depth_points", {})
        if (
            int(depth_cloud.get("width", 0)) != 160
            or int(depth_cloud.get("height", 0)) != 120
            or int(depth_cloud.get("point_count", 0)) != 19200
        ):
            errors.append(f"depth PointCloud2 reduction contract failed at frame {index}")
        if sequence == 0 and record.get("static_tf_same_stamp_observed") is not True:
            errors.append(f"generation first frame {index} lacks same-stamp static TF")
        if generation != current_generation:
            if generation != current_generation + 1 or sequence != 0:
                errors.append(f"frame {index} jumps generation or omits sequence-zero barrier")
            if current_generation >= 0:
                generation_transitions += 1
            current_generation, last_sequence = generation, -1
        elif sequence <= last_sequence:
            errors.append(f"frame {index} replays downstream sequence")
        if stamp <= last_stamp:
            errors.append(f"frame {index} replays global simulation time")
        last_sequence, last_stamp = sequence, stamp

    if downstream:
        last_actual = (int(downstream[-1]["generation"]), int(downstream[-1]["sequence"]))
        if last_actual != producer_identity:
            errors.append("actual downstream last identity differs from four-party snapshot target")

    capture_stage_values = {
        "camera_read_content_sec": [],
        "lidar_query_sec": [],
        "total_capture_sec": [],
    }
    for index, record in enumerate(sensor):
        stamp = int(record["stamp_ns"])
        if set(record.get("stream_stamps_ns", {})) != set(REQUIRED_STREAMS) or any(int(value) != stamp for value in record.get("stream_stamps_ns", {}).values()):
            errors.append(f"sidecar input frame {index} violates atomic same-stamp capture")
        if int(record.get("dynamic_link_count", 0)) != 13:
            errors.append(f"sidecar frame {index} lacks base plus 12 dynamic self-filter centers")
        if int(record.get("self_filter_input_points", 0)) != 640 * 480:
            errors.append(f"sidecar frame {index} lacks self-filter input counts")
        output_points = int(record.get("self_filter_output_points", -1))
        if not 0 < output_points <= 640 * 480:
            errors.append(f"sidecar frame {index} has invalid self-filter output count")
        transport_points = int(record.get("depth_cloud_output_points", -1))
        if (
            record.get("depth_cloud_reduction")
            != "4x4_nearest_valid_after_full_resolution_filter"
            or int(record.get("depth_cloud_tile_stride", 0)) != 4
            or int(record.get("depth_cloud_width", 0)) != 160
            or int(record.get("depth_cloud_height", 0)) != 120
            or int(record.get("depth_cloud_point_count", 0)) != 19200
            or not 0 < transport_points <= min(19200, output_points)
        ):
            errors.append(f"sidecar frame {index} violates bounded depth-cloud transport")
        if (
            int(record.get("depth_valid_in_range_input_points", 0)) < 30720
            or float(record.get("depth_valid_in_range_ratio", 0.0)) < 0.10
        ):
            errors.append(f"sidecar frame {index} fails the 10% valid in-range depth gate")
        if int(record.get("lidar_finite_input_points", 0)) <= 0:
            errors.append(f"sidecar frame {index} has no finite LiDAR input")
        if record.get("diagnostic_geometry_ids") != [item["name"] for item in DIAGNOSTIC_GEOMETRY]:
            errors.append(f"sidecar frame {index} lacks frozen diagnostic geometry identity")
        if record.get("diagnostic_light_evidence") != {
            **DIAGNOSTIC_LIGHT,
            "temperature_enabled": True,
        }:
            errors.append(f"sidecar frame {index} lacks observed frozen DomeLight attributes")
        for stream in ("d435i_rgb_content", "front_rgb_content"):
            content = record.get(stream, {})
            if (
                content.get("finite") is not True
                or content.get("valid") is not True
                or int(content.get("nonzero_count", 0)) <= 0
                or int(content.get("dynamic_range", 0)) <= 0
                or int(content.get("minimum", 0)) >= int(content.get("maximum", 0))
            ):
                errors.append(f"sidecar frame {index} has black/constant {stream}")
        timing = record.get("capture_timing")
        if not isinstance(timing, dict) or timing.get("clock") != "monotonic_perf_counter":
            errors.append(f"sidecar frame {index} lacks monotonic capture timing")
        else:
            valid_timing = True
            observed_values: dict[str, float] = {}
            for name in capture_stage_values:
                raw = timing.get(name)
                if (
                    isinstance(raw, bool)
                    or not isinstance(raw, (int, float))
                    or not math.isfinite(float(raw))
                    or float(raw) < 0.0
                ):
                    valid_timing = False
                else:
                    observed_values[name] = float(raw)
            if (
                int(timing.get("lidar_raycast_count", -1)) != 1440
                or valid_timing
                and observed_values["total_capture_sec"]
                < max(observed_values["camera_read_content_sec"], observed_values["lidar_query_sec"])
            ):
                valid_timing = False
            if not valid_timing:
                errors.append(f"sidecar frame {index} has invalid capture timing values")
            else:
                for name, value in observed_values.items():
                    capture_stage_values[name].append(value)

    for index, record in enumerate(bridge):
        if set(record.get("matched_parts", [])) != {"clock", "tf", "identity", "centers", "lidar", "depth", "front", "front_info"}:
            errors.append(f"bridge frame {index} lacks paired frozen inputs")
        if int(record.get("link_center_count", 0)) != 13:
            errors.append(f"bridge frame {index} lacks finite self-filter centers")
        if int(record.get("lidar_finite_input_points", 0)) <= 0 or int(record.get("lidar_output_points", 0)) <= 0:
            errors.append(f"bridge frame {index} has empty finite/filter LiDAR output")
        if int(record.get("safety_point_count", 0)) <= 0:
            errors.append(f"bridge frame {index} has empty safety cloud")

    active_resets = [item for item in resets if item.get("reason") == "active_periodic"]
    reset_generations = [int(item.get("generation", -1)) for item in resets]
    if not resets or resets[0].get("reason") != "initial" or reset_generations != list(range(len(resets))):
        errors.append("reset audit is incomplete or non-monotonic")
    if any(item.get("atomic_latest_clear") is not True for item in resets):
        errors.append("reset audit lacks atomic latest-only clear evidence")
    if any(item.get("reset_kind") != "continuous_world_articulation_state" for item in resets):
        errors.append("reset audit used a full-world or unknown reset kind")
    if len(active_resets) < profile.minimum_active_resets:
        errors.append("active reset evidence count is too low")
    if int(producer.get("active_reset_count", -1)) != len(active_resets):
        errors.append("producer active reset count differs from reset audit")
    if int(producer.get("generation", -1)) != producer_identity[0]:
        errors.append("producer final generation differs from snapshot generation")
    if generation_transitions != len(active_resets) or boundary_gap_count != len(active_resets):
        errors.append("actual downstream generation boundaries do not match active resets")

    previous_applied = 0
    for index, (sensor_item, stop_item) in enumerate(zip(sensor, controller)):
        generation, sequence = int(sensor_item["generation"]), int(sensor_item["sequence"])
        applied = int(stop_item.get("applied_step_count", 0))
        delta = applied - previous_applied
        if (
            int(stop_item.get("generation", -1)) != generation
            or int(stop_item.get("sequence", -1)) != sequence
            or stop_item.get("identity") != f"sensor-soak:{generation}:{sequence}"
            or float(stop_item.get("linear_x", 1.0)) != 0.0
            or float(stop_item.get("angular_z", 1.0)) != 0.0
            or stop_item.get("emergency_stop") is not True
            or delta <= 0
            or int(stop_item.get("steps_since_previous_capture", 0)) <= 0
            or int(stop_item.get("steps_since_previous_capture", 0)) > delta
        ):
            errors.append(f"safe-stop continuous coverage/identity failed at record {index}")
        previous_applied = applied

    capture_timing_summary = {
        name: {
            "sample_count": len(values),
            "p50_sec": sorted(values)[len(values) // 2] if values else None,
            "p95_sec": _percentile95(values) if values else None,
            "max_sec": max(values) if values else None,
        }
        for name, values in capture_stage_values.items()
    }
    payload = {
        "schema_version": 2,
        "status": "PASS" if not errors else "FAIL",
        "profile": profile_name,
        "sensor_record_count": len(sensor),
        "downstream_record_count": len(downstream),
        "active_reset_count": len(active_resets),
        "generation_boundary_wall_gap_count": boundary_gap_count,
        "effective_duration_sec": effective_sec,
        "observed_rate_hz": observed_rate_hz,
        "p95_wall_gap_sec": p95_gap if math.isfinite(p95_gap) else None,
        "max_wall_gap_sec": maximum_gap if math.isfinite(maximum_gap) else None,
        "capture_stage_timing": capture_timing_summary,
        "tf_lookup_exception_count": int(downstream_ack.get("tf_lookup_exception_count", -1)),
        "generation_contamination_count": sum("generation" in item and ("jump" in item or "replay" in item) for item in errors),
        "safe_stop_continuous_coverage": not any("safe-stop" in item for item in errors),
        "frozen_prefix_reconciled": not any("snapshot" in item or "counts differ" in item for item in errors),
        "errors": errors,
    }
    atomic_write_json(result_dir / "session_validation.json", payload)
    return payload


def validate_result(result_dir: Path, profile_name: str) -> dict[str, Any]:
    """Always produce parseable machine-readable FAIL evidence on bad input."""

    result_dir = Path(result_dir)
    try:
        return _validate_result_impl(result_dir, profile_name)
    except BaseException as exc:
        payload = {
            "schema_version": 2,
            "status": "FAIL",
            "profile": profile_name,
            "errors": [f"validation_exception: {type(exc).__name__}: {exc}"],
        }
        atomic_write_json(result_dir / "session_validation.json", payload)
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    payload = validate_result(args.result_dir.resolve(), args.profile)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
