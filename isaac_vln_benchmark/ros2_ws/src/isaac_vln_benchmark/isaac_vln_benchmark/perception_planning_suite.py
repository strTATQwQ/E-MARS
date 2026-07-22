from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STEP_REQUEST_LATENCY_P95_MAX_SEC = 5.0
THRESHOLD_TO_SAFE_STOP_P95_MAX_SEC = 2.0


TARGET_CLASSES = (
    "fire extinguisher",
    "first aid kit",
    "traffic cone",
)

TARGET_ASSETS = {
    "fire extinguisher": "/home/song/isaacsim_assets/Assets/Isaac/6.0/Isaac/Environments/Simple_Warehouse/Props/SM_FireExtinguisher_02.usd",
    "first aid kit": "/home/song/isaacsim_assets/Assets/Isaac/6.0/Isaac/Environments/Hospital/Props/SM_FirstAidKit_01a.usd",
    "traffic cone": "/home/song/isaacsim_assets/Assets/Isaac/6.0/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
}

TARGET_ASSET_SCALES = {
    "fire extinguisher": 1.0,
    "fire hydrant": 1.0,
    "first aid kit": 1.0,
    "traffic cone": 1.0,
}

ROUTE_LANDMARKS = (
    ("red cone", "blue box"),
    ("green exit sign", "yellow toolbox"),
    ("white first aid sign", "orange barrier"),
    ("purple placard", "silver cabinet"),
)

ORACLE_REQUEST_KEYS = {
    "expected",
    "expected_route",
    "correct_branch",
    "entered_correct_branch",
    "ground_truth",
    "oracle_visibility",
    "target_present",
    "target_visible",
    "distance_to_target_m",
    "distance_to_target",
    "success_judge",
}


def build_controlled_visual_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    variants = ("clear", "similar_color", "partial_occlusion", "dim", "camera_yaw", "far_scale")
    for index in range(24):
        expected = "left" if index % 2 == 0 else "right"
        pair = ROUTE_LANDMARKS[0]
        target = pair[0] if expected == "left" else pair[1]
        cases.append(
            {
                "case_id": f"controlled_route_{index + 1:02d}",
                "domain": "controlled",
                "role": "route_choice",
                "scene_family": ("open_intersection", "static_obstacle", "visual_distractor")[index % 3],
                "visual_variant": variants[index % len(variants)],
                "instruction": f"At the intersection, take the branch marked by the {target}.",
                "target": target,
                "expected": expected,
                "camera_yaw_deg": (-8, 0, 8)[index % 3],
                "brightness": (0.55, 1.0, 1.35)[index % 3],
                "occlusion_fraction": 0.35 if index % len(variants) == 2 else 0.0,
            }
        )
    for index in range(36):
        pair_index = index // 2
        target = TARGET_CLASSES[(pair_index // len(variants)) % len(TARGET_CLASSES)]
        present = index % 2 == 0
        variant = variants[pair_index % len(variants)]
        cases.append(
            {
                "case_id": f"controlled_semantic_{index + 1:02d}",
                "domain": "controlled",
                "role": "semantic_stop",
                "scene_family": "semantic_target",
                "visual_variant": variant,
                "instruction": f"Find the {target} and stop when it is close enough.",
                "target": target,
                "expected": present,
                "target_present": present,
                "distractor": TARGET_CLASSES[(index + 1) % len(TARGET_CLASSES)],
                "camera_yaw_deg": (-15, 0, 15)[index % 3],
                "brightness": (0.6, 1.0, 1.3)[index % 3],
                "occlusion_fraction": 0.4 if variant == "partial_occlusion" and present else 0.0,
            }
        )
    return cases


def build_mp3d_visual_cases(scene_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    scenes = [str(value) for value in (scene_ids or []) if str(value)]
    cases: list[dict[str, Any]] = []
    for index in range(30):
        role = "route_choice" if index < 12 else "semantic_stop"
        scene_id = scenes[index % len(scenes)] if scenes else None
        if role == "route_choice":
            expected = "left" if index % 2 == 0 else "right"
            pair = ROUTE_LANDMARKS[0]
            target = pair[0] if expected == "left" else pair[1]
            instruction = f"At the next junction, take the branch marked by the {target}."
        else:
            semantic_index = index - 12
            expected = semantic_index % 2 == 0
            target = TARGET_CLASSES[semantic_index % len(TARGET_CLASSES)]
            instruction = f"Find the {target} and stop when it is close enough."
        row = {
            "case_id": f"mp3d_pe_{role}_{index + 1:02d}",
            "domain": "mp3d_pe",
            "role": role,
            "scene_id": scene_id,
            "scene_index_required": scene_id is None,
            "prop_injection_audited": True,
            "instruction": instruction,
            "target": target,
            "expected": expected,
            "camera_yaw_deg": (-20, 0, 20)[index % 3],
            "brightness": (0.7, 1.0, 1.25)[index % 3],
            "occlusion_fraction": 0.3 if index % 5 == 0 else 0.0,
        }
        if role == "semantic_stop":
            row["target_present"] = bool(expected)
        cases.append(row)
    return cases


def mp3d_background_transform(bounds: dict[str, Any], *, scale: float = 0.45) -> dict[str, list[float]]:
    minima = [float(value) for value in bounds.get("min", [])]
    center = [float(value) for value in bounds.get("center", [])]
    if len(minima) < 3 or len(center) < 3:
        return {"background_pose": [5.0, 0.0, 0.0], "background_scale": [scale] * 3}
    return {
        "background_pose": [
            round(5.0 - minima[0] * scale, 6),
            round(-center[1] * scale, 6),
            round(1.5 - center[2] * scale, 6),
        ],
        "background_scale": [scale] * 3,
    }


def build_tracking_sequences() -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for index in range(30):
        family = ("static", "moving_camera", "temporary_occlusion")[index // 10]
        frame_count = 8 + (index % 7) * 2
        target = TARGET_CLASSES[index % len(TARGET_CLASSES)]
        frames = []
        for frame_index in range(frame_count):
            visible = True
            if family == "moving_camera":
                visible = True
            elif family == "temporary_occlusion":
                occlusion_start = max(3, frame_count // 3)
                visible = not (occlusion_start <= frame_index < occlusion_start + 3)
            frame_seq = frame_index + 1
            if frame_index == 2:
                frame_seq = 2
                visible = bool(frames[-1]["visible"])
            frames.append(
                {
                    "frame_index": frame_index,
                    "frame_seq": frame_seq,
                    "visible": visible,
                    "confidence_floor": 0.65 if visible else 0.0,
                    "occluded": not visible and family == "temporary_occlusion",
                }
            )
        sequences.append(
            {
                "sequence_id": f"track_{family}_{index % 10 + 1:02d}",
                "family": family,
                "episode_id": f"tracking_episode_{index + 1:02d}",
                "target": target,
                "frames": frames,
                "required_hits": 2,
                "max_misses": 2,
            }
        )
    return sequences


def tracking_visual_case(sequence: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    family = str(sequence["family"])
    visible = bool(frame["visible"])
    frame_count = max(1, len(sequence["frames"]))
    frame_index = int(frame["frame_index"])
    camera_yaw_deg = 0.0
    camera_pose = [0.0, 0.0, 0.0]
    if family == "moving_camera":
        # Move the camera laterally around a fixed, continuously visible target.
        # Loss and reacquisition are tested independently by the occlusion family.
        camera_y = 1.0 - 2.0 * frame_index / max(1, frame_count - 1)
        camera_pose = [0.0, camera_y, 0.0]
    return {
        "case_id": f"{sequence['sequence_id']}_frame_{frame_index:02d}",
        "domain": "tracking",
        "role": "semantic_stop",
        "scene_family": "semantic_target",
        "visual_variant": "temporary_occlusion" if bool(frame.get("occluded")) else family,
        "instruction": f"Find the {sequence['target']} and stop when it is close enough.",
        "target": str(sequence["target"]),
        "expected": visible,
        "target_present": family in {"moving_camera", "temporary_occlusion"} or visible,
        "distractor": TARGET_CLASSES[(TARGET_CLASSES.index(str(sequence["target"])) + 1) % len(TARGET_CLASSES)],
        "camera_yaw_deg": camera_yaw_deg,
        "camera_pose": camera_pose,
        "brightness": 1.0,
        "occlusion_fraction": 1.0 if bool(frame.get("occluded")) else 0.0,
    }


def build_planning_cases() -> dict[str, list[dict[str, Any]]]:
    routes: list[dict[str, Any]] = []
    semantics: list[dict[str, Any]] = []
    families = ("open_intersection", "static_obstacle", "visual_distractor")
    for index in range(30):
        expected = "left" if index % 2 == 0 else "right"
        target = ROUTE_LANDMARKS[0][0 if expected == "left" else 1]
        routes.append(
            {
                "episode_id": f"planning_route_{index + 1:02d}",
                "task_id": "turn_001" if expected == "left" else "turn_002",
                "domain": "planning",
                "role": "route_choice",
                "scene_family": families[(index // 2) % len(families)],
                "seed": 1400 + index,
                "instruction": f"Take the branch marked by the {target}.",
                "target": target,
                "expected": expected,
            }
        )
    for index in range(30):
        present = index < 15
        target = TARGET_CLASSES[index % len(TARGET_CLASSES)]
        semantics.append(
            {
                "episode_id": f"planning_semantic_{index + 1:02d}",
                "domain": "planning",
                "role": "semantic_stop",
                "seed": 2400 + index,
                "instruction": f"Find the {target} and stop when it is close enough.",
                "target": target,
                "expected": present,
                "target_present": present,
                "target_distance_m": 3.5,
                "coverage_distance_m": 2.5,
                "stop_distance_m": 2.0,
            }
        )
    return {"route": routes, "semantic": semantics}


def step_visible_request(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": str(case["role"]),
        "instruction": str(case["instruction"]),
        "active_subgoal": str(case["target"]),
        "target": str(case["target"]),
        "multimodal": True,
        "visual_ground_truth_hidden": True,
        "event": {"type": "visual_decision_due"},
    }


def reset_payload_for_visual_case(case: dict[str, Any], *, episode_id: str) -> dict[str, Any]:
    camera_yaw_deg = float(case.get("camera_yaw_deg", 0.0))
    brightness = float(case.get("brightness", 1.0))
    objects: list[dict[str, Any]] = []
    obstacles: list[dict[str, Any]] = []
    if case.get("role") == "route_choice":
        expected = str(case.get("expected"))
        target = str(case.get("target"))
        pair = next((value for value in ROUTE_LANDMARKS if target in value), ROUTE_LANDMARKS[0])
        other = pair[1] if target == pair[0] else pair[0]
        route_x = 3.8 if case.get("visual_variant") == "far_scale" else 2.8
        route_radius = 0.8 if case.get("visual_variant") == "far_scale" else 1.0
        placements = ((target, route_x, 1.5 if expected == "left" else -1.5), (other, route_x, -1.5 if expected == "left" else 1.5))
        for index, (label, x_pos, y_pos) in enumerate(placements):
            color, object_class = _label_attributes(label)
            landmark_id = "exit_sign_1" if label == pair[0] else "blue_box_1"
            objects.append(
                {
                    "id": landmark_id,
                    "class": object_class,
                    "label": label,
                    "color": color,
                    "pose": [x_pos, y_pos, 0.85],
                    "radius": route_radius,
                    "brightness": brightness,
                }
            )
    else:
        target = str(case.get("target"))
        if bool(case.get("target_present", case.get("expected", False))):
            color, object_class = _label_attributes(target)
            target_object = {
                    "id": "fire_extinguisher_1",
                    "class": object_class,
                    "label": target,
                    "color": color,
                    "pose": [
                        float(case.get("target_distance_m", 2.0 if case.get("visual_variant") != "far_scale" else 3.5)),
                        0.0,
                        0.0,
                    ],
                    "radius": 0.9,
                    "brightness": brightness,
                    "scale": [TARGET_ASSET_SCALES.get(target, 2.0)] * 3,
                }
            if target in TARGET_ASSETS:
                target_object["usd_path"] = TARGET_ASSETS[target]
            objects.append(target_object)
        else:
            objects.append(
                {
                    "id": "fire_extinguisher_1",
                    "class": "box",
                    "label": "hidden judge target",
                    "color": "gray",
                    "pose": [100.0, 100.0, 0.0],
                    "radius": 0.1,
                }
            )
        distractor = str(case.get("distractor") or "blue box")
        color, object_class = _label_attributes(distractor)
        distractor_object = {
                "id": f"semantic_distractor_{distractor.replace(' ', '_')}",
                "class": object_class,
                "label": distractor,
                "color": color,
                "pose": [2.8, 1.25, 0.65],
                "radius": 0.8,
                "brightness": brightness,
                "scale": [TARGET_ASSET_SCALES.get(distractor, 2.0) * 0.8] * 3,
            }
        if distractor in TARGET_ASSETS:
            distractor_object["usd_path"] = TARGET_ASSETS[distractor]
            distractor_object["pose"][2] = 0.0
        objects.append(distractor_object)
    occlusion_fraction = float(case.get("occlusion_fraction", 0.0))
    if occlusion_fraction > 0.0:
        full_occlusion = occlusion_fraction >= 0.9
        occluder_y = (
            (0.8 if str(case.get("expected")) == "left" else -0.8)
            if case.get("role") == "route_choice"
            else (0.0 if full_occlusion else 0.35)
        )
        obstacles.append(
            {
                "id": "visual_occluder",
                "class": "box",
                "color": "black" if full_occlusion else "gray",
                "pose": [1.25 if full_occlusion else 1.7, occluder_y, 3.0 if full_occlusion else 0.48],
                "size": [0.50, 5.0, 6.0] if full_occlusion else [0.20, 0.25, 0.55],
                "brightness": brightness,
                "visual_only": True,
            }
        )
    if str(case.get("domain")) == "planning" and str(case.get("scene_family")) == "static_obstacle":
        obstacles.append(
            {
                "id": "planning_static_obstacle",
                "class": "box",
                "color": "gray",
                "pose": [2.0, 0.0, 0.225],
                "size": [0.25, 0.45, 0.45],
                "visual_only": False,
            }
        )
    if str(case.get("domain")) == "planning" and str(case.get("scene_family")) == "visual_distractor":
        objects.append(
            {
                "id": "planning_visual_distractor",
                "class": "box",
                "label": "red box distractor",
                "color": "red",
                "pose": [2.3, 0.4, 0.45],
                "radius": 0.35,
            }
        )
    camera_pose = list(case.get("camera_pose") or [0.0, 0.0, math.radians(camera_yaw_deg)])
    if len(camera_pose) != 3:
        camera_pose = [0.0, 0.0, math.radians(camera_yaw_deg)]
    payload = {
        "episode_id": episode_id,
        "task_id": str(
            case.get("task_id")
            or ("turn_001" if case.get("role") == "route_choice" else "semantic_001")
        ),
        # Keep the benchmark controller/judge geometry on its audited scene.
        # Visual props and optional MP3D USDs are overlays, not task geometry.
        "scene_id": str(case.get("scene_id") or "intersection_001"),
        "pose": camera_pose,
        "objects": objects,
        "obstacles": obstacles,
        "visual_case_id": str(case.get("case_id") or case.get("name") or ""),
        "visual_variant": str(case.get("visual_variant") or "clear"),
        "brightness": brightness,
        "visual_overlay_only": True,
    }
    background_usd = str(case.get("scene_usd_path") or case.get("background_usd") or "")
    if background_usd:
        payload["background_usd"] = background_usd
        payload["background_pose"] = list(case.get("background_pose") or [0.0, 0.0, 0.0])
        payload["background_scale"] = list(case.get("background_scale") or [1.0, 1.0, 1.0])
        payload["background_visual_only"] = True
    return payload


def audit_step_visible_request(payload: dict[str, Any]) -> dict[str, Any]:
    leaks: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                normalized = str(key).strip().lower()
                if normalized in ORACLE_REQUEST_KEYS or normalized.startswith("oracle_"):
                    leaks.append(child_path)
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "")
    return {"pass": not leaks, "oracle_context_leakage": len(leaks), "leak_paths": sorted(set(leaks))}


def evaluate_visual_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    route = [row for row in rows if row.get("role") == "route_choice"]
    semantic = [row for row in rows if row.get("role") == "semantic_stop"]
    route_correct = sum(str(row.get("predicted")) == str(row.get("expected")) for row in route)
    side_totals = {side: sum(row.get("expected") == side for row in route) for side in ("left", "right")}
    side_correct = {
        side: sum(row.get("expected") == side and row.get("predicted") == side for row in route)
        for side in ("left", "right")
    }
    tp = sum(bool(row.get("expected")) and bool(row.get("predicted")) for row in semantic)
    fp = sum(not bool(row.get("expected")) and bool(row.get("predicted")) for row in semantic)
    fn = sum(bool(row.get("expected")) and not bool(row.get("predicted")) for row in semantic)
    tn = sum(not bool(row.get("expected")) and not bool(row.get("predicted")) for row in semantic)
    class_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in semantic:
        if bool(row.get("expected")):
            class_counts[str(row.get("target"))][1] += 1
            class_counts[str(row.get("target"))][0] += int(bool(row.get("predicted")))
    fresh = sum(bool(row.get("fresh_image")) for row in rows)
    parse_errors = sum(bool(row.get("parse_error")) for row in rows)
    leakage = sum(int(row.get("oracle_context_leakage", 0)) for row in rows)
    background_rows = [row for row in rows if row.get("background_usd")]
    backgrounds_detected = sum(bool(row.get("background_visual_detected")) for row in background_rows)
    latencies = [float(row["step_latency_sec"]) for row in rows if row.get("step_latency_sec") is not None]
    return {
        "total": len(rows),
        "route_total": len(route),
        "route_correct": route_correct,
        "route_accuracy": _rate(route_correct, len(route)),
        "route_side_accuracy": {side: _rate(side_correct[side], side_totals[side]) for side in side_totals},
        "semantic_total": len(semantic),
        "semantic_precision": _rate(tp, tp + fp),
        "semantic_recall": _rate(tp, tp + fn),
        "semantic_false_positive_rate": _rate(fp, fp + tn),
        "semantic_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_target_recall": {name: _rate(values[0], values[1]) for name, values in sorted(class_counts.items())},
        "fresh_image_rate": _rate(fresh, len(rows)),
        "json_parse_errors": parse_errors,
        "oracle_context_leakage": leakage,
        "background_rows": len(background_rows),
        "background_visual_detected_rate": _rate(backgrounds_detected, len(background_rows)),
        "step_latency_p95_sec": percentile(latencies, 0.95),
    }


def visual_rows_from_micro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        observations = [value for value in row.get("observation_responses", []) if isinstance(value, dict)]
        http_events = [value for value in row.get("step_http_events", []) if isinstance(value, dict)]
        role = str(row.get("role") or "")
        predicted = response.get("route_choice") if role == "route_choice" else response.get("target_visible")
        fresh = bool(row.get("reset_ack_received")) and bool(row.get("post_reset_images_ready")) and bool(observations) and all(
            bool(value.get("multimodal"))
            and isinstance(value.get("image_snapshot"), dict)
            and float(value["image_snapshot"].get("age_sec", 99.0)) <= 0.75
            for value in observations
        )
        accepted = bool(http_events) and all(str(value.get("result")) == "accepted" for value in http_events)
        latency_values = [
            float(value.get("latency_s")) for value in http_events if value.get("latency_s") is not None
        ]
        converted.append(
            {
                "case_id": str(row.get("case_id") or row.get("name") or ""),
                "domain": str(row.get("domain") or "controlled"),
                "role": role,
                "target": str(row.get("target") or ""),
                "expected": row.get("expected"),
                "predicted": predicted,
                "fresh_image": fresh,
                "parse_error": not accepted,
                "oracle_context_leakage": sum(
                    int(value.get("oracle_context_leakage", 0)) for value in row.get("request_audits", [])
                ),
                "step_latency_sec": max(latency_values) if latency_values else response.get("step_latency_sec"),
                "step_result": [value.get("result") for value in http_events],
                "image_snapshot": response.get("image_snapshot", {}),
                "visual_attribute_evidence": response.get("visual_attribute_evidence", {}),
                "track": response.get("track", {}),
                "background_usd": row.get("background_usd"),
                "background_visual_detected": bool(
                    (row.get("image_visual_stats") or {}).get("background_visual_detected")
                    or float((row.get("image_visual_stats") or {}).get("upper_half_luminance_stddev", 0.0)) >= 18.0
                    or float((row.get("image_visual_stats") or {}).get("upper_half_edge_mean", 0.0)) >= 5.0
                ),
                "image_visual_stats": dict(row.get("image_visual_stats") or {}),
            }
        )
    return converted


def planning_rows_from_micro(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    route_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for row in rows:
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        role = str(row.get("role") or "")
        primitives = [value for value in row.get("case_primitives", []) if isinstance(value, dict)]
        scoped_primitives = [
            value for value in row.get("primitives_after_response", []) if isinstance(value, dict)
        ]
        safe_cmds = [value for value in row.get("case_safe_cmds", []) if isinstance(value, dict)]
        metrics = [value for value in row.get("case_metrics", []) if isinstance(value, dict)]
        safety = {key: int(row.get(key, 0)) for key in (
            "collision", "runtime_stale", "timebase_error", "stale_action_executed"
        )}
        for metric in metrics:
            for key in safety:
                safety[key] += int(metric.get(key, 0) or 0)
        stop = any(value.get("primitive") == "stop" for value in scoped_primitives)
        safe_motion_seen = any(
            abs(float(value.get("linear_x", 0.0))) > 1.0e-3
            or abs(float(value.get("angular_z", 0.0))) > 1.0e-3
            for value in safe_cmds
        )
        if role == "route_choice":
            decision_confirmed = str(response.get("route_choice") or "") in {"left", "right"}
            bridge_seen = any(value.get("primitive") == "follow_waypoint" for value in scoped_primitives)
            route_rows.append(
                {
                    "episode_id": row.get("episode_id"),
                    "expected": row.get("expected"),
                    "predicted": response.get("route_choice"),
                    "decision_confirmed": decision_confirmed,
                    "entered_correct_branch": bool(row.get("entered_correct_branch")),
                    "branch_membership": row.get("expected") if row.get("entered_correct_branch") else "wrong_or_none",
                    "normal_chain_complete": bool(primitives) and safe_motion_seen,
                    "trajectory": list(row.get("full_trajectory") or row.get("trajectory") or []),
                    "controller_trace": list(row.get("controller_trace") or []),
                    **safety,
                }
            )
        elif role == "semantic_stop":
            positive = bool(row.get("expected"))
            semantic_rows.append(
                {
                    "episode_id": row.get("episode_id"),
                    "expected": positive,
                    "predicted": bool(response.get("target_visible")),
                    "entered_coverage": bool(row.get("coverage_status")) if positive else False,
                    "success": bool(row.get("semantic_success")) if positive else not stop,
                    "stop": stop,
                    "normal_chain_complete": (
                        bool(primitives)
                        and row.get("threshold_to_safe_stop_sec") is not None
                        if positive
                        else bool(response) and not stop
                    ),
                    "threshold_to_safe_stop_sec": row.get("threshold_to_safe_stop_sec"),
                    "coverage_to_stop_sec": row.get("coverage_to_stop_sec"),
                    "approach_steps": list(row.get("approach_steps") or []),
                    "target_distance_m": row.get("distance_m_used"),
                    "trajectory": list(row.get("full_trajectory") or row.get("trajectory") or []),
                    "controller_trace": list(row.get("controller_trace") or []),
                    **safety,
                }
            )
    return {"route": route_rows, "semantic": semantic_rows}


def visual_gate(metrics: dict[str, Any], *, domain: str) -> dict[str, Any]:
    failures: list[str] = []
    route_min = 0.80 if domain == "controlled" else 0.75
    if float(metrics.get("route_accuracy", 0.0)) < route_min:
        failures.append(f"route_accuracy < {route_min:.2f}")
    if domain == "controlled":
        for side in ("left", "right"):
            if float((metrics.get("route_side_accuracy") or {}).get(side, 0.0)) < 0.75:
                failures.append(f"route_{side}_accuracy < 0.75")
    elif int(metrics.get("background_rows", 0)) != int(metrics.get("total", -1)):
        failures.append("one or more mp3d rows have no background USD")
    elif float(metrics.get("background_visual_detected_rate", 0.0)) != 1.0:
        failures.append("mp3d background visual detection rate != 1.0")
    if float(metrics.get("semantic_precision", 0.0)) < 0.90:
        failures.append("semantic_precision < 0.90")
    if float(metrics.get("semantic_recall", 0.0)) < 0.85:
        failures.append("semantic_recall < 0.85")
    if float(metrics.get("semantic_false_positive_rate", 1.0)) > 0.10:
        failures.append("semantic_false_positive_rate > 0.10")
    for target, recall in (metrics.get("per_target_recall") or {}).items():
        if float(recall) < 0.70:
            failures.append(f"target_recall[{target}] < 0.70")
    if float(metrics.get("fresh_image_rate", 0.0)) != 1.0:
        failures.append("fresh_image_rate != 1.0")
    if int(metrics.get("json_parse_errors", 0)) != 0:
        failures.append("json_parse_errors != 0")
    if int(metrics.get("oracle_context_leakage", 0)) != 0:
        failures.append("oracle_context_leakage != 0")
    latency = metrics.get("step_latency_p95_sec")
    if latency is None or float(latency) > STEP_REQUEST_LATENCY_P95_MAX_SEC:
        failures.append(
            f"Step request p95 latency > {STEP_REQUEST_LATENCY_P95_MAX_SEC:g} sec"
        )
    return {"pass": not failures, "domain": domain, "failures": failures, "metrics": metrics}


def evaluate_tracking_sequences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_miscounts = 0
    cross_talk = 0
    false_confirmed = 0
    confirmed_frames = 0
    stale_actions = 0
    occlusion_sequences = set()
    reacquired_sequences = set()
    last_by_track: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("episode_id")), str(row.get("target")))
        previous = last_by_track.get(key)
        if previous and row.get("frame_seq") == previous.get("frame_seq"):
            duplicate_miscounts += int(int(row.get("hits", 0)) != int(previous.get("hits", 0)))
        if str(row.get("track_episode_id")) != key[0] or str(row.get("track_target")) != key[1].lower():
            cross_talk += 1
        confirmed = bool(row.get("confirmed"))
        track_visible = bool(row.get("track_visible", row.get("visible", confirmed)))
        confirmed_visible = bool(confirmed and track_visible)
        confirmed_frames += int(confirmed_visible)
        false_confirmed += int(confirmed_visible and not bool(row.get("visible_truth")))
        stale_actions += int(bool(row.get("action_triggered")) and (not confirmed or bool(row.get("track_stale"))))
        sequence_id = str(row.get("sequence_id"))
        if bool(row.get("occluded")):
            occlusion_sequences.add(sequence_id)
        if bool(row.get("reacquired")):
            reacquired_sequences.add(sequence_id)
        last_by_track[key] = row
    return {
        "frames": len(rows),
        "duplicate_frame_miscounts": duplicate_miscounts,
        "episode_target_cross_talk": cross_talk,
        "confirmed_false_target_rate": _rate(false_confirmed, confirmed_frames),
        "occlusion_sequences": len(occlusion_sequences),
        "reacquired_sequences": len(reacquired_sequences),
        "occlusion_reacquired_sequences": len(reacquired_sequences & occlusion_sequences),
        "reacquisition_rate": _rate(len(reacquired_sequences & occlusion_sequences), len(occlusion_sequences)),
        "stale_track_action_count": stale_actions,
    }


def tracking_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    failures = []
    if int(metrics.get("duplicate_frame_miscounts", 0)) != 0:
        failures.append("duplicate frame counted as a hit")
    if int(metrics.get("episode_target_cross_talk", 0)) != 0:
        failures.append("episode/target track cross-talk")
    if float(metrics.get("confirmed_false_target_rate", 1.0)) > 0.05:
        failures.append("confirmed false target rate > 0.05")
    if float(metrics.get("reacquisition_rate", 0.0)) < 0.90:
        failures.append("occlusion reacquisition rate < 0.90")
    if int(metrics.get("stale_track_action_count", 0)) != 0:
        failures.append("stale track triggered an action")
    return {"pass": not failures, "failures": failures, "metrics": metrics}


def evaluate_planning(route_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_correct = sum(bool(row.get("entered_correct_branch")) for row in route_rows)
    side_totals = {side: sum(row.get("expected") == side for row in route_rows) for side in ("left", "right")}
    side_correct = {
        side: sum(row.get("expected") == side and bool(row.get("entered_correct_branch")) for row in route_rows)
        for side in ("left", "right")
    }
    positive = [row for row in semantic_rows if bool(row.get("expected"))]
    negative = [row for row in semantic_rows if not bool(row.get("expected"))]
    approach_steps = [step for row in positive for step in row.get("approach_steps", [])]
    decreasing = sum(float(step.get("distance_after", math.inf)) < float(step.get("distance_before", -math.inf)) for step in approach_steps)
    threshold_latencies = [
        float(row["threshold_to_safe_stop_sec"])
        for row in positive
        if row.get("threshold_to_safe_stop_sec") is not None
    ]
    safety_keys = ("collision", "runtime_stale", "timebase_error", "stale_action_executed")
    safety = {key: sum(int(row.get(key, 0)) for row in route_rows + semantic_rows) for key in safety_keys}
    return {
        "route_correct": route_correct,
        "route_total": len(route_rows),
        "route_side_correct": side_correct,
        "route_side_total": side_totals,
        "confirmed_wrong_branch_entries": sum(
            bool(row.get("decision_confirmed")) and not bool(row.get("entered_correct_branch")) for row in route_rows
        ),
        "semantic_positive_coverage": sum(bool(row.get("entered_coverage")) for row in positive),
        "semantic_positive_total": len(positive),
        "semantic_success": sum(bool(row.get("success")) for row in positive),
        "semantic_negative_false_stop": sum(bool(row.get("stop")) for row in negative),
        "semantic_negative_total": len(negative),
        "approach_distance_decrease_rate": _rate(decreasing, len(approach_steps)),
        "threshold_to_safe_stop_p95_sec": percentile(threshold_latencies, 0.95),
        "normal_chain_rows": sum(bool(row.get("normal_chain_complete")) for row in route_rows + semantic_rows),
        "total_rows": len(route_rows) + len(semantic_rows),
        "safety": safety,
    }


def planning_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    failures = []
    if int(metrics.get("route_correct", 0)) < 24:
        failures.append("route correct branch < 24/30")
    for side in ("left", "right"):
        if int((metrics.get("route_side_correct") or {}).get(side, 0)) < 12:
            failures.append(f"route {side} correct < 12/15")
    if int(metrics.get("confirmed_wrong_branch_entries", 0)) != 0:
        failures.append("confirmed decision entered wrong branch")
    if int(metrics.get("semantic_positive_coverage", 0)) != int(metrics.get("semantic_positive_total", -1)):
        failures.append("positive semantic coverage is not complete")
    if int(metrics.get("semantic_success", 0)) < 14:
        failures.append("semantic success < 14/15")
    if int(metrics.get("semantic_negative_false_stop", 0)) != 0:
        failures.append("negative semantic false stop != 0")
    if float(metrics.get("approach_distance_decrease_rate", 0.0)) < 0.80:
        failures.append("approach distance decrease rate < 0.80")
    latency = metrics.get("threshold_to_safe_stop_p95_sec")
    if latency is None or float(latency) > THRESHOLD_TO_SAFE_STOP_P95_MAX_SEC:
        failures.append(
            "threshold-to-safe-stop p95 > "
            f"{THRESHOLD_TO_SAFE_STOP_P95_MAX_SEC:g} sec"
        )
    if int(metrics.get("normal_chain_rows", 0)) != int(metrics.get("total_rows", -1)):
        failures.append("one or more actions bypassed the normal safety chain")
    for key, value in (metrics.get("safety") or {}).items():
        if int(value) != 0:
            failures.append(f"{key} != 0")
    return {"pass": not failures, "failures": failures, "metrics": metrics}


def evaluate_robustness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {"delay", "drop", "duplicate", "out_of_order", "reset", "episode_mismatch", "timestamp_mismatch", "horizontal_flip"}
    passed = {str(row.get("profile")) for row in rows if bool(row.get("pass"))}
    stale_action = sum(int(row.get("stale_action_executed", 0)) for row in rows)
    old_track_clear = [bool(row.get("old_track_cleared")) for row in rows if row.get("profile") == "reset"]
    fallback = sum(int(row.get("fallback_or_mock", 0)) for row in rows)
    latencies = [float(row["step_latency_sec"]) for row in rows if row.get("step_latency_sec") is not None]
    failures = []
    missing = sorted(required - passed)
    if missing:
        failures.append(f"missing/passing profiles: {', '.join(missing)}")
    if stale_action:
        failures.append("stale_action_executed != 0")
    if not old_track_clear or not all(old_track_clear):
        failures.append("reset old-track clearing != 100%")
    if fallback:
        failures.append("fallback/mock entered formal evidence")
    p95 = percentile(latencies, 0.95)
    if p95 is None or p95 > STEP_REQUEST_LATENCY_P95_MAX_SEC:
        failures.append(
            f"Step request p95 latency > {STEP_REQUEST_LATENCY_P95_MAX_SEC:g} sec"
        )
    return {
        "pass": not failures,
        "profiles_passed": sorted(passed),
        "missing_profiles": missing,
        "stale_action_executed": stale_action,
        "old_track_clear_rate": _rate(sum(old_track_clear), len(old_track_clear)),
        "fallback_or_mock": fallback,
        "step_latency_p95_sec": p95,
        "failures": failures,
    }


def write_artifact_manifest(output: Path, *, run_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        artifacts.append(
            {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "internnav_identity": "CmaAgent/system1/fallback_static_cma_tokens",
        "real_go2_connected": False,
        "metadata": metadata,
        "artifacts": artifacts,
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, min(len(ordered), math.ceil(float(quantile) * len(ordered))))
    return round(ordered[rank - 1], 6)


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _label_attributes(label: str) -> tuple[str, str]:
    words = str(label).lower().replace("_", " ").split()
    colors = {"red", "blue", "yellow", "green", "orange", "purple", "black", "white", "gray", "grey", "silver"}
    default_color = "red" if "extinguisher" in words or "hydrant" in words else "white"
    color = next((word for word in words if word in colors), default_color)
    category = "_".join(word for word in words if word not in colors) or "box"
    aliases = {
        "hydrant": "cylinder",
        "fire_extinguisher": "fire_extinguisher",
        "traffic_cone": "traffic_cone",
        "cone": "cone",
        "exit_sign": "exit_sign",
        "first_aid_sign": "sign",
        "purple_placard": "sign",
    }
    return color, aliases.get(category, category)
