from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


CONFIG_DIFF_FIELDS = [
    ("mission.max_duration_sec", "mission_timeout"),
    ("omninav.enabled", "omninav_enabled"),
    ("omninav.max_inflight_requests", "max_inflight_requests"),
    ("omninav.queue_size", "queue_size"),
    ("omninav.max_action_age_sec", "omninav_max_action_age"),
    ("omninav.frame_count", "frame_count"),
    ("omninav.use_left_right", "use_left_right"),
    ("omninav.use_history", "use_history"),
    ("model_clients.omninav.config_name", "omninav_config_name"),
    ("model_clients.omninav.ttl_sec", "model_ttl"),
    ("model_clients.omninav.synthetic_fallback", "synthetic_fallback"),
    ("model_clients.omninav.stop_arrive_threshold", "stop_arrive_threshold"),
    ("model_clients.omninav.turn_threshold_deg", "turn_threshold_deg"),
    ("model_clients.omninav.hard_turn_threshold_deg", "hard_turn_threshold_deg"),
    ("model_clients.omninav.waypoint_forward_axis", "waypoint_forward_axis"),
    ("model_clients.omninav.forward_yaw_threshold_deg", "forward_yaw_threshold_deg"),
    ("model_clients.omninav.forward_min_x_m", "forward_min_x_m"),
    ("model_clients.omninav.forward_min_m", "forward_min_m"),
    ("model_clients.omninav.history_limit", "history_limit"),
    ("primitive.normal_speed_mps", "normal_speed_mps"),
    ("primitive.thinking_speed_mps", "thinking_speed_mps"),
    ("primitive.max_linear_x", "max_linear_x"),
    ("primitive.max_linear_y", "max_linear_y"),
    ("primitive.max_yaw_rate", "max_yaw_rate"),
    ("primitive.max_forward_horizon_m", "max_forward_horizon_m"),
    ("primitive.cmd_vel_ttl_sec", "cmd_vel_ttl"),
    ("primitive.lane_centering_enabled", "lane_centering_enabled"),
    ("pending_policy.stop_if_near_goal", "near_goal_stop"),
    ("route_choice_bridge.ttl_sec", "route_stop_bridge_ttl"),
    ("semantic_stop_bridge.ttl_sec", "semantic_stop_gate_ttl"),
    ("stale_gate.omninav_pose_delta_m", "stale_pose_delta"),
    ("stale_gate.omninav_yaw_delta_deg", "stale_yaw_delta"),
    ("safety.dry_run", "dry_run"),
    ("safety.output_topic", "output_topic"),
    ("safety.require_enable_motion_env", "require_enable_motion_env"),
]


def cmd_vel_for_primitive(primitive: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    kind = str(primitive.get("primitive") or "stop")
    normal_speed = _float(deep_get(config, "primitive.normal_speed_mps"), 0.20)
    thinking_speed = _float(deep_get(config, "primitive.thinking_speed_mps"), 0.12)
    max_linear_x = _float(deep_get(config, "primitive.max_linear_x"), 0.25)
    max_yaw_rate = _float(deep_get(config, "primitive.max_yaw_rate"), 0.35)
    max_forward = _float(deep_get(config, "primitive.max_forward_horizon_m"), 0.75)
    linear_x = 0.0
    angular_z = 0.0
    if kind == "move_forward":
        distance = min(abs(_float(primitive.get("distance_m"), 0.25)), max_forward)
        linear_x = min(normal_speed, max_linear_x) if distance > 0.0 else 0.0
    elif kind == "follow_waypoint":
        distance = min(abs(_float(primitive.get("distance_m"), 0.25)), max_forward)
        yaw = max(-45.0, min(45.0, _float(primitive.get("yaw_deg"), 0.0)))
        linear_x = min(normal_speed, max_linear_x) if distance > 0.0 else 0.0
        if abs(yaw) > 20.0:
            linear_x *= 0.5
        angular_z = max(-max_yaw_rate, min(max_yaw_rate, math.radians(yaw)))
    elif kind == "back_off":
        linear_x = -min(thinking_speed, max_linear_x)
    elif kind == "turn_left":
        angular_z = min(max_yaw_rate, math.radians(min(abs(_float(primitive.get("yaw_deg"), 15.0)), 45.0)))
    elif kind == "turn_right":
        angular_z = -min(max_yaw_rate, math.radians(min(abs(_float(primitive.get("yaw_deg"), 15.0)), 45.0)))
    elif kind == "look_around":
        angular_z = min(max_yaw_rate, _float(deep_get(config, "pending_policy.scan_yaw_rate"), 0.15))
    return {"linear_x": float(linear_x), "angular_z": float(angular_z)}


def evaluate_action_parser_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("parsed_primitive") or "unknown") for row in records)
    sample_count = len(records)
    fallback_turn_left = sum(
        1 for row in records if bool(row.get("fallback_used")) and str(row.get("parsed_primitive")) == "turn_left"
    )
    fallback_count = sum(1 for row in records if bool(row.get("fallback_used")))
    move_forward_rows = [row for row in records if str(row.get("parsed_primitive")) == "move_forward"]
    cmd_forward_ok = all(_float(deep_get(row, "cmd_vel_candidate.linear_x"), 0.0) > 0.0 for row in move_forward_rows)
    fallback_turn_left_rate = fallback_turn_left / max(1, fallback_count)
    metrics = {
        "benchmark": "probe_omninav_action_parser",
        "samples": sample_count,
        "move_forward_count": counts.get("move_forward", 0),
        "follow_waypoint_count": counts.get("follow_waypoint", 0),
        "turn_left_count": counts.get("turn_left", 0),
        "turn_right_count": counts.get("turn_right", 0),
        "stop_count": counts.get("stop", 0),
        "unknown_count": counts.get("unknown", 0),
        "fallback_count": fallback_count,
        "fallback_turn_left_rate": fallback_turn_left_rate,
        "cmd_move_forward_linear_ok": cmd_forward_ok,
    }
    failures: list[str] = []
    if sample_count <= 0:
        failures.append("no samples")
    if metrics["move_forward_count"] <= 0:
        failures.append("move_forward_count <= 0")
    if fallback_turn_left_rate >= 0.5:
        failures.append("fallback_turn_left_rate >= 0.5")
    if metrics["unknown_count"] != 0:
        failures.append("unknown_count != 0")
    if not cmd_forward_ok:
        failures.append("move_forward produced nonpositive cmd_vel_candidate.linear_x")
    metrics["pass"] = not failures
    metrics["failures"] = failures
    return metrics


def evaluate_safe_mux_clamp_records(records: list[dict[str, Any]], *, path_m: float, collision_count: int = 0) -> dict[str, Any]:
    candidate_max = max([abs(_float(deep_get(row, "cmd_vel_candidate.linear_x"), 0.0)) for row in records] or [0.0])
    safe_max = max([abs(_float(deep_get(row, "safe_cmd_vel.linear_x"), 0.0)) for row in records] or [0.0])
    stale = sum(1 for row in records if bool(row.get("stale")) or str(row.get("safety_reason") or "") == "stale_cmd")
    metrics = {
        "benchmark": "probe_safe_mux_clamp",
        "samples": len(records),
        "max_candidate_linear_x_mps": candidate_max,
        "max_safe_linear_x_mps": safe_max,
        "max_linear_x_mps": safe_max,
        "path_m": path_m,
        "collision_count": int(collision_count),
        "stale_discard_count": int(stale),
    }
    failures: list[str] = []
    if candidate_max <= 0.0:
        failures.append("cmd_vel_candidate.linear.x <= 0")
    if safe_max <= 0.0:
        failures.append("safe_cmd_vel.linear.x <= 0")
    if path_m <= 0.3:
        failures.append("path_m <= 0.3")
    if collision_count != 0:
        failures.append("collision_count != 0")
    if stale != 0:
        failures.append("stale_discard_count != 0")
    metrics["pass"] = not failures
    metrics["failures"] = failures
    return metrics


def config_diff_rows(configs: dict[str, dict[str, Any]], fields: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    fields = fields or CONFIG_DIFF_FIELDS
    rows: list[dict[str, Any]] = []
    for dotted, label in fields:
        values = {name: deep_get(cfg, dotted) for name, cfg in configs.items()}
        normalized = {name: _normalize(value) for name, value in values.items()}
        changed = len(set(normalized.values())) > 1
        risk = classify_config_risk(label, values)
        rows.append({"field": dotted, "label": label, "changed": changed, "risk": risk, **values})
    return rows


def classify_config_risk(label: str, values: dict[str, Any]) -> str:
    numeric_zero_risks = {"max_linear_x", "normal_speed_mps", "cmd_vel_ttl", "model_ttl", "omninav_max_action_age"}
    latest_name = "v6" if "v6" in values else ("v5" if "v5" in values else next(reversed(values), ""))
    latest = values.get(latest_name)
    if label in numeric_zero_risks and _float(latest, 0.0) <= 0.0:
        return "high:max_or_ttl_zero"
    if label == "dry_run" and bool(latest):
        return "high:dry_run_enabled"
    if label == "synthetic_fallback" and bool(latest):
        return "medium:synthetic_image_fallback"
    if label in {"use_history", "history_limit", "frame_count", "use_left_right"} and len({_normalize(v) for v in values.values()}) > 1:
        return "medium:image_bundle_changed"
    if label in {"near_goal_stop", "route_stop_bridge_ttl", "semantic_stop_gate_ttl"} and len({_normalize(v) for v in values.values()}) > 1:
        return "medium:gate_behavior_changed"
    if len({_normalize(v) for v in values.values()}) > 1:
        return "changed"
    return "none"


def classify_first10_root_cause(records: list[dict[str, Any]]) -> str:
    if not records:
        return "queue_empty_or_no_model_response"
    model_rows = [row for row in records if row.get("omninav_raw_output") or row.get("parsed_primitive")]
    if not model_rows:
        return "queue_empty_or_no_model_response"
    stale_rows = [row for row in records if str(row.get("stale_decision") or "").lower() in {"discarded", "stale"}]
    if len(stale_rows) >= max(1, len(model_rows) // 2):
        return "stale_discards_actions"
    if any(_raw_forward(row) and str(row.get("parsed_primitive")) in {"turn_left", "turn_right"} for row in model_rows):
        return "parser_maps_forward_to_turn"
    primitives = [str(row.get("parsed_primitive") or "") for row in model_rows if row.get("parsed_primitive")]
    if primitives and all(item in {"turn_left", "turn_right", "stop"} for item in primitives) and not any(_raw_forward(row) for row in model_rows):
        if any(item.startswith("turn_") for item in primitives):
            return "model_outputs_turn_only"
    candidate_max = max([abs(_float(deep_get(row, "cmd_vel_candidate.linear_x"), 0.0)) for row in records] or [0.0])
    safe_max = max([abs(_float(deep_get(row, "safe_cmd_vel.linear_x"), 0.0)) for row in records] or [0.0])
    if candidate_max > 0.0 and safe_max <= 1e-4:
        return "safe_mux_clamps_linear_x"
    pose_delta = max([abs(_float(row.get("pose_delta"), 0.0)) for row in records] or [0.0])
    if safe_max > 0.0 and pose_delta < 0.05:
        return "isaac_does_not_move_despite_cmd"
    return "unknown"


def evaluate_omninav_regression_v6(metrics: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    checks = [
        ("success_count", _float(metrics.get("success_count"), 0.0), ">=", 6.0),
        ("mean_path_m", _float(metrics.get("mean_path_m"), 0.0), ">", 3.0),
        ("move_forward_count", _float(metrics.get("move_forward_count"), 0.0), ">", 0.0),
        ("max_linear_x_mps", _float(metrics.get("max_linear_x_mps"), 0.0), ">", 0.0),
        ("stale_discard_count", _float(metrics.get("stale_discard_count"), 999.0), "<=", 20.0),
        ("timebase_error_count", _float(metrics.get("timebase_error_count"), 999.0), "==", 0.0),
        ("collision_count", _float(metrics.get("collision_count"), 999.0), "==", 0.0),
        ("stale_action_executed", _float(metrics.get("stale_action_executed"), 999.0), "==", 0.0),
    ]
    for name, actual, op, expected in checks:
        if op == ">=" and actual < expected:
            failures.append(f"{name}={actual:g} < {expected:g}")
        elif op == ">" and not actual > expected:
            failures.append(f"{name}={actual:g} <= {expected:g}")
        elif op == "<=" and actual > expected:
            failures.append(f"{name}={actual:g} > {expected:g}")
        elif op == "==" and actual != expected:
            failures.append(f"{name}={actual:g} != {expected:g}")
    return metrics | {"pass": not failures, "failures": failures}


def evaluate_sim2real_readiness_v6(metrics: dict[str, Any], gate_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    gate_cfg = gate_cfg or {}
    regression = evaluate_omninav_regression_v6(
        {
            "success_count": metrics.get("success_count", metrics.get("omninav_success_count", 0)),
            "mean_path_m": metrics.get("mean_path_m", 0),
            "move_forward_count": metrics.get("move_forward_count", 0),
            "max_linear_x_mps": metrics.get("max_linear_x_mps", 0),
            "stale_discard_count": metrics.get("stale_discard_count", 999),
            "timebase_error_count": metrics.get("timebase_error_count", 999),
            "collision_count": metrics.get("collision_count", 999),
            "stale_action_executed": metrics.get("stale_action_executed", 999),
        }
    )
    failures = list(regression["failures"])
    max_linear_x = _float(metrics.get("max_linear_x_mps"), -1.0)
    if max_linear_x > _float(gate_cfg.get("max_linear_x_mps"), 0.20):
        failures.append(f"max_linear_x_mps={max_linear_x:g} > {_float(gate_cfg.get('max_linear_x_mps'), 0.20):g}")
    for name in ("episode_mismatch_count", "missing_timestamp_count"):
        if int(metrics.get(name, 999)) != 0:
            failures.append(f"{name}={metrics.get(name)} != 0")
    route_stop_claimed = bool(metrics.get("route_stop_full_claimed") or metrics.get("semantic_nav_claimed"))
    if route_stop_claimed:
        if not bool(metrics.get("route_stop_full_passed", False)):
            failures.append("route_stop_full_passed is not true")
        if not bool(metrics.get("step_full_passed", False)):
            failures.append("step_full_passed is not true")
    return {
        "ready": not failures,
        "status": "READY FOR SENSOR-ONLY DRY-RUN" if not failures else "NOT READY FOR REAL ROBOT AUTONOMY",
        "failures": failures,
        "allowed_next_steps": [
            "sensor-only dry-run",
            "bag replay",
            "offline scoring",
            "stationary camera validation",
            "manual-triggered primitive test",
        ],
        "disallowed_next_steps": [
            "real robot autonomous navigation",
            "dynamic obstacle real test",
            "semantic target approach on real robot",
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = headers or sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_probe_summary(path: Path, title: str, metrics: dict[str, Any], extra_lines: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for key in sorted(metrics):
        if key == "failures":
            continue
        lines.append(f"- {key}: {metrics[key]}")
    failures = metrics.get("failures")
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend([f"- {item}" for item in failures])
    if extra_lines:
        lines.extend(["", *extra_lines])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deep_get(config: dict[str, Any] | None, dotted: str, default: Any = None) -> Any:
    cur: Any = config or {}
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _raw_forward(row: dict[str, Any]) -> bool:
    raw_action = str(row.get("raw_action") or row.get("omninav_raw_output") or "").lower()
    if "forward" in raw_action or "move" in raw_action:
        return True
    waypoint = row.get("raw_waypoint")
    if isinstance(waypoint, (list, tuple)) and len(waypoint) >= 2:
        x = _float(waypoint[0], 0.0)
        y = _float(waypoint[1], 0.0)
        axis = str(row.get("waypoint_forward_axis") or "").lower()
        if axis == "y":
            forward, lateral = y, x
        else:
            forward, lateral = x, y
        return forward > 0.05 and abs(math.degrees(math.atan2(lateral, forward))) < 25.0
    return False


def _normalize(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number
