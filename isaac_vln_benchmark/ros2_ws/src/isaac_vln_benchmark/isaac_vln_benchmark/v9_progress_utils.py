from __future__ import annotations

import json
import math
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from .v8_coverage_utils import (
    ROUTE_COLUMNS,
    analyze_route_episode,
    analyze_stale_attribution_v8,
    analyze_stop_episode,
    cmd_angular_z,
    cmd_is_zero,
    cmd_linear_x,
    count_true,
    distance_at_time,
    episode_dir_for,
    event_time,
    events_for_episode,
    first_event,
    first_event_after,
    load_csv,
    load_json,
    load_jsonl,
    load_trajectory,
    load_yamlish,
    none_or_round,
    render_stale_report,
    truthy,
    visible_at_time,
    write_csv,
    _details,
    _event_type,
    _float,
)


REQUIRED_FACTS = [
    "V6 OmniNav baseline restored.",
    "V7 forced route/stop oracle full did not beat OmniNav golden.",
    "V8 route coverage was 0/6 triggered, 0/6 bridge, 0/6 turn, 0/6 correct branch.",
    "V8 deterministic injection was 0/6 because robot max x was about 1.81-1.88m, below x ~= 3.2m.",
    "V8 semantic stop target was visible 3/3, but distance stayed about 4.78-4.87m > 2.5m.",
    "Alias audit passed.",
    "runtime_stale=0 and timebase_error=0 must be preserved.",
    "InternNav remains CmaAgent/system1/fallback_static_cma_tokens, not full InternVLA-N1.",
    "Step remains frozen in V9.",
    "Sim2Real remains NOT READY FOR REAL ROBOT AUTONOMY.",
]

REACHABILITY_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "seed",
    "start_pose",
    "max_x",
    "max_abs_y",
    "path_m",
    "time_to_x_2m",
    "time_to_x_3m",
    "time_to_trigger_window",
    "reached_injection_window",
    "near_intersection_detected",
    "local_costmap_block_ticks",
    "safety_block_events",
    "safe_cmd_vel_linear_nonzero_count",
    "safe_cmd_vel_angular_nonzero_count",
    "stale_discards",
    "failure_stage",
]

INJECTION_COLUMNS = ROUTE_COLUMNS + [
    "trigger_window",
    "max_x",
    "route_oracle_trigger_time",
    "distance_to_branch_goal",
]

SEMANTIC_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "seed",
    "target_visible_first_time",
    "distance_at_first_visible",
    "min_distance_to_target",
    "reached_stop_threshold",
    "time_to_stop_threshold",
    "stop_oracle_triggered",
    "bridge_received_stop",
    "safe_cmd_vel_zero_time",
    "robot_velocity_below_threshold_time",
    "visible_to_stop_latency",
    "semantic_target_success",
    "failure_stage",
]

WATCHDOG_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "seed",
    "task_type",
    "assist_triggered",
    "assist_reason",
    "assist_primitive",
    "assist_time",
    "path_after_assist",
    "reached_trigger_after_assist",
    "success_after_assist",
    "failure_stage",
]

STALE_COLUMNS = ["class", "episode_id", "event_type", "t", "reason"]

TRAJECTORY_COLUMNS = ["episode_id", "mode", "task_id", "t", "x", "y", "yaw", "source"]


def postprocess_route_reachability_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    rows = route_reachability_rows(output, metrics)
    stale = write_common_v9_artifacts(output, metrics, route_rows=rows)
    summary = evaluate_route_reachability(rows, metrics, stale)
    metrics["v9_route_trigger_reachability"] = summary
    metrics["v9_stale_attribution"] = stale
    _write_v9_metrics(output, metrics)
    (output / "summary.md").write_text(
        render_v9_summary(
            title="V9 Route Trigger Reachability",
            output=output,
            route_reachability=summary,
            stale=stale,
            next_fix="If forced_forward cannot reach the trigger window, debug local safety, geometry, speed, or primitive path before any Step work.",
        ),
        encoding="utf-8",
    )
    return metrics


def postprocess_route_injection_sweep_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    rows = route_injection_sweep_rows(output, metrics)
    stale = write_common_v9_artifacts(output, metrics, injection_rows=rows)
    summary = evaluate_route_injection_sweep(rows, metrics, stale)
    metrics["v9_route_injection_sweep"] = summary
    metrics["v9_v10_route_gate_pass"] = bool(summary.get("pass"))
    metrics["v9_stale_attribution"] = stale
    _write_v9_metrics(output, metrics)
    (output / "summary.md").write_text(
        render_v9_summary(
            title="V9 Route Injection Sweep",
            output=output,
            injection_sweep=summary,
            stale=stale,
            next_fix="Use the earliest effective window if one passes; otherwise debug bridge/primitive/yaw/judge before full oracle.",
        ),
        encoding="utf-8",
    )
    return metrics


def postprocess_semantic_approach_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    rows = semantic_approach_rows(output, metrics)
    stale = write_common_v9_artifacts(output, metrics, semantic_rows=rows)
    summary = evaluate_semantic_approach(rows, metrics, stale)
    metrics["v9_semantic_approach_to_stop"] = summary
    metrics["v9_v10_semantic_gate_pass"] = bool(summary.get("pass"))
    metrics["v9_stale_attribution"] = stale
    _write_v9_metrics(output, metrics)
    (output / "summary.md").write_text(
        render_v9_summary(
            title="V9 Semantic Approach-To-Stop Threshold",
            output=output,
            semantic_approach=summary,
            stale=stale,
            next_fix="If forced approach cannot enter the 2.5m threshold, debug approach motion, scene geometry, or success threshold before Step stop.",
        ),
        encoding="utf-8",
    )
    return metrics


def postprocess_progress_watchdog_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    rows = progress_watchdog_rows(output, metrics)
    stale = write_common_v9_artifacts(output, metrics, watchdog_rows=rows)
    summary = evaluate_progress_watchdog(rows, metrics, stale)
    metrics["v9_progress_to_trigger_watchdog"] = summary
    metrics["v9_stale_attribution"] = stale
    _write_v9_metrics(output, metrics)
    (output / "summary.md").write_text(
        render_v9_summary(
            title="V9 Progress-To-Trigger Watchdog",
            output=output,
            watchdog=summary,
            stale=stale,
            next_fix="If assist triggers but does not unlock route/stop, debug safety blocking and primitive motion before larger benchmarks.",
        ),
        encoding="utf-8",
    )
    return metrics


def route_reachability_rows(output: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        if str(episode.get("task_type") or "") not in {"turn_choice", "turn_microbench"}:
            continue
        episode_dir = episode_dir_for(output, episode)
        scene = load_yamlish(episode_dir / "scene.yaml")
        events = load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode)
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        rows.append(analyze_route_reachability_episode(episode, scene, events, trajectory))
    return rows


def analyze_route_reachability_episode(
    episode: dict[str, Any],
    scene: dict[str, Any],
    events: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    trigger_x = route_trigger_x_from_scene(scene)
    start = trajectory[0] if trajectory else {"x": 0.0, "y": 0.0, "yaw": 0.0}
    max_x = max((_float(row.get("x"), 0.0) for row in trajectory), default=0.0)
    max_abs_y = max((abs(_float(row.get("y"), 0.0)) for row in trajectory), default=0.0)
    reached = max_x >= trigger_x or near_intersection_detected(events)
    safe_linear = sum(1 for event in events if _event_type(event) in {"safe_cmd_vel", "safe_cmd_mux"} and abs(cmd_linear_x(_details(event))) > 0.02)
    safe_angular = sum(1 for event in events if _event_type(event) in {"safe_cmd_vel", "safe_cmd_mux"} and abs(cmd_angular_z(_details(event))) > 0.02)
    local_blocks = sum(1 for event in events if _event_type(event) == "safety_local_status_json" and _details(event).get("local_costmap_clear") is False)
    safety_blocks = sum(1 for event in events if _event_type(event) == "safe_cmd_mux" and str(_details(event).get("result") or "") in {"safety_stop", "failsafe"})
    stale = analyze_stale_attribution_v8(events)
    row = {
        "episode_id": episode.get("episode_id"),
        "mode": episode.get("mode"),
        "task_id": episode.get("task_id"),
        "seed": seed_from_task_id(episode.get("task_id")),
        "start_pose": json.dumps([start.get("x"), start.get("y"), start.get("yaw")], separators=(",", ":")),
        "max_x": round(max_x, 3),
        "max_abs_y": round(max_abs_y, 3),
        "path_m": round(path_length_from_trajectory(trajectory) or _float(episode.get("path_length_m"), 0.0), 3),
        "time_to_x_2m": none_or_round(time_to_x(trajectory, 2.0)),
        "time_to_x_3m": none_or_round(time_to_x(trajectory, 3.0)),
        "time_to_trigger_window": none_or_round(time_to_x(trajectory, trigger_x)),
        "reached_injection_window": reached,
        "near_intersection_detected": near_intersection_detected(events),
        "local_costmap_block_ticks": local_blocks,
        "safety_block_events": safety_blocks,
        "safe_cmd_vel_linear_nonzero_count": safe_linear,
        "safe_cmd_vel_angular_nonzero_count": safe_angular,
        "stale_discards": stale.get("total_stale_discards", 0),
    }
    row["failure_stage"] = reachability_failure_stage(row, episode, stale)
    return row


def route_injection_sweep_rows(output: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        if str(episode.get("task_type") or "") not in {"turn_choice", "turn_microbench"}:
            continue
        episode_dir = episode_dir_for(output, episode)
        task = load_yamlish(episode_dir / "task.yaml")
        scene = load_yamlish(episode_dir / "scene.yaml")
        events = load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode)
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        row = analyze_route_episode(episode, task, scene, events, trajectory)
        row["trigger_window"] = trigger_window_from_episode(episode, events)
        row["max_x"] = round(max((_float(item.get("x"), 0.0) for item in trajectory), default=0.0), 3)
        trigger_event = first_event(
            events,
            lambda e, d, t: t in {"oracle_route_choice_json", "route_oracle_json_published", "v9_route_injection_window"}
            or (t == "metrics_event_jsonl" and d.get("event_type") in {"route_oracle_json_published", "v9_route_injection_window"}),
        )
        row["route_oracle_trigger_time"] = none_or_round(event_time(trigger_event))
        row["distance_to_branch_goal"] = none_or_round(distance_to_branch_goal(task, scene, trajectory))
        row["failure_stage"] = injection_failure_stage(row)
        rows.append(row)
    return rows


def semantic_approach_rows(output: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        if str(episode.get("task_type") or "") != "semantic_target":
            continue
        episode_dir = episode_dir_for(output, episode)
        task = load_yamlish(episode_dir / "task.yaml")
        scene = load_yamlish(episode_dir / "scene.yaml")
        events = load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode)
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        base = analyze_stop_episode(episode, task, scene, events, trajectory)
        min_dist = min_distance_to_target(events, episode)
        threshold = 2.5
        threshold_event = first_event(
            events,
            lambda e, d, t: (
                d.get("event_type") == "v9_semantic_stop_threshold_reached"
                or (d.get("distance_to_target") is not None and _float(d.get("distance_to_target"), 999.0) <= threshold)
            ),
        )
        row = {
            "episode_id": base.get("episode_id"),
            "mode": base.get("mode"),
            "task_id": base.get("task_id"),
            "seed": seed_from_task_id(base.get("task_id")),
            "target_visible_first_time": base.get("target_visible_first_time"),
            "distance_at_first_visible": base.get("distance_at_first_visible"),
            "min_distance_to_target": none_or_round(min_dist),
            "reached_stop_threshold": min_dist is not None and min_dist <= threshold,
            "time_to_stop_threshold": none_or_round(event_time(threshold_event)),
            "stop_oracle_triggered": base.get("semantic_stop_oracle_triggered"),
            "bridge_received_stop": base.get("bridge_received_stop"),
            "safe_cmd_vel_zero_time": base.get("safe_cmd_vel_zero_time"),
            "robot_velocity_below_threshold_time": base.get("robot_velocity_below_threshold_time"),
            "visible_to_stop_latency": base.get("visible_to_stop_latency"),
            "semantic_target_success": base.get("semantic_target_success"),
        }
        row["failure_stage"] = semantic_failure_stage(row)
        rows.append(row)
    return rows


def progress_watchdog_rows(output: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        episode_dir = episode_dir_for(output, episode)
        scene = load_yamlish(episode_dir / "scene.yaml")
        events = load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode)
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        assist = first_event(
            events,
            lambda e, d, t: t in {"v9_progress_watchdog", "v9_progress_assist_forward"}
            or d.get("event_type") in {"v9_progress_watchdog", "v9_progress_assist_forward"},
        )
        assist_t = event_time(assist)
        details = _details(assist)
        reason = str(details.get("assist_reason") or details.get("reason") or "")
        primitive = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
        reached = reached_after_assist(episode, scene, events, trajectory, assist_t)
        path_after = path_length_from_trajectory([row for row in trajectory if assist_t is None or _float(row.get("t"), 0.0) >= assist_t])
        success_after = bool(episode.get("success")) and assist_t is not None
        row = {
            "episode_id": episode.get("episode_id"),
            "mode": episode.get("mode"),
            "task_id": episode.get("task_id"),
            "seed": seed_from_task_id(episode.get("task_id")),
            "task_type": episode.get("task_type"),
            "assist_triggered": assist is not None,
            "assist_reason": reason,
            "assist_primitive": primitive.get("primitive") if primitive else "",
            "assist_time": none_or_round(assist_t),
            "path_after_assist": none_or_round(path_after),
            "reached_trigger_after_assist": reached,
            "success_after_assist": success_after and reached,
        }
        row["failure_stage"] = watchdog_failure_stage(row)
        rows.append(row)
    return rows


def write_common_v9_artifacts(
    output: Path,
    metrics: dict[str, Any],
    *,
    route_rows: list[dict[str, Any]] | None = None,
    injection_rows: list[dict[str, Any]] | None = None,
    semantic_rows: list[dict[str, Any]] | None = None,
    watchdog_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    route_rows = route_rows or []
    injection_rows = injection_rows or []
    semantic_rows = semantic_rows or []
    watchdog_rows = watchdog_rows or []
    write_csv(output / "route_trigger_reachability.csv", route_rows, REACHABILITY_COLUMNS)
    write_csv(output / "route_injection_sweep.csv", injection_rows, INJECTION_COLUMNS)
    write_csv(output / "semantic_approach_probe.csv", semantic_rows, SEMANTIC_COLUMNS)
    write_csv(output / "progress_watchdog.csv", watchdog_rows, WATCHDOG_COLUMNS)
    chain_rows = injection_rows if injection_rows else route_rows if route_rows else semantic_rows if semantic_rows else watchdog_rows
    write_csv(output / "coverage_chain.csv", chain_rows, list(chain_rows[0].keys()) if chain_rows else ["empty"])
    stale = analyze_stale_attribution_v8(load_jsonl(output / "events.jsonl"))
    write_csv(output / "stale_attribution.csv", stale.get("records", []), STALE_COLUMNS)
    (output / "stale_discard_analysis.md").write_text(render_stale_report(stale), encoding="utf-8")
    write_aggregate_trajectory(output, metrics)
    write_visual_artifact(output, metrics, chain_rows)
    return stale


def evaluate_route_reachability(rows: list[dict[str, Any]], metrics: dict[str, Any], stale: dict[str, Any]) -> dict[str, Any]:
    forced = [row for row in rows if str(row.get("mode") or "") == "v9_forced_forward_until_trigger"]
    counts = count_true(forced, ["reached_injection_window"])
    linear_ok = sum(1 for row in forced if _float(row.get("safe_cmd_vel_linear_nonzero_count"), 0.0) > 0.0)
    collision_count = aggregate_collision_count(metrics)
    stale_action_executed = aggregate_stale_action_executed(metrics)
    failures: list[str] = []
    if len(forced) < 6:
        failures.append(f"forced_forward episodes={len(forced)} < 6")
    if counts["reached_injection_window"] < min(5, len(forced)):
        failures.append(f"forced_forward reached_injection_window={counts['reached_injection_window']} < {min(5, len(forced))}")
    if linear_ok < len(forced):
        failures.append(f"safe_cmd_vel_linear_nonzero episodes={linear_ok} < {len(forced)}")
    if collision_count != 0:
        failures.append(f"collision_count={collision_count} != 0")
    if stale_action_executed != 0:
        failures.append(f"stale_action_executed={stale_action_executed} != 0")
    if int(stale.get("runtime_stale_discards", 0) or 0) != 0:
        failures.append(f"runtime_stale_discards={stale.get('runtime_stale_discards')} != 0")
    if int(stale.get("timebase_error", 0) or 0) != 0:
        failures.append(f"timebase_error={stale.get('timebase_error')} != 0")
    return {
        "pass": not failures,
        "episodes": len(rows),
        "forced_forward_episodes": len(forced),
        "forced_forward_reached_injection_window": counts["reached_injection_window"],
        "forced_forward_linear_nonzero_episodes": linear_ok,
        "collision_count": collision_count,
        "stale_action_executed": stale_action_executed,
        "runtime_stale_discards": stale.get("runtime_stale_discards", 0),
        "timebase_error": stale.get("timebase_error", 0),
        "failures": failures,
    }


def evaluate_route_injection_sweep(rows: list[dict[str, Any]], metrics: dict[str, Any], stale: dict[str, Any]) -> dict[str, Any]:
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[str(row.get("trigger_window") or "")].append(row)
    window_order = ["x>=1.5", "x>=2", "x>=2.0", "x>=2.5", "x>=3", "x>=3.0", "intersection_x-0.8", "near_intersection"]
    window_summaries: dict[str, dict[str, Any]] = {}
    earliest = None
    for window, group in by_window.items():
        counts = count_true(
            group,
            [
                "route_oracle_triggered",
                "route_oracle_json_published",
                "route_stop_bridge_received_route",
                "turn_primitive_generated",
                "safe_cmd_vel_turn_nonzero",
                "entered_correct_branch",
            ],
        )
        passed = (
            len(group) >= 6
            and counts["route_stop_bridge_received_route"] >= 6
            and counts["turn_primitive_generated"] >= 6
            and counts["safe_cmd_vel_turn_nonzero"] >= 6
            and counts["entered_correct_branch"] >= 4
        )
        window_summaries[window] = {"episodes": len(group), "pass": passed, **counts}
    for window in window_order:
        key = next((candidate for candidate in window_summaries if normalize_window(candidate) == normalize_window(window)), None)
        if key and window_summaries[key]["pass"]:
            earliest = key
            break
    collision_count = aggregate_collision_count(metrics)
    stale_action_executed = aggregate_stale_action_executed(metrics)
    failures: list[str] = []
    if earliest is None:
        failures.append("no injection window reached entered_correct_branch>=4/6 with full bridge/primitive/safe coverage")
    if collision_count != 0:
        failures.append(f"collision_count={collision_count} != 0")
    if stale_action_executed != 0:
        failures.append(f"stale_action_executed={stale_action_executed} != 0")
    if int(stale.get("runtime_stale_discards", 0) or 0) != 0:
        failures.append(f"runtime_stale_discards={stale.get('runtime_stale_discards')} != 0")
    if int(stale.get("timebase_error", 0) or 0) != 0:
        failures.append(f"timebase_error={stale.get('timebase_error')} != 0")
    return {
        "pass": not failures,
        "episodes": len(rows),
        "earliest_effective_window": earliest,
        "windows": window_summaries,
        "collision_count": collision_count,
        "stale_action_executed": stale_action_executed,
        "runtime_stale_discards": stale.get("runtime_stale_discards", 0),
        "timebase_error": stale.get("timebase_error", 0),
        "failures": failures,
    }


def evaluate_semantic_approach(rows: list[dict[str, Any]], metrics: dict[str, Any], stale: dict[str, Any]) -> dict[str, Any]:
    forced = [row for row in rows if str(row.get("mode") or "") == "v9_semantic_forced_approach"]
    counts = count_true(
        forced,
        ["reached_stop_threshold", "stop_oracle_triggered", "bridge_received_stop", "safe_cmd_vel_zero_time", "semantic_target_success"],
    )
    collision_count = aggregate_collision_count(metrics)
    stale_action_executed = aggregate_stale_action_executed(metrics)
    failures: list[str] = []
    if len(forced) < 3:
        failures.append(f"forced_approach episodes={len(forced)} < 3")
    if counts["reached_stop_threshold"] < min(3, len(forced)):
        failures.append(f"forced_approach reached_stop_threshold={counts['reached_stop_threshold']} < {min(3, len(forced))}")
    if counts["stop_oracle_triggered"] < min(3, len(forced)):
        failures.append(f"forced_approach stop_oracle_triggered={counts['stop_oracle_triggered']} < {min(3, len(forced))}")
    if counts["bridge_received_stop"] < min(3, len(forced)):
        failures.append(f"forced_approach bridge_received_stop={counts['bridge_received_stop']} < {min(3, len(forced))}")
    if counts["safe_cmd_vel_zero_time"] < min(3, len(forced)):
        failures.append(f"forced_approach safe_cmd_vel_zero_time={counts['safe_cmd_vel_zero_time']} < {min(3, len(forced))}")
    if counts["semantic_target_success"] < min(2, len(forced)):
        failures.append(f"forced_approach semantic_target_success={counts['semantic_target_success']} < {min(2, len(forced))}")
    if collision_count != 0:
        failures.append(f"collision_count={collision_count} != 0")
    if stale_action_executed != 0:
        failures.append(f"stale_action_executed={stale_action_executed} != 0")
    if int(stale.get("runtime_stale_discards", 0) or 0) != 0:
        failures.append(f"runtime_stale_discards={stale.get('runtime_stale_discards')} != 0")
    if int(stale.get("timebase_error", 0) or 0) != 0:
        failures.append(f"timebase_error={stale.get('timebase_error')} != 0")
    return {
        "pass": not failures,
        "episodes": len(rows),
        "forced_approach_episodes": len(forced),
        **counts,
        "collision_count": collision_count,
        "stale_action_executed": stale_action_executed,
        "runtime_stale_discards": stale.get("runtime_stale_discards", 0),
        "timebase_error": stale.get("timebase_error", 0),
        "failures": failures,
    }


def evaluate_progress_watchdog(rows: list[dict[str, Any]], metrics: dict[str, Any], stale: dict[str, Any]) -> dict[str, Any]:
    counts = count_true(rows, ["assist_triggered", "reached_trigger_after_assist", "success_after_assist"])
    reasons = sorted({str(row.get("assist_reason") or "") for row in rows if row.get("assist_reason")})
    return {
        "pass": counts["assist_triggered"] > 0 and counts["reached_trigger_after_assist"] > 0,
        "episodes": len(rows),
        **counts,
        "assist_reasons": reasons,
        "collision_count": aggregate_collision_count(metrics),
        "stale_action_executed": aggregate_stale_action_executed(metrics),
        "runtime_stale_discards": stale.get("runtime_stale_discards", 0),
        "timebase_error": stale.get("timebase_error", 0),
        "failures": [
            item
            for item in [
                "assist_triggered=0" if counts["assist_triggered"] == 0 else "",
                "reached_trigger_after_assist=0" if counts["reached_trigger_after_assist"] == 0 else "",
            ]
            if item
        ],
    }


def render_v9_summary(
    *,
    title: str,
    output: Path,
    route_reachability: dict[str, Any] | None = None,
    injection_sweep: dict[str, Any] | None = None,
    semantic_approach: dict[str, Any] | None = None,
    watchdog: dict[str, Any] | None = None,
    stale: dict[str, Any],
    next_fix: str,
) -> str:
    lines = [f"# {title}", "", f"- run_dir: {output}", ""]
    lines.extend(["## Required Facts Preserved", ""])
    lines.extend([f"- {fact}" for fact in REQUIRED_FACTS])
    lines.extend(["", "## Route Trigger Reachability Result", ""])
    lines.extend(render_metric_block(route_reachability))
    lines.extend(["", "## Injection Window Sweep Result", ""])
    lines.extend(render_metric_block(injection_sweep))
    lines.extend(["", "## Semantic Approach-To-Stop Result", ""])
    lines.extend(render_metric_block(semantic_approach))
    lines.extend(["", "## Progress Watchdog Result", ""])
    lines.extend(render_metric_block(watchdog))
    lines.extend(["", "## Stale Attribution", ""])
    lines.append(f"- runtime_stale_discards: {stale.get('runtime_stale_discards')}")
    lines.append(f"- reset_cleanup_discards: {stale.get('reset_cleanup_discards')}")
    lines.append(f"- old_response_after_reset: {stale.get('old_response_after_reset')}")
    lines.append(f"- episode_mismatch: {stale.get('episode_mismatch')}")
    lines.append(f"- timebase_error: {stale.get('timebase_error')}")
    lines.extend(
        [
            "",
            "## Full Oracle / Step Gate",
            "",
            "- V10 forced oracle full retry is allowed only if an injection window reaches entered_correct_branch>=4/6 and forced semantic approach succeeds >=2/3.",
            "- Step remains frozen in V9.",
            "",
            "## Sim2Real Gate Status",
            "",
            "- NOT READY FOR REAL ROBOT AUTONOMY.",
            "- Allowed only: sensor-only dry-run, bag replay, offline scoring, stationary camera validation, manual-triggered primitive test.",
            "",
            "## Next Recommended Fix",
            "",
            f"- {next_fix}",
            "",
            "## Artifacts",
            "",
            f"- coverage_chain.csv: {output / 'coverage_chain.csv'}",
            f"- route_trigger_reachability.csv: {output / 'route_trigger_reachability.csv'}",
            f"- route_injection_sweep.csv: {output / 'route_injection_sweep.csv'}",
            f"- semantic_approach_probe.csv: {output / 'semantic_approach_probe.csv'}",
            f"- progress_watchdog.csv: {output / 'progress_watchdog.csv'}",
            f"- stale_attribution.csv: {output / 'stale_attribution.csv'}",
            f"- trajectory.csv: {output / 'trajectory.csv'}",
            f"- visual: {output / 'visual' / 'viewport.png'}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_metric_block(value: dict[str, Any] | None) -> list[str]:
    if not value:
        return ["- not run in this artifact"]
    compact = {key: val for key, val in value.items() if key not in {"windows"}}
    lines = [f"- {key}: {val}" for key, val in compact.items() if key != "failures"]
    failures = value.get("failures", [])
    lines.append(f"- failures: {', '.join(failures) if failures else 'none'}")
    if isinstance(value.get("windows"), dict):
        lines.append("- windows:")
        for window, row in value["windows"].items():
            lines.append(f"  - {window}: pass={row.get('pass')} entered_correct_branch={row.get('entered_correct_branch')}/{row.get('episodes')}")
    return lines


def reachability_failure_stage(row: dict[str, Any], episode: dict[str, Any], stale: dict[str, Any]) -> str | None:
    if _float(episode.get("num_collisions"), 0.0) > 0:
        return "collision"
    if int(stale.get("runtime_stale_discards", 0) or 0) > 0:
        return "runtime_stale"
    if _float(row.get("safe_cmd_vel_linear_nonzero_count"), 0.0) <= 0:
        return "no_safe_linear_motion"
    if not truthy(row.get("reached_injection_window")):
        if _float(row.get("safety_block_events"), 0.0) > 0 or _float(row.get("local_costmap_block_ticks"), 0.0) > 0:
            return "safety_blocked_before_trigger"
        return "trigger_window_not_reached"
    return None


def injection_failure_stage(row: dict[str, Any]) -> str | None:
    checks = [
        ("route_oracle_triggered", "route_oracle_not_triggered"),
        ("route_oracle_json_published", "route_oracle_not_published"),
        ("route_stop_bridge_received_route", "route_stop_bridge_missing"),
        ("turn_primitive_generated", "turn_primitive_missing"),
        ("safe_cmd_vel_turn_nonzero", "safe_cmd_vel_turn_missing"),
    ]
    for key, stage in checks:
        if not truthy(row.get(key)):
            return stage
    if _float(row.get("yaw_delta_after_turn_deg"), 0.0) < 5.0:
        return "isaac_state_no_yaw_change"
    if not truthy(row.get("entered_correct_branch")):
        return "wrong_branch"
    return None


def semantic_failure_stage(row: dict[str, Any]) -> str | None:
    checks = [
        ("target_visible_first_time", "target_never_visible"),
        ("reached_stop_threshold", "stop_threshold_not_reached"),
        ("stop_oracle_triggered", "semantic_stop_oracle_not_triggered"),
        ("bridge_received_stop", "semantic_stop_bridge_missing"),
        ("safe_cmd_vel_zero_time", "safe_cmd_vel_zero_missing"),
        ("semantic_target_success", "semantic_success_judge_failed"),
    ]
    for key, stage in checks:
        if not truthy(row.get(key)):
            return stage
    return None


def watchdog_failure_stage(row: dict[str, Any]) -> str | None:
    if not truthy(row.get("assist_triggered")):
        return "assist_not_triggered"
    if not truthy(row.get("reached_trigger_after_assist")):
        return "trigger_not_reached_after_assist"
    if not truthy(row.get("success_after_assist")):
        return "no_success_after_assist"
    return None


def near_intersection_detected(events: list[dict[str, Any]]) -> bool:
    return any(
        bool(_details(event).get("near_intersection_detected") or _details(event).get("near_intersection") or _details(event).get("near_doorway"))
        for event in events
    )


def time_to_x(trajectory: list[dict[str, Any]], threshold: float) -> float | None:
    for row in sorted(trajectory, key=lambda item: _float(item.get("t"), 0.0)):
        if _float(row.get("x"), 0.0) >= threshold:
            return _float(row.get("t"), None)
    return None


def route_trigger_x_from_scene(scene: dict[str, Any]) -> float:
    centers: list[float] = []
    for zone in scene.get("semantic_zones", []):
        if str(zone.get("class") or "") in {"intersection", "doorway"}:
            center = zone.get("center", [0.0, 0.0])
            try:
                centers.append(float(center[0]))
            except (TypeError, ValueError, IndexError):
                pass
    return (min(centers) if centers else 4.0) - 0.8


def path_length_from_trajectory(trajectory: list[dict[str, Any]]) -> float:
    total = 0.0
    prev = None
    for row in trajectory:
        x = _float(row.get("x"), 0.0)
        y = _float(row.get("y"), 0.0)
        if prev is not None:
            total += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
    return total


def trigger_window_from_episode(episode: dict[str, Any], events: list[dict[str, Any]]) -> str:
    mode = str(episode.get("mode") or "")
    mode_map = {
        "v9_route_injection_x15": "x>=1.5",
        "v9_route_injection_x20": "x>=2.0",
        "v9_route_injection_x25": "x>=2.5",
        "v9_route_injection_x30": "x>=3.0",
        "v9_route_injection_preintersection": "intersection_x-0.8",
        "v9_route_injection_near_intersection": "near_intersection",
    }
    if mode in mode_map:
        return mode_map[mode]
    for event in events:
        details = _details(event)
        for payload in (details, details.get("payload"), details.get("gate"), details.get("decision")):
            if isinstance(payload, dict) and payload.get("trigger_window"):
                return str(payload.get("trigger_window"))
    return ""


def normalize_window(value: str) -> str:
    return str(value).replace(".0", "").strip()


def distance_to_branch_goal(task: dict[str, Any], scene: dict[str, Any], trajectory: list[dict[str, Any]]) -> float | None:
    if not trajectory:
        return None
    target_id = str(task.get("target_object") or "")
    target = {}
    for obj in scene.get("objects", []):
        if str(obj.get("id") or "") == target_id:
            target = obj
            break
    pose = target.get("pose")
    if not isinstance(pose, list) or len(pose) < 2:
        return None
    final = trajectory[-1]
    return math.hypot(_float(final.get("x"), 0.0) - _float(pose[0], 0.0), _float(final.get("y"), 0.0) - _float(pose[1], 0.0))


def min_distance_to_target(events: list[dict[str, Any]], episode: dict[str, Any]) -> float | None:
    values = [_float(_details(event).get("distance_to_target"), None) for event in events if _details(event).get("distance_to_target") is not None]
    values = [value for value in values if value is not None]
    if values:
        return min(values)
    value = _float(episode.get("final_distance_to_target_m"), None)
    return value


def reached_after_assist(
    episode: dict[str, Any],
    scene: dict[str, Any],
    events: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    assist_t: float | None,
) -> bool:
    if assist_t is None:
        return False
    task_type = str(episode.get("task_type") or "")
    after_events = [event for event in events if event_time(event) is None or _float(event_time(event), 0.0) >= assist_t]
    if task_type in {"turn_choice", "turn_microbench"}:
        trigger_x = route_trigger_x_from_scene(scene)
        after_trajectory = [row for row in trajectory if _float(row.get("t"), 0.0) >= assist_t]
        return any(_float(row.get("x"), 0.0) >= trigger_x for row in after_trajectory) or near_intersection_detected(after_events)
    if task_type == "semantic_target":
        return any(_details(event).get("distance_to_target") is not None and _float(_details(event).get("distance_to_target"), 999.0) <= 2.5 for event in after_events)
    return bool(episode.get("success"))


def write_aggregate_trajectory(output: Path, metrics: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        episode_dir = episode_dir_for(output, episode)
        for point in load_trajectory(episode_dir / "trajectory.csv"):
            rows.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "mode": episode.get("mode"),
                    "task_id": episode.get("task_id"),
                    "t": point.get("t"),
                    "x": point.get("x"),
                    "y": point.get("y"),
                    "yaw": point.get("yaw"),
                    "source": point.get("source", ""),
                }
            )
    write_csv(output / "trajectory.csv", rows, TRAJECTORY_COLUMNS)


def write_visual_artifact(output: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    visual = output / "visual"
    visual.mkdir(parents=True, exist_ok=True)
    overlay_rows = []
    for row in rows[:50]:
        overlay_rows.append({"episode_id": row.get("episode_id"), "mode": row.get("mode"), "failure_stage": row.get("failure_stage")})
    (visual / "overlay_state.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in overlay_rows) + ("\n" if overlay_rows else ""), encoding="utf-8")
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (960, 540), (248, 250, 252))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 959, 72), fill=(20, 24, 35))
        draw.text((24, 22), "V9 route/stop probe viewport", fill=(255, 255, 255))
        draw.text((24, 96), f"episodes: {len(metrics.get('episodes', []))}", fill=(20, 24, 35))
        y = 128
        for row in rows[:12]:
            text = f"{row.get('mode','')} {row.get('task_id','')} stage={row.get('failure_stage')}"
            draw.text((24, y), text[:120], fill=(35, 50, 70))
            y += 28
        draw.line((90, 440, 840, 440), fill=(30, 120, 210), width=4)
        draw.ellipse((820, 420, 860, 460), fill=(220, 60, 70))
        image.save(visual / "viewport.png")
    except Exception:
        write_minimal_png(visual / "viewport.png")


def write_minimal_png(path: Path) -> None:
    width, height = 8, 8
    raw = b"".join(b"\x00" + bytes([240, 248, 255]) * width for _ in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def aggregate_collision_count(metrics: dict[str, Any]) -> int:
    return sum(int(_float(row.get("num_collisions"), 0.0)) for row in metrics.get("episodes", []))


def aggregate_stale_action_executed(metrics: dict[str, Any]) -> int:
    return sum(1 for row in metrics.get("episodes", []) if str(row.get("failure_reason") or "") == "stale_action_executed")


def seed_from_task_id(task_id: Any) -> str:
    text = str(task_id or "")
    if "_seed" in text:
        return text.rsplit("_seed", 1)[-1]
    return ""


def _write_v9_metrics(output: Path, metrics: dict[str, Any]) -> None:
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
