from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .metrics import aggregate_metrics


def group_by(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get(key, "")), []).append(item)
    return grouped


def write_diagnostic_artifacts(output_dir: str | Path, config: dict[str, Any]) -> None:
    output = Path(output_dir)
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        return
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = data.get("episodes", [])
    if not isinstance(metrics, list):
        return
    write_summary_aggregate(output, metrics)
    name = str(config.get("benchmark", {}).get("name") or config.get("name") or "")
    if name == "turn_microbench":
        write_turn_microbench(output, metrics)
    elif name == "target_verify_bench":
        write_target_verify(output, metrics)
    elif name == "step_trigger_ablation":
        write_step_trigger_ablation(output, metrics)
    elif name == "live_success_small_v2":
        append_live_v2_sections(output, metrics)


def write_summary_aggregate(output: Path, metrics: list[dict[str, Any]]) -> None:
    rows = []
    for mode, mode_rows in sorted(group_by(metrics, "mode").items()):
        agg = aggregate_metrics(mode_rows)
        rows.append(
            {
                "mode": mode,
                "episodes": agg.get("episodes", 0),
                "success": sum(1 for row in mode_rows if row.get("success")),
                "success_rate": agg.get("success_rate", 0.0),
                "clean_success_rate": agg.get("clean_success_rate", 0.0),
                "recovered_success_rate": agg.get("recovered_success_rate", 0.0),
                "mean_time_s": agg.get("mean_mission_time", 0.0),
                "mean_final_dist_m": mean(row.get("final_distance_to_target_m", 0.0) for row in mode_rows),
                "mean_path_m": agg.get("mean_path_length", 0.0),
                "mean_step_calls": agg.get("mean_step_calls", 0.0),
                "mean_omninav_calls": agg.get("mean_omninav_calls", 0.0),
                "mean_internnav_calls": agg.get("mean_internnav_calls", 0.0),
                "failure_top1": agg.get("failure_top1"),
                "safety_block_ticks": agg.get("safety_block_ticks", 0),
                "safety_intervention_events": agg.get("safety_intervention_events", 0),
                "recovery_override_ticks": agg.get("recovery_override_ticks", 0),
                "recovery_override_events": agg.get("recovery_override_events", 0),
                "episodes_with_recovery": agg.get("episodes_with_recovery", 0),
            }
        )
    (output / "summary_aggregate.json").write_text(json.dumps({"summary": rows}, indent=2, sort_keys=True), encoding="utf-8")


def write_turn_microbench(output: Path, metrics: list[dict[str, Any]]) -> None:
    rows = []
    for row in metrics:
        if row.get("task_type") not in {"turn_microbench", "turn_choice"}:
            continue
        episode_dir = episode_dir_for(output, row)
        events = load_jsonl(episode_dir / "events.jsonl")
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        instruction = load_task_instruction(episode_dir)
        expected = "left" if "left" in instruction.lower() else "right" if "right" in instruction.lower() else "unknown"
        first_turn = first_turn_action(events)
        yaw5 = yaw_at(trajectory, 5.0)
        yaw10 = yaw_at(trajectory, 10.0)
        y20 = pose_at(trajectory, 20.0)[1] if trajectory else 0.0
        branch = "left" if y20 > 0.4 else "right" if y20 < -0.4 else "center"
        rows.append(
            {
                "mode": row.get("mode"),
                "task_id": row.get("task_id"),
                "success": row.get("success"),
                "first_turn_action": first_turn,
                "first_turn_time_sec": first_turn_time(events),
                "yaw_after_5s": yaw5,
                "yaw_after_10s": yaw10,
                "entered_correct_branch": branch == expected,
                "branch_id_after_20s": branch,
                "distance_to_target_after_20s": row.get("final_distance_to_target_m"),
                "near_intersection_block_ticks": near_intersection_block_ticks(events),
                "near_intersection_block_events": near_intersection_block_events(events),
                "turn_primitive_angle_deg": turn_primitive_angle(events),
                "turn_cmd_vel_count": turn_cmd_vel_count(events),
            }
        )
    write_csv(output / "turn_microbench_metrics.csv", rows)
    cause = classify_turn_failure(rows)
    lines = [
        "# Turn Microbench Summary",
        "",
        f"- episodes: {len(rows)}",
        f"- entered_correct_branch_rate: {rate(row.get('entered_correct_branch') for row in rows):.3f}",
        f"- turn_cmd_vel_total: {sum(int(row.get('turn_cmd_vel_count') or 0) for row in rows)}",
        f"- near_intersection_block_events: {sum(int(row.get('near_intersection_block_events') or 0) for row in rows)}",
        "",
        f"Turn failure appears dominated by: {cause}",
    ]
    (output / "turn_microbench_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_target_verify(output: Path, metrics: list[dict[str, Any]]) -> None:
    rows = []
    for row in metrics:
        if row.get("task_type") != "target_verify":
            continue
        reason = str(row.get("failure_reason") or "")
        final_distance = float(row.get("final_distance_to_target_m", 0.0) or 0.0)
        visible = bool(row.get("target_visible_at_completion"))
        rows.append(
            {
                "mode": row.get("mode"),
                "task_id": row.get("task_id"),
                "stop_decision_accuracy": bool(row.get("success")),
                "target_visible_at_stop": visible,
                "final_distance_to_target": final_distance,
                "never_stopped": reason == "never_stopped",
                "stopped_too_early": reason == "stopped_too_early",
                "stopped_too_late": reason in {"never_stopped", "timeout"} and final_distance <= 2.5,
                "step_verify_latency": row.get("step_mean_latency_ms", 0.0),
                "verify_calls": row.get("num_step_calls", 0),
            }
        )
    write_csv(output / "target_verify_metrics.csv", rows)
    failure_counts = Counter()
    for row in rows:
        if row["never_stopped"]:
            failure_counts["never_stopped"] += 1
        if row["stopped_too_early"]:
            failure_counts["stopped_too_early"] += 1
        if row["stopped_too_late"]:
            failure_counts["stopped_too_late"] += 1
        if not row["target_visible_at_stop"]:
            failure_counts["target_not_visible"] += 1
    top = failure_counts.most_common(1)[0][0] if failure_counts else "none"
    lines = [
        "# Target Verify Summary",
        "",
        f"- episodes: {len(rows)}",
        f"- stop_decision_accuracy: {rate(row.get('stop_decision_accuracy') for row in rows):.3f}",
        f"- target_visible_at_stop_rate: {rate(row.get('target_visible_at_stop') for row in rows):.3f}",
        f"- semantic_stop_failure_top1: {top}",
    ]
    (output / "target_verify_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_step_trigger_ablation(output: Path, metrics: list[dict[str, Any]]) -> None:
    by_mode = group_by(metrics, "mode")
    rows = []
    for mode, mode_rows in sorted(by_mode.items()):
        if mode in {"omninav_only", "internnav_only"}:
            continue
        baseline_name = "internnav_only" if "internnav" in mode else "omninav_only"
        baseline_by_task = {row.get("task_id"): row for row in by_mode.get(baseline_name, [])}
        helped = hurt = no_effect = 0
        for row in mode_rows:
            baseline = baseline_by_task.get(row.get("task_id"))
            if baseline is None:
                no_effect += 1
            elif row.get("success") and not baseline.get("success"):
                helped += 1
            elif not row.get("success") and baseline.get("success"):
                hurt += 1
            else:
                no_effect += 1
        agg = aggregate_metrics(mode_rows)
        rows.append(
            {
                "mode": mode,
                "success_rate": agg.get("success_rate", 0.0),
                "clean_success_rate": agg.get("clean_success_rate", 0.0),
                "recovered_success_rate": agg.get("recovered_success_rate", 0.0),
                "mean_step_calls": agg.get("mean_step_calls", 0.0),
                "mean_fast_executor_calls": agg.get("mean_internnav_calls", 0.0) or agg.get("mean_omninav_calls", 0.0),
                "mean_time_s": agg.get("mean_mission_time", 0.0),
                "failure_top1": agg.get("failure_top1"),
                "step_helped_count": helped,
                "step_hurt_count": hurt,
                "step_no_effect_count": no_effect,
                "net_help": helped - hurt,
            }
        )
    write_csv(output / "step_trigger_ablation_metrics.csv", rows)
    positive = [row for row in rows if int(row["net_help"]) > 0]
    best = max(positive, key=lambda row: (int(row["net_help"]), float(row["success_rate"])), default=None)
    harmful = sorted(rows, key=lambda row: int(row["net_help"]))[:3]
    lines = [
        "# Step Trigger Ablation Summary",
        "",
        f"- modes: {len(rows)}",
        f"- positive_trigger_top1: {best['mode'] if best else 'none'}",
        f"- strongest_harmful: {harmful[0]['mode'] if harmful else 'none'}",
        "",
        "Current Step event supervision should only be promoted to v2 if a trigger has positive net_help and non-worse success.",
    ]
    (output / "step_trigger_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_live_v2_sections(output: Path, metrics: list[dict[str, Any]]) -> None:
    summary = output / "summary.md"
    marker = "## Live V2 Diagnostic Required Sections"
    by_mode = group_by(metrics, "mode")
    omni = aggregate_metrics(by_mode.get("omninav_only", []))
    intern = aggregate_metrics(by_mode.get("internnav_only", []))
    step_modes = [mode for mode in by_mode if mode.startswith("step")]
    best_step = max((aggregate_metrics(by_mode[mode]) | {"mode": mode} for mode in step_modes), key=lambda row: row.get("success_rate", 0.0), default={})
    text = summary.read_text(encoding="utf-8") if summary.exists() else "# Live Success Small V2 Summary\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    text += f"\n\n{marker}\n\n"
    text += f"- OmniNav-only success_rate: {omni.get('success_rate', 0.0):.3f}\n"
    text += f"- InternNav-only success_rate: {intern.get('success_rate', 0.0):.3f}\n"
    text += f"- Best Step mode in v2: {best_step.get('mode', 'none')} at {best_step.get('success_rate', 0.0):.3f}\n"
    text += "- Caveat: InternNav remains CmaAgent/system1/fallback_static_cma_tokens unless audit proves otherwise.\n"
    summary.write_text(text, encoding="utf-8")


def episode_dir_for(output: Path, metric: dict[str, Any]) -> Path:
    return output / str(metric.get("mode")) / str(metric.get("task_id"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def load_trajectory(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: float(value) if value not in {"", None} and key != "source" else value for key, value in row.items()} for row in csv.DictReader(handle)]


def load_task_instruction(episode_dir: Path) -> str:
    task_path = episode_dir / "task.yaml"
    if not task_path.exists():
        return ""
    text = task_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return str(data.get("instruction", ""))
    except json.JSONDecodeError:
        marker = '"instruction":'
        if marker in text:
            return text.split(marker, 1)[1].split(",", 1)[0].strip().strip('"')
    return ""


def first_turn_action(events: list[dict[str, Any]]) -> str:
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        for key in ("model_action", "applied_action", "action_type", "primitive"):
            value = str(details.get(key) or "")
            if "left" in value:
                return "left"
            if "right" in value:
                return "right"
    return "none"


def first_turn_time(events: list[dict[str, Any]]) -> float | None:
    for event in events:
        if first_turn_action([event]) != "none":
            return float(event.get("t", 0.0) or 0.0)
    return None


def yaw_at(trajectory: list[dict[str, float]], timestamp: float) -> float | None:
    if not trajectory:
        return None
    row = min(trajectory, key=lambda item: abs(float(item.get("t", 0.0)) - timestamp))
    return float(row.get("yaw", 0.0))


def pose_at(trajectory: list[dict[str, float]], timestamp: float) -> tuple[float, float, float]:
    if not trajectory:
        return (0.0, 0.0, 0.0)
    row = min(trajectory, key=lambda item: abs(float(item.get("t", 0.0)) - timestamp))
    return (float(row.get("x", 0.0)), float(row.get("y", 0.0)), float(row.get("yaw", 0.0)))


def near_intersection_block_ticks(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if details.get("near_intersection") and not bool(details.get("local_costmap_clear", True)):
            count += 1
    return count


def near_intersection_block_events(events: list[dict[str, Any]]) -> int:
    active = False
    total = 0
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        blocked = bool(details.get("near_intersection")) and not bool(details.get("local_costmap_clear", True))
        if blocked and not active:
            total += 1
        active = blocked
    return total


def turn_primitive_angle(events: list[dict[str, Any]]) -> float:
    best = 0.0
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        try:
            best = max(best, abs(float(details.get("yaw_deg", 0.0) or 0.0)))
        except (TypeError, ValueError):
            pass
    return best


def turn_cmd_vel_count(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        cmd = details.get("cmd_vel") if isinstance(details.get("cmd_vel"), dict) else details
        angular = cmd.get("angular") if isinstance(cmd.get("angular"), dict) else {}
        try:
            if abs(float(angular.get("z", 0.0))) > 0.02:
                count += 1
        except (TypeError, ValueError):
            pass
    return count


def classify_turn_failure(rows: list[dict[str, Any]]) -> str:
    failures = [row for row in rows if not row.get("success")]
    if not failures:
        return "none"
    if sum(int(row.get("near_intersection_block_events") or 0) for row in failures) > len(failures) * 0.3:
        return "safety_block"
    if sum(int(row.get("turn_cmd_vel_count") or 0) for row in failures) == 0:
        return "model_action"
    if rate(row.get("entered_correct_branch") for row in failures) < 0.35:
        return "route_choice_prompt"
    yaw_values = [abs(float(row.get("yaw_after_10s") or 0.0)) for row in failures]
    if yaw_values and sum(1 for value in yaw_values if value < 0.25) / len(yaw_values) > 0.5:
        return "yaw_control"
    return "unknown"


def rate(values: Any) -> float:
    vals = list(values)
    return sum(1 for value in vals if value) / max(len(vals), 1)


def mean(values: Any) -> float:
    vals = [float(value or 0.0) for value in values]
    return sum(vals) / max(len(vals), 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = sorted({key for row in rows for key in row.keys()})
    if not headers:
        headers = ["empty"]
        rows = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
