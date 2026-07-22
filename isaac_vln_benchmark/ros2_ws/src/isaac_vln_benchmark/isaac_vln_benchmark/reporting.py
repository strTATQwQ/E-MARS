from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .metrics import aggregate_metrics, markdown_table


def group_by(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get(key, "")), []).append(item)
    return grouped


def first_existing_mode(by_mode: dict[str, list[dict[str, Any]]], names: list[str]) -> str:
    for name in names:
        if by_mode.get(name):
            return name
    return names[0] if names else ""


def build_summary(metrics: list[dict[str, Any]], config: dict[str, Any]) -> str:
    by_mode = group_by(metrics, "mode")
    step_omninav_mode = first_existing_mode(
        by_mode,
        ["step_omninav_event", "step_omninav_best_trigger", "omninav_step_verify", "step_goal_verify_only"],
    )
    step_internnav_mode = first_existing_mode(
        by_mode,
        ["step_internnav_event", "step_internnav_best_trigger", "internnav_step_verify"],
    )
    mode_rows = []
    for mode, rows in sorted(by_mode.items()):
        agg = aggregate_metrics(rows)
        mode_rows.append(
            [
                mode,
                agg.get("episodes", 0),
                agg.get("success_rate", 0.0),
                agg.get("clean_success_rate", 0.0),
                agg.get("recovered_success_rate", 0.0),
                agg.get("mean_mission_time", 0.0),
                agg.get("mean_path_length", 0.0),
                agg.get("mean_step_calls", 0.0),
                agg.get("mean_omninav_calls", 0.0),
                agg.get("mean_internnav_calls", 0.0),
                agg.get("mean_model_latency_s", 0.0),
                agg.get("stale_action_rate", 0.0),
                agg.get("recovery_override_rate", 0.0),
                agg.get("safety_block_ticks", 0),
                agg.get("safety_intervention_events", 0),
                agg.get("recovery_override_ticks", 0),
                agg.get("recovery_override_events", 0),
                agg.get("episodes_with_recovery", 0),
                agg.get("failure_top1", ""),
            ]
        )

    pending_modes = {
        "step_omninav_stop_only": "stop_only",
        "step_omninav_move_while_thinking": "move_while_thinking",
        "step_omninav_safe_scan": "safe_scan",
        "step_omninav_event": "auto",
    }
    pending_rows = []
    for mode, label in pending_modes.items():
        rows = by_mode.get(mode, [])
        if not rows:
            continue
        agg = aggregate_metrics(rows)
        pending_rows.append(
            [
                label,
                agg.get("success_rate", 0.0),
                agg.get("mean_mission_time", 0.0),
                sum(int(m.get("num_safety_stops", 0)) for m in rows),
                sum(int(m.get("num_stale_step_results", 0)) for m in rows),
                sum(int(m.get("num_collisions", 0)) for m in rows),
                sum(1 for m in rows if m.get("target_visible_at_completion")) / len(rows),
            ]
        )

    by_delay = group_by(metrics, "delay_profile")
    delay_rows = []
    for delay, rows in sorted(by_delay.items()):
        agg = aggregate_metrics(rows)
        delay_rows.append(
            [
                delay,
                agg.get("success_rate", 0.0),
                agg.get("stale_discard_count", 0),
                agg.get("unsafe_action_blocked_count", 0),
                agg.get("collision_count", 0),
                agg.get("timeout_rate", 0.0),
            ]
        )

    by_task_type = group_by(metrics, "task_type")
    value_rows = []
    for task_type, rows in sorted(by_task_type.items()):
        modes = group_by(rows, "mode")
        step = aggregate_metrics(modes.get("step_only", [])).get("success_rate", 0.0) if modes.get("step_only") else 0.0
        omni = (
            aggregate_metrics(modes.get("omninav_only", [])).get("success_rate", 0.0)
            if modes.get("omninav_only")
            else 0.0
        )
        task_step_omni_mode = first_existing_mode(
            modes,
            ["step_omninav_event", "step_omninav_best_trigger", "omninav_step_verify", "step_goal_verify_only"],
        )
        event = aggregate_metrics(modes.get(task_step_omni_mode, [])).get("success_rate", 0.0)
        avg_step_calls = aggregate_metrics(modes.get(task_step_omni_mode, [])).get("mean_step_calls", 0.0)
        value_rows.append([task_type, step, omni, event, event - omni, avg_step_calls])

    fast_rows = []
    for label, modes in (
        ("internnav", ["internnav_only", "step_internnav_event", "step_internnav_best_trigger", "step_internnav_periodic_4_1", "internnav_step_verify"]),
        ("omninav", ["omninav_only", "step_omninav_event", "step_omninav_best_trigger", "step_omninav_periodic_4_1", "omninav_step_verify"]),
    ):
        rows = [row for mode in modes for row in by_mode.get(mode, [])]
        if not rows:
            continue
        agg = aggregate_metrics(rows)
        fast_rows.append(
            [
                label,
                agg.get("success_rate", 0.0),
                agg.get("clean_success_rate", 0.0),
                agg.get("mean_model_latency_s", 0.0) or (agg.get("mean_omninav_latency", 0.0) / 1000.0),
                agg.get("action_entropy", 0.0),
                agg.get("forward_ratio", 0.0),
                agg.get("recovery_override_count", 0),
                agg.get("failure_top1", ""),
            ]
        )

    step_value_rows = []
    for fast, only_mode, event_mode in (
        ("internnav", "internnav_only", step_internnav_mode),
        ("omninav", "omninav_only", step_omninav_mode),
    ):
        only = aggregate_metrics(by_mode.get(only_mode, []))
        event = aggregate_metrics(by_mode.get(event_mode, []))
        if not only and not event:
            continue
        step_value_rows.append(
            [
                fast,
                only.get("success_rate", 0.0),
                event.get("success_rate", 0.0),
                event.get("success_rate", 0.0) - only.get("success_rate", 0.0),
                event.get("mean_step_calls", 0.0),
                event.get("mean_mission_time", 0.0) - only.get("mean_mission_time", 0.0),
            ]
        )

    failure_counts = Counter(str(m.get("failure_reason")) for m in metrics if m.get("failure_reason"))
    taxonomy = [
        "timeout",
        "wrong_direction",
        "wrong_target",
        "forward_bias",
        "never_stopped",
        "target_not_visible",
        "collision",
        "stale_action_executed",
        "no_progress",
        "parse_error",
    ]
    taxonomy_rows = [[reason, failure_counts.get(reason, 0)] for reason in taxonomy]

    event_rows = by_mode.get(step_omninav_mode, [])
    event_agg = aggregate_metrics(event_rows)
    omni_rows = by_mode.get("omninav_only", [])
    rr_rows = by_mode.get("step_omninav_roundrobin_1_1", [])
    rr_step = aggregate_metrics(rr_rows).get("mean_step_calls", 0.0) if rr_rows else 0.0
    event_step = event_agg.get("mean_step_calls", 0.0)
    reduction = 0.0 if rr_step == 0 else 100.0 * (rr_step - event_step) / rr_step
    stale_blocked = sum(int(m.get("num_stale_step_results", 0)) + int(m.get("num_stale_omninav_actions", 0)) for m in event_rows)
    pending_policy = "auto"
    if event_rows:
        pending_policy = str(event_rows[0].get("pending_policy", "auto"))
    conclusion = (
        f"In this run, {step_internnav_mode} achieved "
        f"{100.0 * aggregate_metrics(by_mode.get(step_internnav_mode, [])).get('success_rate', 0.0):.1f}% success and "
        f"{100.0 * aggregate_metrics(by_mode.get(step_internnav_mode, [])).get('clean_success_rate', 0.0):.1f}% clean success, "
        f"compared with InternNav-only at {100.0 * aggregate_metrics(by_mode.get('internnav_only', [])).get('success_rate', 0.0):.1f}% / "
        f"{100.0 * aggregate_metrics(by_mode.get('internnav_only', [])).get('clean_success_rate', 0.0):.1f}%. "
        f"Step supervision changed success by "
        f"{100.0 * (aggregate_metrics(by_mode.get(step_internnav_mode, [])).get('success_rate', 0.0) - aggregate_metrics(by_mode.get('internnav_only', [])).get('success_rate', 0.0)):.1f} points, "
        f"while requiring {aggregate_metrics(by_mode.get(step_internnav_mode, [])).get('mean_step_calls', 0.0):.2f} Step calls per episode. "
        f"The dominant failure mode was {max(failure_counts.items(), key=lambda item: item[1])[0] if failure_counts else 'none'}."
    )

    failure_rows = []
    for m in sorted(metrics, key=lambda row: (row.get("success", True), -float(row.get("mission_time_sec", 0.0))))[:10]:
        if not m.get("success"):
            failure_rows.append([m.get("mode"), m.get("task_id"), m.get("failure_reason"), m.get("mission_time_sec")])

    return "\n\n".join(
        [
            "# Isaac VLN Benchmark Summary",
            "## Experiment Configuration\n\n```json\n" + json.dumps(config, indent=2, sort_keys=True) + "\n```",
            "## Machine And Model Latency\n\n"
            f"- Step mean latency target: {config.get('latency_model', {}).get('step_mean_ms', 'n/a')} ms\n"
            f"- OmniNav mean latency target: {config.get('latency_model', {}).get('omninav_mean_ms', 'n/a')} ms\n"
            f"- Oracle semantics: {config.get('oracle_semantics', True)}",
            "## Table 1: Mode Comparison\n\n"
            + markdown_table(
                [
                    "mode",
                    "episodes",
                    "success_rate",
                    "clean_success_rate",
                    "recovered_success_rate",
                    "mean_time_s",
                    "mean_path_m",
                    "mean_step_calls",
                    "mean_omninav_calls",
                    "mean_internnav_calls",
                    "mean_model_latency_s",
                    "stale_action_rate",
                    "recovery_override_rate",
                    "safety_block_ticks",
                    "safety_intervention_events",
                    "recovery_override_ticks",
                    "recovery_override_events",
                    "episodes_with_recovery",
                    "failure_top1",
                ],
                mode_rows,
            ),
            "## Table 2: Step Pending Strategy Comparison\n\n"
            + markdown_table(
                [
                    "pending_policy",
                    "success_rate",
                    "mean_mission_time",
                    "safety_stop_count",
                    "stale_step_result_count",
                    "collision_count",
                    "target_visible_at_completion",
                ],
                pending_rows,
            ),
            "## Table 3: Delay Stress Test\n\n"
            + markdown_table(
                [
                    "delay_profile",
                    "success_rate",
                    "stale_discard_count",
                    "unsafe_action_blocked_count",
                    "collision_count",
                    "timeout_rate",
                ],
                delay_rows,
            ),
            "## Table 4: Step Calling Value\n\n"
            + markdown_table(
                [
                    "task_type",
                    "step_only_success",
                    "omninav_only_success",
                    "event_step_omninav_success",
                    "delta_event_vs_omninav",
                    "avg_step_calls",
                ],
                value_rows,
            ),
            "## InternNav vs OmniNav\n\n"
            + markdown_table(
                [
                    "fast_executor",
                    "success_rate",
                    "clean_success_rate",
                    "mean_latency",
                    "action_entropy",
                    "forward_ratio",
                    "recovery_overrides",
                    "failure_top1",
                ],
                fast_rows,
            ),
            "## Step Value\n\n"
            + markdown_table(
                [
                    "fast_executor",
                    "fast_only_success",
                    "step_event_success",
                    "delta_success",
                    "extra_step_calls",
                    "mean_time_delta",
                ],
                step_value_rows,
            ),
            "## Failure Taxonomy\n\n" + markdown_table(["failure_reason", "count"], taxonomy_rows),
            "## Failure Cases Top-K\n\n" + markdown_table(["mode", "task_id", "reason", "mission_time_sec"], failure_rows),
            "## Recommendation\n\n"
            "- Use event-triggered Step calls for route choices, ambiguity, blocked paths, and completion verification.\n"
            "- Keep stale gate enabled for both Step and OmniNav actions.\n"
            "- Prefer safe_scan when dynamic obstacles or target visibility uncertainty dominates; use move_slow only in clear corridors.\n"
            "- Run a real-Isaac dry-run before any real robot transfer. This package does not publish to real Unitree motion topics.",
            "## Conclusion\n\n" + conclusion,
        ]
    )


def write_summary(output_dir: str | Path, metrics: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    summary = build_summary(metrics, config)
    path = output_dir / "summary.md"
    path.write_text(summary + "\n", encoding="utf-8")
    write_csv_artifacts(output_dir, metrics)
    return path


def write_csv_artifacts(output_dir: str | Path, metrics: list[dict[str, Any]]) -> None:
    output_dir = Path(output_dir)
    by_mode = group_by(metrics, "mode")
    mode_headers = [
        "mode",
        "episodes",
        "success_rate",
        "clean_success_rate",
        "recovered_success_rate",
        "mean_time_s",
        "mean_path_m",
        "mean_step_calls",
        "mean_omninav_calls",
        "mean_internnav_calls",
        "mean_model_latency_s",
        "stale_action_rate",
        "recovery_override_rate",
        "safety_block_ticks",
        "safety_intervention_events",
        "recovery_override_ticks",
        "recovery_override_events",
        "episodes_with_recovery",
        "failure_top1",
    ]
    mode_rows = []
    for mode, rows in sorted(by_mode.items()):
        agg = aggregate_metrics(rows)
        mode_rows.append(
            {
                "mode": mode,
                "episodes": agg.get("episodes", 0),
                "success_rate": agg.get("success_rate", 0.0),
                "clean_success_rate": agg.get("clean_success_rate", 0.0),
                "recovered_success_rate": agg.get("recovered_success_rate", 0.0),
                "mean_time_s": agg.get("mean_mission_time", 0.0),
                "mean_path_m": agg.get("mean_path_length", 0.0),
                "mean_step_calls": agg.get("mean_step_calls", 0.0),
                "mean_omninav_calls": agg.get("mean_omninav_calls", 0.0),
                "mean_internnav_calls": agg.get("mean_internnav_calls", 0.0),
                "mean_model_latency_s": agg.get("mean_model_latency_s", 0.0),
                "stale_action_rate": agg.get("stale_action_rate", 0.0),
                "recovery_override_rate": agg.get("recovery_override_rate", 0.0),
                "safety_block_ticks": agg.get("safety_block_ticks", 0),
                "safety_intervention_events": agg.get("safety_intervention_events", 0),
                "recovery_override_ticks": agg.get("recovery_override_ticks", 0),
                "recovery_override_events": agg.get("recovery_override_events", 0),
                "episodes_with_recovery": agg.get("episodes_with_recovery", 0),
                "failure_top1": agg.get("failure_top1", ""),
            }
        )
    _write_csv(output_dir / "mode_table.csv", mode_headers, mode_rows)

    failure_headers = ["mode", "task_id", "failure_reason", "mission_time_sec", "success_class"]
    failure_rows = [
        {
            "mode": row.get("mode"),
            "task_id": row.get("task_id"),
            "failure_reason": row.get("failure_reason"),
            "mission_time_sec": row.get("mission_time_sec"),
            "success_class": row.get("success_class"),
        }
        for row in metrics
        if row.get("failure_reason")
    ]
    _write_csv(output_dir / "failure_table.csv", failure_headers, failure_rows)

    action_headers = [
        "mode",
        "episodes",
        "forward_ratio",
        "left_ratio",
        "right_ratio",
        "stop_ratio",
        "action_entropy",
        "recovery_override_count",
    ]
    action_rows = []
    for mode, rows in sorted(by_mode.items()):
        agg = aggregate_metrics(rows)
        action_rows.append(
            {
                "mode": mode,
                "episodes": agg.get("episodes", 0),
                "forward_ratio": agg.get("forward_ratio", 0.0),
                "left_ratio": agg.get("left_ratio", 0.0),
                "right_ratio": agg.get("right_ratio", 0.0),
                "stop_ratio": agg.get("stop_ratio", 0.0),
                "action_entropy": agg.get("action_entropy", 0.0),
                "recovery_override_count": agg.get("recovery_override_count", 0),
            }
        )
    _write_csv(output_dir / "action_distribution.csv", action_headers, action_rows)

    safety_headers = [
        "mode",
        "episodes",
        "safety_block_ticks",
        "safety_intervention_events",
        "recovery_override_ticks",
        "recovery_override_events",
        "episodes_with_recovery",
        "max_continuous_safety_block_sec",
        "max_continuous_recovery_sec",
    ]
    safety_rows = []
    for mode, rows in sorted(by_mode.items()):
        agg = aggregate_metrics(rows)
        safety_rows.append(
            {
                "mode": mode,
                "episodes": agg.get("episodes", 0),
                "safety_block_ticks": agg.get("safety_block_ticks", 0),
                "safety_intervention_events": agg.get("safety_intervention_events", 0),
                "recovery_override_ticks": agg.get("recovery_override_ticks", 0),
                "recovery_override_events": agg.get("recovery_override_events", 0),
                "episodes_with_recovery": agg.get("episodes_with_recovery", 0),
                "max_continuous_safety_block_sec": agg.get("max_continuous_safety_block_sec", 0.0),
                "max_continuous_recovery_sec": agg.get("max_continuous_recovery_sec", 0.0),
            }
        )
    _write_csv(output_dir / "safety_recovery_events.csv", safety_headers, safety_rows)


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def load_run_metrics(output_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(output_dir) / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("episodes", [])
