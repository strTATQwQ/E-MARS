from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .v7_value_utils import load_json, load_jsonl, mode_success, postprocess_v7_run, task_success_count
from .v8_coverage_utils import analyze_stale_attribution_v8


BASELINE_MODE = "omninav_only_v11_matched"
ORACLE_MODE = "omninav_forced_route_stop_oracle_v11"


def paired_effect_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in metrics.get("episodes", []):
        by_mode.setdefault(str(episode.get("mode") or ""), {})[str(episode.get("task_id") or "")] = episode
    baseline = by_mode.get(BASELINE_MODE, {})
    oracle = by_mode.get(ORACLE_MODE, {})
    rows = []
    for task_id in sorted(set(baseline) | set(oracle)):
        base = baseline.get(task_id, {})
        forced = oracle.get(task_id, {})
        base_success = bool(base.get("success"))
        forced_success = bool(forced.get("success"))
        effect = "helped" if forced_success and not base_success else "hurt" if base_success and not forced_success else "no_effect"
        rows.append(
            {
                "task_id": task_id,
                "task_type": forced.get("task_type") or base.get("task_type"),
                "baseline_success": base_success,
                "oracle_success": forced_success,
                "baseline_clean": bool(base.get("clean_success")),
                "oracle_clean": bool(forced.get("clean_success")),
                "effect": effect,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0]) if rows else ["task_id", "task_type", "effect"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_v11(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = postprocess_v7_run(output, run_kind="v11_forced_oracle_upper_bound")
    baseline = mode_success(metrics, BASELINE_MODE)
    oracle = mode_success(metrics, ORACLE_MODE)
    effects = paired_effect_rows(metrics)
    _write_csv(output / "helped_hurt_no_effect.csv", effects)
    events = load_jsonl(output / "events.jsonl")
    stale = analyze_stale_attribution_v8(events)
    _write_csv(output / "stale_attribution_v11.csv", stale.get("records", []))
    v7 = metrics.get("v7_summary", {})
    turn_success = task_success_count(metrics, ORACLE_MODE, "turn_choice")
    semantic_success = task_success_count(metrics, ORACLE_MODE, "semantic_target")
    base_simple = int(baseline.get("task_type_success", {}).get("simple_navigation", {}).get("success", 0))
    oracle_simple = int(oracle.get("task_type_success", {}).get("simple_navigation", {}).get("success", 0))
    success_delta = int(oracle.get("success_count", 0)) - int(baseline.get("success_count", 0))
    clean_delta = int(oracle.get("clean_success_count", 0)) - int(baseline.get("clean_success_count", 0))
    failures = []
    if int(oracle.get("success_count", 0)) < 8:
        failures.append("forced oracle success < 8/15")
    if int(oracle.get("clean_success_count", 0)) < 6:
        failures.append("forced oracle clean < 6/15")
    if success_delta < 2:
        failures.append("paired success delta < +2")
    if clean_delta < 2:
        failures.append("paired clean delta < +2")
    if turn_success <= 0:
        failures.append("turn_choice success <= 0")
    if semantic_success <= 0:
        failures.append("semantic_target success <= 0")
    if oracle_simple < base_simple:
        failures.append("simple_navigation regressed")
    if int(stale.get("runtime_stale_discards", 0)) != 0:
        failures.append(f"runtime_stale_discards={stale.get('runtime_stale_discards')} != 0")
    if int(stale.get("timebase_error", 0)) != 0:
        failures.append(f"timebase_error={stale.get('timebase_error')} != 0")
    if int(v7.get("collision_count", 0)) != 0:
        failures.append(f"collision_count={v7.get('collision_count')} != 0")
    if int(v7.get("stale_action_executed", 0)) != 0:
        failures.append(f"stale_action_executed={v7.get('stale_action_executed')} != 0")
    result = {
        "schema_version": 1,
        "pass": not failures,
        "baseline": baseline,
        "oracle": oracle,
        "success_delta": success_delta,
        "clean_delta": clean_delta,
        "oracle_turn_choice_success": turn_success,
        "oracle_semantic_target_success": semantic_success,
        "simple_navigation_delta": oracle_simple - base_simple,
        "effect_counts": {
            key: sum(1 for row in effects if row["effect"] == key)
            for key in ("helped", "hurt", "no_effect")
        },
        "stale": stale,
        "collision_count": int(v7.get("collision_count", 0)),
        "stale_action_executed": int(v7.get("stale_action_executed", 0)),
        "failures": failures,
        "step": "MAY UNFREEZE FOR MICRO TESTS" if not failures else "FROZEN",
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
    }
    metrics["v11_evaluation"] = result
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "v11_evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_v11_summary(output, result), encoding="utf-8")
    return result


def render_v11_summary(output: Path, result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    oracle = result["oracle"]
    lines = [
        "# V11 Forced-Oracle Full Upper Bound",
        "",
        f"- run: {output}",
        f"- pass: {result['pass']}",
        f"- baseline: {baseline['success_count']}/15 success, {baseline['clean_success_count']}/15 clean",
        f"- forced oracle: {oracle['success_count']}/15 success, {oracle['clean_success_count']}/15 clean",
        f"- delta: {result['success_delta']:+d} success, {result['clean_delta']:+d} clean",
        f"- turn_choice success: {result['oracle_turn_choice_success']}",
        f"- semantic_target success: {result['oracle_semantic_target_success']}",
        f"- effects: {result['effect_counts']}",
        f"- runtime_stale: {result['stale']['runtime_stale_discards']}",
        f"- timebase_error: {result['stale']['timebase_error']}",
        f"- collision: {result['collision_count']}",
        f"- stale_action_executed: {result['stale_action_executed']}",
        "",
        "## Gate",
        "",
        f"- Step: {result['step']}",
        "- Sim2Real: NOT READY FOR REAL ROBOT AUTONOMY",
    ]
    if result["failures"]:
        lines += ["", "## Failures", ""] + [f"- {item}" for item in result["failures"]]
    return "\n".join(lines) + "\n"
