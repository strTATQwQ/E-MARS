from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .v7_value_utils import load_jsonl, mode_rows, mode_success, postprocess_v7_run, task_success_count
from .v8_coverage_utils import analyze_stale_attribution_v8


BASELINE_MODE = "omninav_only_v11_matched"
ORACLE_MODE = "omninav_forced_route_stop_oracle_v11"
ROUTE_MODE = "omninav_step_route_only_v12_screen"
STOP_MODE = "omninav_step_stop_only_v12_screen"
ROUTE_STOP_MODE = "omninav_step_route_stop_v12_screen"
STEP_ONLY_MODE = "step_only_v12_sanity"
SCREEN_MODES = [BASELINE_MODE, ORACLE_MODE, ROUTE_MODE, STOP_MODE, ROUTE_STOP_MODE, STEP_ONLY_MODE]


def paired_effect_rows(metrics: dict[str, Any], candidate_mode: str) -> list[dict[str, Any]]:
    baseline = {str(row.get("task_id") or ""): row for row in mode_rows(metrics, BASELINE_MODE)}
    candidate = {str(row.get("task_id") or ""): row for row in mode_rows(metrics, candidate_mode)}
    rows: list[dict[str, Any]] = []
    for task_id in sorted(set(baseline) | set(candidate)):
        base = baseline.get(task_id, {})
        current = candidate.get(task_id, {})
        base_success = bool(base.get("success"))
        current_success = bool(current.get("success"))
        effect = "helped" if current_success and not base_success else "hurt" if base_success and not current_success else "no_effect"
        rows.append(
            {
                "candidate_mode": candidate_mode,
                "task_id": task_id,
                "task_type": current.get("task_type") or base.get("task_type"),
                "baseline_success": base_success,
                "candidate_success": current_success,
                "baseline_clean": bool(base.get("clean_success")),
                "candidate_clean": bool(current.get("clean_success")),
                "effect": effect,
            }
        )
    return rows


def real_step_http_by_mode(metrics: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    episode_mode = {
        str(row.get("episode_id") or ""): str(row.get("mode") or "")
        for row in metrics.get("episodes", [])
    }
    result = {
        mode: {
            "accepted": 0,
            "accepted_multimodal": 0,
            "fresh_multimodal": 0,
            "route_choice": 0,
            "semantic_stop": 0,
            "errors": 0,
            "fallback_or_mock": 0,
            "latency_s": [],
        }
        for mode in SCREEN_MODES
    }
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else event
        event_type = str(details.get("event_type") or event.get("event") or "")
        episode_id = str(event.get("episode_id") or details.get("episode_id") or "")
        mode = episode_mode.get(episode_id, "")
        if mode not in result:
            continue
        model = str(details.get("model") or "").lower()
        if event_type == "step_http_response":
            if str(details.get("result") or "") == "accepted" and model == "step_http":
                result[mode]["accepted"] += 1
                if bool(details.get("multimodal")):
                    result[mode]["accepted_multimodal"] += 1
                    snapshot = details.get("image_snapshot") if isinstance(details.get("image_snapshot"), dict) else {}
                    try:
                        if int(snapshot.get("frame_seq", -1)) >= 0 and float(snapshot.get("age_sec", 99.0)) <= 0.75:
                            result[mode]["fresh_multimodal"] += 1
                    except (TypeError, ValueError):
                        pass
                role = str(details.get("role") or "")
                if role in {"route_choice", "semantic_stop"}:
                    result[mode][role] += 1
                try:
                    result[mode]["latency_s"].append(float(details.get("latency_s")))
                except (TypeError, ValueError):
                    pass
            else:
                result[mode]["errors"] += 1
        step_scoped = event_type.lower().startswith("step_") or model.startswith("step") or "mock_step" in model
        if step_scoped and ("fallback" in event_type.lower() or "mock" in model):
            result[mode]["fallback_or_mock"] += 1
    for values in result.values():
        latencies = values.pop("latency_s")
        values["mean_latency_s"] = round(sum(latencies) / len(latencies), 6) if latencies else None
        values["p95_latency_s"] = _percentile_nearest_rank(latencies, 0.95)
    return result


def evaluate_v12_screening(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = postprocess_v7_run(output, run_kind="v12_step_value_screening")
    events = load_jsonl(output / "events.jsonl")
    modes = {mode: mode_success(metrics, mode) for mode in SCREEN_MODES}
    all_effects: list[dict[str, Any]] = []
    effect_counts: dict[str, dict[str, int]] = {}
    for mode in [ORACLE_MODE, ROUTE_MODE, STOP_MODE, ROUTE_STOP_MODE, STEP_ONLY_MODE]:
        rows = paired_effect_rows(metrics, mode)
        all_effects.extend(rows)
        effect_counts[mode] = {
            key: sum(1 for row in rows if row["effect"] == key)
            for key in ("helped", "hurt", "no_effect")
        }
    _write_csv(output / "helped_hurt_no_effect_v12.csv", all_effects)
    stale = analyze_stale_attribution_v8(events)
    _write_csv(output / "stale_attribution_v12.csv", stale.get("records", []))
    step_http = real_step_http_by_mode(metrics, events)
    baseline = modes[BASELINE_MODE]
    candidate = modes[ROUTE_STOP_MODE]
    success_delta = int(candidate["success_count"]) - int(baseline["success_count"])
    clean_delta = int(candidate["clean_success_count"]) - int(baseline["clean_success_count"])
    turn_success = task_success_count(metrics, ROUTE_STOP_MODE, "turn_choice")
    semantic_success = task_success_count(metrics, ROUTE_STOP_MODE, "semantic_target")
    simple_success = task_success_count(metrics, ROUTE_STOP_MODE, "simple_navigation")
    candidate_effects = effect_counts[ROUTE_STOP_MODE]
    v7 = metrics.get("v7_summary", {})
    collision_count = sum(int(row.get("num_collisions", 0) or 0) for row in metrics.get("episodes", []))
    stale_action_executed = int(v7.get("stale_action_executed", 0) or 0)
    failures: list[str] = []
    expected_episodes = 15
    for mode in SCREEN_MODES:
        if int(modes[mode]["episodes"]) != expected_episodes:
            failures.append(f"{mode} episodes={modes[mode]['episodes']} != {expected_episodes}")
    if int(candidate["success_count"]) < 8:
        failures.append("route+stop success < 8/15")
    if int(candidate["clean_success_count"]) < 6:
        failures.append("route+stop clean < 6/15")
    if success_delta < 2:
        failures.append("paired success delta < +2")
    if clean_delta < 2:
        failures.append("paired clean delta < +2")
    if turn_success <= 0:
        failures.append("turn_choice success <= 0")
    if semantic_success <= 0:
        failures.append("semantic_target success <= 0")
    if simple_success < 5:
        failures.append("simple_navigation success < 5/6")
    if int(step_http[ROUTE_STOP_MODE]["accepted"]) <= 0:
        failures.append("route+stop real Step call count <= 0")
    expected_step_calls = {
        ROUTE_MODE: {"route_choice": 6, "semantic_stop": 0, "accepted": 6},
        STOP_MODE: {"route_choice": 0, "semantic_stop": 3, "accepted": 3},
        ROUTE_STOP_MODE: {"route_choice": 6, "semantic_stop": 3, "accepted": 9},
    }
    for mode, expected in expected_step_calls.items():
        for field, value in expected.items():
            if int(step_http[mode][field]) != value:
                failures.append(f"{mode} {field}={step_http[mode][field]} != {value}")
        if int(step_http[mode]["fallback_or_mock"]) != 0:
            failures.append(f"{mode} contains mock/fallback Step evidence")
        if int(step_http[mode]["errors"]) != 0:
            failures.append(f"{mode} Step HTTP errors != 0")
    if candidate_effects["helped"] <= candidate_effects["hurt"]:
        failures.append("route+stop helped <= hurt")
    if int(stale.get("runtime_stale_discards", 0)) != 0:
        failures.append(f"runtime_stale_discards={stale.get('runtime_stale_discards')} != 0")
    if int(stale.get("timebase_error", 0)) != 0:
        failures.append(f"timebase_error={stale.get('timebase_error')} != 0")
    if collision_count != 0:
        failures.append(f"collision_count={collision_count} != 0")
    if stale_action_executed != 0:
        failures.append(f"stale_action_executed={stale_action_executed} != 0")
    result = {
        "schema_version": 1,
        "pass": not failures,
        "modes": modes,
        "route_stop_success_delta": success_delta,
        "route_stop_clean_delta": clean_delta,
        "route_stop_turn_choice_success": turn_success,
        "route_stop_semantic_target_success": semantic_success,
        "route_stop_simple_navigation_success": simple_success,
        "effect_counts": effect_counts,
        "real_step_http": step_http,
        "stale": stale,
        "collision_count": collision_count,
        "stale_action_executed": stale_action_executed,
        "failures": failures,
        "confirmation": "ALLOWED" if not failures else "LOCKED",
        "value_claim": "SCREENING PASS; PAIRED CONFIRMATION REQUIRED" if not failures else "UNPROVEN",
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
    }
    metrics["v12_screening"] = result
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "v12_screening_evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_summary(output, result), encoding="utf-8")
    return result


def render_summary(output: Path, result: dict[str, Any]) -> str:
    lines = [
        "# V12 OmniNav+Step Screening",
        "",
        f"- run: {output}",
        f"- pass: {result['pass']}",
        f"- value claim: {result['value_claim']}",
        f"- paired confirmation: {result['confirmation']}",
        f"- route+stop delta: {result['route_stop_success_delta']:+d} success, {result['route_stop_clean_delta']:+d} clean",
        f"- route+stop task successes: simple={result['route_stop_simple_navigation_success']}, turn={result['route_stop_turn_choice_success']}, semantic={result['route_stop_semantic_target_success']}",
        f"- route+stop effects: {result['effect_counts'][ROUTE_STOP_MODE]}",
        f"- route+stop real Step HTTP: {result['real_step_http'][ROUTE_STOP_MODE]}",
        f"- runtime stale: {result['stale']['runtime_stale_discards']}",
        f"- timebase errors: {result['stale']['timebase_error']}",
        f"- collision: {result['collision_count']}",
        f"- stale action executed: {result['stale_action_executed']}",
        "- Sim2Real: NOT READY FOR REAL ROBOT AUTONOMY",
        "",
        "## Modes",
        "",
    ]
    for mode in SCREEN_MODES:
        values = result["modes"][mode]
        lines.append(f"- {mode}: {values['success_count']}/15 success, {values['clean_success_count']}/15 clean")
    if result["failures"]:
        lines += ["", "## Failures", ""] + [f"- {item}" for item in result["failures"]]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percentile_nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int((len(ordered) * quantile) + 0.999999)))
    return round(ordered[rank - 1], 6)
