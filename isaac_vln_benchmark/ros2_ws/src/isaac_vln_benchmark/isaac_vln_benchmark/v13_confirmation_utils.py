from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from .v7_value_utils import load_jsonl, mode_rows, mode_success, postprocess_v7_run
from .v8_coverage_utils import analyze_stale_attribution_v8
from .v12_step_value_utils import BASELINE_MODE, real_step_http_by_mode


def paired_rows(metrics: dict[str, Any], candidate_mode: str) -> list[dict[str, Any]]:
    baseline = {str(row.get("task_id") or ""): row for row in mode_rows(metrics, BASELINE_MODE)}
    candidate = {str(row.get("task_id") or ""): row for row in mode_rows(metrics, candidate_mode)}
    rows: list[dict[str, Any]] = []
    for task_id in sorted(set(baseline) | set(candidate)):
        base = baseline.get(task_id, {})
        current = candidate.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "task_type": current.get("task_type") or base.get("task_type"),
                "baseline_success": bool(base.get("success")),
                "candidate_success": bool(current.get("success")),
                "success_delta": int(bool(current.get("success"))) - int(bool(base.get("success"))),
                "baseline_clean": bool(base.get("clean_success")),
                "candidate_clean": bool(current.get("clean_success")),
                "clean_delta": int(bool(current.get("clean_success"))) - int(bool(base.get("clean_success"))),
                "baseline_time_sec": base.get("mission_time_sec"),
                "candidate_time_sec": current.get("mission_time_sec"),
                "baseline_path_m": base.get("path_length_m"),
                "candidate_path_m": current.get("path_length_m"),
            }
        )
    return rows


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    samples: int = 20000,
    seed: int = 20260711,
) -> dict[str, float | int | None]:
    if not deltas:
        return {"pairs": 0, "mean": None, "ci95_lower": None, "ci95_upper": None, "samples": samples, "seed": seed}
    rng = random.Random(seed)
    count = len(deltas)
    means = [sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)]
    means.sort()
    lower = means[max(0, int(samples * 0.025) - 1)]
    upper = means[min(samples - 1, int(samples * 0.975))]
    return {
        "pairs": count,
        "mean": round(sum(deltas) / count, 8),
        "ci95_lower": round(lower, 8),
        "ci95_upper": round(upper, 8),
        "samples": samples,
        "seed": seed,
    }


def evaluate_v13_confirmation(output: Path, candidate_mode: str) -> dict[str, Any]:
    output = Path(output)
    metrics = postprocess_v7_run(output, run_kind="v13_paired_confirmation")
    events = load_jsonl(output / "events.jsonl", limit=2_000_000)
    pairs = paired_rows(metrics, candidate_mode)
    _write_csv(output / "paired_results.csv", pairs)
    success_bootstrap = paired_bootstrap_ci([float(row["success_delta"]) for row in pairs])
    baseline = mode_success(metrics, BASELINE_MODE)
    candidate = mode_success(metrics, candidate_mode)
    clean_delta = int(candidate["clean_success_count"]) - int(baseline["clean_success_count"])
    subset_delta = {
        task_type: sum(int(row["success_delta"]) for row in pairs if row["task_type"] == task_type)
        for task_type in ("simple_navigation", "turn_choice", "semantic_target")
    }
    step_http = real_step_http_by_mode(metrics, events).get(candidate_mode, {})
    stale = analyze_stale_attribution_v8(events)
    collision_count = sum(int(row.get("num_collisions", 0) or 0) for row in metrics.get("episodes", []))
    v7 = metrics.get("v7_summary", {})
    stale_action_executed = int(v7.get("stale_action_executed", 0) or 0)
    failures: list[str] = []
    if len(pairs) != 45 or int(baseline["episodes"]) != 45 or int(candidate["episodes"]) != 45:
        failures.append("paired episode count != 45 per mode")
    if success_bootstrap["ci95_lower"] is None or float(success_bootstrap["ci95_lower"]) <= 0.0:
        failures.append("paired success bootstrap CI95 lower <= 0")
    if clean_delta < 0:
        failures.append("clean delta < 0")
    if subset_delta["turn_choice"] < 0:
        failures.append("turn_choice subset regressed")
    if subset_delta["semantic_target"] < 0:
        failures.append("semantic_target subset regressed")
    if int(step_http.get("accepted", 0) or 0) < 27:
        failures.append("accepted real Step calls < 27 route+semantic pairs")
    if int(step_http.get("errors", 0) or 0) != 0 or int(step_http.get("fallback_or_mock", 0) or 0) != 0:
        failures.append("Step HTTP errors/mock/fallback present")
    if int(stale.get("runtime_stale_discards", 0)) != 0:
        failures.append("runtime stale discards != 0")
    if int(stale.get("timebase_error", 0)) != 0:
        failures.append("timebase errors != 0")
    if collision_count != 0:
        failures.append("collision count != 0")
    if stale_action_executed != 0:
        failures.append("stale action executed != 0")
    result = {
        "schema_version": 1,
        "pass": not failures,
        "candidate_mode": candidate_mode,
        "baseline": baseline,
        "candidate": candidate,
        "paired_success": success_bootstrap,
        "clean_delta": clean_delta,
        "subset_success_delta": subset_delta,
        "real_step_http": step_http,
        "stale": stale,
        "collision_count": collision_count,
        "stale_action_executed": stale_action_executed,
        "failures": failures,
        "value_claim": "PROVEN" if not failures else "UNPROVEN",
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
    }
    metrics["v13_confirmation"] = result
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_summary(output, result), encoding="utf-8")
    return result


def render_summary(output: Path, result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V13 Paired Confirmation",
            "",
            f"- run: {output}",
            f"- pass: {result['pass']}",
            f"- value claim: {result['value_claim']}",
            f"- candidate: {result['candidate_mode']}",
            f"- paired success: {result['paired_success']}",
            f"- clean delta: {result['clean_delta']:+d}",
            f"- subset deltas: {result['subset_success_delta']}",
            f"- real Step HTTP: {result['real_step_http']}",
            f"- runtime stale: {result['stale']['runtime_stale_discards']}",
            f"- collision: {result['collision_count']}",
            f"- failures: {result['failures']}",
            "- Sim2Real: NOT READY FOR REAL ROBOT AUTONOMY",
            "",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
