from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config_loader import load_data


BASELINE_MODE = "omninav_only_v20_semantic_screen"
HEURISTIC_MODE = "omninav_public_heuristic_v20_semantic_screen"
ORACLE_MODE = "omninav_forced_semantic_oracle_v20_semantic_screen"
STEP_MODE = "omninav_step_semantic_executive_v20_semantic_screen"
SCREEN_MODES = (BASELINE_MODE, HEURISTIC_MODE, ORACLE_MODE, STEP_MODE)
SCREEN_TASK_IDS = tuple(
    [f"ref_{index:02d}" for index in range(1, 6)]
    + [f"comp_{index:02d}" for index in range(1, 6)]
    + [f"rec_{index:02d}" for index in range(1, 6)]
)


def materialize_v20_screen_tasks(source: str | Path, output: str | Path) -> dict[str, Any]:
    document = load_data(source)
    by_id = {str(task.get("task_id")): dict(task) for task in document.get("tasks", [])}
    missing = [task_id for task_id in SCREEN_TASK_IDS if task_id not in by_id]
    if missing:
        raise KeyError(f"missing V20 screening tasks: {missing}")
    tasks = []
    for task_id in SCREEN_TASK_IDS:
        task = by_id[task_id]
        task["original_instruction"] = str(task.get("instruction") or "")
        task["instruction"] = _explicit_public_instruction(task.get("oracle_plan") or [])
        task["v20_screen_scope"] = {
            "instruction_uses_public_marker_labels": True,
            "explicit_stage_decomposition": True,
            "oracle_plan_is_judge_only": True,
            "actual_viewport_required_for_step": True,
            "open_vocabulary_claim": False,
        }
        tasks.append(task)
    result = {
        "schema_version": 1,
        "source_benchmark": "v19_local_marker_v2",
        "evidence_scope": "instrumented_explicit_stage_semantic_value_screen",
        "qualification_evidence": False,
        "tasks": tasks,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _explicit_public_instruction(plan: list[dict[str, Any]]) -> str:
    clauses = []
    verbs = {
        "find": "find",
        "pass": "pass",
        "enter": "enter",
        "approach": "approach",
        "verify": "verify arrival at",
        "ask": "ask about",
    }
    for index, row in enumerate(plan):
        subgoal_type = str(row.get("subgoal_type") or "ask")
        target = str(row.get("target") or "operator clarification")
        recovery = str(row.get("recovery") or "stop")
        clauses.append(f"({index}) {verbs.get(subgoal_type, 'ask about')} {target} with recovery {recovery}")
    return "Execute exactly one numbered stage per decision without merging or skipping: " + "; ".join(clauses) + "."


def evaluate_v20_screen(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    metrics = load_data(root / "metrics.json")
    task_doc = load_data(root / "selected_tasks.yaml")
    categories = {str(task["task_id"]): str(task.get("category") or "unknown") for task in task_doc.get("tasks", [])}
    rows = [row for row in metrics.get("episodes", []) if str(row.get("mode") or "") in SCREEN_MODES]
    by_mode: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in SCREEN_MODES}
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_mode[str(row["mode"])][str(row["task_id"])] = row
    for mode in SCREEN_MODES:
        mode_rows = list(by_mode[mode].values())
        category_success = Counter()
        category_total = Counter()
        safety = {key: 0 for key in ("num_collisions", "runtime_stale", "timebase_error", "stale_action_executed")}
        for row in mode_rows:
            category = categories.get(str(row.get("task_id") or ""), "unknown")
            category_total[category] += 1
            category_success[category] += int(bool(row.get("success")))
            for key in safety:
                safety[key] += int(row.get(key, 0) or 0)
        summaries[mode] = {
            "episodes": len(mode_rows),
            "success": sum(bool(row.get("success")) for row in mode_rows),
            "clean": sum(bool(row.get("clean_success")) for row in mode_rows),
            "category_success": dict(category_success),
            "category_total": dict(category_total),
            "safety": safety,
            "failure_top1": _failure_top1(mode_rows),
        }

    paired_rows = []
    helped = hurt = 0
    for task_id in categories:
        baseline = by_mode[BASELINE_MODE].get(task_id, {})
        candidate = by_mode[STEP_MODE].get(task_id, {})
        baseline_success = bool(baseline.get("success"))
        candidate_success = bool(candidate.get("success"))
        helped += int(candidate_success and not baseline_success)
        hurt += int(baseline_success and not candidate_success)
        paired_rows.append(
            {
                "task_id": task_id,
                "category": categories[task_id],
                "baseline_success": baseline_success,
                "step_success": candidate_success,
                "baseline_clean": bool(baseline.get("clean_success")),
                "step_clean": bool(candidate.get("clean_success")),
                "outcome": "helped" if candidate_success and not baseline_success else (
                    "hurt" if baseline_success and not candidate_success else "no_effect"
                ),
            }
        )

    evidence = _step_evidence(root, by_mode[STEP_MODE], categories)
    baseline = summaries[BASELINE_MODE]
    step = summaries[STEP_MODE]
    oracle = summaries[ORACLE_MODE]
    category_nonregression = all(
        step["category_success"].get(category, 0) >= baseline["category_success"].get(category, 0)
        for category in sorted(set(categories.values()))
    )
    gates = {
        "complete": all(summaries[mode]["episodes"] == 15 for mode in SCREEN_MODES),
        "oracle_upper_bound": oracle["success"] >= 12,
        "step_success": step["success"] >= 8,
        "step_clean": step["clean"] >= 6,
        "success_delta": step["success"] - baseline["success"] >= 2,
        "clean_delta": step["clean"] - baseline["clean"] >= 2,
        "helped_gt_hurt": helped > hurt,
        "category_nonregression": category_nonregression,
        "real_fresh_step": evidence["all_step_episodes_called"] and evidence["all_responses_fresh_multimodal"],
        "strict_step": evidence["responses"] > 0 and evidence["accepted_responses"] == evidence["responses"],
        "step_latency": evidence["step_latency_p95_sec"] <= 7.0,
        "oracle_leakage": not evidence["oracle_leakage_findings"],
        "safety": all(
            value == 0
            for mode in SCREEN_MODES
            for value in summaries[mode]["safety"].values()
        ),
    }
    return {
        "schema_version": 1,
        "pass": all(gates.values()),
        "modes": summaries,
        "paired": paired_rows,
        "helped": helped,
        "hurt": hurt,
        "success_delta": step["success"] - baseline["success"],
        "clean_delta": step["clean"] - baseline["clean"],
        "step_evidence": evidence,
        "gates": gates,
        "paired_confirmation_allowed": all(gates.values()),
        "evidence_scope": "instrumented_explicit_stage_semantic_value_screen",
        "full_natural_language_value_claim": "unproven",
        "qualification_evidence": False,
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
    }


def _step_evidence(root: Path, episodes: dict[str, dict[str, Any]], categories: dict[str, str]) -> dict[str, Any]:
    responses = []
    leakage = []
    called = set()
    for task_id in categories:
        path = root / STEP_MODE / task_id / "events.jsonl"
        for event in _read_jsonl(path):
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            event_type = str(details.get("event_type") or "")
            if event.get("event") == "step_request":
                leakage.extend(_request_leakage(details, task_id))
            if event_type == "step_http_response" and str(details.get("role") or "") == "semantic_executive":
                responses.append(details)
                called.add(task_id)
    latencies = [float(row.get("latency_s", 0.0) or 0.0) for row in responses if float(row.get("latency_s", 0.0) or 0.0) > 0]
    fresh = 0
    for row in responses:
        snapshot = row.get("image_snapshot") if isinstance(row.get("image_snapshot"), dict) else {}
        fresh += int(
            bool(row.get("multimodal"))
            and str(snapshot.get("source") or "") == "primary"
            and int(snapshot.get("frame_seq", 0) or 0) > 0
            and float(snapshot.get("age_sec", 999.0) or 999.0) <= 0.75
        )
    return {
        "step_episodes": len(episodes),
        "episodes_with_call": len(called),
        "all_step_episodes_called": len(episodes) == 15 and len(called) == 15,
        "responses": len(responses),
        "accepted_responses": sum(str(row.get("result") or "").startswith("accepted") for row in responses),
        "fresh_multimodal_responses": fresh,
        "all_responses_fresh_multimodal": bool(responses) and fresh == len(responses),
        "completion_evidence_defaulted_responses": sum(
            str((row.get("output") or {}).get("completion_evidence_source") or "") == "client_safe_default"
            for row in responses
        ),
        "step_latency_p95_sec": _percentile(latencies, 0.95),
        "oracle_leakage_findings": sorted(set(leakage)),
    }


def _request_leakage(request: dict[str, Any], task_id: str) -> list[str]:
    forbidden = {"oracle_plan", "judge", "target_pose", "expected_branch", "correct_branch", "ground_truth"}
    findings = []
    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + (str(index),))
        elif isinstance(value, dict):
            for key, item in value.items():
                current = path + (str(key).lower(),)
                if str(key).lower() in forbidden:
                    findings.append(f"{task_id}:{'.'.join(current)}")
                visit(item, current)
    visit(request)
    return findings


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _failure_top1(rows: list[dict[str, Any]]) -> str:
    failures = Counter(str(row.get("failure_reason") or "unknown") for row in rows if not row.get("success"))
    return failures.most_common(1)[0][0] if failures else "none"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)
