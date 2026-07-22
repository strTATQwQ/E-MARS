from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .config_loader import load_data


PUBLIC_INSTRUCTIONS = {
    "ref_01": (
        "Find the nearest red fire extinguisher marker, approach it, then verify arrival at the same nearest red fire "
        "extinguisher marker while stopped. "
        "Scan if it cannot be found."
    ),
    "ref_04": (
        "Find the nearest red fire extinguisher marker, then enter the nearest blue box marker region, "
        "and verify arrival at the same nearest blue box marker while stopped. Backtrack if entry is blocked."
    ),
    "comp_01": (
        "Execute exactly one numbered stage per decision without merging or skipping: (0) pass the nearest red fire "
        "extinguisher marker with recovery scan; (1) enter the nearest blue box marker region with recovery backtrack; "
        "(2) approach the nearest red traffic cone marker with recovery scan; (3) verify arrival at the same nearest "
        "red traffic cone marker while stopped with recovery stop."
    ),
    "comp_07": (
        "Execute exactly one numbered stage per decision without merging or skipping: (0) pass the nearest red fire "
        "extinguisher marker with recovery scan; (1) pass the nearest blue box marker with recovery scan; (2) enter "
        "the nearest red traffic cone marker region with recovery backtrack; (3) approach the nearest green exit sign "
        "marker with recovery stop and use 'green exit sign marker approached' as completion evidence; (4) verify "
        "arrival at the same nearest green exit sign marker while stopped with recovery stop."
    ),
    "rec_01": (
        "First find the nearest red fire extinguisher marker. Then approach it. Finally verify arrival at the same "
        "nearest red fire extinguisher marker while stopped. Use recovery backtrack for both find and approach, and "
        "stop for verify."
    ),
    "rec_02": (
        "First find the nearest red fire extinguisher marker. Then approach it. Finally verify arrival at the same "
        "nearest red fire extinguisher marker while stopped. Use recovery scan for both find and approach, and stop "
        "for verify."
    ),
}


def materialize_public_marker_tasks(source: str | Path, output: str | Path) -> dict[str, Any]:
    document = load_data(source)
    by_id = {str(task.get("task_id")): dict(task) for task in document.get("tasks", [])}
    tasks: list[dict[str, Any]] = []
    for task_id, instruction in PUBLIC_INSTRUCTIONS.items():
        if task_id not in by_id:
            raise KeyError(f"missing V19 task {task_id}")
        task = by_id[task_id]
        if task_id == "comp_01":
            task = _without_plan_stages(task, {2})
        elif task_id == "comp_07":
            task = _without_plan_stages(task, {3})
        task["instruction"] = instruction
        task["step_quality_scope"] = {
            "public_marker_language": True,
            "actual_viewport_required": True,
            "oracle_plan_is_judge_only": True,
            "open_vocabulary_claim": False,
        }
        tasks.append(task)
    result = {
        "schema_version": 1,
        "source_benchmark": "v19_local_marker_v2",
        "evidence_scope": "instrumented_public_marker_semantic_instruction_following",
        "qualification_evidence": False,
        "tasks": tasks,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _without_plan_stages(task: dict[str, Any], removed: set[int]) -> dict[str, Any]:
    result = dict(task)
    original_plan = list(result.get("oracle_plan") or [])
    kept = [index for index in range(len(original_plan)) if index not in removed]
    result["oracle_plan"] = [dict(original_plan[index]) for index in kept]
    judge = dict(result.get("judge") or {})
    judge["required_sequence"] = [
        str(original_plan[index].get("subgoal_type") or "") for index in kept
    ]
    result["judge"] = judge
    runtime = dict(result.get("semantic_runtime") or {})
    for field in ("subgoal_object_ids", "execution_targets"):
        source = runtime.get(field) if isinstance(runtime.get(field), dict) else {}
        runtime[field] = {
            str(new_index): source[str(old_index)]
            for new_index, old_index in enumerate(kept)
            if str(old_index) in source
        }
    result["semantic_runtime"] = runtime
    result["step_quality_removed_redundant_find_indices"] = sorted(removed)
    return result


def evaluate_step_semantic_quality(
    run_dir: str | Path,
    *,
    max_image_age_sec: float = 0.75,
    latency_gate_sec: float | None = None,
    closed_loop_success_min: int | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    config = load_data(root / "config.yaml") if (root / "config.yaml").is_file() else {}
    gate_config = config.get("gates") if isinstance(config.get("gates"), dict) else {}
    if latency_gate_sec is None:
        latency_gate_sec = float(gate_config.get("step_latency_p95_max_sec", 7.0))
    task_doc = load_data(root / "selected_tasks.yaml")
    tasks = {str(task["task_id"]): task for task in task_doc.get("tasks", [])}
    metrics_doc = load_data(root / "metrics.json")
    episodes = [
        row for row in metrics_doc.get("episodes", [])
        if str(row.get("mode")) == "omninav_step_semantic_executive"
    ]
    mode_root = root / "omninav_step_semantic_executive"
    comparisons: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    leakage_findings: list[str] = []
    response_total = 0
    accepted_total = 0
    evidence_defaulted_total = 0
    multimodal_total = 0
    fresh_image_total = 0
    latencies: list[float] = []

    for episode in episodes:
        task_id = str(episode.get("task_id") or "")
        task = tasks.get(task_id, {})
        expected_plan = list(task.get("oracle_plan") or [])
        events_path = mode_root / task_id / "events.jsonl"
        events = _read_jsonl(events_path)
        requests = [row.get("details") or {} for row in events if row.get("event") == "step_request"]
        for request in requests:
            leakage_findings.extend(_oracle_request_findings(request, task_id))
        responses = [
            row.get("details") or {}
            for row in events
            if str((row.get("details") or {}).get("event_type") or "") == "step_http_response"
            and str((row.get("details") or {}).get("role") or "") == "semantic_executive"
        ]
        response_total += len(responses)
        accepted = [row for row in responses if str(row.get("result") or "").startswith("accepted")]
        accepted_total += len(accepted)
        evidence_defaulted_total += sum(
            str((row.get("output") or {}).get("completion_evidence_source") or "")
            == "client_safe_default"
            for row in accepted
        )
        multimodal_total += sum(bool(row.get("multimodal")) for row in responses)
        for row in responses:
            latency = float(row.get("latency_s", 0.0) or 0.0)
            if latency > 0:
                latencies.append(latency)
            snapshot = row.get("image_snapshot") if isinstance(row.get("image_snapshot"), dict) else {}
            fresh_image_total += int(
                int(snapshot.get("frame_seq", -1) or -1) > 0
                and float(snapshot.get("age_sec", 999.0) or 999.0) <= max_image_age_sec
                and str(snapshot.get("source") or "") == "primary"
            )
        outputs = [row.get("output") or {} for row in accepted]
        for index, expected in enumerate(expected_plan):
            predicted = outputs[index] if index < len(outputs) else {}
            type_ok = str(predicted.get("subgoal_type") or "").lower() == str(
                expected.get("subgoal_type") or ""
            ).lower()
            target_ok = _target_matches(str(predicted.get("target") or ""), str(expected.get("target") or ""))
            relation_ok = _relation_matches(
                str(predicted.get("relation") or ""), str(expected.get("relation") or "")
            )
            recovery_ok = str(predicted.get("recovery") or "").lower() == str(
                expected.get("recovery") or ""
            ).lower()
            comparisons.append(
                {
                    "task_id": task_id,
                    "subgoal_index": index,
                    "expected_type": expected.get("subgoal_type"),
                    "predicted_type": predicted.get("subgoal_type"),
                    "type_ok": type_ok,
                    "expected_target": expected.get("target"),
                    "predicted_target": predicted.get("target"),
                    "target_relation_ok": target_ok and relation_ok,
                    "expected_recovery": expected.get("recovery"),
                    "predicted_recovery": predicted.get("recovery"),
                    "recovery_ok": recovery_ok,
                }
            )
        sequence_ok = len(outputs) >= len(expected_plan) and all(
            str(outputs[index].get("subgoal_type") or "").lower()
            == str(expected_plan[index].get("subgoal_type") or "").lower()
            for index in range(len(expected_plan))
        )
        sequence_rows.append(
            {
                "task_id": task_id,
                "expected_count": len(expected_plan),
                "accepted_count": len(outputs),
                "sequence_ok": sequence_ok,
                "closed_loop_success": bool(episode.get("success")),
            }
        )

    expected_total = len(comparisons)
    type_accuracy = _ratio(sum(bool(row["type_ok"]) for row in comparisons), expected_total)
    target_relation_accuracy = _ratio(
        sum(bool(row["target_relation_ok"]) for row in comparisons), expected_total
    )
    recovery_accuracy = _ratio(sum(bool(row["recovery_ok"]) for row in comparisons), expected_total)
    sequence_accuracy = _ratio(sum(bool(row["sequence_ok"]) for row in sequence_rows), len(sequence_rows))
    p95_latency = _percentile(latencies, 0.95)
    safety = {
        key: sum(int(row.get(key, 0) or 0) for row in episodes)
        for key in ("num_collisions", "runtime_stale", "timebase_error", "stale_action_executed")
    }
    closed_loop_success = sum(bool(row.get("success")) for row in episodes)
    if closed_loop_success_min is None:
        configured_minimum = gate_config.get("closed_loop_success_min")
        closed_loop_success_min = (
            int(configured_minimum)
            if configured_minimum is not None
            else math.ceil(0.67 * len(episodes))
        )
    gates = {
        "type_accuracy": type_accuracy >= 0.80,
        "target_relation_accuracy": target_relation_accuracy >= 0.80,
        "sequence_accuracy": sequence_accuracy >= 0.80,
        "recovery_accuracy": recovery_accuracy >= 0.80,
        "strict_parse": response_total > 0 and accepted_total == response_total,
        "real_multimodal": response_total > 0 and multimodal_total == response_total,
        "fresh_images": response_total > 0 and fresh_image_total == response_total,
        "latency_p95": bool(latencies) and p95_latency <= latency_gate_sec,
        "oracle_leakage": not leakage_findings,
        "safety": all(value == 0 for value in safety.values()),
        "closed_loop": bool(episodes) and closed_loop_success >= closed_loop_success_min,
    }
    return {
        "pass": all(gates.values()),
        "evidence_scope": "instrumented_public_marker_semantic_instruction_following",
        "open_vocabulary_claim": False,
        "episodes": len(episodes),
        "closed_loop_success": closed_loop_success,
        "closed_loop_success_min": closed_loop_success_min,
        "expected_subgoals": expected_total,
        "responses": response_total,
        "accepted_responses": accepted_total,
        "completion_evidence_defaulted_responses": evidence_defaulted_total,
        "multimodal_responses": multimodal_total,
        "fresh_image_responses": fresh_image_total,
        "type_accuracy": type_accuracy,
        "target_relation_accuracy": target_relation_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "recovery_accuracy": recovery_accuracy,
        "step_latency_p95_sec": p95_latency,
        "latency_gate_sec": latency_gate_sec,
        "oracle_leakage_findings": sorted(set(leakage_findings)),
        "safety": safety,
        "gates": gates,
        "comparisons": comparisons,
        "sequence_rows": sequence_rows,
        "step_full_screening_allowed": all(gates.values()),
        "qualification_evidence": False,
        "value_claim": "unproven",
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
    }


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


def _tokens(value: str) -> set[str]:
    ignored = {
        "nearest", "the", "a", "an", "marker", "target", "operator", "alternative",
        "pass", "toward", "arrival", "arrived", "at",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value).lower()) if token not in ignored
    }


def _target_matches(predicted: str, expected: str) -> bool:
    expected_tokens = _tokens(expected)
    return bool(expected_tokens) and expected_tokens.issubset(_tokens(predicted))


def _relation_matches(predicted: str, expected: str) -> bool:
    expected_tokens = _tokens(expected)
    predicted_tokens = _tokens(predicted)
    return not predicted_tokens if not expected_tokens else expected_tokens.issubset(predicted_tokens)


def _oracle_request_findings(request: dict[str, Any], task_id: str) -> list[str]:
    forbidden = {
        "oracle_plan",
        "original_oracle_plan",
        "target_pose",
        "expected_branch",
        "correct_branch",
        "branch_polygon",
        "branch_polygons",
        "ground_truth",
        "judge",
        "success_truth",
    }
    findings: list[str] = []

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + (str(index),))
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            normalized = str(key).lower()
            current = path + (normalized,)
            if normalized in forbidden:
                findings.append(f"{task_id}:{'.'.join(current)}")
            visit(item, current)

    visit(request)
    return findings


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[rank]
