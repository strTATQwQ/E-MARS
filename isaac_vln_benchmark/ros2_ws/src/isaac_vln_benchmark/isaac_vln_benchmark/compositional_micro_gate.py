from __future__ import annotations

from collections import Counter
from typing import Any


SAFETY_FIELDS = {
    "collision": "num_collisions",
    "runtime_stale": "runtime_stale",
    "timebase_error": "timebase_error",
    "stale_action_executed": "stale_action_executed",
}


def evaluate_v19_compositional_micro(
    metrics: dict[str, Any],
    task_set: dict[str, Any],
    model_evidence: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    tasks = {str(task["task_id"]): task for task in task_set.get("tasks", [])}
    episodes = {
        str(episode.get("task_id") or ""): episode
        for episode in metrics.get("episodes", [])
        if str(episode.get("task_id") or "") in tasks
    }
    parent_total: Counter[str] = Counter()
    parent_success: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    safety = {field: 0 for field in SAFETY_FIELDS}
    semantic_errors = 0
    success = 0
    clean = 0
    for task_id, episode in episodes.items():
        runtime = tasks[task_id].get("semantic_runtime") or {}
        parent = str(runtime.get("parent_task_id") or task_id)
        episode_success = bool(episode.get("success", False))
        parent_total[parent] += 1
        parent_success[parent] += int(episode_success)
        success += int(episode_success)
        clean += int(bool(episode.get("clean_success", False)))
        semantic_errors += int(episode.get("semantic_errors", 0) or 0)
        for name, source in SAFETY_FIELDS.items():
            safety[name] += int(episode.get(source, 0) or 0)
        if not episode_success:
            failures[str(episode.get("failure_reason") or "unknown")] += 1
    expected = len(tasks)
    complete = len(episodes) == expected
    success_rate = success / expected if expected else 0.0
    success_rate_min = float(gate_config.get("success_rate_min", 0.90))
    parent_success_min = int(gate_config.get("parent_success_min", 2))
    parent_episodes = int(gate_config.get("parent_episodes", 3))
    parent_gate = {
        parent: parent_total[parent] == parent_episodes and parent_success[parent] >= parent_success_min
        for parent in sorted(parent_total)
    }
    expected_parents = {
        str((task.get("semantic_runtime") or {}).get("parent_task_id") or task_id)
        for task_id, task in tasks.items()
    }
    parent_complete = set(parent_gate) == expected_parents
    safety_pass = all(value == int(gate_config.get(name, 0)) for name, value in safety.items())
    model_evidence_pass = bool(model_evidence.get("pass", False))
    passed = bool(
        complete
        and success_rate >= success_rate_min
        and parent_complete
        and all(parent_gate.values())
        and semantic_errors == 0
        and safety_pass
        and model_evidence_pass
    )
    first_runtime = next(
        (
            task.get("semantic_runtime")
            for task in tasks.values()
            if isinstance(task.get("semantic_runtime"), dict)
        ),
        {},
    )
    return {
        "schema_version": 1,
        "mode": "forced_semantic_oracle",
        "micro_stage": str(task_set.get("micro_stage") or first_runtime.get("micro_stage") or "unknown"),
        "runtime_profile": str(task_set.get("runtime_profile") or first_runtime.get("runtime_profile") or "unknown"),
        "episodes_expected": expected,
        "episodes_scored": len(episodes),
        "complete": complete,
        "success": success,
        "clean": clean,
        "success_rate": success_rate,
        "success_rate_min": success_rate_min,
        "parent_success": dict(parent_success),
        "parent_total": dict(parent_total),
        "parent_gate": parent_gate,
        "semantic_errors": semantic_errors,
        "safety": safety,
        "safety_pass": safety_pass,
        "model_evidence_gate": model_evidence_pass,
        "omninav_real_image_calls": int(model_evidence.get("real_image_calls", 0) or 0),
        "omninav_fallback_calls": int(model_evidence.get("fallback_calls", 0) or 0),
        "failure_counts": dict(failures),
        "failure_top1": failures.most_common(1)[0][0] if failures else "none",
        "pass": passed,
        "formal_compositional_gate_unlocked": False,
        "step_screening_allowed": False,
        "qualification_evidence": False,
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
    }


def evaluate_v19_compositional_full(
    metrics: dict[str, Any],
    task_set: dict[str, Any],
    model_evidence: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    task_ids = {str(task["task_id"]) for task in task_set.get("tasks", [])}
    episodes = [episode for episode in metrics.get("episodes", []) if str(episode.get("task_id") or "") in task_ids]
    success = sum(int(bool(episode.get("success", False))) for episode in episodes)
    clean = sum(int(bool(episode.get("clean_success", False))) for episode in episodes)
    semantic_errors = sum(int(episode.get("semantic_errors", 0) or 0) for episode in episodes)
    safety = {
        name: sum(int(episode.get(source, 0) or 0) for episode in episodes)
        for name, source in SAFETY_FIELDS.items()
    }
    failures = Counter(
        str(episode.get("failure_reason") or "unknown")
        for episode in episodes
        if not bool(episode.get("success", False))
    )
    expected = len(task_ids)
    success_min = int(gate_config.get("success_min", 8))
    complete = len(episodes) == expected == 10
    safety_pass = all(value == int(gate_config.get(name, 0)) for name, value in safety.items())
    model_evidence_pass = bool(model_evidence.get("pass", False))
    passed = bool(
        complete
        and success >= success_min
        and semantic_errors == 0
        and safety_pass
        and model_evidence_pass
    )
    first_runtime = next(
        (
            task.get("semantic_runtime")
            for task in task_set.get("tasks", [])
            if isinstance(task.get("semantic_runtime"), dict)
        ),
        {},
    )
    return {
        "schema_version": 1,
        "mode": "forced_semantic_oracle",
        "gate_name": "v19_compositional_full10",
        "runtime_profile": str(first_runtime.get("runtime_profile") or "unknown"),
        "episodes_expected": expected,
        "episodes_scored": len(episodes),
        "complete": complete,
        "success": success,
        "clean": clean,
        "success_min": success_min,
        "semantic_errors": semantic_errors,
        "safety": safety,
        "safety_pass": safety_pass,
        "model_evidence_gate": model_evidence_pass,
        "omninav_real_image_calls": int(model_evidence.get("real_image_calls", 0) or 0),
        "omninav_fallback_calls": int(model_evidence.get("fallback_calls", 0) or 0),
        "failure_counts": dict(failures),
        "failure_top1": failures.most_common(1)[0][0] if failures else "none",
        "pass": passed,
        "forced_oracle_full30_retry_allowed": passed,
        "step_screening_allowed": False,
        "qualification_evidence": False,
        "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
    }
