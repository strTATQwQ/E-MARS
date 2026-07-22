from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CATEGORIES = {"referential", "compositional", "recovery"}
PUBLIC_TASK_KEYS = {"task_id", "category", "scene_template", "instruction", "timeout_sec"}
SAFETY_KEYS = {
    "collision",
    "runtime_stale",
    "timebase_error",
    "stale_action_executed",
}


def load_task_set(path: str | Path, *, require_full: bool = True) -> dict[str, Any]:
    task_path = Path(path)
    text = task_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("semantic navigation task set must be an object")
    validate_task_set(payload, require_full=require_full)
    return payload


def validate_task_set(payload: dict[str, Any], *, require_full: bool = True) -> None:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("semantic navigation task set must contain at least one task")
    if require_full and len(tasks) != 30:
        raise ValueError("semantic navigation task set must contain exactly 30 tasks")
    ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each task must be an object")
        for key in PUBLIC_TASK_KEYS | {"judge", "oracle_plan"}:
            if key not in task:
                raise ValueError(f"task missing required key: {key}")
        task_id = str(task["task_id"])
        if task_id in ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        ids.add(task_id)
        category = str(task["category"])
        if category not in CATEGORIES:
            raise ValueError(f"unsupported category: {category}")
        category_counts[category] += 1
        if not str(task["instruction"]).strip():
            raise ValueError(f"empty instruction: {task_id}")
        if not isinstance(task["judge"], dict):
            raise ValueError(f"judge must be an object: {task_id}")
        plan = task["oracle_plan"]
        if not isinstance(plan, list) or not plan:
            raise ValueError(f"oracle_plan must be a non-empty list: {task_id}")
        for subgoal in plan:
            _parse_semantic_subgoal(subgoal)
    expected = {category: 10 for category in CATEGORIES}
    if require_full and dict(category_counts) != expected:
        raise ValueError(f"expected 10 tasks per category, got {dict(category_counts)}")


def public_task_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task[key] for key in PUBLIC_TASK_KEYS}


def oracle_plan_payload(task: dict[str, Any], *, episode_id: str = "") -> list[dict[str, Any]]:
    result = []
    for index, raw in enumerate(task["oracle_plan"]):
        subgoal = _parse_semantic_subgoal(raw)
        subgoal.update(
            {
                "task_id": str(task["task_id"]),
                "episode_id": str(episode_id),
                "subgoal_index": index,
                "source": "forced_semantic_oracle",
            }
        )
        result.append(subgoal)
    return result


def task_set_fingerprint(path: str | Path) -> dict[str, Any]:
    task_path = Path(path)
    data = task_path.read_bytes()
    payload = load_task_set(task_path)
    return {
        "path": str(task_path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "task_count": len(payload["tasks"]),
        "category_counts": dict(Counter(str(task["category"]) for task in payload["tasks"])),
    }


def evaluate_semantic_navigation_run(
    records: list[dict[str, Any]],
    task_set: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    tasks_by_id = {str(task["task_id"]): task for task in task_set["tasks"]}
    seen: set[str] = set()
    category_success: Counter[str] = Counter()
    category_total: Counter[str] = Counter()
    successes = 0
    clean = 0
    semantic_errors = 0
    helped = 0
    hurt = 0
    fresh_calls = 0
    model_calls = 0
    omninav_real_image_calls = 0
    omninav_fallback_calls = 0
    episodes_with_real_omninav = 0
    safety_totals = {key: 0 for key in SAFETY_KEYS}
    failures: Counter[str] = Counter()

    for record in records:
        task_id = str(record.get("task_id") or "")
        if task_id not in tasks_by_id or task_id in seen:
            failures["invalid_or_duplicate_task"] += 1
            continue
        seen.add(task_id)
        category = str(tasks_by_id[task_id]["category"])
        category_total[category] += 1
        success = bool(record.get("success", False))
        is_clean = bool(record.get("clean", success))
        successes += int(success)
        clean += int(is_clean)
        category_success[category] += int(success)
        semantic_errors += int(record.get("semantic_errors", 0) or 0)
        helped += int(bool(record.get("helped", False)))
        hurt += int(bool(record.get("hurt", False)))
        model_calls += int(record.get("step_calls", 0) or 0)
        fresh_calls += int(record.get("fresh_multimodal_calls", 0) or 0)
        real_omninav = int(record.get("real_omninav_calls", 0) or 0)
        omninav_real_image_calls += real_omninav
        omninav_fallback_calls += int(record.get("omninav_fallback_calls", 0) or 0)
        episodes_with_real_omninav += int(real_omninav > 0)
        for key in SAFETY_KEYS:
            safety_totals[key] += int(record.get(key, 0) or 0)
        if not success:
            failures[str(record.get("failure_reason") or "unknown")] += 1

    category_gate = {
        category: category_total[category] == 10 and category_success[category] >= 8
        for category in sorted(CATEGORIES)
    }
    safety_pass = all(value == 0 for value in safety_totals.values())
    complete = len(seen) == len(tasks_by_id)
    forced_oracle_gate = complete and successes >= 24 and all(category_gate.values()) and safety_pass
    if mode == "step":
        model_evidence_gate = model_calls >= len(seen) and fresh_calls == model_calls
    elif mode == "forced_semantic_oracle":
        model_evidence_gate = (
            complete
            and episodes_with_real_omninav == len(seen)
            and omninav_fallback_calls == 0
        )
    else:
        model_evidence_gate = True
    passed = forced_oracle_gate and model_evidence_gate
    return {
        "schema_version": 1,
        "mode": mode,
        "episodes_expected": len(tasks_by_id),
        "episodes_scored": len(seen),
        "success": successes,
        "clean": clean,
        "success_rate": successes / len(seen) if seen else 0.0,
        "category_success": dict(category_success),
        "category_total": dict(category_total),
        "category_gate": category_gate,
        "semantic_errors": semantic_errors,
        "helped": helped,
        "hurt": hurt,
        "step_calls": model_calls,
        "fresh_multimodal_calls": fresh_calls,
        "omninav_real_image_calls": omninav_real_image_calls,
        "omninav_fallback_calls": omninav_fallback_calls,
        "episodes_with_real_omninav": episodes_with_real_omninav,
        "safety": safety_totals,
        "safety_pass": safety_pass,
        "complete": complete,
        "forced_semantic_oracle_gate": forced_oracle_gate,
        "model_evidence_gate": model_evidence_gate,
        "pass": passed,
        "screening_allowed": passed if mode == "forced_semantic_oracle" else False,
        "qualification_evidence": False,
        "failure_top1": failures.most_common(1)[0][0] if failures else "none",
        "failure_counts": dict(failures),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            result.append(payload)
    return result


def _parse_semantic_subgoal(raw: dict[str, Any]) -> dict[str, Any]:
    # Import lazily so benchmark data validation also works from a source checkout.
    try:
        from omninav_step_scheduler.semantic_executive import parse_semantic_subgoal_json
    except ImportError as exc:
        raise RuntimeError("omninav_step_scheduler must be on PYTHONPATH") from exc
    return parse_semantic_subgoal_json(raw).to_dict(include_metadata=False)
