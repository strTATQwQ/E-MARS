#!/usr/bin/env python3
"""Diagnose T5 evaluator starvation without changing the navigation runtime.

The analyzer deliberately separates evaluator/physics steps from model-facing
decisions.  A motion-gate-only response is never counted as a model decision.
It also correlates the final client response with the evaluator termination so
that a safe-stop cannot silently masquerade as a model STOP.

The optional T0/T2/T3 inputs are historical, descriptive references.  The
strict promotion gate is applied only to the T5 ``--run`` inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTION_SOURCES = {
    0: "unknown",
    1: "system2",
    2: "system1_new",
    3: "system1_queue",
}
CAMERA_FIELDS = (
    "camera_frame",
    "camera_height_above_support_m",
    "camera_hfov_deg",
    "camera_pitch_down_deg",
    "camera_translation_from_base_m",
    "camera_vfov_deg",
    "depth_camera_frame",
    "depth_camera_model",
    "depth_height_above_support_m",
    "depth_hfov_deg",
    "depth_minimum_m",
    "depth_pitch_down_deg",
    "depth_translation_from_base_m",
    "depth_vfov_deg",
)
SAFE_STOP_GATE_KINDS = {
    "cancel_ack",
    "hold_post_stop",
    "inherited_cancel_ack",
    "safe_stop_complete",
    "safe_stop_stale",
    "safe_stop_timeout",
}


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        value["_evidence_line"] = line_number
        values.append(value)
    if not values:
        raise ValueError(f"{path} is empty")
    return values


def _unique(root: Path, name: str) -> Path:
    matches = sorted(
        path for path in root.rglob(name) if path.is_file() and not path.is_symlink()
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def _physics_hz(path: Path | None) -> float | None:
    if path is None:
        return None
    observed = {
        _finite(row["physics_hz"], "physics_hz")
        for row in _rows(path)
        if "physics_hz" in row
    }
    if len(observed) != 1 or next(iter(observed), 0.0) <= 0.0:
        raise ValueError(f"{path} must contain one consistent positive physics_hz")
    return next(iter(observed))


def _metrics(raw: dict[str, Any]) -> dict[str, float | None]:
    official = raw.get("official_metrics")
    if not isinstance(official, dict):
        official = raw
    return {
        "tl_m": _optional_number(raw.get("TL", raw.get("tl_m"))),
        "shortest_path_length_m": _optional_number(raw.get("shortest_path_length")),
        "ne_m": _optional_number(official.get("ne_m", official.get("NE"))),
        "os": _optional_number(official.get("os", official.get("OS"))),
        "sr": _optional_number(official.get("sr", official.get("SR"))),
        "spl": _optional_number(official.get("spl", official.get("SPL"))),
        "ndtw": _optional_number(raw.get("ndtw")),
    }


def _aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    names = (
        "tl_m",
        "shortest_path_length_m",
        "ne_m",
        "os",
        "sr",
        "spl",
        "ndtw",
    )
    return {
        name: _mean(episode["metrics"].get(name) for episode in episodes)
        for name in names
    }


def _normalized_result_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    value: dict[str, Any] = result
    if "val_unseen" in result and isinstance(result["val_unseen"], dict):
        value = result["val_unseen"]
    return {
        "tl_m": _optional_number(value.get("TL")),
        "shortest_path_length_m": None,
        "ne_m": _optional_number(value.get("NE")),
        "os": _optional_number(value.get("OS")),
        "sr": _optional_number(value.get("SR")),
        "spl": _optional_number(value.get("SPL")),
        "ndtw": _optional_number(value.get("nDTW", value.get("ndtw"))),
    }


def _episode_token(raw: dict[str, Any], ordinal: int) -> str:
    for key in ("trajectory_id", "episode_key", "episode_id"):
        value = raw.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value).rsplit("_", 1)[-1].rsplit("::", 1)[-1]
    raise ValueError(f"episode ordinal {ordinal} has no explicit identity")


def _action_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row.get("action_source", 0)) for row in rows)
    return {name: counts.get(number, 0) for number, name in ACTION_SOURCES.items()}


def _gate_kind(row: dict[str, Any]) -> str | None:
    decision = row.get("motion_observation_gate_decision")
    if isinstance(decision, dict) and isinstance(decision.get("kind"), str):
        return str(decision["kind"])
    return None


def _last_response_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_line": row.get("_evidence_line"),
        "request_id": row.get("request_id"),
        "sequence_id": row.get("sequence_id"),
        "discrete_action": row.get("discrete_action"),
        "model_discrete_action": row.get("model_discrete_action"),
        "stop": row.get("stop"),
        "model_stop": row.get("model_stop"),
        "motion_gate_only": row.get("motion_observation_gate_only") is True,
        "motion_gate_kind": _gate_kind(row),
        "status_message": row.get("status_message"),
    }


def _ipc_step_error_code(error: object) -> str:
    """Reduce an IPC exception to a stable code without embedding log content."""

    normalized = str(error).casefold()
    if "fresh recovery trajectory timed out" in normalized:
        return "fresh_recovery_trajectory_timeout"
    if (
        "system 1 queue" in normalized
        and "no identity-bound absolute target" in normalized
    ):
        return "system1_queue_missing_identity_bound_target"
    return "other_ipc_step_error"


def _evaluator_log_corroboration(path: Path | None) -> dict[str, Any]:
    """Return bounded, non-content-bearing corroboration from an optional log."""

    if path is None:
        return {
            "available": False,
            "inference_basis": (
                "identity-correlated final client response only; evaluator action-zero "
                "cause cannot be attributed because its log was not archived"
            ),
        }
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"optional evaluator log is not a regular file: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ipc_marker = "INTERNVLA_LOCAL_IPC_STEP_ERROR"
    ipc_events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        marker_index = line.find(ipc_marker)
        if marker_index < 0:
            continue
        encoded = line[marker_index + len(ipc_marker) :].lstrip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(encoded)
        except json.JSONDecodeError:
            ipc_events.append(
                {
                    "line_index": index,
                    "episode_ordinal": None,
                    "error_code": "unparseable_ipc_step_error",
                }
            )
            continue
        ordinal = payload.get("episode_ordinal") if isinstance(payload, dict) else None
        ipc_events.append(
            {
                "line_index": index,
                "episode_ordinal": (
                    ordinal if isinstance(ordinal, int) and ordinal >= 0 else None
                ),
                "error_code": _ipc_step_error_code(
                    payload.get("error") if isinstance(payload, dict) else None
                ),
            }
        )
    client_minus_one = re.compile(
        r"(?:internvla_model_action_ok|ipc).*?"
        r"(?:discrete_)?action[\"']?\s*(?:[:=]\s*)?(?<![0-9])-1(?![0-9])",
        re.IGNORECASE,
    )
    evaluator_action_zero = re.compile(
        r"(?:now\s+action|ideal_flag).*?[\"']?action[\"']?\s*:\s*\[\s*0\s*\]",
        re.IGNORECASE,
    )
    not_reach_goal = re.compile(r"not_reach_goal", re.IGNORECASE)
    finished_not_reach_goal = re.compile(
        r"finish:.*result:not_reach_goal", re.IGNORECASE
    )
    finished_episode = re.compile(
        r"\[(?P<ordinal>[1-9][0-9]*)/[1-9][0-9]*\].*?"
        r"finish:.*?\[result:(?P<result>[^\]]+)\]",
        re.IGNORECASE,
    )
    minus_one_lines = [
        index
        for index, line in enumerate(lines)
        if ipc_marker not in line and client_minus_one.search(line)
    ]
    zero_lines = [
        index for index, line in enumerate(lines) if evaluator_action_zero.search(line)
    ]
    not_reach_lines = [
        index for index, line in enumerate(lines) if not_reach_goal.search(line)
    ]
    finished_not_reach_lines = [
        index for index, line in enumerate(lines) if finished_not_reach_goal.search(line)
    ]
    finish_events = [
        {
            "line_index": index,
            "episode_ordinal": int(match.group("ordinal")) - 1,
            "result": match.group("result").strip().casefold(),
        }
        for index, line in enumerate(lines)
        if (match := finished_episode.search(line)) is not None
    ]
    # Isaac/Kit commonly emits the same evaluator line twice.  Adjacent action
    # zero lines are one semantic evaluator event, not two episodes.
    zero_events: list[int] = []
    for index in zero_lines:
        if not zero_events or index - zero_events[-1] > 2:
            zero_events.append(index)
    adjacency_count = 0
    full_chain_count = 0
    terminal_events: list[dict[str, Any]] = []
    lookbehind_lines = 30
    for event_index, zero_index in enumerate(zero_events):
        segment_end = (
            zero_events[event_index + 1]
            if event_index + 1 < len(zero_events)
            else len(lines)
        )
        finish = next(
            (
                event
                for event in finish_events
                if zero_index <= event["line_index"] < segment_end
            ),
            None,
        )
        followed_by_not_reach = any(
            zero_index <= index < segment_end
            for index in (finished_not_reach_lines or not_reach_lines)
        )
        preceded_by_minus_one = any(
            zero_index - lookbehind_lines <= index <= zero_index
            for index in minus_one_lines
        )
        nearby_ipc = [
            event
            for event in ipc_events
            if zero_index - lookbehind_lines <= event["line_index"] <= zero_index
        ]
        matching_ipc = next(
            (
                event
                for event in reversed(nearby_ipc)
                if finish is None
                or event["episode_ordinal"] is None
                or event["episode_ordinal"] == finish["episode_ordinal"]
            ),
            None,
        )
        episode_ordinal = (
            finish["episode_ordinal"]
            if finish is not None
            else (
                matching_ipc["episode_ordinal"] if matching_ipc is not None else None
            )
        )
        ipc_step_error = matching_ipc is not None
        direct_gate_response_chain = preceded_by_minus_one and not ipc_step_error
        terminal_events.append(
            {
                "event_index": event_index,
                "episode_ordinal": episode_ordinal,
                "finish_result": finish["result"] if finish is not None else None,
                "ipc_step_error": ipc_step_error,
                "ipc_step_error_code": (
                    matching_ipc["error_code"] if matching_ipc is not None else None
                ),
                "direct_gate_response_chain": direct_gate_response_chain,
                "terminal_cause": (
                    "ipc_step_error"
                    if ipc_step_error
                    else (
                        "gate_response_chain"
                        if direct_gate_response_chain
                        else "unattributed_evaluator_action_zero"
                    )
                ),
            }
        )
        adjacency_count += int(followed_by_not_reach)
        full_chain_count += int(followed_by_not_reach and preceded_by_minus_one)
    ipc_code_counts = Counter(event["error_code"] for event in ipc_events)
    return {
        "available": True,
        "sha256": _sha256(path),
        "line_count": len(lines),
        "client_discrete_action_minus_one_line_count": len(minus_one_lines),
        "evaluator_action_zero_raw_line_count": len(zero_lines),
        "evaluator_action_zero_event_count": len(zero_events),
        "finished_not_reach_goal_event_count": len(finished_not_reach_lines),
        "action_zero_then_not_reach_goal_event_count": adjacency_count,
        "discrete_minus_one_then_action_zero_then_not_reach_goal_event_count": full_chain_count,
        "ipc_step_error_marker_count": len(ipc_events),
        "ipc_step_error_code_counts": dict(sorted(ipc_code_counts.items())),
        "episode_terminal_events": terminal_events,
        "lookbehind_lines": lookbehind_lines,
        "terminal_segment_boundary": "next evaluator action-zero event or end-of-log",
        "raw_log_content_embedded": False,
        "inference_basis": (
            "episode-ordinal evaluator action-zero cause, optional log corroboration, "
            "and identity-correlated final client response"
        ),
    }


def _episode_log_terminal_event(
    corroboration: dict[str, Any], episode_ordinal: int, episode_count: int
) -> dict[str, Any] | None:
    """Return an ordinal-bound terminal event, with ordered legacy fallback."""

    events = corroboration.get("episode_terminal_events")
    if not isinstance(events, list):
        return None
    exact = [
        event for event in events if event.get("episode_ordinal") == episode_ordinal
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"evaluator log has duplicate terminal events for ordinal {episode_ordinal}"
        )
    if len(events) == episode_count and 0 <= episode_ordinal < len(events):
        event = events[episode_ordinal]
        if event.get("episode_ordinal") is None:
            return event
    return None


def _episode_safety(
    controller_rows: list[dict[str, Any]], max_linear: float, max_angular: float
) -> dict[str, Any]:
    linear = [
        abs(value)
        for row in controller_rows
        if (value := _optional_number(row.get("desired_linear_x"))) is not None
    ]
    angular = [
        abs(value)
        for row in controller_rows
        if (value := _optional_number(row.get("desired_angular_z"))) is not None
    ]
    stale_motion = sum(
        row.get("motion_enabled") is True and row.get("command_fresh") is False
        for row in controller_rows
    )
    invalid_identity_motion = sum(
        row.get("motion_enabled") is True and row.get("identity_valid") is not True
        for row in controller_rows
    )
    nan_motion = sum(
        row.get("motion_enabled") is True and row.get("nan_detected") is True
        for row in controller_rows
    )
    estop_motion = sum(
        row.get("motion_enabled") is True and row.get("emergency_stop") is True
        for row in controller_rows
    )
    maximum_linear = max(linear, default=None)
    maximum_angular = max(angular, default=None)
    checks = {
        "controller_evidence_present": bool(controller_rows),
        "linear_velocity_within_bound": maximum_linear is not None
        and maximum_linear <= max_linear + 1e-9,
        "angular_velocity_within_bound": maximum_angular is not None
        and maximum_angular <= max_angular + 1e-9,
        "stale_motion_zero": stale_motion == 0,
        "invalid_identity_motion_zero": invalid_identity_motion == 0,
        "nan_motion_zero": nan_motion == 0,
        "estop_motion_zero": estop_motion == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "record_count": len(controller_rows),
        "maximum_abs_desired_linear_mps": maximum_linear,
        "maximum_abs_desired_angular_radps": maximum_angular,
        "limits": {
            "maximum_abs_desired_linear_mps": max_linear,
            "maximum_abs_desired_angular_radps": max_angular,
        },
        "stale_motion_record_count": stale_motion,
        "invalid_identity_motion_record_count": invalid_identity_motion,
        "nan_motion_record_count": nan_motion,
        "estop_motion_record_count": estop_motion,
        "physical_collision_record_count": sum(
            row.get("physical_collision") is True for row in controller_rows
        ),
    }


def _model_fingerprint(root: Path) -> dict[str, Any]:
    candidates = [
        root / "remote" / "dgx" / "model" / "model_weight_audit.json",
        root / "remote" / "dgx" / "model_identity_audit.json",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {
            "model_revision": None,
            "checkpoint_revision": None,
            "inventory_sha256": None,
        }
    value = _object(path)
    return {
        "model_revision": value.get("model_revision"),
        "checkpoint_revision": value.get("checkpoint_revision"),
        "inventory_sha256": value.get("inventory_sha256"),
    }


def _run_fingerprint(
    root: Path,
    evaluator_episodes: list[dict[str, Any]],
    controller_summary: dict[str, Any],
) -> dict[str, Any]:
    keys = [str(episode.get("trajectory_id")) for episode in evaluator_episodes]
    ordered_manifest = root / "remote" / "x86" / "ordered_episode_manifest.json"
    dataset_sha256 = None
    if ordered_manifest.is_file():
        dataset_sha256 = _object(ordered_manifest).get("dataset_sha256")
    camera_contract = {
        key: controller_summary.get(key)
        for key in CAMERA_FIELDS
        if key in controller_summary
    }
    model = _model_fingerprint(root)
    value = {
        "ordered_episode_keys_sha256": _canonical_sha256(keys),
        "dataset_sha256": dataset_sha256,
        "model": model,
        "camera_contract_sha256": _canonical_sha256(camera_contract),
        "camera_contract": camera_contract,
    }
    value["comparable_input_sha256"] = _canonical_sha256(
        {
            "ordered_episode_keys_sha256": value["ordered_episode_keys_sha256"],
            "dataset_sha256": dataset_sha256,
            "model": model,
            "camera_contract_sha256": value["camera_contract_sha256"],
        }
    )
    return value


def analyze_t5_run(
    label: str,
    result_dir: Path,
    *,
    minimum_true_decisions: int,
    max_linear: float,
    max_angular: float,
    evaluator_log: Path | None,
) -> dict[str, Any]:
    root = result_dir.resolve()
    evaluator_root = root / "remote" / "x86" / "evaluator"
    per_episode_path = _unique(evaluator_root, "per_episode.json")
    go2_path = _unique(evaluator_root, "go2_runtime_audit.jsonl")
    client_path = root / "remote" / "dgx" / "client" / "client_records.jsonl"
    controller_path = root / "remote" / "dgx" / "onboard" / "controller_records.jsonl"
    controller_summary_path = (
        root / "remote" / "dgx" / "onboard" / "controller_summary.json"
    )
    for path in (client_path, controller_path, controller_summary_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required evidence is missing: {path}")

    per_episode = _object(per_episode_path)
    raw_episodes = per_episode.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{per_episode_path} has no episodes")
    if any(not isinstance(value, dict) for value in raw_episodes):
        raise ValueError(f"{per_episode_path} episodes must be JSON objects")
    evaluator_episodes: list[dict[str, Any]] = list(raw_episodes)
    client_rows = _rows(client_path)
    controller_rows = _rows(controller_path)
    controller_summary = _object(controller_summary_path)
    physics_hz = _physics_hz(go2_path)
    assert physics_hz is not None
    log_corroboration = _evaluator_log_corroboration(evaluator_log)

    client_identities = sorted(
        {
            (str(row["episode_id"]), int(row["reset_generation"]))
            for row in client_rows
            if isinstance(row.get("episode_id"), str)
            and isinstance(row.get("reset_generation"), int)
        }
    )
    episodes: list[dict[str, Any]] = []
    seen_identities: set[tuple[str, int]] = set()
    for index, raw in enumerate(evaluator_episodes):
        ordinal = int(raw.get("ordinal", index + 1))
        token = _episode_token(raw, ordinal)
        candidates = [
            identity
            for identity in client_identities
            if identity[0].rsplit("::", 1)[-1] == token
        ]
        hinted = [identity for identity in candidates if identity[1] == ordinal - 1]
        identity_candidates = hinted or candidates
        if len(identity_candidates) != 1:
            raise ValueError(
                f"episode {raw.get('trajectory_id')} maps to {identity_candidates}, "
                "expected one client identity"
            )
        episode_id, reset_generation = identity_candidates[0]
        if (episode_id, reset_generation) in seen_identities:
            raise ValueError(f"duplicate client identity {episode_id}/{reset_generation}")
        seen_identities.add((episode_id, reset_generation))
        selected_client = [
            row
            for row in client_rows
            if row.get("episode_id") == episode_id
            and row.get("reset_generation") == reset_generation
        ]
        selected_controller = [
            row
            for row in controller_rows
            if row.get("episode_id") == episode_id
            and row.get("reset_generation") == reset_generation
            and row.get("state_only") is False
        ]
        if not selected_client:
            raise ValueError(f"episode {episode_id}/{reset_generation} has no client rows")

        gate_rows = [
            row
            for row in selected_client
            if row.get("motion_observation_gate_only") is True
        ]
        decisions = [
            row
            for row in selected_client
            if row.get("motion_observation_gate_only") is not True
        ]
        gate_kinds = Counter(_gate_kind(row) or "unknown" for row in gate_rows)
        model_stop_count = sum(row.get("model_stop") is True for row in decisions)
        natural_model_stop = model_stop_count > 0
        last = selected_client[-1]
        last_kind = _gate_kind(last)
        client_terminal_stop_without_model_stop = (
            last.get("stop") is True and last.get("model_stop") is not True
        )
        log_terminal_event = _episode_log_terminal_event(
            log_corroboration, ordinal - 1, len(evaluator_episodes)
        )
        evaluator_action_zero_without_model_stop = (
            log_terminal_event is not None and not natural_model_stop
        )
        ipc_step_error_termination = bool(
            log_terminal_event is not None
            and log_terminal_event.get("ipc_step_error") is True
        )
        direct_gate_response_chain = bool(
            log_terminal_event is not None
            and log_terminal_event.get("direct_gate_response_chain") is True
        )
        terminal_stop_without_model_stop = (
            client_terminal_stop_without_model_stop
            or evaluator_action_zero_without_model_stop
        )
        safe_stop_terminal_conflation = (
            terminal_stop_without_model_stop
            and last.get("motion_observation_gate_only") is True
            and last_kind in SAFE_STOP_GATE_KINDS
            and not ipc_step_error_termination
            and direct_gate_response_chain
        )
        terminal_cause = (
            "ipc_step_error"
            if ipc_step_error_termination
            else (
                "safe_stop_terminal_conflation"
                if safe_stop_terminal_conflation
                else (
                    "natural_model_stop"
                    if natural_model_stop
                    else (
                        "unattributed_evaluator_stop"
                        if evaluator_action_zero_without_model_stop
                        else (
                            "client_stop_without_model_stop"
                            if client_terminal_stop_without_model_stop
                            else "no_terminal_stop"
                        )
                    )
                )
            )
        )
        low_decision_termination = (
            len(decisions) < minimum_true_decisions and not natural_model_stop
        )
        safety = _episode_safety(selected_controller, max_linear, max_angular)
        failures: list[str] = []
        if ipc_step_error_termination:
            failures.append("ipc_step_error_termination")
        elif safe_stop_terminal_conflation:
            failures.append("safe_stop_terminal_conflation")
        if terminal_stop_without_model_stop:
            failures.append("terminal_stop_without_model_stop")
        if low_decision_termination:
            failures.append("low_true_model_decision_termination")
        if safety["status"] != "PASS":
            failures.append("safety_or_velocity_boundary_failure")

        step_count = int(raw.get("step_count", -1))
        wall_duration = _finite(raw.get("duration_sec"), "duration_sec")
        if step_count < 0 or wall_duration <= 0.0:
            raise ValueError(f"episode {episode_id} has invalid step_count/duration")
        sim_duration = step_count / physics_hz
        metrics = _metrics(raw)
        if metrics["sr"] is None and isinstance(raw.get("success"), bool):
            metrics["sr"] = float(bool(raw["success"]))
        episodes.append(
            {
                "ordinal": ordinal,
                "trajectory_id": raw.get("trajectory_id"),
                "client_identity": {
                    "episode_id": episode_id,
                    "reset_generation": reset_generation,
                },
                "evaluator_step_count": step_count,
                "true_model_decisions": len(decisions),
                "gate_holds": {
                    "total": len(gate_rows),
                    "by_kind": dict(sorted(gate_kinds.items())),
                },
                "action_sources": _action_source_counts(decisions),
                "model_stop_count": model_stop_count,
                "natural_model_stop": natural_model_stop,
                "termination": {
                    "evaluator_reason": raw.get("termination_reason"),
                    "evaluator_success": raw.get("success"),
                    "last_response": _last_response_summary(last),
                    "terminal_cause": terminal_cause,
                    "ipc_step_error_termination": ipc_step_error_termination,
                    "ipc_step_error_code": (
                        log_terminal_event.get("ipc_step_error_code")
                        if ipc_step_error_termination and log_terminal_event is not None
                        else None
                    ),
                    "evaluator_action_zero_correlated": log_terminal_event is not None,
                    "direct_gate_response_chain": direct_gate_response_chain,
                    "client_terminal_stop_without_model_stop": (
                        client_terminal_stop_without_model_stop
                    ),
                    "terminal_stop_without_model_stop": terminal_stop_without_model_stop,
                    "safe_stop_terminal_conflation": safe_stop_terminal_conflation,
                    "false_termination": terminal_stop_without_model_stop,
                    "low_true_model_decision_termination": low_decision_termination,
                    "inference_basis": log_corroboration["inference_basis"],
                },
                "timing": {
                    "wall_duration_sec": wall_duration,
                    "sim_duration_sec": sim_duration,
                    "sim_duration_estimated": True,
                    "sim_duration_method": "evaluator_step_count / physics_hz",
                    "physics_hz": physics_hz,
                    "rtf": sim_duration / wall_duration,
                },
                "metrics": metrics,
                "safety": safety,
                "evaluation_valid": not failures,
                "failures": failures,
            }
        )

    aggregate_decisions = sum(row["true_model_decisions"] for row in episodes)
    aggregate_steps = sum(row["evaluator_step_count"] for row in episodes)
    aggregate_wall = math.fsum(row["timing"]["wall_duration_sec"] for row in episodes)
    aggregate_sim = math.fsum(row["timing"]["sim_duration_sec"] for row in episodes)
    aggregate_gate_kinds: Counter[str] = Counter()
    for episode in episodes:
        aggregate_gate_kinds.update(episode["gate_holds"]["by_kind"])
    summary_checks = {
        "stale_motion_execution_zero": int(
            controller_summary.get("stale_motion_execution_count", -1)
        )
        == 0,
        "direct_motion_bypass_zero": int(
            controller_summary.get("direct_motion_bypass_count", -1)
        )
        == 0,
        "cmd_vel_quantization_zero": int(
            controller_summary.get("cmd_vel_quantization_count", -1)
        )
        == 0,
        "nan_zero": int(controller_summary.get("nan_count", -1)) == 0,
    }
    all_episode_safety = all(
        episode["safety"]["status"] == "PASS" for episode in episodes
    )
    run_valid = all(episode["evaluation_valid"] for episode in episodes) and all(
        summary_checks.values()
    )
    sources = [
        per_episode_path,
        go2_path,
        client_path,
        controller_path,
        controller_summary_path,
    ]
    return {
        "label": label,
        "run_id": root.name,
        "evidence_level": "raw_episode_correlated",
        "status": "PASS" if run_valid else "EVALUATION_INVALID",
        "fingerprint": _run_fingerprint(root, evaluator_episodes, controller_summary),
        "evaluator_log_corroboration": log_corroboration,
        "episodes": episodes,
        "aggregate": {
            "episode_count": len(episodes),
            "evaluator_step_count": aggregate_steps,
            "true_model_decisions": aggregate_decisions,
            "evaluator_steps_per_true_model_decision": (
                aggregate_steps / aggregate_decisions if aggregate_decisions else None
            ),
            "gate_holds": {
                "total": sum(row["gate_holds"]["total"] for row in episodes),
                "by_kind": dict(sorted(aggregate_gate_kinds.items())),
            },
            "action_sources": {
                name: sum(row["action_sources"][name] for row in episodes)
                for name in ACTION_SOURCES.values()
            },
            "model_stop_count": sum(row["model_stop_count"] for row in episodes),
            "terminal_stop_without_model_stop_count": sum(
                row["termination"]["terminal_stop_without_model_stop"]
                for row in episodes
            ),
            "safe_stop_terminal_conflation_count": sum(
                row["termination"]["safe_stop_terminal_conflation"]
                for row in episodes
            ),
            "ipc_step_error_termination_count": sum(
                row["termination"]["ipc_step_error_termination"]
                for row in episodes
            ),
            "terminal_causes": dict(
                sorted(
                    Counter(
                        row["termination"]["terminal_cause"] for row in episodes
                    ).items()
                )
            ),
            "low_true_model_decision_termination_count": sum(
                row["termination"]["low_true_model_decision_termination"]
                for row in episodes
            ),
            "false_termination_count": sum(
                row["termination"]["false_termination"] for row in episodes
            ),
            "valid_episode_count": sum(row["evaluation_valid"] for row in episodes),
            "timing": {
                "wall_duration_sec": aggregate_wall,
                "sim_duration_sec": aggregate_sim,
                "sim_duration_estimated": True,
                "rtf": aggregate_sim / aggregate_wall,
            },
            "metrics": _aggregate_metrics(episodes),
            "safety": {
                "status": (
                    "PASS"
                    if all_episode_safety and all(summary_checks.values())
                    else "FAIL"
                ),
                "episode_safety_all_pass": all_episode_safety,
                "controller_summary_checks": summary_checks,
                "physical_collision_count_warn_only": controller_summary.get(
                    "physical_collision_count"
                ),
            },
        },
        "sources": {
            path.relative_to(root).as_posix(): {"sha256": _sha256(path)}
            for path in sources
        },
    }


def _legacy_episode(
    raw: dict[str, Any],
    *,
    ordinal: int,
    decisions: list[dict[str, Any]] | None,
    physics_hz: float | None,
    model_stop_field: str,
) -> dict[str, Any]:
    step_count = int(raw.get("step_count", raw.get("steps", -1)))
    wall_duration = _optional_number(
        raw.get("duration_sec", raw.get("wall_time_seconds"))
    )
    metrics = _metrics(raw)
    if metrics["sr"] is None and isinstance(raw.get("success"), bool):
        metrics["sr"] = float(raw["success"])
    if decisions is None:
        decision_count = None
        action_sources = {name: None for name in ACTION_SOURCES.values()}
        model_stop_count = None
        natural_stop = None
    else:
        decision_count = len(decisions)
        action_sources = _action_source_counts(decisions)
        if model_stop_field == "model_action":
            model_stop_count = sum(row.get("model_action") == 0 for row in decisions)
        else:
            model_stop_count = sum(row.get("model_stop") is True for row in decisions)
        natural_stop = model_stop_count > 0
    sim_duration = (
        step_count / physics_hz
        if physics_hz is not None and step_count >= 0
        else None
    )
    return {
        "ordinal": ordinal,
        "trajectory_id": raw.get("trajectory_id", raw.get("episode_key")),
        "evaluator_step_count": step_count if step_count >= 0 else None,
        "true_model_decisions": decision_count,
        "gate_holds": {"total": 0, "by_kind": {}},
        "action_sources": action_sources,
        "model_stop_count": model_stop_count,
        "natural_model_stop": natural_stop,
        "termination": {
            "evaluator_reason": raw.get("termination_reason"),
            "evaluator_success": raw.get("success", metrics["sr"] == 1.0),
            "terminal_stop_without_model_stop": None,
            "safe_stop_terminal_conflation": False,
            "false_termination": None,
            "low_true_model_decision_termination": None,
            "assessment": "historical descriptive evidence; strict T5 gate not applied",
        },
        "timing": {
            "wall_duration_sec": wall_duration,
            "sim_duration_sec": sim_duration,
            "sim_duration_estimated": sim_duration is not None,
            "sim_duration_method": (
                "evaluator_step_count / physics_hz" if sim_duration is not None else None
            ),
            "physics_hz": physics_hz,
            "rtf": (
                sim_duration / wall_duration
                if sim_duration is not None and wall_duration not in (None, 0.0)
                else None
            ),
        },
        "metrics": metrics,
        "evaluation_valid": None,
    }


def analyze_t0_status(path: Path) -> dict[str, Any]:
    value = _object(path)
    raw_episodes = value.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{path} has no historical episode rows")
    episodes = [
        _legacy_episode(
            raw,
            ordinal=index + 1,
            decisions=None,
            physics_hz=None,
            model_stop_field="model_stop",
        )
        for index, raw in enumerate(raw_episodes)
        if isinstance(raw, dict)
    ]
    metrics = _normalized_result_metrics(value.get("metrics", {}))
    true_decisions = int(value.get("protocol_requests_parsed", 0))
    return {
        "label": "t0_official_pilot",
        "evidence_level": "historical_episode_summary_with_aggregate_protocol_count",
        "strict_t5_gate_applied": False,
        "episodes": episodes,
        "aggregate": {
            "episode_count": len(episodes),
            "evaluator_step_count": sum(
                row["evaluator_step_count"] or 0 for row in episodes
            ),
            "true_model_decisions": true_decisions,
            "per_episode_true_model_decisions_available": False,
            "gate_holds": {"total": 0, "by_kind": {}},
            "model_stop_count": value.get("action_distribution", {}).get("stop_0"),
            "metrics": metrics,
        },
        "limitations": [
            "T0 preserves aggregate parsed model responses but not their episode boundaries.",
            "T0 has no T5 motion-observation gate, so gate_holds is structurally zero.",
        ],
        "sources": {path.name: {"sha256": _sha256(path)}},
    }


def analyze_legacy_run(label: str, root: Path, kind: str) -> dict[str, Any]:
    per_episode_path = root / "per_episode.json"
    records_name = "active_records.jsonl" if kind == "t2" else "client_records.jsonl"
    records_path = root / records_name
    result_path = root / "result.json"
    for path in (per_episode_path, records_path, result_path):
        if not path.is_file():
            raise ValueError(f"historical evidence is missing: {path}")
    per_episode = _object(per_episode_path)
    raw_episodes = per_episode.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{per_episode_path} has no episodes")
    records = _rows(records_path)
    by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        generation = row.get("reset_generation")
        if isinstance(generation, int):
            by_generation[generation].append(row)
    physics_path = root / "go2_runtime_audit.jsonl"
    physics_hz = _physics_hz(physics_path if physics_path.is_file() else None)
    episodes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_episodes):
        if not isinstance(raw, dict):
            raise ValueError(f"{per_episode_path} episode {index} is not an object")
        ordinal = int(raw.get("ordinal", index + 1))
        episodes.append(
            _legacy_episode(
                raw,
                ordinal=ordinal,
                decisions=by_generation.get(ordinal - 1, []),
                physics_hz=physics_hz,
                model_stop_field="model_action" if kind == "t2" else "model_stop",
            )
        )
    result = _object(result_path)
    sources = [per_episode_path, records_path, result_path]
    if physics_path.is_file():
        sources.append(physics_path)
    return {
        "label": label,
        "evidence_level": "historical_raw_episode_descriptive",
        "strict_t5_gate_applied": False,
        "episodes": episodes,
        "aggregate": {
            "episode_count": len(episodes),
            "evaluator_step_count": sum(
                row["evaluator_step_count"] or 0 for row in episodes
            ),
            "true_model_decisions": sum(
                row["true_model_decisions"] or 0 for row in episodes
            ),
            "per_episode_true_model_decisions_available": True,
            "gate_holds": {"total": 0, "by_kind": {}},
            "action_sources": {
                name: sum((row["action_sources"].get(name) or 0) for row in episodes)
                for name in ACTION_SOURCES.values()
            },
            "model_stop_count": sum(row["model_stop_count"] or 0 for row in episodes),
            "metrics": _normalized_result_metrics(result),
        },
        "limitations": [
            "Historical termination is descriptive and is not retroactively judged by the new T5 gate.",
            "Historical evidence has no T5 safe-stop terminal conflation field.",
        ],
        "sources": {path.name: {"sha256": _sha256(path)} for path in sources},
    }


def _historical_differential(
    run: dict[str, Any], historical: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    current = run["aggregate"]
    current_metrics = current["metrics"]
    values: list[dict[str, Any]] = []
    for reference in historical:
        aggregate = reference["aggregate"]
        reference_metrics = aggregate["metrics"]
        current_decisions = current.get("true_model_decisions")
        reference_decisions = aggregate.get("true_model_decisions")
        values.append(
            {
                "reference": reference["label"],
                "paired": False,
                "note": "descriptive only; manifests and runtime semantics differ",
                "sr_delta": (
                    current_metrics["sr"] - reference_metrics["sr"]
                    if current_metrics["sr"] is not None
                    and reference_metrics["sr"] is not None
                    else None
                ),
                "ne_delta_m": (
                    current_metrics["ne_m"] - reference_metrics["ne_m"]
                    if current_metrics["ne_m"] is not None
                    and reference_metrics["ne_m"] is not None
                    else None
                ),
                "true_model_decision_ratio": (
                    current_decisions / reference_decisions
                    if isinstance(current_decisions, int)
                    and isinstance(reference_decisions, int)
                    and reference_decisions > 0
                    else None
                ),
                "current_true_model_decisions": current_decisions,
                "reference_true_model_decisions": reference_decisions,
            }
        )
    return values


def _paired_gate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) == 1:
        return {
            "status": "NOT_APPLICABLE_SINGLE_RUN",
            "same_manifest_model_camera": None,
            "all_runs_evaluation_valid": runs[0]["status"] == "PASS",
        }
    fingerprints = {
        run["fingerprint"]["comparable_input_sha256"] for run in runs
    }
    same_inputs = len(fingerprints) == 1
    all_valid = all(run["status"] == "PASS" for run in runs)
    return {
        "status": "PASS" if same_inputs and all_valid else "FAIL",
        "same_manifest_model_camera": same_inputs,
        "all_runs_evaluation_valid": all_valid,
        "fingerprints": {
            run["label"]: run["fingerprint"]["comparable_input_sha256"]
            for run in runs
        },
    }


def analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    evaluator_logs: dict[str, Path] = {}
    for specification in arguments.evaluator_log:
        if "=" not in specification:
            raise ValueError("--evaluator-log must use LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        if not label or not raw_path or label in evaluator_logs:
            raise ValueError("--evaluator-log must use unique, non-empty LABEL=PATH")
        evaluator_logs[label] = Path(raw_path)
    runs: list[dict[str, Any]] = []
    for specification in arguments.run:
        if "=" not in specification:
            raise ValueError("--run must use LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        if not label or not raw_path:
            raise ValueError("--run must use non-empty LABEL=PATH")
        runs.append(
            analyze_t5_run(
                label,
                Path(raw_path),
                minimum_true_decisions=arguments.minimum_true_decisions,
                max_linear=arguments.max_linear_velocity,
                max_angular=arguments.max_angular_velocity,
                evaluator_log=evaluator_logs.get(label),
            )
        )
    unknown_log_labels = sorted(set(evaluator_logs) - {run["label"] for run in runs})
    if unknown_log_labels:
        raise ValueError(f"--evaluator-log labels have no matching --run: {unknown_log_labels}")
    historical: list[dict[str, Any]] = []
    if arguments.t0_status:
        historical.append(analyze_t0_status(arguments.t0_status))
    if arguments.t2_run:
        historical.append(analyze_legacy_run("t2_go2_nav2_pilot", arguments.t2_run, "t2"))
    if arguments.t3_run:
        historical.append(
            analyze_legacy_run(
                "t3_obstacle_aware_continuous_pilot", arguments.t3_run, "t3"
            )
        )
    paired = _paired_gate(runs)
    run_valid = all(run["status"] == "PASS" for run in runs)
    paired_valid = paired["status"] in {"PASS", "NOT_APPLICABLE_SINGLE_RUN"}
    status = "PASS" if run_valid and paired_valid else "EVALUATION_INVALID"
    return {
        "schema_version": 1,
        "analysis_kind": "t5_evaluation_starvation_differential",
        "status": status,
        "policy": {
            "minimum_true_model_decisions_or_natural_stop": arguments.minimum_true_decisions,
            "motion_gate_only_excluded_from_true_model_decisions": True,
            "terminal_stop_without_model_stop_maximum": 0,
            "safe_stop_terminal_conflation_maximum": 0,
            "ipc_step_error_termination_maximum": 0,
            "false_termination_maximum": 0,
            "paired_same_manifest_model_camera_required": True,
            "maximum_abs_desired_linear_mps": arguments.max_linear_velocity,
            "maximum_abs_desired_angular_radps": arguments.max_angular_velocity,
            "safety_invariants": [
                "no stale motion",
                "no invalid-identity motion",
                "no motion while emergency-stop is asserted",
                "no NaN motion",
                "bounded desired linear and angular velocity",
            ],
            "physical_collision_policy": "completion_sim metric/WARN; not a gate relaxation for stale or unbounded motion",
        },
        "runs": runs,
        "paired_gate": paired,
        "historical_references": historical,
        "historical_differential": (
            _historical_differential(runs[0], historical) if runs else []
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="T5 run as LABEL=RESULT_DIR; repeat for paired A/B analysis",
    )
    parser.add_argument("--t0-status", type=Path)
    parser.add_argument("--t2-run", type=Path)
    parser.add_argument("--t3-run", type=Path)
    parser.add_argument(
        "--evaluator-log",
        action="append",
        default=[],
        help="optional evaluator log as LABEL=PATH; content is counted but never embedded",
    )
    parser.add_argument("--minimum-true-decisions", type=int, default=50)
    parser.add_argument("--max-linear-velocity", type=float, default=0.25)
    parser.add_argument("--max-angular-velocity", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.minimum_true_decisions <= 0:
            raise ValueError("--minimum-true-decisions must be positive")
        if arguments.max_linear_velocity <= 0 or arguments.max_angular_velocity <= 0:
            raise ValueError("velocity bounds must be positive")
        payload = analyze(arguments)
        exit_code = 0 if payload["status"] == "PASS" else 2
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": 1,
            "analysis_kind": "t5_evaluation_starvation_differential",
            "status": "ERROR",
            "error": str(exc),
        }
        exit_code = 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
