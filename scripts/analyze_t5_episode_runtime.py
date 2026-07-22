#!/usr/bin/env python3
"""Summarize per-episode T5 timing, yaw, command-age, and motion-gate evidence.

This is a diagnostic-only, offline analyzer.  Evaluator ``duration_sec`` is
treated as wall duration.  Sim duration is deliberately labelled an estimate
and is computed only as ``step_count / physics_hz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ACTION_NAMES = {1: "forward", 2: "left", 3: "right"}
TERMINAL_GATE_KINDS = {
    "safe_stop_complete": "completed",
    "safe_stop_timeout": "timeout",
    "safe_stop_stale": "stale",
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
        values.append(value)
    if not values:
        raise ValueError(f"{path} is empty")
    return values


def _unique(root: Path, name: str) -> Path:
    matches = sorted(
        path for path in root.rglob(name) if path.is_file() and not path.is_symlink()
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0] if ordered else None,
        "mean": statistics.fmean(ordered) if ordered else None,
        "p50": _percentile(ordered, 0.50) if ordered else None,
        "p95": _percentile(ordered, 0.95) if ordered else None,
        "maximum": ordered[-1] if ordered else None,
        "percentile_method": "linear_interpolation",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_token(row: dict[str, Any], ordinal: int) -> str:
    for name in ("episode_id", "trajectory_id", "episode_key"):
        value = row.get(name)
        if isinstance(value, (str, int)) and str(value):
            return str(value).rsplit("::", 1)[-1].rsplit("_", 1)[-1]
    raise ValueError(f"episode ordinal {ordinal} has no explicit identity")


def _physics_hz(rows: list[dict[str, Any]]) -> float:
    observed = [
        _finite(row["physics_hz"], "go2 physics_hz")
        for row in rows
        if "physics_hz" in row
    ]
    if not observed or any(value <= 0.0 for value in observed):
        raise ValueError("go2 runtime audit has no positive physics_hz")
    reference = observed[0]
    if any(not math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-9) for value in observed):
        raise ValueError("go2 runtime audit contains inconsistent physics_hz values")
    return reference


def _controller_identity(
    rows: list[dict[str, Any]], token: str, reset_hint: int
) -> tuple[str, int]:
    identities = sorted(
        {
            (str(row.get("episode_id")), int(row.get("reset_generation")))
            for row in rows
            if row.get("identity_valid") is True
            and row.get("state_only") is False
            and isinstance(row.get("episode_id"), str)
            and isinstance(row.get("reset_generation"), int)
            and str(row["episode_id"]).rsplit("::", 1)[-1] == token
        }
    )
    hinted = [identity for identity in identities if identity[1] == reset_hint]
    if len(hinted) == 1:
        return hinted[0]
    if len(identities) == 1:
        return identities[0]
    raise ValueError(
        f"episode token {token} does not map to one controller identity; "
        f"observed={identities}"
    )


def _controller_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    commanded: list[float] = []
    measured: list[float] = []
    ages: list[float] = []
    active_commanded: list[float] = []
    active_measured: list[float] = []
    active_ages: list[float] = []
    wall_times: list[float] = []
    for row in rows:
        if "desired_angular_z" in row:
            commanded_value = _finite(row["desired_angular_z"], "desired_angular_z")
            commanded.append(commanded_value)
        else:
            commanded_value = None
        actual = row.get("actual_angular_velocity")
        if isinstance(actual, list) and len(actual) >= 3:
            measured_value = _finite(actual[2], "actual_angular_velocity[2]")
            measured.append(measured_value)
        else:
            measured_value = None
        if "command_age_sec" in row:
            age = _finite(row["command_age_sec"], "command_age_sec")
            if age < 0.0:
                raise ValueError("command_age_sec must be nonnegative")
            ages.append(age)
        else:
            age = None
        if "wall_time_unix" in row:
            wall_times.append(_finite(row["wall_time_unix"], "wall_time_unix"))
        if row.get("motion_enabled") is True:
            if commanded_value is not None:
                active_commanded.append(commanded_value)
            if measured_value is not None:
                active_measured.append(measured_value)
            if age is not None:
                active_ages.append(age)
    return {
        "record_count": len(rows),
        "observed_wall_span_sec": (
            max(wall_times) - min(wall_times) if len(wall_times) >= 2 else None
        ),
        "all_updates": {
            "commanded_yaw_rate_rad_s": _stats(commanded),
            "measured_yaw_rate_rad_s": _stats(measured),
            "command_age_sec": _stats(ages),
        },
        "motion_enabled_updates": {
            "record_count": sum(row.get("motion_enabled") is True for row in rows),
            "commanded_yaw_rate_rad_s": _stats(active_commanded),
            "measured_yaw_rate_rad_s": _stats(active_measured),
            "command_age_sec": _stats(active_ages),
        },
    }


def _gate_action_rows(
    rows: list[dict[str, Any]], episode_id: str, reset_generation: int
) -> list[dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            row.get("episode_id") != episode_id
            or row.get("reset_generation") != reset_generation
        ):
            continue
        gate = row.get("gate")
        decision = row.get("decision")
        if not isinstance(gate, dict) or not isinstance(decision, dict):
            continue
        source = gate.get("pending")
        if not isinstance(source, dict):
            source = gate.get("stop_barrier")
        if not isinstance(source, dict):
            continue
        request_id = source.get("request_id")
        action = source.get("action")
        if not isinstance(request_id, str) or action not in ACTION_NAMES:
            continue
        key = request_id
        item = requests.setdefault(
            key,
            {
                "request_id": request_id,
                "action": int(action),
                "action_name": ACTION_NAMES[int(action)],
                "outcome": "unterminated",
                "decision_kind": None,
                "elapsed_sim_sec": None,
                "progress": None,
                "required_progress": None,
                "commanded_progress": None,
            },
        )
        if item["action"] != action:
            raise ValueError(f"motion gate request {request_id} changed action")
        kind = decision.get("kind")
        if kind in TERMINAL_GATE_KINDS:
            outcome = TERMINAL_GATE_KINDS[str(kind)]
            if item["outcome"] not in {"unterminated", outcome}:
                raise ValueError(f"motion gate request {request_id} has conflicting outcomes")
            item.update(
                {
                    "outcome": outcome,
                    "decision_kind": kind,
                    "elapsed_sim_sec": _finite(
                        decision.get("elapsed_sim_sec"), "gate elapsed_sim_sec"
                    ),
                    "progress": _finite(decision.get("progress"), "gate progress"),
                    "required_progress": _finite(
                        decision.get("required_progress"), "gate required_progress"
                    ),
                    "commanded_progress": _finite(
                        decision.get("commanded_progress"), "gate commanded_progress"
                    ),
                }
            )
    return sorted(requests.values(), key=lambda item: item["request_id"])


def _gate_summary(actions: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(selected: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "attempted": len(selected),
            "completed": sum(item["outcome"] == "completed" for item in selected),
            "timeout": sum(item["outcome"] == "timeout" for item in selected),
            "stale": sum(item["outcome"] == "stale" for item in selected),
            "unterminated": sum(item["outcome"] == "unterminated" for item in selected),
        }

    return {
        **counts(actions),
        "by_action": {
            name: counts([item for item in actions if item["action_name"] == name])
            for name in ACTION_NAMES.values()
        },
        "completion_elapsed_sim_sec": _stats(
            [
                float(item["elapsed_sim_sec"])
                for item in actions
                if item["outcome"] == "completed"
                and item["elapsed_sim_sec"] is not None
            ]
        ),
        "actions": actions,
    }


def analyze(result_dir: Path) -> dict[str, Any]:
    root = result_dir.resolve()
    evaluator_root = root / "remote" / "x86" / "evaluator"
    per_episode_path = _unique(evaluator_root, "per_episode.json")
    go2_audit_path = _unique(evaluator_root, "go2_runtime_audit.jsonl")
    controller_path = root / "remote" / "dgx" / "onboard" / "controller_records.jsonl"
    gate_candidates = [
        root / "remote" / "dgx" / "onboard" / "motion_observation_gate_records.jsonl",
        root / "remote" / "dgx" / "client" / "motion_observation_gate_records.jsonl",
    ]
    gate_paths = [path for path in gate_candidates if path.is_file() and not path.is_symlink()]
    if len(gate_paths) != 1:
        raise ValueError(
            "expected exactly one DGX motion_observation_gate_records.jsonl, "
            f"found {len(gate_paths)}"
        )
    gate_path = gate_paths[0]

    per_episode = _object(per_episode_path)
    go2_rows = _rows(go2_audit_path)
    controller_rows = _rows(controller_path)
    gate_rows = _rows(gate_path)
    physics_hz = _physics_hz(go2_rows)
    raw_episodes = per_episode.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("per_episode.json has no episode records")

    episodes: list[dict[str, Any]] = []
    all_controller_rows: list[dict[str, Any]] = []
    all_gate_actions: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for index, raw in enumerate(raw_episodes):
        if not isinstance(raw, dict):
            raise ValueError(f"episode index {index} is not a JSON object")
        ordinal = int(raw.get("ordinal", index + 1))
        token = _episode_token(raw, ordinal)
        if token in seen_tokens:
            raise ValueError(f"duplicate episode token {token}")
        seen_tokens.add(token)
        episode_id, reset_generation = _controller_identity(
            controller_rows, token, ordinal - 1
        )
        selected_controller = [
            row
            for row in controller_rows
            if row.get("episode_id") == episode_id
            and row.get("reset_generation") == reset_generation
            and row.get("identity_valid") is True
            and row.get("state_only") is False
        ]
        wall_duration = _finite(raw.get("duration_sec"), "episode duration_sec")
        step_count = int(raw.get("step_count", -1))
        if wall_duration <= 0.0 or step_count < 0:
            raise ValueError(f"episode {token} has invalid duration or step_count")
        sim_duration = step_count / physics_hz
        gate_actions = _gate_action_rows(
            gate_rows, episode_id=episode_id, reset_generation=reset_generation
        )
        all_controller_rows.extend(selected_controller)
        all_gate_actions.extend(gate_actions)
        episodes.append(
            {
                "ordinal": ordinal,
                "trajectory_id": raw.get("trajectory_id"),
                "episode_token": token,
                "controller_identity": {
                    "episode_id": episode_id,
                    "reset_generation": reset_generation,
                },
                "wall_duration_sec": wall_duration,
                "wall_duration_source": "per_episode.json:duration_sec",
                "step_count": step_count,
                "physics_hz": physics_hz,
                "sim_duration_sec": sim_duration,
                "sim_duration_estimated": True,
                "sim_duration_method": "step_count / physics_hz",
                "rtf": sim_duration / wall_duration,
                "rtf_uses_estimated_sim_duration": True,
                "controller": _controller_values(selected_controller),
                "motion_gate": _gate_summary(gate_actions),
            }
        )

    total_wall = math.fsum(item["wall_duration_sec"] for item in episodes)
    total_sim = math.fsum(item["sim_duration_sec"] for item in episodes)
    aggregate_gate = _gate_summary(all_gate_actions)
    checks = {
        "episode_count_matches_summary": len(episodes)
        == int(per_episode.get("completed_episode_count", -1)),
        "every_episode_has_controller_samples": all(
            item["controller"]["record_count"] > 0 for item in episodes
        ),
        "every_episode_has_command_age_samples": all(
            item["controller"]["all_updates"]["command_age_sec"]["count"] > 0
            for item in episodes
        ),
        "physics_hz_consistent": True,
        "sim_duration_explicitly_estimated": True,
    }
    sources = [per_episode_path, go2_audit_path, controller_path, gate_path]
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "INCOMPLETE",
        "diagnostic_only": True,
        "run_id": root.name,
        "checks": checks,
        "methodology": {
            "wall_duration": "per_episode.json duration_sec",
            "sim_duration": "estimated as step_count / physics_hz",
            "rtf": "estimated sim duration / wall duration",
            "controller_filter": (
                "exact episode_id/reset_generation, identity_valid=true, "
                "state_only=false"
            ),
            "measured_yaw_rate": "actual_angular_velocity[2]",
            "commanded_yaw_rate": "desired_angular_z",
            "gate_deduplication": "unique episode/reset/request_id",
        },
        "physics_hz": physics_hz,
        "episodes": episodes,
        "aggregate": {
            "episode_count": len(episodes),
            "wall_duration_sec": total_wall,
            "sim_duration_sec": total_sim,
            "sim_duration_estimated": True,
            "rtf": total_sim / total_wall,
            "rtf_uses_estimated_sim_duration": True,
            "controller": _controller_values(all_controller_rows),
            "motion_gate": aggregate_gate,
        },
        "sources": {
            path.relative_to(root).as_posix(): {"sha256": _sha256(path)}
            for path in sources
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", "--result-root", dest="result_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or arguments.result_dir / "analysis" / "episode_runtime_metrics.json"
    try:
        payload = analyze(arguments.result_dir)
        exit_code = 0 if payload["status"] == "PASS" else 2
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        payload = {"schema_version": 1, "status": "FAIL", "error": str(exc)}
        exit_code = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(payload, sort_keys=True, allow_nan=False),
        file=sys.stdout if exit_code == 0 else sys.stderr,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
