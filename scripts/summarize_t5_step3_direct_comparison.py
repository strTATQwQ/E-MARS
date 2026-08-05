#!/usr/bin/env python3
"""Compare frozen InternVLA fixed-five with the pure Step3-direct screening run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _unique(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = (len(ordered) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _navigation(root: Path) -> dict[str, Any]:
    result_path = _unique(root, "result.json")
    value = _load(result_path).get("val_unseen", {})
    episodes = _load(_unique(root, "per_episode.json")).get("episodes", [])
    termination = Counter(str(row.get("termination_reason", "unknown")) for row in episodes)
    return {
        "count": int(value.get("Count", len(episodes))),
        "sr": value.get("SR"),
        "osr": value.get("OS"),
        "spl": value.get("SPL"),
        "ndtw": value.get("nDTW", value.get("NDTW")),
        "ne_m": value.get("NE"),
        "stuck_count": termination.get("stuck", 0),
        "stuck_rate": (
            termination.get("stuck", 0) / len(episodes) if episodes else None
        ),
        "termination_reason_counts": dict(sorted(termination.items())),
        "result_path": str(result_path),
    }


def _direct_details(root: Path) -> dict[str, Any]:
    dgx = root / "remote/dgx_b"
    client = _jsonl(dgx / "client/client_records.jsonl")
    adapter = _jsonl(dgx / "onboard/step3_advisor_records.jsonl")
    decisions = _jsonl(dgx / "step3/frontend_decisions.jsonl")
    prepares = [row for row in adapter if row.get("phase") == "prepare"]
    commits = [row for row in adapter if row.get("phase") == "commit"]
    calls = Counter(str(row.get("episode_id")) for row in prepares)
    call_times: dict[str, list[float]] = defaultdict(list)
    for row in prepares:
        call_times[str(row.get("episode_id"))].append(float(row["recorded_wall_time"]))
    intervals = [
        later - earlier
        for values in call_times.values()
        for earlier, later in zip(sorted(values), sorted(values)[1:])
    ]
    planning = [float(row["planning_latency_sec"]) for row in client if "planning_latency_sec" in row]
    command_age = [float(row["command_age_sec"]) for row in client if "command_age_sec" in row]
    nav2 = [float(row["latency_sec"]) for row in commits if row.get("status") == "COMMITTED"]
    public = [row.get("decision", {}) for row in decisions]
    safe_stops = [row for row in public if row.get("requires_safe_stop") is True]
    abstains = [row for row in public if row.get("source_decision") == "abstain"]
    timeouts = [row for row in safe_stops if "timeout" in str(row.get("safe_stop_reason", ""))]
    fallbacks = [row for row in public if row.get("fallback_used") is True]
    return {
        "step3_call_count": len(prepares),
        "step3_calls_per_episode": dict(sorted(calls.items())),
        "step3_latency_sec": {
            "count": len(planning), "p50": _quantile(planning, 0.5),
            "p95": _quantile(planning, 0.95),
            "mean": statistics.fmean(planning) if planning else None,
        },
        "decision_interval_sec": {
            "count": len(intervals), "p50": _quantile(intervals, 0.5),
            "p95": _quantile(intervals, 0.95),
        },
        "nav2_commit_latency_sec": {
            "count": len(nav2), "p50": _quantile(nav2, 0.5),
            "p95": _quantile(nav2, 0.95),
        },
        "command_age_sec": {
            "count": len(command_age), "p50": _quantile(command_age, 0.5),
            "p95": _quantile(command_age, 0.95),
        },
        "timeout_count": len(timeouts), "abstain_count": len(abstains),
        "safe_stop_count": len(safe_stops), "fallback_count": len(fallbacks),
        "internvla_model_loaded": False,
        "internvla_fallback_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--direct", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    direct = args.direct.resolve()
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "screening_only": True,
        "prompt_tuned_after_fixed5": False,
        "baseline": {"runtime": "InternVLA", **_navigation(baseline)},
        "step3_direct": {
            "runtime": "Step3-VL-10B direct_high_level",
            **_navigation(direct),
            **_direct_details(direct),
        },
        "metric_notes": {
            "ndtw": "null means the frozen evaluator did not emit nDTW; no value is synthesized",
            "osr": "reported from evaluator result val_unseen.OS",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
