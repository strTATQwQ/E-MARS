#!/usr/bin/env python3
"""Fail-closed T5 cuVSLAM fixed-3 takeover acceptance analyzer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_EPISODES = 3
MINIMUM_ORACLE_SUCCESSES = 2
MAXIMUM_SAFE_STOP_WALL_SEC = 0.30


def _object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("{}:{} must contain an object".format(path, number))
        rows.append(value)
    return rows


def _unique(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one {}, found {}".format(name, len(matches))
        )
    return matches[0]


def analyze(result_root: Path) -> Dict[str, Any]:
    root = result_root.resolve()
    fast = _object(root / "fast_lane_final_summary.json")
    dgx = root / "remote" / "dgx"
    controller = _object(dgx / "onboard" / "controller_summary.json")
    runtime = _object(dgx / "cuvslam" / "takeover_runtime_contract.json")
    supervisor = _rows(dgx / "cuvslam" / "odometry_supervisor_records.jsonl")
    episodes = _object(_unique(root / "remote" / "x86" / "evaluator", "per_episode.json"))

    completed = int(episodes.get("completed_episode_count", -1))
    expected = int(episodes.get("expected_episode_count", -1))
    success_count = int(episodes.get("success_count", -1))
    fallback_count = int(runtime.get("gt_fallback_count", -1))
    reset_pollution_count = int(runtime.get("reset_pollution_count", -1))
    stop_events = [
        row
        for row in supervisor
        if row.get("event") in {"tracking_lost", "fatal_unexpected_exit"}
    ]
    stop_latencies = [
        float(row["safe_stop_latency_wall_sec"])
        for row in stop_events
        if isinstance(row.get("safe_stop_latency_wall_sec"), (int, float))
    ]
    stop_evidence_complete = len(stop_latencies) == len(stop_events)
    safe_stop_pass = (
        stop_evidence_complete
        and all(row.get("fail_safe_stop_published") is True for row in stop_events)
        and all(0.0 <= value <= MAXIMUM_SAFE_STOP_WALL_SEC for value in stop_latencies)
    )

    checks = {
        "fast_lane_screen3_oracle_pass": fast.get("status") == "PASS"
        and fast.get("runtime_summary", {}).get("profile") == "screen3"
        and fast.get("runtime_summary", {}).get("lane") == "a",
        "exact_takeover_contract": runtime.get("status") == "PASS"
        and runtime.get("mode") == "cuvslam_takeover"
        and runtime.get("primary_source") == "cuvslam"
        and runtime.get("navigation_odometry_topic") == "/odom",
        "single_odom_authority": runtime.get("single_odom_authority") is True,
        "controller_uses_external_odometry": controller.get("pose_source")
        == "external_odometry"
        and controller.get("ground_truth_pose_used_for_nav") is False
        and int(controller.get("external_odometry_nav_publish_count", 0)) > 0,
        "fixed3_complete": expected == EXPECTED_EPISODES
        and completed == EXPECTED_EPISODES
        and isinstance(episodes.get("episodes"), list)
        and len(episodes["episodes"]) == EXPECTED_EPISODES,
        "oracle_success_threshold": success_count >= MINIMUM_ORACLE_SUCCESSES,
        "safe_stop_within_wall_deadline": safe_stop_pass,
        "reset_pollution_zero": reset_pollution_count == 0,
        "fallback_count_valid": fallback_count >= 0,
    }
    safety_checks = dict(checks)
    safety_checks.pop("oracle_success_threshold")
    base_safe = all(safety_checks.values())
    if base_safe and checks["oracle_success_threshold"] and fallback_count == 0:
        status = "NAV_CANDIDATE_PASS"
        nav_candidate_pass = True
    elif base_safe and fallback_count > 0:
        status = "SAFE_FALLBACK_PASS"
        nav_candidate_pass = False
    else:
        status = "FAIL"
        nav_candidate_pass = False
    return {
        "schema_version": 1,
        "status": status,
        "mode": "cuvslam_takeover",
        "checks": checks,
        "navigation_pose_authority": "cuvslam"
        if checks["single_odom_authority"]
        else "unproven",
        "nav_candidate_pass": nav_candidate_pass,
        "online_navigation_acceptance": status,
        "expected_episode_count": EXPECTED_EPISODES,
        "completed_episode_count": completed,
        "minimum_oracle_success_count": MINIMUM_ORACLE_SUCCESSES,
        "success_count": success_count,
        "gt_fallback_count": fallback_count,
        "tracking_loss_or_exit_count": len(stop_events),
        "safe_stop_latency_wall_sec": stop_latencies,
        "maximum_safe_stop_wall_sec": MAXIMUM_SAFE_STOP_WALL_SEC,
        "reset_pollution_count": reset_pollution_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = analyze(args.result_root)
        exit_code = 0 if payload["status"] in {
            "NAV_CANDIDATE_PASS",
            "SAFE_FALLBACK_PASS",
        } else 2
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        payload = {"schema_version": 1, "status": "FAIL", "error": str(exc)}
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
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
