#!/usr/bin/env python3
"""Score T4.5 stuck, refresh, fresh-path, and post-recovery progress evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-episode", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--replan-records", type=Path)
    parser.add_argument("--adapter-records", type=Path)
    args = parser.parse_args()
    per_episode_path = args.per_episode or args.result_dir / "per_episode.json"
    records_path = args.records or args.result_dir / "recovery_records.jsonl"
    replan_path = (
        args.replan_records or args.result_dir / "replan_request_records.jsonl"
    )
    adapter_path = args.adapter_records or args.result_dir / "active_records.jsonl"
    per_episode = json.loads(
        per_episode_path.read_text(encoding="utf-8")
    )
    records = jsonl(records_path)
    replan_records = jsonl(replan_path)
    adapter_records = jsonl(adapter_path)
    episodes = per_episode.get("episodes", [])
    expected = int(per_episode.get("completed_episode_count", len(episodes)))
    stuck_episodes = sum(
        str(item.get("termination_reason", item.get("reason", ""))) == "stuck"
        for item in episodes
    )
    recovery_events = [item for item in records if item.get("event") == "recovery"]
    recovery_profile_ids = sorted(
        {
            str(item.get("recovery_profile_id", ""))
            for item in recovery_events
            if str(item.get("recovery_profile_id", ""))
        }
    )
    recovery_profile_sha256 = sorted(
        {
            str(item.get("recovery_profile_sha256", ""))
            for item in recovery_events
            if str(item.get("recovery_profile_sha256", ""))
        }
    )
    full_recoveries = [item for item in recovery_events if item.get("full_recovery") is True]
    routine_refreshes = [item for item in recovery_events if item.get("full_recovery") is False]
    fresh_paths = [
        item for item in records if item.get("event") == "post_recovery_trajectory"
    ]
    positive = {
        (str(item.get("episode_id", "")), int(item.get("recovery_index", -1)))
        for item in records
        if item.get("event") == "post_recovery_positive_progress"
    }
    full_keys = {
        (str(item.get("episode_id", "")), int(item.get("recovery_index", -1)))
        for item in full_recoveries
    }
    recovery_ids = {
        str(item.get("recovery_id", "")) for item in recovery_events
    }
    fresh_accepts = [
        item
        for item in adapter_records
        if item.get("event") == "recovery_latch_fresh_trajectory_accepted"
    ]
    fresh_accept_ids = {
        str(item.get("recovery_id", "")) for item in fresh_accepts
    }
    excluded_by_recovery = {
        str(item.get("recovery_id", "")): {
            *[str(value) for value in item.get("excluded_absolute_sha256", [])],
            *[str(value) for value in item.get("excluded_shape_sha256", [])],
        }
        for item in recovery_events
    }
    old_trajectory_execution_count = sum(
        str(item.get("absolute_sha256", ""))
        in excluded_by_recovery.get(str(item.get("recovery_id", "")), set())
        or str(item.get("shape_sha256", ""))
        in excluded_by_recovery.get(str(item.get("recovery_id", "")), set())
        for item in fresh_accepts
    )
    fresh_path_ok = (
        bool(recovery_ids)
        and fresh_accept_ids == recovery_ids
        and old_trajectory_execution_count == 0
    )
    full_progress_ok = full_keys.issubset(positive)
    terminal_stops = sum(
        item.get("event") == "terminal_safe_stop" for item in records
    )
    requested_indices = {
        int(item["request_index"])
        for item in replan_records
        if item.get("event") == "requested"
    }
    consumed_indices = {
        int(item["request_index"])
        for item in replan_records
        if item.get("event") == "consumed_by_fresh_model_step"
    }
    replan_consumption_ok = (
        len(requested_indices) == len(recovery_events)
        and consumed_indices == requested_indices
    )
    typed_transaction_ok = bool(recovery_events) and all(
        item.get("typed_transaction") is True
        and item.get("cancel_success") is True
        and item.get("history_clear_success") is True
        and item.get("fresh_trajectory_requested") is True
        and item.get("adapter_replan_gate_armed") is True
        and int(item.get("cache_epoch", 0)) > 0
        and "error" not in item
        for item in recovery_events
    )
    stuck_rate = stuck_episodes / expected if expected else 1.0
    functional_pass = (
        expected > 0
        and len(recovery_events) > 0
        and typed_transaction_ok
        and fresh_path_ok
        and replan_consumption_ok
    )
    quality_pass = (
        stuck_rate < 0.25
        and full_progress_ok
        and terminal_stops == 0
        and len(fresh_paths) == len(recovery_events)
        and all(item.get("fresh_path_hash") is True for item in fresh_paths)
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if functional_pass else "FAIL",
        "quality_status": "PASS" if quality_pass else "WARN",
        "required": {
            "minimum_recovery_or_refresh_event_count": 1,
            "typed_identity_scoped_transaction": True,
            "old_trajectory_execution_count": 0,
            "every_refresh_consumed_by_fresh_model_step": True,
        },
        "quality_targets": {
            "maximum_stuck_episode_rate_exclusive": 0.25,
            "fresh_path_observation_after_every_refresh": True,
            "positive_progress_after_every_full_recovery": True,
            "terminal_safe_stop_count": 0,
        },
        "episode_count": expected,
        "stuck_episode_count": stuck_episodes,
        "stuck_episode_rate": stuck_rate,
        "full_recovery_count": len(full_recoveries),
        "routine_refresh_count": len(routine_refreshes),
        "recovery_profile_ids": recovery_profile_ids,
        "recovery_profile_sha256": recovery_profile_sha256,
        "fresh_path_event_count": len(fresh_paths),
        "typed_fresh_accept_count": len(fresh_accepts),
        "fresh_path_ok": fresh_path_ok,
        "old_trajectory_execution_count": old_trajectory_execution_count,
        "typed_transaction_ok": typed_transaction_ok,
        "full_recovery_positive_progress_ok": full_progress_ok,
        "terminal_safe_stop_count": terminal_stops,
        "replan_request_count": len(requested_indices),
        "replan_consumed_count": len(consumed_indices),
        "replan_consumption_ok": replan_consumption_ok,
        "evidence_paths": {
            "per_episode": per_episode_path.as_posix(),
            "recovery_records": records_path.as_posix(),
            "replan_records": replan_path.as_posix(),
            "adapter_records": adapter_path.as_posix(),
        },
    }
    output = args.output or args.result_dir / "recovery_metrics.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if functional_pass else 1)


if __name__ == "__main__":
    main()
