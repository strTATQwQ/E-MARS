#!/usr/bin/env python3
"""Validate a bounded T5 engineering canary without claiming episode PASS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


MAX_SAMPLE_GAP_SECONDS = 15
MAX_START_OFFSET_SECONDS = MAX_SAMPLE_GAP_SECONDS
MAX_CLOCK_STATE_AGE_SECONDS = 15.0
MAX_SOAK_SAFE_STOP_TIMEOUT_WARNINGS = 12


def analyze(
    samples_path: Path,
    ready_path: Path,
    cutoff_path: Path,
    lane: str,
    mode: str,
    configured: int,
    observed: int,
    started_unix: float,
    timebase: str = "wall",
) -> dict[str, object]:
    if timebase not in {"wall", "sim"}:
        raise ValueError("engineering duration timebase must be wall or sim")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready_model_action_count = ready.get("ready_model_action_count")
    ready_model_step_error_count = ready.get("ready_model_step_error_count")
    ready_safe_stop_timeout_warning_count = ready.get(
        "ready_safe_stop_timeout_warning_count", 0
    )
    ready_fatal_error_count = ready.get("ready_fatal_error_count", 0)
    ready_baselines_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (
            ready_model_action_count,
            ready_model_step_error_count,
            ready_safe_stop_timeout_warning_count,
            ready_fatal_error_count,
        )
    ) and (
        ready_safe_stop_timeout_warning_count + ready_fatal_error_count
        == ready_model_step_error_count
    )
    cutoff = json.loads(cutoff_path.read_text(encoding="utf-8"))
    cutoff_elapsed = cutoff.get("elapsed_seconds")
    cutoff_model_action_count = cutoff.get("cutoff_model_action_count")
    cutoff_model_step_error_count = cutoff.get("cutoff_model_step_error_count")
    cutoff_new_model_step_error_count = cutoff.get("new_model_step_error_count")
    cutoff_evaluator_total_path_count = cutoff.get("evaluator_total_path_count")
    cutoff_safe_stop_timeout_warning_count = cutoff.get(
        "safe_stop_timeout_warning_count", 0
    )
    cutoff_fatal_error_count = cutoff.get("fatal_error_count", 0)
    cutoff_new_safe_stop_timeout_warning_count = cutoff.get(
        "new_safe_stop_timeout_warning_count", 0
    )
    cutoff_new_fatal_error_count = cutoff.get("new_fatal_error_count", 0)
    evaluator_cycle_count = cutoff.get("evaluator_cycle_count", 1)
    cutoff_elapsed_valid = (
        isinstance(cutoff_elapsed, (int, float))
        and not isinstance(cutoff_elapsed, bool)
        and cutoff_elapsed >= 0
    )
    cutoff_counters_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (
            cutoff_model_action_count,
            cutoff_model_step_error_count,
            cutoff_new_model_step_error_count,
            cutoff_evaluator_total_path_count,
            cutoff_safe_stop_timeout_warning_count,
            cutoff_fatal_error_count,
            cutoff_new_safe_stop_timeout_warning_count,
            cutoff_new_fatal_error_count,
            evaluator_cycle_count,
        )
    )
    cutoff_values_valid = (
        cutoff.get("status") == "RECORDED"
        and cutoff_elapsed_valid
        and cutoff_counters_valid
        and ready_baselines_valid
        and cutoff.get("ready_model_action_count") == ready_model_action_count
        and cutoff.get("ready_model_step_error_count")
        == ready_model_step_error_count
        and cutoff_new_model_step_error_count
        == cutoff_model_step_error_count - ready_model_step_error_count
        and cutoff.get("ready_safe_stop_timeout_warning_count", 0)
        == ready_safe_stop_timeout_warning_count
        and cutoff.get("ready_fatal_error_count", 0) == ready_fatal_error_count
        and cutoff_safe_stop_timeout_warning_count + cutoff_fatal_error_count
        == cutoff_model_step_error_count
        and cutoff_new_safe_stop_timeout_warning_count
        == cutoff_safe_stop_timeout_warning_count
        - ready_safe_stop_timeout_warning_count
        and cutoff_new_fatal_error_count
        == cutoff_fatal_error_count - ready_fatal_error_count
        and cutoff.get("duration_timebase", "wall") == timebase
    )
    samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    received = [row["clock"]["received_step_count"] for row in samples]
    published = [row["clock"]["publish_count"] for row in samples]
    clock_ns = [row["clock"]["last_clock_ns"] for row in samples]
    elapsed = [row["elapsed_seconds"] for row in samples]
    sample_timebases_match = bool(samples) and all(
        row.get("duration_timebase", "wall") == timebase for row in samples
    )
    model_action_counts = [row.get("model_action_count") for row in samples]
    model_action_counts_valid = bool(samples) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in model_action_counts
    )
    model_action_counts_with_cutoff = model_action_counts + [
        cutoff_model_action_count
    ]
    model_action_progress_intervals = (
        sum(
            current > previous
            for previous, current in zip(
                model_action_counts_with_cutoff,
                model_action_counts_with_cutoff[1:],
            )
        )
        if model_action_counts_valid and cutoff_values_valid
        else 0
    )
    model_step_error_counts = [
        row.get("model_step_error_count") for row in samples
    ]
    model_step_error_counts_valid = bool(samples) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in model_step_error_counts
    )
    model_step_error_counts_with_cutoff = model_step_error_counts + [
        cutoff_model_step_error_count
    ]
    reported_ready_action_counts = [
        row.get("ready_model_action_count") for row in samples
    ]
    reported_ready_error_counts = [
        row.get("ready_model_step_error_count") for row in samples
    ]
    reported_new_error_counts = [
        row.get("new_model_step_error_count") for row in samples
    ]
    reported_new_error_counts_with_cutoff = reported_new_error_counts + [
        cutoff_new_model_step_error_count
    ]
    sample_baselines_match_ready = bool(samples) and ready_baselines_valid and all(
        action == ready_model_action_count and error == ready_model_step_error_count
        for action, error in zip(
            reported_ready_action_counts, reported_ready_error_counts
        )
    )
    new_model_step_error_counts_valid = (
        model_step_error_counts_valid
        and cutoff_values_valid
        and ready_baselines_valid
        and bool(samples)
        and all(
            isinstance(reported, int)
            and not isinstance(reported, bool)
            and reported >= 0
            and reported == absolute - ready_model_step_error_count
            for reported, absolute in zip(
                reported_new_error_counts_with_cutoff,
                model_step_error_counts_with_cutoff,
            )
        )
    )
    safe_stop_timeout_warning_counts = [
        row.get("safe_stop_timeout_warning_count", 0) for row in samples
    ]
    fatal_error_counts = [row.get("fatal_error_count", 0) for row in samples]
    new_safe_stop_timeout_warning_counts = [
        row.get("new_safe_stop_timeout_warning_count", 0) for row in samples
    ]
    new_fatal_error_counts = [
        row.get("new_fatal_error_count", 0) for row in samples
    ]
    classified_error_counts_valid = (
        bool(samples)
        and ready_baselines_valid
        and cutoff_values_valid
        and all(
            all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in (warning, fatal, new_warning, new_fatal)
            )
            and warning + fatal == total
            and new_warning == warning - ready_safe_stop_timeout_warning_count
            and new_fatal == fatal - ready_fatal_error_count
            for warning, fatal, new_warning, new_fatal, total in zip(
                safe_stop_timeout_warning_counts,
                fatal_error_counts,
                new_safe_stop_timeout_warning_counts,
                new_fatal_error_counts,
                model_step_error_counts,
            )
        )
    )
    soak_safe_stop_timeout_warning_limit = MAX_SOAK_SAFE_STOP_TIMEOUT_WARNINGS
    bounded_error_policy_pass = (
        classified_error_counts_valid
        and cutoff_new_fatal_error_count == 0
        and (
            cutoff_new_safe_stop_timeout_warning_count
            <= soak_safe_stop_timeout_warning_limit
            if timebase == "sim"
            else cutoff_new_safe_stop_timeout_warning_count == 0
        )
    )
    evaluator_total_path_counts = [
        row.get("evaluator_total_path_count") for row in samples
    ]
    evaluator_total_path_counts_valid = bool(samples) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in evaluator_total_path_counts
    )
    clock_state_ages = []
    for row in samples:
        explicit_age = row.get("clock_state_age_seconds")
        sampled_unix = row.get("sampled_unix")
        updated_unix = row.get("clock", {}).get("updated_unix")
        if isinstance(explicit_age, (int, float)) and not isinstance(
            explicit_age, bool
        ):
            age = float(explicit_age)
        elif (
            isinstance(sampled_unix, (int, float))
            and not isinstance(sampled_unix, bool)
            and isinstance(updated_unix, (int, float))
            and not isinstance(updated_unix, bool)
        ):
            age = float(sampled_unix) - float(updated_unix)
        else:
            age = float("inf")
        clock_state_ages.append(age)
    max_clock_state_age_seconds = max(clock_state_ages) if clock_state_ages else None
    sample_gaps = [
        current - previous for previous, current in zip(elapsed, elapsed[1:])
    ]
    elapsed_strictly_increasing = bool(samples) and all(
        gap > 0 for gap in sample_gaps
    )
    max_sample_gap_seconds = max(sample_gaps) if sample_gaps else None
    cutoff_gap_seconds = (
        cutoff_elapsed - elapsed[-1]
        if cutoff_values_valid and elapsed
        else None
    )
    step_progress_intervals = sum(
        current > previous for previous, current in zip(received, received[1:])
    )
    last_progress_elapsed = elapsed[0] if elapsed else 0
    max_step_stagnation_seconds = 0
    for index in range(1, len(samples)):
        max_step_stagnation_seconds = max(
            max_step_stagnation_seconds, elapsed[index] - last_progress_elapsed
        )
        if received[index] > received[index - 1]:
            last_progress_elapsed = elapsed[index]
    minimum_periodic_sample_count = (
        max(
            2,
            (configured + MAX_SAMPLE_GAP_SECONDS - 1)
            // MAX_SAMPLE_GAP_SECONDS,
        )
        if timebase == "sim"
        else configured // 5
    )
    checks = {
        "bounded_duration_valid": 30 <= configured <= 600,
        "duration_timebase_valid": timebase in {"wall", "sim"},
        "sim_soak_is_600_seconds": timebase != "sim" or configured == 600,
        "sample_timebases_match": sample_timebases_match,
        "sim_duration_reaches_configured": timebase != "sim"
        or (
            isinstance(cutoff.get("elapsed_sim_seconds"), (int, float))
            and not isinstance(cutoff.get("elapsed_sim_seconds"), bool)
            and cutoff.get("elapsed_sim_seconds") >= configured
        ),
        "ready_interval_complete": observed >= configured,
        "ready_probe_pass": ready.get("status") == "PASS"
        and ready.get("lane") == lane
        and ready.get("execution_profile") == "engineering_canary",
        "ready_baselines_valid": ready_baselines_valid,
        "cutoff_values_valid": cutoff_values_valid,
        "cutoff_counter_not_rolled_back": cutoff_values_valid
        and model_action_counts_valid
        and model_step_error_counts_valid
        and cutoff_model_action_count >= model_action_counts[-1]
        and cutoff_model_step_error_count >= model_step_error_counts[-1],
        "sample_baselines_match_ready": sample_baselines_match_ready,
        "periodic_sample_count": len(samples) >= minimum_periodic_sample_count,
        "timeline_starts_near_zero": bool(samples)
        and 0 <= elapsed[0] <= MAX_START_OFFSET_SECONDS,
        "timeline_strictly_increasing": elapsed_strictly_increasing,
        "timeline_covers_configured_window": cutoff_values_valid
        and cutoff_elapsed >= configured
        and cutoff_elapsed <= observed,
        "timeline_sample_gap_bounded": bool(sample_gaps)
        and max_sample_gap_seconds is not None
        and max_sample_gap_seconds <= MAX_SAMPLE_GAP_SECONDS
        and cutoff_gap_seconds is not None
        and 0 <= cutoff_gap_seconds <= MAX_SAMPLE_GAP_SECONDS,
        "health_running_every_sample": bool(samples)
        and all(
            row.get("lane") == lane and row.get("health_state") == "RUNNING"
            for row in samples
        ),
        "model_action_counts_valid": model_action_counts_valid,
        "model_action_count_positive_at_ready": model_action_counts_valid
        and ready_baselines_valid
        and ready_model_action_count > 0
        and model_action_counts[0] >= ready_model_action_count,
        "model_action_counts_monotonic": model_action_counts_valid
        and cutoff_values_valid
        and all(
            current >= previous
            for previous, current in zip(
                model_action_counts_with_cutoff,
                model_action_counts_with_cutoff[1:],
            )
        ),
        "model_actions_advance": model_action_counts_valid
        and cutoff_values_valid
        and len(model_action_counts_with_cutoff) >= 2
        and cutoff_model_action_count > model_action_counts[0]
        and model_action_progress_intervals >= 1,
        "model_step_error_counts_valid": model_step_error_counts_valid,
        "model_step_error_counts_monotonic_from_ready": (
            model_step_error_counts_valid
            and ready_baselines_valid
            and model_step_error_counts[0] >= ready_model_step_error_count
            and all(
                current >= previous
                for previous, current in zip(
                    model_step_error_counts_with_cutoff,
                    model_step_error_counts_with_cutoff[1:],
                )
            )
        ),
        "new_model_step_error_counts_valid": new_model_step_error_counts_valid,
        "classified_error_counts_valid": classified_error_counts_valid,
        "bounded_safe_stop_timeout_warning_policy": bounded_error_policy_pass,
        "model_step_errors_absent": new_model_step_error_counts_valid
        and all(value == 0 for value in reported_new_error_counts_with_cutoff),
        "new_model_step_errors_absent": new_model_step_error_counts_valid
        and all(value == 0 for value in reported_new_error_counts_with_cutoff),
        "fresh_five_episode_evaluator_state": evaluator_total_path_counts_valid
        and cutoff_values_valid
        and all(value == 5 for value in evaluator_total_path_counts)
        and cutoff_evaluator_total_path_count == 5,
        "clock_state_age_seconds_le_15": bool(clock_state_ages)
        and all(
            0 <= age <= MAX_CLOCK_STATE_AGE_SECONDS for age in clock_state_ages
        ),
        "clock_counters_monotonic": bool(samples)
        and all(
            current >= previous
            for previous, current in zip(received, received[1:])
        )
        and all(
            current > previous
            for previous, current in zip(published, published[1:])
        )
        and all(
            current >= previous
            for previous, current in zip(clock_ns, clock_ns[1:])
        ),
        "physics_steps_advance": len(samples) >= 2
        and received[-1] > received[0]
        and clock_ns[-1] > clock_ns[0]
        and step_progress_intervals >= 2,
        "physics_not_stagnant_over_30_seconds": bool(samples)
        and max_step_stagnation_seconds <= 30,
        "evaluator_stop_completed_before_analysis": True,
    }
    diagnostic_only_checks = (
        {"model_step_errors_absent", "new_model_step_errors_absent"}
        if timebase == "sim"
        else set()
    )
    required_check_names = [
        name for name in checks if name not in diagnostic_only_checks
    ]
    return {
        "schema_version": 1,
        "status": (
            "PASS" if all(checks[name] for name in required_check_names) else "FAIL"
        ),
        "profile": "engineering_canary",
        "lane": lane,
        "mode": mode,
        "configured_seconds": configured,
        "observed_seconds": observed,
        "duration_timebase": timebase,
        "observed_wall_seconds": cutoff.get("elapsed_wall_seconds"),
        "observed_sim_seconds": cutoff.get("elapsed_sim_seconds"),
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "termination": (
            "coordinator_after_600_sim_second_soak"
            if timebase == "sim"
            else "coordinator_after_bounded_canary"
        ),
        "episode_acceptance_claimed": False,
        "evaluation_completed_naturally": False,
        "sample_path": str(samples_path),
        "cutoff_path": str(cutoff_path),
        "sample_count": len(samples),
        "minimum_periodic_sample_count": minimum_periodic_sample_count,
        "first_sample_elapsed_seconds": elapsed[0] if elapsed else None,
        "last_sample_elapsed_seconds": elapsed[-1] if elapsed else None,
        "cutoff_elapsed_seconds": cutoff_elapsed if cutoff_values_valid else None,
        "cutoff_gap_seconds": cutoff_gap_seconds,
        "max_start_offset_limit_seconds": MAX_START_OFFSET_SECONDS,
        "max_sample_gap_seconds": max_sample_gap_seconds,
        "max_sample_gap_limit_seconds": MAX_SAMPLE_GAP_SECONDS,
        "max_clock_state_age_seconds": max_clock_state_age_seconds,
        "max_clock_state_age_limit_seconds": MAX_CLOCK_STATE_AGE_SECONDS,
        "first_model_action_count": model_action_counts[0]
        if model_action_counts_valid else None,
        "ready_model_action_count": ready_model_action_count
        if ready_baselines_valid else None,
        "ready_model_step_error_count": ready_model_step_error_count
        if ready_baselines_valid else None,
        "final_model_action_count": cutoff_model_action_count
        if cutoff_values_valid else None,
        "model_action_progress_intervals": model_action_progress_intervals,
        "final_model_step_error_count": cutoff_model_step_error_count
        if cutoff_values_valid else None,
        "final_new_model_step_error_count": cutoff_new_model_step_error_count
        if new_model_step_error_counts_valid else None,
        "safe_stop_timeout_warnings": {
            "status": (
                "WARN"
                if cutoff_new_safe_stop_timeout_warning_count > 0
                else "PASS"
            ),
            "exact_safe_stop_required": True,
            "count_after_ready": cutoff_new_safe_stop_timeout_warning_count,
            "limit": (
                soak_safe_stop_timeout_warning_limit if timebase == "sim" else 0
            ),
            "policy_scope": (
                "soak600_sim_only" if timebase == "sim" else "not_relaxed"
            ),
        },
        "final_new_fatal_error_count": cutoff_new_fatal_error_count
        if classified_error_counts_valid else None,
        "evaluator_cycle_count": evaluator_cycle_count
        if cutoff_values_valid else None,
        "required_check_names": required_check_names,
        "diagnostic_only_check_names": sorted(diagnostic_only_checks),
        "evaluator_total_path_count": cutoff_evaluator_total_path_count
        if cutoff_values_valid else None,
        "step_progress_intervals": step_progress_intervals,
        "max_step_stagnation_seconds": max_step_stagnation_seconds,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--cutoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("a", "b"), required=True)
    parser.add_argument("--mode", choices=("oracle", "model"), required=True)
    parser.add_argument("--configured-sec", type=int, required=True)
    parser.add_argument("--observed-sec", type=int, required=True)
    parser.add_argument("--timebase", choices=("wall", "sim"), default="wall")
    parser.add_argument("--started-unix", type=float, required=True)
    args = parser.parse_args()
    payload = analyze(
        args.samples,
        args.ready,
        args.cutoff,
        args.lane,
        args.mode,
        args.configured_sec,
        args.observed_sec,
        args.started_unix,
        args.timebase,
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if payload["status"] == "PASS" else 75)


if __name__ == "__main__":
    main()
