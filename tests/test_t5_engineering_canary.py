from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_t5_engineering_canary.py"


def run_case(
    tmp_path: Path,
    mutate=None,
    mutate_cutoff=None,
    configured: int = 60,
    timebase: str = "wall",
) -> dict[str, object]:
    ready = tmp_path / "ready.json"
    samples = tmp_path / "samples.jsonl"
    cutoff = tmp_path / "cutoff.json"
    output = tmp_path / "summary.json"
    ready.write_text(
        json.dumps(
            {
                "status": "PASS",
                "lane": "a",
                "execution_profile": "engineering_canary",
                "ready_model_action_count": 2,
                "ready_model_step_error_count": 0,
                "ready_safe_stop_timeout_warning_count": 0,
                "ready_fatal_error_count": 0,
                "pre_ready_nav2_timeout_warning_count": 0,
                "ready_consecutive_action_count": 2,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for index, elapsed in enumerate(range(0, configured + 1, 5)):
        row = {
                "elapsed_seconds": elapsed,
                "sampled_unix": 1_000.0 + elapsed,
                "lane": "a",
                "health_state": "RUNNING",
                "model_action_count": index + 2,
                "model_step_error_count": 0,
                "ready_model_action_count": 2,
                "ready_model_step_error_count": 0,
                "new_model_step_error_count": 0,
                "safe_stop_timeout_warning_count": 0,
                "fatal_error_count": 0,
                "ready_safe_stop_timeout_warning_count": 0,
                "ready_fatal_error_count": 0,
                "new_safe_stop_timeout_warning_count": 0,
                "new_fatal_error_count": 0,
                "evaluator_total_path_count": 5,
                "clock": {
                    "received_step_count": index * 10,
                    "publish_count": 100 + index * 5,
                    "last_clock_ns": (
                        1_000_000_000 + elapsed * 1_000_000_000
                        if timebase == "sim"
                        else 1_000_000_000 + index * 100_000_000
                    ),
                    "updated_unix": 999.0 + elapsed,
                },
            }
        if timebase == "sim":
            row.update(
                duration_timebase="sim",
                elapsed_wall_seconds=elapsed * 5,
                elapsed_sim_seconds=float(elapsed),
            )
        rows.append(row)
    if mutate is not None:
        mutate(rows, ready)
    ready_value = json.loads(ready.read_text(encoding="utf-8"))
    cutoff_value = {
        "schema_version": 1,
        "status": "RECORDED",
        "elapsed_seconds": configured,
        "cutoff_model_action_count": rows[-1]["model_action_count"],
        "cutoff_model_step_error_count": rows[-1]["model_step_error_count"],
        "ready_model_action_count": ready_value["ready_model_action_count"],
        "ready_model_step_error_count": ready_value[
            "ready_model_step_error_count"
        ],
        "new_model_step_error_count": rows[-1]["model_step_error_count"]
        - ready_value["ready_model_step_error_count"],
        "safe_stop_timeout_warning_count": rows[-1][
            "safe_stop_timeout_warning_count"
        ],
        "fatal_error_count": rows[-1]["fatal_error_count"],
        "ready_safe_stop_timeout_warning_count": ready_value[
            "ready_safe_stop_timeout_warning_count"
        ],
        "ready_fatal_error_count": ready_value["ready_fatal_error_count"],
        "new_safe_stop_timeout_warning_count": rows[-1][
            "new_safe_stop_timeout_warning_count"
        ],
        "new_fatal_error_count": rows[-1]["new_fatal_error_count"],
        "evaluator_cycle_count": 1,
        "evaluator_total_path_count": rows[-1]["evaluator_total_path_count"],
    }
    if timebase == "sim":
        cutoff_value.update(
            duration_timebase="sim",
            elapsed_wall_seconds=configured * 5,
            elapsed_sim_seconds=float(configured),
        )
    if mutate_cutoff is not None:
        mutate_cutoff(cutoff_value)
    cutoff.write_text(json.dumps(cutoff_value), encoding="utf-8")
    samples.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--samples",
            str(samples),
            "--ready",
            str(ready),
            "--cutoff",
            str(cutoff),
            "--output",
            str(output),
            "--lane",
            "a",
            "--mode",
            "model",
            "--configured-sec",
            str(configured),
            "--observed-sec",
            str(configured),
            "--started-unix",
            "1.0",
            "--timebase",
            timebase,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["returncode"] = completed.returncode
    return value


def test_progressing_canary_passes(tmp_path: Path) -> None:
    value = run_case(tmp_path)
    assert value["status"] == "PASS"
    assert value["returncode"] == 0
    assert value["episode_acceptance_claimed"] is False
    assert value["evaluation_completed_naturally"] is False
    assert value["checks"]["evaluator_stop_completed_before_analysis"] is True
    assert "clean_stop_delegated_to_finalizer" not in value["checks"]
    assert value["first_sample_elapsed_seconds"] == 0
    assert value["last_sample_elapsed_seconds"] == 60
    assert value["max_sample_gap_seconds"] == 5
    assert value["checks"]["model_action_counts_valid"] is True
    assert value["checks"]["model_action_count_positive_at_ready"] is True
    assert value["checks"]["model_action_counts_monotonic"] is True
    assert value["checks"]["model_actions_advance"] is True
    assert value["first_model_action_count"] == 2
    assert value["final_model_action_count"] == 14
    assert value["model_action_progress_intervals"] == 12
    assert value["checks"]["model_step_error_counts_valid"] is True
    assert value["checks"]["model_step_errors_absent"] is True
    assert value["checks"]["new_model_step_errors_absent"] is True
    assert value["final_model_step_error_count"] == 0
    assert value["final_new_model_step_error_count"] == 0
    assert value["checks"]["fresh_five_episode_evaluator_state"] is True
    assert value["evaluator_total_path_count"] == 5
    assert value["minimum_periodic_sample_count"] == 12


def test_600_second_soak_uses_the_same_bounded_canary_contract(
    tmp_path: Path,
) -> None:
    value = run_case(tmp_path, configured=600)
    assert value["status"] == "PASS"
    assert value["returncode"] == 0
    assert value["configured_seconds"] == 600
    assert value["observed_seconds"] == 600
    assert value["episode_acceptance_claimed"] is False
    assert value["evaluation_completed_naturally"] is False
    assert value["checks"]["bounded_duration_valid"] is True
    assert value["checks"]["timeline_covers_configured_window"] is True


def test_600_second_soak_can_be_bound_to_simulation_time(tmp_path: Path) -> None:
    value = run_case(tmp_path, configured=600, timebase="sim")
    assert value["status"] == "PASS"
    assert value["duration_timebase"] == "sim"
    assert value["observed_sim_seconds"] == 600.0
    assert value["observed_wall_seconds"] == 3000
    assert value["checks"]["sim_duration_reaches_configured"] is True
    assert value["termination"] == "coordinator_after_600_sim_second_soak"


def test_sim_soak_records_bounded_exact_safe_stop_timeout_as_warning(
    tmp_path: Path,
) -> None:
    def one_safe_stop(rows, _ready):
        for row in rows[8:]:
            row["model_step_error_count"] = 1
            row["new_model_step_error_count"] = 1
            row["safe_stop_timeout_warning_count"] = 1
            row["new_safe_stop_timeout_warning_count"] = 1

    value = run_case(
        tmp_path, one_safe_stop, configured=600, timebase="sim"
    )
    assert value["status"] == "PASS"
    assert value["returncode"] == 0
    assert value["safe_stop_timeout_warnings"]["status"] == "WARN"
    assert value["safe_stop_timeout_warnings"]["count_after_ready"] == 1
    assert value["safe_stop_timeout_warnings"]["limit"] == 12
    assert value["final_new_fatal_error_count"] == 0
    assert value["checks"]["model_step_errors_absent"] is False
    assert value["checks"]["new_model_step_errors_absent"] is False
    assert "model_step_errors_absent" in value["diagnostic_only_check_names"]
    assert "model_step_errors_absent" not in value["required_check_names"]


def test_wall_canary_does_not_relax_safe_stop_timeout(tmp_path: Path) -> None:
    def one_safe_stop(rows, _ready):
        for row in rows[8:]:
            row["model_step_error_count"] = 1
            row["new_model_step_error_count"] = 1
            row["safe_stop_timeout_warning_count"] = 1
            row["new_safe_stop_timeout_warning_count"] = 1

    value = run_case(tmp_path, one_safe_stop)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["bounded_safe_stop_timeout_warning_policy"] is False


def test_sim_soak_rejects_more_than_global_safe_stop_timeout_limit(
    tmp_path: Path,
) -> None:
    def thirteen_safe_stops(rows, _ready):
        for row in rows[8:]:
            row["model_step_error_count"] = 13
            row["new_model_step_error_count"] = 13
            row["safe_stop_timeout_warning_count"] = 13
            row["new_safe_stop_timeout_warning_count"] = 13

    value = run_case(
        tmp_path, thirteen_safe_stops, configured=600, timebase="sim"
    )
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["safe_stop_timeout_warnings"]["limit"] == 12


def test_sim_soak_accepts_minimum_samples_implied_by_gap_limit(
    tmp_path: Path,
) -> None:
    def keep_fifteen_second_samples(rows, _ready):
        rows[:] = [row for row in rows if row["elapsed_seconds"] % 15 == 0]

    value = run_case(
        tmp_path,
        keep_fifteen_second_samples,
        configured=600,
        timebase="sim",
    )
    assert value["status"] == "PASS"
    assert value["sample_count"] == 41
    assert value["minimum_periodic_sample_count"] == 40
    assert value["max_sample_gap_seconds"] == 15


def test_sim_soak_rejects_fatal_error_even_when_safe_stop_warnings_are_allowed(
    tmp_path: Path,
) -> None:
    def one_fatal(rows, _ready):
        for row in rows[8:]:
            row["model_step_error_count"] = 1
            row["new_model_step_error_count"] = 1
            row["fatal_error_count"] = 1
            row["new_fatal_error_count"] = 1

    value = run_case(tmp_path, one_fatal, configured=600, timebase="sim")
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["final_new_fatal_error_count"] == 1


def test_frozen_physics_clock_fails(tmp_path: Path) -> None:
    def freeze(rows, _ready):
        for row in rows:
            row["clock"]["received_step_count"] = 1
            row["clock"]["last_clock_ns"] = 1_000_000_000

    value = run_case(tmp_path, freeze)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["physics_steps_advance"] is False


def test_stale_health_fails(tmp_path: Path) -> None:
    def stale(rows, _ready):
        rows[4]["health_state"] = "STOPPING"

    value = run_case(tmp_path, stale)
    assert value["status"] == "FAIL"
    assert value["checks"]["health_running_every_sample"] is False


def test_model_action_count_not_increasing_fails(tmp_path: Path) -> None:
    def frozen_actions(rows, _ready):
        for row in rows:
            row["model_action_count"] = 4

    value = run_case(tmp_path, frozen_actions)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["model_action_counts_monotonic"] is True
    assert value["checks"]["model_actions_advance"] is False
    assert value["model_action_progress_intervals"] == 0


def test_model_action_count_rollback_fails(tmp_path: Path) -> None:
    def rollback(rows, _ready):
        rows[7]["model_action_count"] = rows[6]["model_action_count"] - 1

    value = run_case(tmp_path, rollback)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["model_action_counts_valid"] is True
    assert value["checks"]["model_action_counts_monotonic"] is False
    assert value["checks"]["model_actions_advance"] is True


def test_model_step_error_fails(tmp_path: Path) -> None:
    def step_error(rows, _ready):
        rows[8]["model_step_error_count"] = 1
        rows[8]["new_model_step_error_count"] = 1
        rows[8]["fatal_error_count"] = 1
        rows[8]["new_fatal_error_count"] = 1
        for row in rows[9:]:
            row["model_step_error_count"] = 1
            row["new_model_step_error_count"] = 1
            row["fatal_error_count"] = 1
            row["new_fatal_error_count"] = 1

    value = run_case(tmp_path, step_error)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["model_step_error_counts_valid"] is True
    assert value["checks"]["model_step_errors_absent"] is False
    assert value["checks"]["new_model_step_errors_absent"] is False
    assert value["final_model_step_error_count"] == 1
    assert value["final_new_model_step_error_count"] == 1


def test_pre_ready_safe_stop_errors_are_baselined_not_counted_in_window(
    tmp_path: Path,
) -> None:
    def pre_ready_warnings(rows, ready):
        value = json.loads(ready.read_text(encoding="utf-8"))
        value["ready_model_step_error_count"] = 3
        value["ready_safe_stop_timeout_warning_count"] = 3
        value["pre_ready_nav2_timeout_warning_count"] = 3
        ready.write_text(json.dumps(value), encoding="utf-8")
        for row in rows:
            row["model_step_error_count"] = 3
            row["ready_model_step_error_count"] = 3
            row["new_model_step_error_count"] = 0
            row["safe_stop_timeout_warning_count"] = 3
            row["ready_safe_stop_timeout_warning_count"] = 3

    value = run_case(tmp_path, pre_ready_warnings)
    assert value["status"] == "PASS"
    assert value["returncode"] == 0
    assert value["ready_model_step_error_count"] == 3
    assert value["final_model_step_error_count"] == 3
    assert value["final_new_model_step_error_count"] == 0
    assert value["checks"]["new_model_step_errors_absent"] is True


def test_error_observed_only_by_deadline_cutoff_fails(tmp_path: Path) -> None:
    def cutoff_error(value):
        value["cutoff_model_step_error_count"] = 1
        value["new_model_step_error_count"] = 1
        value["fatal_error_count"] = 1
        value["new_fatal_error_count"] = 1

    value = run_case(tmp_path, mutate_cutoff=cutoff_error)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["cutoff_counter_not_rolled_back"] is True
    assert value["checks"]["new_model_step_errors_absent"] is False
    assert value["final_new_model_step_error_count"] == 1


def test_resumed_four_episode_evaluator_state_fails(tmp_path: Path) -> None:
    def resumed(rows, _ready):
        for row in rows:
            row["evaluator_total_path_count"] = 4

    value = run_case(tmp_path, resumed)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["fresh_five_episode_evaluator_state"] is False
    assert value["evaluator_total_path_count"] == 4


def test_clock_state_age_above_15_seconds_fails(tmp_path: Path) -> None:
    def stale_clock_state(rows, _ready):
        rows[6]["clock"]["updated_unix"] = rows[6]["sampled_unix"] - 15.1

    value = run_case(tmp_path, stale_clock_state)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["clock_state_age_seconds_le_15"] is False
    assert value["max_clock_state_age_seconds"] > 15


def test_more_than_30_seconds_without_physics_progress_fails(tmp_path: Path) -> None:
    def pause(rows, _ready):
        for index, row in enumerate(rows):
            row["clock"]["received_step_count"] = 0 if index <= 7 else index - 7
            row["clock"]["last_clock_ns"] = (
                1_000_000_000 if index <= 7 else 1_000_000_000 + (index - 7) * 10
            )

    value = run_case(tmp_path, pause)
    assert value["status"] == "FAIL"
    assert value["checks"]["physics_not_stagnant_over_30_seconds"] is False


def test_cutoff_covers_window_without_blocking_deadline_health_sample(
    tmp_path: Path,
) -> None:
    def omit_end(rows, _ready):
        rows.pop()

    value = run_case(tmp_path, omit_end)
    assert value["status"] == "PASS"
    assert value["returncode"] == 0
    assert value["last_sample_elapsed_seconds"] == 55
    assert value["cutoff_elapsed_seconds"] == 60
    assert value["checks"]["timeline_covers_configured_window"] is True


def test_first_sample_at_15_second_probe_bound_passes(tmp_path: Path) -> None:
    def start_at_probe_bound(rows, _ready):
        elapsed = [15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 58, 60]
        for row, value in zip(rows, elapsed):
            row["elapsed_seconds"] = value

    value = run_case(tmp_path, start_at_probe_bound)
    assert value["status"] == "PASS"
    assert value["returncode"] == 0
    assert value["first_sample_elapsed_seconds"] == 15
    assert value["max_start_offset_limit_seconds"] == 15
    assert value["checks"]["timeline_starts_near_zero"] is True
    assert value["checks"]["timeline_covers_configured_window"] is True


def test_first_sample_above_15_second_probe_bound_fails(tmp_path: Path) -> None:
    def start_after_probe_bound(rows, _ready):
        elapsed = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 58, 60]
        for row, value in zip(rows, elapsed):
            row["elapsed_seconds"] = value

    value = run_case(tmp_path, start_after_probe_bound)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["first_sample_elapsed_seconds"] == 16
    assert value["max_start_offset_limit_seconds"] == 15
    assert value["checks"]["timeline_starts_near_zero"] is False
    assert value["checks"]["timeline_sample_gap_bounded"] is True
    assert value["checks"]["timeline_covers_configured_window"] is True


def test_samples_clustered_away_from_start_fail(tmp_path: Path) -> None:
    def cluster_at_end(rows, _ready):
        for index, row in enumerate(rows):
            row["elapsed_seconds"] = 30 + (index * 5) // 2

    value = run_case(tmp_path, cluster_at_end)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["timeline_starts_near_zero"] is False
    assert value["checks"]["timeline_strictly_increasing"] is True
    assert value["checks"]["timeline_covers_configured_window"] is True


def test_excessive_sample_gap_fails(tmp_path: Path) -> None:
    def add_large_gap(rows, _ready):
        elapsed = [0, 5, 10, 15, 20, 25, 41, 46, 51, 56, 61, 66, 71]
        for row, value in zip(rows, elapsed):
            row["elapsed_seconds"] = value

    value = run_case(tmp_path, add_large_gap)
    assert value["status"] == "FAIL"
    assert value["returncode"] == 75
    assert value["checks"]["timeline_strictly_increasing"] is True
    assert value["checks"]["timeline_covers_configured_window"] is True
    assert value["checks"]["timeline_sample_gap_bounded"] is False
    assert value["max_sample_gap_seconds"] == 16
