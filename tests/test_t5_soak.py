import ast
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_t5_soak.sh"
RUNNER = ROOT / "coordination" / "run_t5_fast_lane_online.sh"
ISAAC = ROOT / "scripts" / "run_t5_distributed_isaac.sh"
ANALYZER = ROOT / "scripts" / "analyze_t5_engineering_canary.py"


def wrapper_summary_program() -> str:
    lines = WRAPPER.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if "<<'PY'" in line)
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == "PY")
    return "\n".join(lines[start + 1 : end]) + "\n"


def test_soak_wrapper_is_bash_syntax_valid() -> None:
    completed = subprocess.run(
        ["bash", "-n", WRAPPER.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    ast.parse(wrapper_summary_program(), filename=str(WRAPPER))


def test_soak_wrapper_defaults_baseline_but_allows_recovery_a() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'candidate_profile="${6:-baseline}"' in text
    assert 'case "$candidate_profile" in baseline|recovery_a)' in text
    assert 'INTERNNAV_T5_CANDIDATE_PROFILE="$candidate_profile"' in text
    assert '"$lane" soak600 "$run_id" "$code_sha"' in text


def test_soak_wrapper_delegates_all_resource_ownership_to_fast_lane() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "with_resource_lease.sh" not in text
    assert "all-lanes" not in text
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'with_resource_lease.sh" "$resource_profile"' in runner
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK="$resource_profile"' in runner


def test_soak_profile_reuses_engineering_canary_for_600_sim_seconds() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert (
        "PROFILE: canary60 | soak600 | screen1 | screen3 | fixed5 | "
        "pilot-screen1 | final10"
    ) in runner
    assert (
        'case "$profile" in '
        "canary60|soak600|screen1|screen3|fixed5|pilot-screen1|final10)"
    ) in runner
    assert "soak600)" in runner
    assert "engineering_canary_sec=600" in runner
    assert "engineering_canary_timebase=sim" in runner
    assert 'soak600) run_timeout="${INTERNNAV_T5_FAST_SOAK_TIMEOUT_SEC:-7800}"' in runner
    assert "run_timeout >= 3000 && run_timeout <= 9000" in runner
    assert '{"canary60":60,"soak600":600}.get(profile)' in runner
    assert 'engineering_canary.get("configured_seconds")==expected_engineering_seconds' in runner
    assert 'engineering_canary.get("duration_timebase")==expected_engineering_timebase' in runner
    assert 'engineering_canary.get("episode_acceptance_claimed") is False' in runner
    assert 'x86_status.get("evaluation_completed_naturally") is False' in runner
    assert "engineering_canary_sec >= 30 && engineering_canary_sec <= 600" in ISAAC.read_text(
        encoding="utf-8"
    )
    assert '"bounded_duration_valid": 30 <= configured <= 600' in ANALYZER.read_text(
        encoding="utf-8"
    )
    assert '"sim_duration_reaches_configured"' in ANALYZER.read_text(encoding="utf-8")


def test_sim_time_soak_sample_accepts_fractional_elapsed_seconds() -> None:
    text = ISAAC.read_text(encoding="utf-8")
    assert '"elapsed_seconds": float(sys.argv[5])' in text
    assert '"elapsed_seconds": int(sys.argv[5])' not in text


def test_sim_time_soak_continues_after_frozen_five_and_accumulates_steps() -> None:
    text = ISAAC.read_text(encoding="utf-8")
    assert "start_evaluator_cycle()" in text
    assert "reap_evaluator_cycle()" in text
    assert "ensure_engineering_evaluator_running()" in text
    assert 'if wait "$gate_pid"; then' in text
    assert 'test "$evaluator_cycle_exit_code" = 0' in text
    assert 'test "$probe_rc" = 1' in text
    assert 'test ! -S "$ipc_alias/${ipc_token}_${socket_name}.sock"' in text
    assert "ordered_episode_manifest_soak_${attempt}.json" in text
    assert 'evaluator_cycle_index=$((evaluator_cycle_index + 1))' in text
    assert 'continuation_reset=1' in text
    assert 'INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET="$continuation_reset"' in text
    assert "canary_accumulated_sim_ns=0" in text
    assert (
        "canary_accumulated_sim_ns + current_clock_ns - canary_last_clock_ns"
        in text
    )
    assert "canary_sim_observed_ns=\"$canary_accumulated_sim_ns\"" in text
    assert "canary_last_clock_ns - canary_started_clock_ns" not in text
    assert 'INTERNVLA_T5_SIM_CLOCK_START_NS="$completed_clock_ns"' in text
    assert (
        'INTERNVLA_T5_CAMERA_SOURCE_SEQUENCE_START="$completed_received_step_count"'
        in text
    )
    assert 'completed evaluator clock did not quiesce after reap' in text
    assert 'updated >= not_before' in text
    assert 'canary_accumulated_sim_ns >= canary_target_clock_ns' in text
    assert 'INTERNNAV_T5_EVALUATOR_RESTART_TIMEOUT_SEC:-180' in text
    assert 'test "$engineering_canary_timebase" != sim || return 0' in text
    assert "evaluator_cycle_count" in text


def test_wall_canary_accepts_clean_early_frozen_five_completion() -> None:
    text = ISAAC.read_text(encoding="utf-8")
    function_start = text.index("ensure_engineering_evaluator_running() {")
    function_end = text.index("\n}\n\nstart_evaluator_cycle", function_start)
    function = text[function_start:function_end]
    assert 'leader_is_alive "$gate_pid" && return 0' in function
    assert "reap_evaluator_cycle" in function
    assert 'test "$engineering_canary_timebase" = sim || return 0' in function
    assert function.index("reap_evaluator_cycle") < function.index(
        'test "$engineering_canary_timebase" = sim || return 0'
    )
    assert 'test "$engineering_canary_timebase" = sim || return 1' not in function


def test_soak_summary_requires_runtime_cleanup_and_lease_pass() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    for check in (
        '"runtime_pass"',
        '"ready_window_600_pass"',
        '"runtime_cleanup_pass"',
        '"lease_release_pass"',
        '"final_summary_pass"',
    ):
        assert check in text
    assert '"profile": "soak600"' in text
    assert '"configured_seconds": 600' in text
    assert '"episode_acceptance_claimed": False' in text
    assert '"fault_injection": "not_requested"' in text
    assert 'engineering.get("duration_timebase") == "sim"' in text
    assert 'engineering.get("observed_sim_seconds", 0.0) >= 600.0' in text
    assert 'output.write_text(' in text


def test_invalid_soak_identity_stops_before_fast_runner() -> None:
    completed = subprocess.run(
        [
            "bash",
            WRAPPER.relative_to(ROOT).as_posix(),
            "a",
            "short",
            "a" * 40,
            "results/internnav_t5/d0-0-prepare-t5d0020260719t000000",
            "results/internnav_t5/fast-lane-a-soak600-short",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "ACQUIRED" not in completed.stdout + completed.stderr
