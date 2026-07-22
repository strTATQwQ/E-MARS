from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_ros2"))

from internvla_ros2.fault_injection import (  # noqa: E402
    FAULT_KINDS,
    FAULT_PROFILE,
    FaultControlReader,
    build_fault_restart_session,
    load_fault_plan,
    validate_control_snapshot,
    validate_fault_plan,
    validate_fault_restart_session,
)
from internvla_ros2.identity import CHECKPOINT_REVISION, MODEL_REVISION  # noqa: E402
from internvla_ros2.protocol import GenerationBarrier, ProtocolError  # noqa: E402


CONFIG = ROOT / "configs/internnav_t5/fault_injection_minimal_v1.json"


def test_frozen_minimal_plan_has_exact_six_sim_time_events() -> None:
    plan = load_fault_plan(CONFIG)
    assert plan["profile"] == FAULT_PROFILE
    assert plan["soak_duration_sim_sec"] == 600
    assert plan["schedule_timebase"] == "sim_time"
    assert plan["wall_time_scope"] == "actuator_and_process_liveness_only"
    assert tuple(event["kind"] for event in plan["events"]) == FAULT_KINDS
    assert plan["required_invariants"] == {
        "safe_stop_before_destructive_restart": True,
        "sensor_and_action_freshness_timebase": "sim_time",
        "stale_response_discard": True,
        "wall_time_used_only_for_liveness": True,
        "final_residual_count": 0,
        "real_go2_eligible": False,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schedule_timebase="wall_time"),
        lambda value: value["events"].pop(),
        lambda value: value["events"][1].update(kind="unknown"),
        lambda value: value["events"][1].update(start_sim_offset_sec=65),
        lambda value: value["events"][4].update(duration_sim_sec=30),
    ],
)
def test_plan_rejects_scope_or_schedule_drift(mutation) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutation(value)
    with pytest.raises(ValueError):
        validate_fault_plan(value)


def test_control_reader_records_activation_consumption_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = tmp_path / "state.json"
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("INTERNNAV_T5_FAULT_INJECTION_PROFILE", FAULT_PROFILE)
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNNAV_T5_FAULT_CONTROL_PATH", str(control))
    monkeypatch.setenv("INTERNNAV_T5_FAULT_EVENT_PATH", str(events))
    reader = FaultControlReader("unit_component")
    active = {
        "schema_version": 1,
        "profile": FAULT_PROFILE,
        "lane": "a",
        "revision": 1,
        "observed_sim_ns": 10,
        "active": [
            {
                "event_id": "fi-unit-model-timeout",
                "kind": "model_request_timeout",
            }
        ],
    }
    control.write_text(json.dumps(active), encoding="utf-8")
    assert reader.consume_once("model_request_timeout") == "fi-unit-model-timeout"
    assert reader.consume_once("model_request_timeout") is None
    active.update(revision=2, observed_sim_ns=20, active=[])
    control.write_text(json.dumps(active), encoding="utf-8")
    reader.read()
    phases = [json.loads(line)["phase"] for line in events.read_text().splitlines()]
    assert phases == ["observed_active", "consumed", "observed_recovered"]
    active.update(revision=1)
    control.write_text(json.dumps(active), encoding="utf-8")
    with pytest.raises(RuntimeError, match="revision regressed"):
        reader.read()


def test_control_reader_records_recovery_after_component_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = tmp_path / "state.json"
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("INTERNNAV_T5_FAULT_INJECTION_PROFILE", FAULT_PROFILE)
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNNAV_T5_FAULT_CONTROL_PATH", str(control))
    monkeypatch.setenv("INTERNNAV_T5_FAULT_EVENT_PATH", str(events))
    state = {
        "schema_version": 1,
        "profile": FAULT_PROFILE,
        "lane": "a",
        "revision": 1,
        "observed_sim_ns": 10,
        "active": [
            {
                "event_id": "fi-unit-network-outage",
                "kind": "network_short_outage_and_recovery",
            }
        ],
    }
    control.write_text(json.dumps(state), encoding="utf-8")
    FaultControlReader("isaac_controller").read()

    restarted = FaultControlReader("isaac_controller")
    state.update(revision=2, observed_sim_ns=20, active=[])
    control.write_text(json.dumps(state), encoding="utf-8")
    restarted.read()

    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert [record["phase"] for record in records] == [
        "observed_active",
        "observed_recovered",
    ]
    assert records[-1]["event_id"] == "fi-unit-network-outage"


def test_control_snapshot_rejects_cross_lane_and_duplicate_kind() -> None:
    value = {
        "schema_version": 1,
        "profile": FAULT_PROFILE,
        "lane": "b",
        "revision": 1,
        "observed_sim_ns": 1,
        "active": [],
    }
    with pytest.raises(ValueError, match="lane mismatch"):
        validate_control_snapshot(value, expected_lane="a")
    value["lane"] = "a"
    value["active"] = [
        {"event_id": "fi-one", "kind": "episode_reset"},
        {"event_id": "fi-two", "kind": "episode_reset"},
    ]
    with pytest.raises(ValueError, match="duplicated"):
        validate_control_snapshot(value, expected_lane="a")


def test_restart_session_restores_one_exact_quiesced_generation() -> None:
    payload = build_fault_restart_session(
        {
            "status_code": 0,
            "status_message": "ready",
            "initialized": True,
            "lifecycle_state": 2,
            "episode_id": "a::episode-4",
            "reset_generation": 3,
            "last_sequence_id": 17,
            "model_revision": MODEL_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        },
        lane="a",
        event_id="fi-model-restart",
        action="model_service_restart",
        captured_unix=1.0,
    )
    barrier = GenerationBarrier()
    barrier.restore(
        payload["episode_id"],
        payload["reset_generation"],
        payload["last_sequence_id"],
    )
    assert (
        barrier.episode_id,
        barrier.reset_generation,
        barrier.last_sequence_id,
    ) == ("a::episode-4", 3, 17)
    with pytest.raises(ProtocolError, match="initialized barrier"):
        barrier.restore("a::episode-4", 3, 17)

    for field, bad in (
        ("lane", "b"),
        ("episode_id", "episode-without-lane"),
        ("last_sequence_id", -2),
        ("model_revision", "drift"),
    ):
        mutated = dict(payload)
        mutated[field] = bad
        with pytest.raises(ValueError, match="session identity mismatch"):
            validate_fault_restart_session(mutated, expected_lane="a")


def test_fault_overlay_is_opt_in_and_preserves_default_runtime(tmp_path: Path) -> None:
    source = ROOT / "scripts/internnav_go2_runtime.py"
    builder = ROOT / "scripts/build_t4_r3_sensor_runtime_overlay.py"
    default_output = tmp_path / "default.py"
    default_manifest = tmp_path / "default.json"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--source",
            str(source),
            "--output",
            str(default_output),
            "--manifest",
            str(default_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "FaultControlReader" not in default_output.read_text(encoding="utf-8")
    assert "fault_injection_profile" not in json.loads(default_manifest.read_text())

    output = tmp_path / "fault.py"
    manifest = tmp_path / "fault.json"
    environment = os.environ.copy()
    environment.update(
        INTERNNAV_T5_FAULT_INJECTION_PROFILE=FAULT_PROFILE,
        INTERNNAV_RUNTIME_POLICY="completion_sim",
        INTERNNAV_SIMULATION_TARGET="isaac",
        INTERNNAV_T5_LANE="a",
    )
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--source",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    generated = output.read_text(encoding="utf-8")
    compile(generated, str(output), "exec")
    assert 'FaultControlReader("isaac_controller")' in generated
    assert "if not t5_sensor_outage:" in generated
    assert "injected T5 lane network data-plane outage" in generated
    assert (
        "if state_only and not (\n"
        "                    t5_network_outage\n"
        "                    and isinstance(error, ConnectionError)\n"
        "                ):\n"
        "                    raise"
    ) in generated
    assert (
        "t5_network_outage\n"
        "                    and isinstance(error, ConnectionError)"
        not in default_output.read_text(encoding="utf-8")
    )
    assert json.loads(manifest.read_text())["fault_injection_profile"] == FAULT_PROFILE


def test_existing_fast_runner_owns_fault_director_and_bounded_restarts() -> None:
    fast = (ROOT / "coordination/run_t5_fast_lane_online.sh").read_text()
    dgx = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text()
    isaac = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    agent = (ROOT / "scripts/internvla_ipc_agent_client.py").read_text()
    model = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/model_node.py"
    ).read_text()
    wrapper = (ROOT / "scripts/run_t5_fault_injection.sh").read_text()
    assert 'test "$profile" = soak600' in fast
    assert "run_t5_fault_injection.py" in fast
    assert "--wall-liveness-timeout-sec \"$fault_wall_liveness_timeout\"" in fast
    assert dgx.index('fault_safe_stop "$event_id" || return 1') < dgx.index(
        'stop_group model "$old_pid"'
    )
    assert dgx.index('fault_safe_stop "$event_id" || return 1', dgx.index("restart_onboard_for_fault")) < dgx.index(
        'stop_group onboard "$old_pid"'
    )
    assert "residual_after_stop" in dgx
    assert "TERM" in dgx and "KILL" in dgx
    assert "INTERNNAV_T5_FAULT_CONTROL_PATH" in isaac
    assert "fault control remained active during cleanup" in isaac
    assert "fault_restart_maintenance_active" in isaac
    assert 'in {"model_service_restart", "dgx_ros_node_restart"}' in isaac
    assert "canary_fault_restart_deadline" in isaac
    assert "SECONDS <= canary_fault_restart_deadline" in isaac
    assert "canary_fault_restart_deadline=0" in isaac
    assert "simulation clock made no progress for 60 wall seconds" in isaac
    assert 'consume_once("episode_reset")' in agent
    assert 'consume_once("model_request_timeout")' in model
    assert "maintenance_wait_started" in agent
    assert "maintenance_wait_finished" in agent
    assert "INTERNVLA_EXPECTED_FAULT_SAFE_STOP" in agent
    assert "session_continuity_confirmed" in dgx
    assert "INTERNVLA_T5_MODEL_RESTART_SESSION_FILE" in dgx
    assert "INTERNVLA_T5_ONBOARD_RESTART_SESSION_FILE" in dgx
    assert (
        'start_onboard "$onboard_result_dir" "$onboard_log_path" "$session_path"'
        in dgx
    )
    assert "t5_fault_restart_session_probe.py" in dgx
    assert dgx.count('test ! -e "$launch_result_dir"') == 2
    assert 'mkdir -p "$launch_result_dir"' not in dgx
    assert "600-sim-second soak" in wrapper
    combined = fast + dgx + isaac + wrapper
    assert "iptables" not in combined
    assert "nft " not in combined


def test_fault_control_starts_inactive_before_either_lane_runtime() -> None:
    dgx = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text()
    isaac = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    for source, launch_marker in (
        (dgx, 'static_map_manifest="$(readlink -f "$static_map_manifest")"'),
        (isaac, 'pid_ledger="$result_root/pid_ledger.jsonl"'),
    ):
        initialization = source.index('"revision": 0')
        assert initialization < source.index(launch_marker)
        assert '"observed_sim_ns": 0' in source
        assert '"active": []' in source
        assert "os.replace(temporary, path)" in source


def test_fault_director_quiesces_restart_and_binds_expected_timeout() -> None:
    director = (ROOT / "scripts/run_t5_fault_injection.py").read_text()
    restart = director[
        director.index('elif kind in {"model_service_restart"') :
        director.index('elif kind == "episode_reset"')
    ]
    assert restart.index('target="both"') < restart.index(
        '"maintenance_wait_started"'
    ) < restart.index("_request_dgx_restart")
    assert restart.index("_request_dgx_restart") < restart.index(
        'self._write_state(None'
    ) < restart.index('"maintenance_wait_finished"')

    timeout = director[
        director.index('if kind == "model_request_timeout"') :
        director.index('elif kind in {"model_service_restart"')
    ]
    assert 'target="both"' in timeout
    assert '"expected_timeout_safe_stop"' in timeout


def test_fault_wrapper_reuses_fast_runner_frozen_candidate_allowlist() -> None:
    fast = (ROOT / "coordination/run_t5_fast_lane_online.sh").read_text()
    wrapper = (ROOT / "scripts/run_t5_fault_injection.sh").read_text()
    selector_case = (
        'case "$candidate_profile" in baseline|recovery_a) ;; '
        'a0|a1|a0+b0|a0+b1|a1+b0|a1+b1|a1+b0+c0|a1+b0+c1|a1+b0+c2|'
        'a1+b1+c0|a1+b1+c1|a1+b1+c2) ;; *) usage ;; esac'
    )
    assert fast.count(selector_case) == 1
    assert wrapper.count(selector_case) == 1
    assert "a1+b1+c1" in selector_case


def test_t5_isaac_overlay_import_path_is_bound_to_deployment(
    tmp_path: Path,
) -> None:
    distributed = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    assert 't5_isaac_python_package_root="$root/internvla_ros2"' in distributed
    assert (
        'export INTERNVLA_T5_ISAAC_PYTHON_PACKAGE_ROOT='
        '"$t5_isaac_python_package_root"'
    ) in distributed
    assert (
        'test -f "$t5_isaac_python_package_root/internvla_ros2/'
        'fault_injection.py"'
    ) in distributed

    output = tmp_path / "t5_phase.sh"
    manifest = tmp_path / "t5_phase.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_t5_isaac_remote_phase_overlay.py"),
            "--source",
            str(ROOT / "scripts/run_go2_continuous_phase.sh"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = output.read_text(encoding="utf-8")
    assert (
        'ISAAC_PYTHONPATH="$INTERNVLA_T5_ISAAC_PYTHON_PACKAGE_ROOT:'
        '$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT:'
        '$CONTROL_ROOT/internvla_go2_controller"'
    ) in generated
    assert (
        generated.index("$INTERNVLA_T5_ISAAC_PYTHON_PACKAGE_ROOT")
        < generated.index("$COMPAT_OVERLAY")
    )


def test_fault_model_restart_clean_loads_after_hf_token_is_erased() -> None:
    dgx = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text()
    start = dgx.index("start_model() {")
    model_start = dgx[start : dgx.index("\nstart_onboard() {", start)]
    guard_start = model_start.index('    if test -n "${hf_token:-}"; then')
    guard_end = model_start.index("    fi", guard_start) + len("    fi")
    actual_guard = model_start[guard_start:guard_end]
    if os.name != "posix":
        pytest.skip("runtime guard replay requires a native POSIX bash")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "set -eu\nunset hf_token HF_TOKEN\n"
            + actual_guard
            + '\ntest -z "${HF_TOKEN+x}"\n',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_fault_wrapper_summary_python_compiles_and_director_validates() -> None:
    wrapper = (ROOT / "scripts/run_t5_fault_injection.sh").read_text().splitlines()
    starts = [index for index, line in enumerate(wrapper) if "<<'PY'" in line]
    assert len(starts) == 1
    end = wrapper.index("PY", starts[0] + 1)
    ast.parse("\n".join(wrapper[starts[0] + 1 : end]) + "\n")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_t5_fault_injection.py"),
            "validate",
            "--config",
            str(CONFIG),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["profile"] == FAULT_PROFILE
