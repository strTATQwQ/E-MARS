from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_t4_run.py"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _result_dir(tmp_path: Path, *, collisions: int = 0, falls: int = 0) -> Path:
    _write_json(tmp_path / "result.json", {"val_unseen": {"SR": 1.0, "Count": 1}})
    _write_json(
        tmp_path / "active_summary.json",
        {"status": "FINISHED", "failure_count": 0},
    )
    _write_json(
        tmp_path / "controller_summary.json",
        {
            "status": "FINISHED",
            "measured_control_hz": 30.0,
            "nan_count": 0,
            "fall_count": falls,
            "physical_collision_count": collisions,
            "stale_motion_execution_count": 0,
            "identity_safe_stop_count": 0,
            "cmd_vel_quantization_count": 0,
            "direct_motion_bypass_count": 0,
            "map_source": "static_map",
            "pose_source": "ground_truth",
            "static_map_publish_count": 1,
            "static_map_selection_count": 1,
            "static_map_selections": [
                {"selection": "dataset_episode_id_no_runtime_pose"}
            ],
            "ground_truth_pose_used_for_nav": True,
        },
    )
    _write_json(tmp_path / "per_episode.json", {"completed_episode_count": 1})
    return tmp_path


def _validate(result_dir: Path, policy: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-B",
            str(VALIDATOR),
            str(result_dir),
            "--expected",
            "1",
            "--minimum-sr",
            "0.8",
            "--map-source",
            "static_map",
            "--pose-source",
            "ground_truth",
            "--runtime-policy",
            policy,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_completion_sim_records_physical_collision_as_quality_warning(
    tmp_path: Path,
) -> None:
    completed = _validate(_result_dir(tmp_path, collisions=2), "completion_sim")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["quality_status"] == "WARN"
    assert payload["required"]["physical_collision_policy"] == "metric_only_warn"
    assert payload["quality_warnings"] == [
        {"code": "PHYSICAL_COLLISION_METRIC_ONLY", "count": 2}
    ]


def test_strict_evidence_keeps_physical_collision_fatal(tmp_path: Path) -> None:
    completed = _validate(_result_dir(tmp_path, collisions=1), "strict_evidence")

    assert completed.returncode == 1
    payload = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["quality_status"] == "PASS"
    assert payload["required"]["physical_collision_policy"] == "must_be_zero"


def test_completion_sim_keeps_fall_fatal(tmp_path: Path) -> None:
    completed = _validate(_result_dir(tmp_path, falls=1), "completion_sim")

    assert completed.returncode == 1
    payload = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "fall_count" in payload["required"]["fatal_safety_zero"]


def test_generated_phase_overlay_forwards_runtime_policy_and_uses_ros_shim(
    tmp_path: Path,
) -> None:
    builder = ROOT / "scripts" / "build_t4_sensor_phase_overlay.py"
    source = builder.read_text(encoding="utf-8")
    assert '--runtime-policy "${INTERNNAV_RUNTIME_POLICY:-strict_evidence}"' in source
    output = tmp_path / "run_go2_continuous_phase.sh"
    manifest = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(builder),
            "--source",
            str(ROOT / "scripts" / "run_go2_continuous_phase.sh"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    generated = output.read_text(encoding="utf-8")
    assert "source /opt/ros/humble/setup.bash" not in generated
    assert 'source "$ROS_WS/install/setup.bash"' not in generated
    assert 'export PATH="$CONTROL_ROOT/runtime/t4_ros2_shim:$PATH"' in generated
    assert generated.count(
        'bash "$CONTROL_ROOT/scripts/t4_python_container_shim.sh"'
    ) == 5
    assert '"$CONTROL_ROOT/t4_completion/map/warn_relay.py"' in generated
    relay_start = generated.index(
        '"$CONTROL_ROOT/t4_completion/map/warn_relay.py"'
    )
    relay_end = generated.index("relay_pid=$!", relay_start)
    relay_launch = generated[relay_start:relay_end]
    assert "use_sim_time:=false" in relay_launch
    assert "use_sim_time:=true" not in relay_launch
    assert 'kill -0 "$relay_pid"' in generated
    assert 'stop_group "$relay_pid"' in generated
    assert '--validate-evidence "$RESULT_DIR/warn_only_relay.jsonl"' in generated
    assert '--summary "$RESULT_DIR/warn_only_relay_summary.json"' in generated
    assert generated.count(
        'bash "$CONTROL_ROOT/scripts/t4_cleanup_container_processes.sh"'
    ) == 2
    assert '-p "ablation_dataset_file:=${INTERNVLA_T4_ABLATION_DATASET_FILE:-}"' not in generated
    assert 'if test -n "${INTERNVLA_T4_ABLATION_DATASET_FILE:-}"; then' in generated
    assert '-p "ablation_dataset_file:=$INTERNVLA_T4_ABLATION_DATASET_FILE"' in generated
    for enum_parameter in (
        "ablation_variant_id",
        "ablation_config_sha256",
        "system_mode",
        "trajectory_mode",
        "termination_mode",
        "history_mode",
        "recovery_mode",
        "view_mode",
    ):
        matching_lines = [
            line for line in generated.splitlines() if f'-p "{enum_parameter}:=' in line
        ]
        assert len(matching_lines) == 1
        assert ":='${" in matching_lines[0]
        assert matching_lines[0].endswith("}'\"")
    runtime_builder = ROOT / "scripts" / "build_t4_sensor_runtime_overlay.py"
    runtime_output = tmp_path / "internnav_go2_runtime.py"
    runtime_manifest = tmp_path / "runtime_manifest.json"
    runtime_completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(runtime_builder),
            "--source",
            str(ROOT / "scripts" / "internnav_go2_runtime.py"),
            "--output",
            str(runtime_output),
            "--manifest",
            str(runtime_manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert runtime_completed.returncode == 0, runtime_completed.stderr
    runtime = runtime_output.read_text(encoding="utf-8")
    base_runtime = (ROOT / "scripts" / "internnav_go2_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "timeout_sec=0.25" in base_runtime
    assert 'INTERNVLA_T4_CONTROLLER_IPC_TIMEOUT_SEC", "5.0"' in runtime
    assert "if not 0.25 <= controller_ipc_timeout_sec <= 5.0:" in runtime
    assert "timeout_sec=controller_ipc_timeout_sec" in runtime
    assert '"depth_values": sample.reshape(-1).astype(float).tolist()' not in runtime
    assert '"depth_encoding": "uint16_mm_zlib_b64_v1"' in runtime
    assert 'if len(depth_encoded) > 480 * 1024:' in runtime
    assert '"INTERNVLA_GO2_CONTROLLER_ENDPOINT"' in runtime
    assert 'rgba.shape != (240, 320, 4)' not in runtime
    assert 'tuple(rgba.shape[:2]) != (240, 320)' in runtime
    assert 'or rgba.shape[2] < 3' in runtime
    assert 'INTERNNAV_T4_STEREO_DIAGNOSTIC' in runtime
    assert '"reason": "sensor_missing"' in runtime
    assert '"reason": "invalid_rgba"' in runtime
    runtime_manifest_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert (
        "dgx_onboard_tcp_controller_endpoint"
        in runtime_manifest_payload["changes"]
    )
    assert (
        "            if not state_only:\n"
        "                request.update(self._sample_depth())\n"
        "                request.update(self._sample_stereo())\n"
        "            try:\n"
    ) in runtime
    assert "bounded_uint16_depth_ipc" in runtime_manifest.read_text(encoding="utf-8")
    assert "bounded_completion_controller_ipc_timeout" in runtime_manifest.read_text(
        encoding="utf-8"
    )
    assert "state_only_sensor_fast_path" in runtime_manifest.read_text(
        encoding="utf-8"
    )
    assert (
        "bounded_stereo_sensor_diagnostic_and_rgb_rgba_acceptance"
        in runtime_manifest.read_text(encoding="utf-8")
    )
    phase_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert phase_manifest["bounded_completion_safe_cmd_relay"] is True
    gate = (ROOT / "scripts" / "run_t4_sensor_gate.sh").read_text(encoding="utf-8")
    assert (
        'install -m 700 "$CONTROL_ROOT/scripts/t4_ros2_container_shim.sh" '
        '"$SHIM_DIR/ros2"'
    ) in gate
    assert 'IPC_ALIAS="${INTERNVLA_T4_IPC_ALIAS_OVERRIDE:-/tmp/internnav_t4_ipc}"' in gate
    assert "/tmp/internnav_t5_a_ipc|/tmp/internnav_t5_b_ipc" in gate
    assert 'export INTERNVLA_ORACLE_SOCKET="$IPC_ALIAS/${IPC_TOKEN}_oracle.sock"' in gate
    assert 'export INTERNVLA_GO2_CONTROLLER_SOCKET="$IPC_ALIAS/${IPC_TOKEN}_controller.sock"' in gate
    assert 'test "$(printf %s "$socket_path" | wc -c)" -le "$IPC_PATH_MAX_BYTES"' in gate
    longest_socket = "/tmp/internnav_t4_ipc/" + ("f" * 16) + "_controller.sock"
    assert len(longest_socket.encode()) <= 100
    python_shim = (ROOT / "scripts" / "t4_python_container_shim.sh").read_text(
        encoding="utf-8"
    )
    assert "INTERNNAV_RUNTIME_POLICY" in python_shim
    assert "INTERNNAV_SIMULATION_TARGET" in python_shim
    assert "INTERNNAV_T4_MAP_COMPANION_ACK" in python_shim
    ros2_shim = (ROOT / "scripts" / "t4_ros2_container_shim.sh").read_text(
        encoding="utf-8"
    )
    assert (
        '-e "INTERNNAV_RUNTIME_POLICY=${INTERNNAV_RUNTIME_POLICY:-}"'
        in ros2_shim
    )
    assert (
        '-e "INTERNNAV_SIMULATION_TARGET=${INTERNNAV_SIMULATION_TARGET:-}"'
        in ros2_shim
    )
    cleanup_shim = (
        ROOT / "scripts" / "t4_cleanup_container_processes.sh"
    ).read_text(encoding="utf-8")
    assert '--control-root "$CONTROL_ROOT"' in cleanup_shim
    cleanup_source = (
        ROOT / "scripts" / "t4_container_process_cleanup.py"
    ).read_text(encoding="utf-8")
    assert (
        'f"{control_root.as_posix()}/t4_completion/map/warn_relay.py"'
        in cleanup_source
    )
    strict_phase = (ROOT / "scripts" / "run_go2_continuous_phase.sh").read_text(
        encoding="utf-8"
    )
    assert "t4_completion/map/warn_relay.py" not in strict_phase
