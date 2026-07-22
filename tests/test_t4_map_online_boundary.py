from __future__ import annotations

from pathlib import Path

import pytest

from t4_completion.map.companion import require_managed_boundary


ROOT = Path(__file__).resolve().parents[1]


def test_online_wrapper_fails_closed_until_shared_hook_exists() -> None:
    text = (ROOT / "scripts/t4_map_online_smoke.sh").read_text(encoding="utf-8")
    assert "scripts/with_resource_lease.sh" in text
    assert "INTERNNAV_SENSOR_SESSION_LEASE_ACK" in text
    assert "INTERNNAV_T4_MAP_COMPANION_MODULE" in text
    assert "completion_sim_map" in text
    assert "exit 78" in text
    assert "ssh " not in text.lower()
    assert "docker exec" not in text.lower()
    assert "real_go2" not in text


def test_companion_requires_exact_sensor_session_result_and_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    monkeypatch.delenv("INTERNNAV_T4_MAP_COMPANION_ACK", raising=False)
    monkeypatch.setenv("RESULT_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        require_managed_boundary(tmp_path)


def test_companion_children_remain_in_shared_process_group() -> None:
    text = (ROOT / "t4_completion/map/companion.py").read_text(encoding="utf-8")
    assert "start_new_session=False" in text
    assert "shared supervisor owns this" in text
    assert "os.getpgrp()" in text
    assert "SMOKE_DURATION_SEC = 60.0" in text
    assert "runtime_validation.json" in text
    assert "continues_until_shared_stop" in text
    assert '"/internvla/stop"' in text
    assert 'line.strip() == "data: true"' in text
    assert "os.link(temporary, path)" in text
    assert 'environment["ROS2CLI_NO_DAEMON"] = "1"' in text
    assert '"--qos-reliability", "best_effort"' in text
    assert '"--qos-durability", "transient_local"' in text


def test_shared_session_owns_map_profile_and_companion() -> None:
    session = (ROOT / "sensor_runtime/session.py").read_text(encoding="utf-8")
    inner = (ROOT / "sensor_runtime/ros_inner_supervisor.py").read_text(
        encoding="utf-8"
    )
    sidecar = (ROOT / "sensor_runtime/ros_sidecar.py").read_text(encoding="utf-8")
    local_runner = (ROOT / "coordination/run_01r_online.sh").read_text(
        encoding="utf-8"
    )
    remote_runner = (ROOT / "coordination/remote_01r_deploy_and_run.sh").read_text(
        encoding="utf-8"
    )
    container_runner = (ROOT / "sensor_runtime/run_ros_container.sh").read_text(
        encoding="utf-8"
    )
    archive_validator = (
        ROOT / "coordination/validate_payload_archive.py"
    ).read_text(encoding="utf-8")
    assert 'return "10" if profile == "completion_sim_map" else "01R"' in session
    assert '"t4_map_companion"' in inner
    assert '"/internvla/stop"' in sidecar
    assert 'self.create_publisher(Bool, "/internvla/stop", sim_estop_qos)' in sidecar
    assert 'self.create_publisher(TFMessage, "/tf", tf_qos)' in sidecar
    assert '"tf_dynamic": qos_snapshot(tf_qos)' in sidecar
    assert "reliability=ReliabilityPolicy.RELIABLE" in sidecar
    assert "results/parallel/t4_map/online-smoke-10" in local_runner
    assert "t4_completion configs" in local_runner
    assert "scripts/t4_map_online_smoke.sh" in remote_runner
    assert 'include_map=profile == "completion_sim_map"' in remote_runner
    assert 'INTERNNAV_SESSION_PROFILE=${INTERNNAV_SESSION_PROFILE:-}' in container_runner
    assert '"INTERNNAV_SESSION_PROFILE="' in session
    assert "environment.get('INTERNNAV_SESSION_PROFILE', '')" in session
    assert (
        'INTERNNAV_T4_MAP_COMPANION_MODULE='
        '${INTERNNAV_T4_MAP_COMPANION_MODULE:-}' in container_runner
    )
    assert 'INTERNNAV_SIMULATION_TARGET=${INTERNNAV_SIMULATION_TARGET:-}' in container_runner
    assert '"t4_completion"' in archive_validator
    assert '"configs"' in archive_validator
