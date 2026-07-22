from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "coordination/run_t5_fast_lane_online.sh"
WRAPPER = ROOT / "coordination/run_t5_cuvslam_shadow_online.sh"
DGX = ROOT / "scripts/run_t5_dgx_lane.sh"
ISAAC = ROOT / "scripts/run_t5_distributed_isaac.sh"
SENSOR_GATE = ROOT / "scripts/run_t4_sensor_gate.sh"
ONBOARD = ROOT / "scripts/run_t4_dgx_onboard.sh"
SUPERVISOR = (
    ROOT
    / "internvla_t4_sensors/internvla_t4_sensors/odometry_supervisor_node.py"
)
ANALYZER = ROOT / "scripts/analyze_t5_cuvslam_shadow.py"
MATERIALIZER = ROOT / "scripts/materialize_t5_cuvslam_shadow_params.py"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_fast_runner_exposes_only_lane_a_screen3_cuvslam_shadow() -> None:
    text = FAST.read_text(encoding="utf-8")
    assert "INTERNNAV_T5_STRICT_EXTENSION_PROFILE: off | cuvslam_shadow" in text
    assert 'strict_extension_profile="${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}"' in text
    assert 'test "$lane" = a || usage' in text
    assert 'test "$profile" = screen3 || usage' in text
    assert text.count(
        'INTERNNAV_T5_STRICT_EXTENSION_PROFILE="$strict_extension_profile"'
    ) >= 3
    assert "run_t5_fast_lane_online.sh" in WRAPPER.read_text(encoding="utf-8")
    assert "a screen3" in WRAPPER.read_text(encoding="utf-8")


def test_runtime_wiring_is_explicit_and_gt_authority_remains_default() -> None:
    dgx = DGX.read_text(encoding="utf-8")
    isaac = ISAAC.read_text(encoding="utf-8")
    sensor_gate = SENSOR_GATE.read_text(encoding="utf-8")
    onboard = ONBOARD.read_text(encoding="utf-8")
    assert 'strict_extension_profile="${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}"' in dgx
    assert "cuvslam_shadow) cuvslam_mode=shadow" in dgx
    assert "internvla_t4_odometry_supervisor" in dgx
    assert "-p use_sim_time:=true -p launch_use_sim_time:=true" in dgx
    assert "-p shadow_only:=true" in dgx
    assert "materialize_t5_cuvslam_shadow_params.py" in dgx
    assert 'nav2_params="$cuvslam_profile_dir/nav2_params.yaml"' in dgx
    assert 'strict_extension_profile="${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}"' in isaac
    assert "export INTERNVLA_T4_ENABLE_STEREO_ODOMETRY=1" in isaac
    assert '"strict_extension_profile": os.environ[' in isaac
    assert '"stereo_odometry_enabled": os.environ[' in isaac
    assert '"${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}" = cuvslam_shadow' in sensor_gate
    assert 'test "$INTERNVLA_T4_POSE_SOURCE" = ground_truth' in sensor_gate
    assert '--params-file "$PARAMS" -p use_sim_time:=true' in onboard
    assert "-p map_source:=static_map -p pose_source:=ground_truth" in onboard


def test_shadow_supervisor_uses_sim_freshness_and_never_stops_navigation() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert 'self.declare_parameter("shadow_only", False)' in source
    assert 'self.declare_parameter("launch_use_sim_time", False)' in source
    assert 'command.append("use_sim_time:=true")' in source
    assert 'child_environment.pop("ROS_NAMESPACE", None)' in source
    assert 'self.create_subscription(Odometry, "/odom", self._on_truth, 20)' in source
    assert '"freshness_clock": "sim" if self.shadow_only else "wall"' in source
    assert "if not self.shadow_only:" in source
    assert '"truth_usage": "shadow_scoring_only_gt_remains_navigation_authority"' in source


def test_materializer_preserves_base_bridge_safety_and_adds_shadow_feed(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "internvla_go2_controller_bridge:\n"
        "  ros__parameters:\n"
        "    command_timeout_sec: 0.30\n"
        "    expected_control_hz: 50.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "effective.yaml"
    receipt = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--base",
            str(base),
            "--overlay",
            str(ROOT / "configs/internnav_t5/cuvslam_shadow.yaml"),
            "--output",
            str(output),
            "--receipt",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    text = output.read_text(encoding="utf-8")
    assert "command_timeout_sec: 0.3" in text
    assert "enable_stereo_feed: true" in text
    assert "pose_source: ground_truth" in text
    assert "/**/internvla_go2_controller_bridge:" not in text
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_value["cuvslam_has_navigation_authority"] is False
    assert (
        receipt_value["namespaced_bridge_parameter_delivery"]
        == "onboard_explicit_argv"
    )


def test_namespaced_bridge_stereo_switch_is_explicit_argv_not_nav2_wildcard() -> None:
    dgx = DGX.read_text(encoding="utf-8")
    onboard = ONBOARD.read_text(encoding="utf-8")
    assert 'INTERNVLA_T4_ENABLE_STEREO_FEED="$([[ $cuvslam_mode = shadow ]]' in dgx
    assert 'ENABLE_STEREO_FEED="${INTERNVLA_T4_ENABLE_STEREO_FEED:-0}"' in onboard
    assert '-p enable_stereo_feed:="$([[ $ENABLE_STEREO_FEED = 1 ]]' in onboard
    assert "base[OVERLAY_BRIDGE_KEY]" not in (
        ROOT / "scripts/materialize_t5_cuvslam_shadow_params.py"
    ).read_text(encoding="utf-8")


def _fixture(root: Path, duration_sec: float = 120.0) -> None:
    dgx = root / "remote/dgx"
    _write_rows(
        dgx / "cuvslam/odometry_supervisor_records.jsonl",
        [
            {"event": "start", "reset_generation": 0, "start_index": 2},
            {"event": "start", "reset_generation": 1, "start_index": 3},
        ],
    )
    first_duration = duration_sec / 2.0
    rows = []
    for generation, start_index, start_sec in ((0, 2, 10.0), (1, 3, 100.0)):
        for offset in (0.0, first_duration):
            stamp = int((start_sec + offset) * 1_000_000_000)
            rows.append(
                {
                    "reset_generation": generation,
                    "source_start_index": start_index,
                    "truth_stamp_ns": stamp,
                    "estimate_stamp_ns": stamp,
                    "truth_xy_yaw": [offset * 0.01, 0.0, 0.0],
                    "estimated_xy_yaw": [offset * 0.01, 0.0, 0.0],
                    "xy_error_m": 0.0,
                    "yaw_error_rad": 0.0,
                    "tf_age_sec": 0.1,
                }
            )
    _write_rows(dgx / "cuvslam/cuvslam_shadow_samples.jsonl", rows)
    _write(
        dgx / "onboard/controller_summary.json",
        {
            "pose_source": "ground_truth",
            "ground_truth_pose_used_for_nav": True,
            "external_odometry_nav_publish_count": 0,
            "stereo_pair_max_sync_error_ms": 0.0,
        },
    )
    _write(dgx / "lane_status.json", {"lane": "a", "status": "PASS"})
    _write(
        root / "remote/x86/isaac_contract.json",
        {
            "strict_extension_profile": "cuvslam_shadow",
            "stereo_odometry_enabled": True,
        },
    )


def test_analyzer_accepts_complete_shadow_without_claiming_takeover(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    output = tmp_path / "metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--result-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPONENT_SMOKE_PASS"
    assert payload["sim_duration_sec"] == 120.0
    assert payload["reset_count"] == 2
    assert payload["navigation_pose_authority"] == "ground_truth"
    assert payload["nav_candidate_pass"] is False


def test_analyzer_fails_closed_before_120_sim_seconds(tmp_path: Path) -> None:
    _fixture(tmp_path, duration_sec=119.0)
    output = tmp_path / "metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--result-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["checks"]["minimum_120_sim_seconds"] is False
