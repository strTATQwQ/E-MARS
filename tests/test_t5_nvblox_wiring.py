from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "internvla_t4_sensors"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from internvla_t4_sensors.t5_nvblox_runtime import (  # noqa: E402
    NvbloxReadinessState,
    T5NvbloxContractError,
    materialize_profile,
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _params(document: dict, name: str) -> dict:
    value = document[name]
    if name in {"local_costmap", "global_costmap"}:
        value = value[name]
    return value["ros__parameters"]


def test_shadow_materialization_starts_real_fused_node_without_nav_authority(
    tmp_path: Path,
) -> None:
    output = tmp_path / "shadow"
    payload = materialize_profile(ROOT, output, "shadow")
    assert payload["starts_real_nvblox_node"] is True
    assert payload["feeds_navigation"] is False
    assert payload["navigation"] == {
        "static_global_map_retained": True,
        "lidar_voxel_local_retained": True,
        "nvblox_global_layer_present": False,
        "nvblox_local_layer_loaded": False,
        "nvblox_local_layer_initially_enabled": False,
    }
    nav = _load(output / "nav2_params.yaml")
    assert nav == _load(ROOT / "configs/internnav_t5/nav2_static_lidar.yaml")
    assert _params(nav, "local_costmap")["plugins"] == [
        "voxel_layer",
        "inflation_layer",
    ]
    nvblox = _load(output / "nvblox_params.yaml")["/**"]["ros__parameters"]
    assert nvblox["use_sim_time"] is True
    assert nvblox["use_depth"] is True
    assert nvblox["use_lidar"] is True
    assert nvblox["integrate_depth_rate_hz"] > 0
    assert nvblox["integrate_lidar_rate_hz"] > 0


def test_active_local_preserves_static_global_and_lidar_voxel(tmp_path: Path) -> None:
    output = tmp_path / "active"
    payload = materialize_profile(
        ROOT, output, "active_local_gt", lane_namespace="/t5/lane_a"
    )
    base = _load(ROOT / "configs/internnav_t5/nav2_static_lidar.yaml")
    active = _load(output / "nav2_params.yaml")
    assert _params(active, "global_costmap") == _params(base, "global_costmap")
    assert _params(active, "local_costmap")["voxel_layer"] == _params(
        base, "local_costmap"
    )["voxel_layer"]
    local = _params(active, "local_costmap")
    assert local["plugins"] == ["voxel_layer", "nvblox_layer", "inflation_layer"]
    assert local["nvblox_layer"] == {
        "plugin": "nvblox::nav2::NvbloxCostmapLayer",
        "enabled": False,
        "nav2_costmap_global_frame": "odom",
        "nvblox_map_slice_topic": "/t5/lane_a/nvblox_node/static_map_slice",
        "convert_to_binary_costmap": True,
    }
    assert payload["lane_namespace"] == "/t5/lane_a"
    assert payload["slice_topic_absolute"] == (
        "/t5/lane_a/nvblox_node/static_map_slice"
    )
    assert payload["pose_source"] == "isaac_ground_truth"
    assert payload["claims"]["lidar_only_counts_as_fused"] is False


def test_materializer_is_fresh_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    materialize_profile(
        ROOT, first, "active_local_gt", lane_namespace="/t5/lane_a"
    )
    materialize_profile(
        ROOT, second, "active_local_gt", lane_namespace="/t5/lane_a"
    )
    assert {path.name for path in first.iterdir()} == {
        "nav2_params.yaml",
        "nvblox_params.yaml",
        "runtime_contract.json",
    }
    for name in ("nav2_params.yaml", "nvblox_params.yaml", "runtime_contract.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(FileExistsError):
        materialize_profile(ROOT, first, "shadow")
    with pytest.raises(T5NvbloxContractError):
        materialize_profile(ROOT, tmp_path / "bad", "real_go2")


def test_active_overlay_composes_after_selected_candidate_nav2(tmp_path: Path) -> None:
    candidate = _load(ROOT / "configs/internnav_t5/nav2_static_lidar.yaml")
    candidate["controller_server"]["ros__parameters"]["progress_checker"][
        "movement_time_allowance"
    ] = 77.0
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
    )
    output = tmp_path / "active-candidate"
    materialize_profile(
        ROOT,
        output,
        "active_local_gt",
        candidate_path,
        "/t5/lane_a",
    )
    rendered = _load(output / "nav2_params.yaml")
    assert (
        rendered["controller_server"]["ros__parameters"]["progress_checker"][
            "movement_time_allowance"
        ]
        == 77.0
    )
    assert _params(rendered, "local_costmap")["nvblox_layer"]["enabled"] is False


def test_active_materialization_requires_and_binds_exact_lane_namespace(
    tmp_path: Path,
) -> None:
    with pytest.raises(T5NvbloxContractError, match="requires lane_namespace"):
        materialize_profile(ROOT, tmp_path / "missing", "active_local_gt")
    with pytest.raises(T5NvbloxContractError, match="lane_namespace must be"):
        materialize_profile(
            ROOT,
            tmp_path / "invalid",
            "active_local_gt",
            lane_namespace="/t5/lane_c",
        )
    output = tmp_path / "lane-b"
    payload = materialize_profile(
        ROOT, output, "active_local_gt", lane_namespace="/t5/lane_b"
    )
    rendered = _load(output / "nav2_params.yaml")
    assert _params(rendered, "local_costmap")["nvblox_layer"][
        "nvblox_map_slice_topic"
    ] == "/t5/lane_b/nvblox_node/static_map_slice"
    assert payload["slice_topic_absolute"] == (
        "/t5/lane_b/nvblox_node/static_map_slice"
    )


def _feed_fused(state: NvbloxReadinessState, stamp_ns: int) -> None:
    for name in ("depth", "depth_camera_info", "lidar", "ground_truth_odometry"):
        state.observe_sensor(name, stamp_ns)


def test_active_gate_requires_fused_inputs_classes_and_ten_new_slices() -> None:
    state = NvbloxReadinessState("active_local_gt")
    state.reset(0, 1_000_000_000)
    _feed_fused(state, 2_000_000_000)
    for index in range(9):
        stamp = 2_000_000_000 + index + 1
        assert state.observe_slice(stamp, stamp, 1, 1, 1) == "OBSERVED"
    stamp = 2_000_000_010
    assert state.observe_slice(stamp, stamp, 1, 1, 1) == "ENABLE_LAYER"
    assert state.layer_enabled is False
    state.mark_layer_enabled()
    assert state.ready(stamp) is True
    assert state.layer_enabled is True


def test_lidar_only_never_arms_and_stale_active_layer_requests_fallback() -> None:
    state = NvbloxReadinessState("active_local_gt")
    state.reset(0, 1)
    state.observe_sensor("lidar", 10)
    state.observe_sensor("ground_truth_odometry", 10)
    for stamp in range(11, 21):
        state.observe_slice(stamp, stamp, 1, 1, 1)
    assert state.ready(21) is False
    _feed_fused(state, 1_000_000_000)
    for stamp in range(1_000_000_001, 1_000_000_011):
        state.observe_slice(stamp, stamp, 1, 1, 1)
    state.mark_layer_enabled()
    assert state.tick(4_000_000_001, child_alive=True) == "DISABLE_LAYER"
    assert state.layer_enabled is True  # authority remains assumed until Nav2 ACK
    state.mark_layer_disabled("fused_input_or_slice_stale")
    assert state.layer_enabled is False
    assert state.fallback_reason == "fused_input_or_slice_stale"


def test_reset_clears_authority_and_rejects_old_slices() -> None:
    state = NvbloxReadinessState("active_local_gt")
    state.reset(0, 100)
    _feed_fused(state, 200)
    for stamp in range(201, 211):
        state.observe_slice(stamp, stamp, 1, 1, 1)
    state.mark_layer_enabled()
    assert state.reset(1, 1_000) is True
    assert state.layer_enabled is False
    assert state.consecutive_valid_slices == 0
    assert state.observe_slice(999, 1_001, 1, 1, 1) == "REJECTED_SLICE"


def test_shadow_readiness_expires_and_requires_ten_new_slices() -> None:
    state = NvbloxReadinessState("shadow")
    state.reset(0, 1)
    _feed_fused(state, 1_000_000_000)
    for stamp in range(1_000_000_001, 1_000_000_011):
        action = state.observe_slice(stamp, stamp, 1, 1, 1)
    assert action == "SHADOW_READY"
    assert state.tick(4_000_000_001, child_alive=True) == "DEGRADED"
    assert state.ready(4_000_000_001) is False
    assert state.consecutive_valid_slices == 0


def test_duplicate_slice_neither_counts_nor_breaks_unique_valid_run() -> None:
    state = NvbloxReadinessState("shadow")
    state.reset(0, 1)
    _feed_fused(state, 1_000_000_000)
    first = 1_000_000_001
    assert state.observe_slice(first, first, 1, 1, 1) == "OBSERVED"
    assert state.observe_slice(first, first, 1, 1, 1) == "DUPLICATE_SLICE"
    assert state.consecutive_valid_slices == 1
    for stamp in range(first + 1, first + 9):
        assert state.observe_slice(stamp, stamp, 1, 1, 1) == "OBSERVED"
        assert state.observe_slice(stamp, stamp, 1, 1, 1) == "DUPLICATE_SLICE"
    final = first + 9
    assert state.observe_slice(final, final, 1, 1, 1) == "SHADOW_READY"
    assert state.consecutive_valid_slices == 10


def test_cli_runs_offline_and_runner_hook_is_opt_in(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/materialize_t5_nvblox_profile.py"),
            "--mode",
            "shadow",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "READY"
    runner = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text(encoding="utf-8")
    assert 'nvblox_mode="${INTERNNAV_T5_NVBLOX_MODE:-off}"' in runner
    assert "off|shadow|active_local_gt" in runner
    assert "materialize_t5_nvblox_profile.py" in runner
    assert '--lane-namespace "$lane_namespace"' in runner
    assert "internvla_t5_nvblox_supervisor" in runner
    assert 'INTERNVLA_T4_ENABLE_D435I="$enable_d435i"' in runner
    assert "stop_group nvblox" in runner
    t4_runner = (ROOT / "scripts/run_t4_sensor_gate.sh").read_text(encoding="utf-8")
    assert "INTERNNAV_T5_NVBLOX_MODE" not in t4_runner
    fast_runner = (
        ROOT / "coordination/run_t5_fast_lane_online.sh"
    ).read_text(encoding="utf-8")
    assert 'INTERNNAV_T5_NVBLOX_MODE="$nvblox_mode"' in fast_runner
    assert 'INTERNNAV_T5_RUN_MODE="$run_mode"' in fast_runner
    assert 'case "$profile" in screen3|fixed5)' in fast_runner
    assert '(nvblox_mode == "shadow" and profile in {"screen3", "fixed5"})' in fast_runner
    assert '(nvblox_mode == "active_local_gt" and profile == "screen3")' in fast_runner
    assert '"$lane" "$run_mode" "$result" "$map"' in fast_runner
    assert '"$lane" "$run_mode" "$result" "$dataset"' in fast_runner


def test_supervisor_has_no_motion_authority_and_uses_lane_relative_slice() -> None:
    source = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/t5_nvblox_supervisor_node.py"
    ).read_text(encoding="utf-8")
    assert 'namespace + "/nvblox_node/static_map_slice"' in source
    assert 'namespace + "/local_costmap/costmap_raw"' in source
    assert 'namespace + "/local_costmap/get_costmap"' in source
    assert '"get_costmap_service"' in source
    assert '"nvblox_backed_costmap_update"' in source
    assert '"ready_window_sample"' in source
    assert '"/go2/d435i/depth/image_rect"' in source
    assert '"/go2/lidar/points"' in source
    assert '"/odom"' in source
    assert "cmd_vel" not in source
    assert "terminal_stop" not in source
    assert '"nvblox_layer.enabled"' in source
    assert (
        '"local_costmap_node", "local_costmap/local_costmap"' in source
    )
    assert '"successful" in completed.stdout.lower()' in source
    assert 'pose_source != "isaac_ground_truth"' in source
    assert "start_new_session=False" in source
    assert "child.terminate()" in source
    assert "child.kill()" in source


def test_analyzer_reports_component_readiness_without_nav_overclaim(
    tmp_path: Path,
) -> None:
    result = tmp_path / "runtime"
    result.mkdir()
    events = [
        {"event": "nvblox_child_start"},
        {
            "event": "nvblox_layer_set",
            "enabled": True,
            "success": True,
        },
    ]
    (result / "nvblox_runtime.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    (result / "nvblox_slice_classes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "unknown_count": 1,
                    "free_positive_count": 1,
                    "occupied_nonpositive_count": 1,
                    "consecutive_valid_slices": index,
                }
            )
            + "\n"
            for index in range(1, 11)
        ),
        encoding="utf-8",
    )
    (result / "nvblox_ready.json").write_text(
        json.dumps(
            {
                "status": "ACTIVE_LOCAL_READY",
                "real_nvblox_node": True,
                "depth_and_lidar_required": True,
            }
        ),
        encoding="utf-8",
    )
    evidence = [
        {
            "event": "ready_window_sample",
            "generation": 0,
            "ready": index != 0,
        }
        for index in range(20)
    ]
    evidence.extend(
        {
            "event": "nvblox_backed_costmap_update",
            "generation": 0,
            "consecutive_updates": index,
        }
        for index in range(1, 11)
    )
    (result / "nvblox_costmap_evidence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evidence), encoding="utf-8"
    )
    output = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_t5_nvblox_run.py"),
            "--result-dir",
            str(result),
            "--mode",
            "active_local_gt",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "ACTIVE_LOCAL_COMPONENT_READY"
    assert summary["online_navigation_acceptance"] == "NOT_EVALUATED"
    assert summary["nav_candidate_pass"] is False
    assert summary["costmap_ready_window_fraction"] == 0.95
    assert summary["maximum_consecutive_costmap_layer_updates"] == 10


def test_active_analyzer_fails_closed_without_direct_costmap_evidence(
    tmp_path: Path,
) -> None:
    result = tmp_path / "runtime"
    result.mkdir()
    (result / "nvblox_runtime.jsonl").write_text(
        json.dumps({"event": "nvblox_child_start"}) + "\n"
        + json.dumps(
            {"event": "nvblox_layer_set", "enabled": True, "success": True}
        )
        + "\n",
        encoding="utf-8",
    )
    (result / "nvblox_slice_classes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "unknown_count": 1,
                    "free_positive_count": 1,
                    "occupied_nonpositive_count": 1,
                    "consecutive_valid_slices": index,
                }
            )
            + "\n"
            for index in range(1, 11)
        ),
        encoding="utf-8",
    )
    (result / "nvblox_ready.json").write_text(
        json.dumps(
            {
                "status": "ACTIVE_LOCAL_READY",
                "real_nvblox_node": True,
                "depth_and_lidar_required": True,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_t5_nvblox_run.py"),
            "--result-dir",
            str(result),
            "--mode",
            "active_local_gt",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["checks"]["ready_window_sampled"] is False
    assert summary["checks"]["ten_consecutive_costmap_layer_updates"] is False
