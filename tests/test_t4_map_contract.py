from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from t4_completion.map.contract import (
    DEFAULT_CONFIG_DIR,
    ContractError,
    audit_runtime_isolation,
    core_nav_projection,
    load_and_validate,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_configs(tmp_path: Path) -> Path:
    target = tmp_path / "config copy with spaces"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


def test_default_config_is_static_global_filtered_lidar_local() -> None:
    configs = load_and_validate()
    profile = configs.profile
    assert profile["runtime_policy"] == "completion_sim"
    assert profile["target"] == "isaac_simulation_only"
    assert profile["default_nvblox_mode"] == "shadow"
    assert profile["core_nav"]["global"] == "static_map"
    assert profile["core_nav"]["local"] == "lidar_voxel"
    assert profile["topics"]["lidar"] == "/go2/lidar/points_base"

    local = configs.nav2_shadow["local_costmap"]["local_costmap"]["ros__parameters"]
    global_costmap = configs.nav2_shadow["global_costmap"]["global_costmap"][
        "ros__parameters"
    ]
    assert local["plugins"] == ["voxel_layer", "inflation_layer"]
    assert local["voxel_layer"]["lidar"]["topic"] == "/go2/lidar/points_base"
    assert local["voxel_layer"]["lidar"]["clearing"] is True
    assert local["voxel_layer"]["lidar"]["marking"] is True
    assert global_costmap["plugins"] == ["static_layer", "inflation_layer"]
    assert global_costmap["static_layer"]["map_topic"] == "/map"
    docking = configs.nav2_shadow["docking_server"]["ros__parameters"]
    assert docking["dock_plugins"] == ["simple_charging_dock"]
    assert docking["simple_charging_dock"]["use_battery_status"] is False
    assert docking["simple_charging_dock"]["use_stall_detection"] is False
    assert docking["controller"]["v_linear_max"] == 0.15
    assert docking["controller"]["use_collision_detection"] is True


def test_completion_thresholds_and_warn_only_route_are_frozen() -> None:
    configs = load_and_validate()
    profile = configs.profile
    assert profile["timing"] == {
        "use_sim_time": True,
        "source_timeout_sec": 5.0,
        "observation_persistence_sec": 5.0,
        "costmap_current_timeout_sec": 5.0,
        "transform_tolerance_sec": 2.5,
    }
    assert profile["safety"]["robot_radius_m"] == 0.30
    assert profile["safety"]["footprint_padding_m"] == 0.02
    assert profile["safety"]["inflation_radius_m"] == 0.40
    assert profile["safety"]["collision_monitor_mode"] == "warn_only"
    assert profile["safety"]["collision_monitor_in_navigation_command_path"] is False
    monitor = configs.nav2_shadow["collision_monitor"]["ros__parameters"]
    assert monitor["cmd_vel_in_topic"] == "/cmd_vel_nav"
    assert monitor["cmd_vel_out_topic"] == (
        "/completion_sim/collision_monitor/warn_only_cmd_vel"
    )
    assert monitor["cmd_vel_out_topic"] != "/cmd_vel_safe"


def test_active_overlay_retains_exact_fallback_core() -> None:
    configs = load_and_validate()
    active = configs.nav2_for_mode("active")
    active_local = active["local_costmap"]["local_costmap"]["ros__parameters"]
    assert active_local["plugins"] == [
        "voxel_layer",
        "nvblox_layer",
        "inflation_layer",
    ]
    assert core_nav_projection(active) == core_nav_projection(configs.nav2_shadow)
    assert configs.nvblox_shadow["feeds_navigation_costmap"] is False
    assert configs.nvblox_active["failure_fallback"] == "shadow"


def test_strict_evidence_remains_exact_and_enforcing() -> None:
    audit = audit_runtime_isolation(ROOT)
    assert audit["strict_evidence_unchanged"] is True
    strict = yaml.safe_load(
        (ROOT / "configs/runtime/strict_evidence.yaml").read_text(encoding="utf-8")
    )
    assert strict["recorder"]["mode"] == "fatal_exact_batch"
    assert strict["navigation"]["source_timeout_sec"] == 0.35
    assert strict["navigation"]["collision_monitor"] == "enforce"
    assert strict["tf"] == {"lookup": "exact", "transform_tolerance_sec": 0.0}
    assert strict["mapping"] == {"nvblox": "active_required", "global_map": "nvblox"}


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "unknown", "hardware", "boolean_radius"),
)
def test_invalid_profile_is_rejected(tmp_path: Path, mutation: str) -> None:
    config_dir = _copy_configs(tmp_path)
    path = config_dir / "profile.yaml"
    text = path.read_text(encoding="utf-8")
    if mutation == "duplicate":
        text += "\nruntime_policy: completion_sim\n"
    elif mutation == "unknown":
        text += "\nunknown_policy: true\n"
    elif mutation == "hardware":
        text = text.replace("target: isaac_simulation_only", "target: real_go2")
    else:
        text = text.replace("robot_radius_m: 0.30", "robot_radius_m: true")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ContractError):
        load_and_validate(config_dir)
