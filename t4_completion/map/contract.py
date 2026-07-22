"""Fail-closed configuration contract for completion_sim mapping.

This module validates only the explicitly isolated completion profile.  It
does not read or alter the strict_evidence configuration and it rejects every
target except Isaac simulation.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "configs" / "completion_sim" / "map"

PROFILE_FILES = {
    "profile": "profile.yaml",
    "nav2_shadow": "nav2_static_lidar.yaml",
    "nav2_active": "nav2_nvblox_active.yaml",
    "nvblox_shadow": "nvblox_shadow.yaml",
    "nvblox_active": "nvblox_active.yaml",
    "smoke_map": "smoke_static_map.yaml",
}


class ContractError(ValueError):
    """Raised when an owned mapping configuration violates a frozen rule."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> MutableMapping[Any, Any]:
    loader.flatten_mapping(node)
    value: MutableMapping[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in value:
            raise ContractError("duplicate YAML key: {!r}".format(key))
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class LoadedConfigs:
    config_dir: Path
    profile: Mapping[str, Any]
    nav2_shadow: Mapping[str, Any]
    nav2_active: Mapping[str, Any]
    nvblox_shadow: Mapping[str, Any]
    nvblox_active: Mapping[str, Any]
    smoke_map: Mapping[str, Any]

    def nav2_for_mode(self, mode: str) -> Mapping[str, Any]:
        if mode == "shadow":
            return self.nav2_shadow
        if mode == "active":
            return deep_merge(self.nav2_shadow, self.nav2_active)
        raise ContractError("nvblox mode must be shadow or active")

    def nvblox_for_mode(self, mode: str) -> Mapping[str, Any]:
        if mode == "shadow":
            return self.nvblox_shadow
        if mode == "active":
            return self.nvblox_active
        raise ContractError("nvblox mode must be shadow or active")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("{} must be a mapping".format(label))
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise ContractError(
            "{} keys mismatch: missing={} unknown={}".format(
                label,
                sorted(expected_set - actual),
                sorted(actual - expected_set),
            )
        )


def _number(value: Any, expected: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("{} must be numeric".format(label))
    if not math.isfinite(float(value)) or not math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ContractError("{} must equal {}".format(label, expected))


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError("required config is unreadable: {}".format(path)) from exc
    if "\t" in raw:
        raise ContractError("YAML must not contain tab characters: {}".format(path))
    try:
        value = yaml.load(raw, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, ContractError) as exc:
        raise ContractError("invalid YAML {}: {}".format(path, exc)) from exc
    return _mapping(value, path.name)


def _resolve_config_dir(config_dir: Path) -> Path:
    resolved = config_dir.resolve()
    if not resolved.is_dir():
        raise ContractError("config directory is missing: {}".format(resolved))
    return resolved


def _params(document: Mapping[str, Any], node_name: str) -> Mapping[str, Any]:
    node = _mapping(document.get(node_name), node_name)
    if node_name in {"local_costmap", "global_costmap"}:
        node = _mapping(node.get(node_name), "{}.{}".format(node_name, node_name))
    return _mapping(node.get("ros__parameters"), "{}.ros__parameters".format(node_name))


def _validate_profile(profile: Mapping[str, Any]) -> None:
    _exact_keys(
        profile,
        {
            "schema_version",
            "runtime_policy",
            "target",
            "default_nvblox_mode",
            "core_nav",
            "topics",
            "frames",
            "safety",
            "timing",
            "nvblox",
        },
        "profile",
    )
    if profile["schema_version"] != 1:
        raise ContractError("profile.schema_version must equal 1")
    if profile["runtime_policy"] != "completion_sim":
        raise ContractError("map profile must be completion_sim")
    if profile["target"] != "isaac_simulation_only":
        raise ContractError("completion map target must be isaac_simulation_only")
    if profile["default_nvblox_mode"] != "shadow":
        raise ContractError("default Nvblox mode must remain shadow")

    core = _mapping(profile["core_nav"], "profile.core_nav")
    if core.get("global") != "static_map" or core.get("local") != "lidar_voxel":
        raise ContractError("default core must be static global plus LiDAR local")
    topics = _mapping(profile["topics"], "profile.topics")
    if topics.get("lidar") != "/go2/lidar/points_base":
        raise ContractError("local costmap must consume filtered base-frame LiDAR")
    if topics.get("static_map") != "/map":
        raise ContractError("global costmap must consume /map")
    if topics.get("collision_points") != "/go2/safety/points":
        raise ContractError("Collision Monitor must observe the audited safety cloud")

    frames = _mapping(profile["frames"], "profile.frames")
    expected_frames = {"global": "map", "local": "odom", "robot": "base_link"}
    for key, expected in expected_frames.items():
        if frames.get(key) != expected:
            raise ContractError("profile.frames.{} must equal {}".format(key, expected))

    safety = _mapping(profile["safety"], "profile.safety")
    _number(safety.get("robot_radius_m"), 0.30, "profile.safety.robot_radius_m")
    _number(safety.get("footprint_padding_m"), 0.02, "profile.safety.footprint_padding_m")
    _number(safety.get("inflation_radius_m"), 0.40, "profile.safety.inflation_radius_m")
    _number(safety.get("max_linear_velocity_mps"), 0.25, "profile.safety.max_linear_velocity_mps")
    _number(safety.get("max_angular_velocity_radps"), 1.0, "profile.safety.max_angular_velocity_radps")
    if safety.get("collision_monitor_mode") != "warn_only":
        raise ContractError("completion Collision Monitor must be warn_only")
    if safety.get("collision_monitor_in_navigation_command_path") is not False:
        raise ContractError("warn-only Collision Monitor must not gate navigation")
    if safety.get("collision_monitor_shadow_output_has_motion_consumer") is not False:
        raise ContractError("warn-only output must have no motion consumer")
    if safety.get("simulation_estop_required") is not True:
        raise ContractError("simulation estop must remain required")
    if safety.get("bounded_velocity_required") is not True:
        raise ContractError("bounded velocity must remain required")
    if safety.get("real_go2_allowed") is not False:
        raise ContractError("completion_sim must explicitly reject real Go2")

    timing = _mapping(profile["timing"], "profile.timing")
    _number(timing.get("source_timeout_sec"), 5.0, "profile.timing.source_timeout_sec")
    _number(timing.get("observation_persistence_sec"), 5.0, "profile.timing.observation_persistence_sec")
    _number(timing.get("transform_tolerance_sec"), 2.5, "profile.timing.transform_tolerance_sec")

    nvblox = _mapping(profile["nvblox"], "profile.nvblox")
    if nvblox.get("active_opt_in") is not True:
        raise ContractError("Nvblox active must be opt-in")
    if nvblox.get("active_failure_fallback") != "shadow":
        raise ContractError("Nvblox active failure must fall back to shadow")
    if nvblox.get("fallback_core_nav") != "nav2_static_lidar.yaml":
        raise ContractError("Nvblox fallback must preserve the default core")


def _validate_sim_time(document: Mapping[str, Any], label: str) -> None:
    required_nodes = (
        "bt_navigator",
        "controller_server",
        "local_costmap",
        "global_costmap",
        "planner_server",
        "smoother_server",
        "behavior_server",
        "waypoint_follower",
        "velocity_smoother",
        "docking_server",
        "collision_monitor",
        "internvla_go2_controller_bridge",
    )
    for name in required_nodes:
        if _params(document, name).get("use_sim_time") is not True:
            raise ContractError("{} {} must set use_sim_time=true".format(label, name))


def _validate_common_nav2(document: Mapping[str, Any], label: str) -> None:
    _validate_sim_time(document, label)
    local = _params(document, "local_costmap")
    global_params = _params(document, "global_costmap")
    if local.get("global_frame") != "odom" or local.get("rolling_window") is not True:
        raise ContractError("{} local costmap must be a rolling odom window".format(label))
    if global_params.get("global_frame") != "map" or global_params.get("rolling_window") is not False:
        raise ContractError("{} global costmap must be a fixed map window".format(label))
    for params, prefix in ((local, "local"), (global_params, "global")):
        _number(params.get("robot_radius"), 0.30, "{}.{}.robot_radius".format(label, prefix))
        _number(params.get("footprint_padding"), 0.02, "{}.{}.footprint_padding".format(label, prefix))
        inflation = _mapping(params.get("inflation_layer"), "{}.{}.inflation_layer".format(label, prefix))
        _number(inflation.get("inflation_radius"), 0.40, "{}.{}.inflation_radius".format(label, prefix))

    docking = _params(document, "docking_server")
    if docking.get("dock_plugins") != ["simple_charging_dock"]:
        raise ContractError("{} docking server must have one inert plugin".format(label))
    dock_plugin = _mapping(
        docking.get("simple_charging_dock"), "{}.docking.simple_charging_dock".format(label)
    )
    if dock_plugin.get("plugin") != "opennav_docking::SimpleChargingDock":
        raise ContractError("{} docking plugin identity changed".format(label))
    if dock_plugin.get("use_battery_status") is not False:
        raise ContractError("{} docking battery path must remain disabled".format(label))
    if dock_plugin.get("use_stall_detection") is not False:
        raise ContractError("{} docking stall path must remain disabled".format(label))
    dock_controller = _mapping(
        docking.get("controller"), "{}.docking.controller".format(label)
    )
    _number(
        dock_controller.get("v_linear_max"),
        0.15,
        "{}.docking.controller.v_linear_max".format(label),
    )
    if dock_controller.get("use_collision_detection") is not True:
        raise ContractError("{} docking collision detection must remain enabled".format(label))

    voxel = _mapping(local.get("voxel_layer"), "{}.local.voxel_layer".format(label))
    if voxel.get("plugin") != "nav2_costmap_2d::VoxelLayer":
        raise ContractError("local fast path must use nav2 VoxelLayer")
    source_name = voxel.get("observation_sources")
    if source_name != "lidar":
        raise ContractError("LiDAR must be the sole local observation source")
    source = _mapping(voxel.get(source_name), "{}.local.voxel_layer.lidar".format(label))
    expected_source = {
        "topic": "/go2/lidar/points_base",
        "data_type": "PointCloud2",
        "clearing": True,
        "marking": True,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ContractError("LiDAR source {} must equal {!r}".format(key, expected))
    _number(source.get("expected_update_rate"), 5.0, "LiDAR expected_update_rate")
    _number(source.get("observation_persistence"), 5.0, "LiDAR observation_persistence")

    global_plugins = global_params.get("plugins")
    if global_plugins != ["static_layer", "inflation_layer"]:
        raise ContractError("global costmap must remain static plus inflation only")
    static = _mapping(global_params.get("static_layer"), "{}.global.static_layer".format(label))
    if static.get("plugin") != "nav2_costmap_2d::StaticLayer":
        raise ContractError("global costmap must use StaticLayer")
    if static.get("map_subscribe_transient_local") is not True:
        raise ContractError("StaticLayer must accept the transient-local /map publisher")

    smoother = _params(document, "velocity_smoother")
    max_velocity = smoother.get("max_velocity")
    min_velocity = smoother.get("min_velocity")
    if max_velocity != [0.25, 0.0, 1.0] or min_velocity != [-0.25, 0.0, -1.0]:
        raise ContractError("velocity bounds must remain +/-0.25 m/s and +/-1.0 rad/s")

    monitor = _params(document, "collision_monitor")
    if monitor.get("cmd_vel_in_topic") != "/cmd_vel_nav":
        raise ContractError("Collision Monitor must observe cmd_vel_nav")
    if monitor.get("cmd_vel_out_topic") != "/completion_sim/collision_monitor/warn_only_cmd_vel":
        raise ContractError("warn-only Collision Monitor output must not drive cmd_vel_safe")
    _number(monitor.get("source_timeout"), 5.0, "Collision Monitor source_timeout")
    pointcloud = _mapping(monitor.get("pointcloud"), "collision_monitor.pointcloud")
    if pointcloud.get("topic") != "/go2/safety/points":
        raise ContractError("Collision Monitor must consume /go2/safety/points")

    bridge = _params(document, "internvla_go2_controller_bridge")
    _number(bridge.get("command_timeout_sec"), 0.30, "controller command timeout")


def _validate_nav2_modes(shadow: Mapping[str, Any], active_overlay: Mapping[str, Any]) -> None:
    active = deep_merge(shadow, active_overlay)
    _validate_common_nav2(shadow, "nav2_shadow")
    _validate_common_nav2(active, "nav2_active")
    shadow_local = _params(shadow, "local_costmap")
    active_local = _params(active, "local_costmap")
    if shadow_local.get("plugins") != ["voxel_layer", "inflation_layer"]:
        raise ContractError("default/shadow local costmap must not contain Nvblox")
    if "nvblox_layer" in shadow_local:
        raise ContractError("default/shadow Nav2 config contains an Nvblox layer")
    if active_local.get("plugins") != [
        "voxel_layer",
        "nvblox_layer",
        "inflation_layer",
    ]:
        raise ContractError("active local costmap must retain LiDAR around Nvblox")
    nvblox_layer = _mapping(active_local.get("nvblox_layer"), "active nvblox_layer")
    if nvblox_layer.get("plugin") != "nvblox::nav2::NvbloxCostmapLayer":
        raise ContractError("active Nvblox layer plugin is invalid")
    if nvblox_layer.get("enabled") is not True:
        raise ContractError("explicit active config must enable Nvblox")
    # The active overlay may add only the optional layer.  Every frozen core
    # value must remain byte-for-byte equal after removing that layer.
    active_core = dict(active_local)
    active_core.pop("nvblox_layer", None)
    active_core["plugins"] = ["voxel_layer", "inflation_layer"]
    if active_core != dict(shadow_local):
        raise ContractError("active overlay mutates the static+LiDAR fallback core")
    if _params(active, "global_costmap") != _params(shadow, "global_costmap"):
        raise ContractError("Nvblox active must not replace the static global map")


def _mode_value(document: Mapping[str, Any], label: str) -> str:
    mode = document.get("mode")
    if mode not in {"shadow", "active"}:
        raise ContractError("{}.mode must be shadow or active".format(label))
    return str(mode)


def _validate_nvblox_modes(
    shadow: Mapping[str, Any], active: Mapping[str, Any]
) -> None:
    if _mode_value(shadow, "nvblox_shadow") != "shadow":
        raise ContractError("shadow file must explicitly select shadow")
    if _mode_value(active, "nvblox_active") != "active":
        raise ContractError("active file must explicitly select active")
    if shadow.get("feeds_navigation_costmap") is not False:
        raise ContractError("shadow Nvblox must have no navigation influence")
    if shadow.get("required_for_navigation") is not False:
        raise ContractError("shadow Nvblox must not be required for navigation")
    if shadow.get("failure_fallback") != "default_core":
        raise ContractError("shadow failure must preserve static+LiDAR")
    if active.get("feeds_navigation_costmap") is not True:
        raise ContractError("active config must explicitly feed its optional layer")
    if active.get("preflight_required") is not True:
        raise ContractError("active config requires a readiness barrier")
    if active.get("failure_fallback") != "shadow":
        raise ContractError("active failure must downgrade to shadow")
    if active.get("fallback_navigation_config") != "nav2_static_lidar.yaml":
        raise ContractError("active fallback must select the default Nav2 config")
    if shadow.get("params") != active.get("params"):
        raise ContractError("shadow/active mapper parameters must be identical")
    for document, label in ((shadow, "shadow"), (active, "active")):
        node_params = _mapping(document.get("params"), "nvblox_{}.params".format(label))
        if node_params.get("use_sim_time") is not True:
            raise ContractError("Nvblox must use simulation time")
        if node_params.get("use_tf_transforms") is not True:
            raise ContractError("Nvblox must use observed TF transforms")
        if node_params.get("global_frame") != "map":
            raise ContractError("Nvblox global frame must be map")
        if node_params.get("use_lidar") is not True:
            raise ContractError("Nvblox mode must retain LiDAR input")
        if node_params.get("use_depth") is not False:
            raise ContractError("completion Nvblox config is explicitly LiDAR-only")


def _validate_smoke_map(document: Mapping[str, Any]) -> None:
    _exact_keys(
        document,
        {
            "schema_version",
            "name",
            "target",
            "topic",
            "frame_id",
            "odom_frame_id",
            "map_to_odom",
            "qos",
            "resolution_m",
            "width",
            "height",
            "origin_xy",
            "free_value",
            "occupied_value",
            "geometry_contract",
            "obstacles",
        },
        "smoke_static_map",
    )
    if document.get("schema_version") != 1:
        raise ContractError("smoke map schema_version must equal 1")
    if document.get("target") != "isaac_completion_diagnostic_scene_only":
        raise ContractError("smoke map must be limited to the Isaac diagnostic scene")
    if document.get("topic") != "/map":
        raise ContractError("smoke map must publish /map")
    if document.get("frame_id") != "map" or document.get("odom_frame_id") != "odom":
        raise ContractError("smoke map frames must be map and odom")
    if document.get("map_to_odom") != "identity":
        raise ContractError("diagnostic smoke map requires identity map->odom")
    qos = _mapping(document.get("qos"), "smoke_static_map.qos")
    if qos != {"reliability": "reliable", "durability": "transient_local", "depth": 1}:
        raise ContractError("smoke map QoS must be reliable transient-local depth 1")
    _number(document.get("resolution_m"), 0.05, "smoke map resolution")
    if document.get("width") != 80 or document.get("height") != 80:
        raise ContractError("smoke map must be the frozen 80x80 diagnostic grid")
    if document.get("origin_xy") != [-1.0, -2.0]:
        raise ContractError("smoke map origin must equal [-1.0, -2.0]")
    if document.get("free_value") != 0 or document.get("occupied_value") != 100:
        raise ContractError("smoke map occupancy values must be 0 and 100")
    if document.get("geometry_contract") != "sensor_runtime.contract.DIAGNOSTIC_GEOMETRY":
        raise ContractError("smoke map must identify its frozen geometry source")
    obstacles = document.get("obstacles")
    expected = [
        {
            "name": "near_depth_lidar_box",
            "center_xy_m": [1.0, 0.0],
            "size_xy_m": [0.30, 0.40],
        },
        {
            "name": "far_diagnostic_wall",
            "center_xy_m": [2.0, 0.0],
            "size_xy_m": [0.10, 2.00],
        },
    ]
    if obstacles != expected:
        raise ContractError("smoke map obstacles differ from DIAGNOSTIC_GEOMETRY")
    from sensor_runtime.contract import DIAGNOSTIC_GEOMETRY

    upstream_projection = [
        {
            "name": item["name"],
            "center_xy_m": list(item["position_world_m"][:2]),
            "size_xy_m": list(item["size_m"][:2]),
        }
        for item in DIAGNOSTIC_GEOMETRY
    ]
    if upstream_projection != expected:
        raise ContractError("upstream diagnostic geometry changed; smoke map is stale")


def load_and_validate(config_dir: Path = DEFAULT_CONFIG_DIR) -> LoadedConfigs:
    resolved = _resolve_config_dir(config_dir)
    values: Dict[str, Mapping[str, Any]] = {}
    for key, filename in PROFILE_FILES.items():
        values[key] = _load_yaml(resolved / filename)
    loaded = LoadedConfigs(config_dir=resolved, **values)
    _validate_profile(loaded.profile)
    _validate_nav2_modes(loaded.nav2_shadow, loaded.nav2_active)
    _validate_nvblox_modes(loaded.nvblox_shadow, loaded.nvblox_active)
    _validate_smoke_map(loaded.smoke_map)
    return loaded


def audit_runtime_isolation(root: Path = ROOT) -> Mapping[str, Any]:
    """Read-only proof that completion settings did not leak into strict mode."""

    strict_path = root / "configs" / "runtime" / "strict_evidence.yaml"
    completion_path = root / "configs" / "runtime" / "completion_sim.yaml"
    strict = _load_yaml(strict_path)
    completion = _load_yaml(completion_path)
    strict_nav = _mapping(strict.get("navigation"), "strict.navigation")
    strict_tf = _mapping(strict.get("tf"), "strict.tf")
    strict_mapping = _mapping(strict.get("mapping"), "strict.mapping")
    if strict.get("runtime_policy") != "strict_evidence":
        raise ContractError("strict runtime policy identity changed")
    if strict.get("recorder", {}).get("mode") != "fatal_exact_batch":
        raise ContractError("strict exact recorder changed")
    _number(strict_nav.get("source_timeout_sec"), 0.35, "strict source timeout")
    _number(strict_nav.get("costmap_current_timeout_sec"), 0.35, "strict costmap timeout")
    if strict_nav.get("collision_monitor") != "enforce":
        raise ContractError("strict Collision Monitor must enforce")
    if strict_tf.get("lookup") != "exact":
        raise ContractError("strict TF lookup must remain exact")
    _number(strict_tf.get("transform_tolerance_sec"), 0.0, "strict TF tolerance")
    if strict_mapping != {"nvblox": "active_required", "global_map": "nvblox"}:
        raise ContractError("strict mapping policy changed")
    if completion.get("runtime_policy") != "completion_sim":
        raise ContractError("completion runtime policy identity changed")
    if completion.get("forbidden_targets") != ["real_go2", "hardware_motion"]:
        raise ContractError("completion runtime no longer rejects hardware")
    completion_nav = _mapping(completion.get("navigation"), "completion.navigation")
    completion_mapping = _mapping(completion.get("mapping"), "completion.mapping")
    if completion_nav.get("collision_monitor") != "warn_only":
        raise ContractError("completion Collision Monitor policy drifted")
    if completion_mapping != {
        "nvblox": "shadow",
        "global_map": "static_with_lidar_local",
    }:
        raise ContractError("completion map policy drifted")
    return {
        "strict_evidence_sha256": hashlib.sha256(strict_path.read_bytes()).hexdigest(),
        "completion_sim_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        "strict_evidence_unchanged": True,
        "completion_sim_isolated": True,
    }


def core_nav_projection(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the navigation core with an optional Nvblox layer removed."""

    local = dict(_params(document, "local_costmap"))
    local.pop("nvblox_layer", None)
    local["plugins"] = ["voxel_layer", "inflation_layer"]
    return {
        "local_costmap": local,
        "global_costmap": dict(_params(document, "global_costmap")),
        "velocity_smoother": dict(_params(document, "velocity_smoother")),
        "controller_bridge": dict(_params(document, "internvla_go2_controller_bridge")),
    }


def deep_merge(
    base: Mapping[str, Any], overlay: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Recursively merge an owned overlay without mutating either input."""

    merged: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(
                _mapping(merged[key], "base.{}".format(key)), value
            )
        else:
            merged[key] = value
    return merged
