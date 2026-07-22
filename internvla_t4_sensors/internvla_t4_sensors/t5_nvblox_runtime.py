"""T5-only Nvblox profile materialization and readiness state.

The module deliberately has no ROS imports so its failover and configuration
contracts can be tested on the coordinator.  The ROS supervisor is a thin
adapter around :class:`NvbloxReadinessState`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Set

import yaml


VALID_MODES = {"off", "shadow", "active_local_gt"}
FUSED_INPUTS = ("depth", "depth_camera_info", "lidar", "ground_truth_odometry")


class T5NvbloxContractError(ValueError):
    """Raised when an opt-in T5 Nvblox profile violates its frozen boundary."""


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
            raise T5NvbloxContractError("duplicate YAML key: {!r}".format(key))
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError, T5NvbloxContractError) as exc:
        raise T5NvbloxContractError("invalid YAML {}: {}".format(path, exc)) from exc
    if not isinstance(value, Mapping):
        raise T5NvbloxContractError("{} must contain a mapping".format(path))
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise T5NvbloxContractError("invalid JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, Mapping):
        raise T5NvbloxContractError("{} must contain a mapping".format(path))
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256_bytes(canonical)


def _nav_params(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    node = document.get(name)
    if not isinstance(node, Mapping):
        raise T5NvbloxContractError("Nav2 config is missing {}".format(name))
    if name in {"local_costmap", "global_costmap"}:
        node = node.get(name)
        if not isinstance(node, Mapping):
            raise T5NvbloxContractError(
                "Nav2 config is missing {}.{}".format(name, name)
            )
    params = node.get("ros__parameters")
    if not isinstance(params, Mapping):
        raise T5NvbloxContractError(
            "Nav2 config is missing {} ros__parameters".format(name)
        )
    return params


def _resolved_source(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise T5NvbloxContractError("{} path is invalid".format(label))
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise T5NvbloxContractError("{} escapes the repository".format(label)) from exc
    if not candidate.is_file():
        raise T5NvbloxContractError("{} is missing: {}".format(label, candidate))
    return candidate


def load_profile_sources(root: Path, nav2_base: Optional[Path] = None) -> Mapping[str, Any]:
    """Load and validate the T5-only fused/active-local configuration."""

    resolved_root = root.resolve()
    manifest_path = resolved_root / "configs/internnav_t5/nvblox_runtime_contract.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise T5NvbloxContractError("Nvblox contract schema_version must equal 1")
    if manifest.get("runtime_policy") != "completion_sim":
        raise T5NvbloxContractError("Nvblox contract must be completion_sim")
    if manifest.get("target") != "isaac_simulation_only":
        raise T5NvbloxContractError("Nvblox contract must reject real hardware")
    if manifest.get("default_mode") != "off":
        raise T5NvbloxContractError("T5 Nvblox must remain opt-in")
    if set(manifest.get("allowed_modes", [])) != VALID_MODES:
        raise T5NvbloxContractError("Nvblox allowed modes drifted")

    node = manifest.get("node")
    navigation = manifest.get("navigation")
    if not isinstance(node, Mapping) or not isinstance(navigation, Mapping):
        raise T5NvbloxContractError("Nvblox node/navigation contract is missing")
    nvblox_path = _resolved_source(resolved_root, node.get("parameter_file"), "Nvblox params")
    overlay_path = _resolved_source(
        resolved_root, navigation.get("active_local_overlay"), "active-local overlay"
    )
    base_path = (
        nav2_base.resolve()
        if nav2_base is not None
        else _resolved_source(resolved_root, navigation.get("base"), "Nav2 base")
    )
    if not base_path.is_file():
        raise T5NvbloxContractError("Nav2 base is missing: {}".format(base_path))

    nvblox = _load_yaml(nvblox_path)
    overlay = _load_yaml(overlay_path)
    base = _load_yaml(base_path)
    wildcard = nvblox.get("/**")
    params = wildcard.get("ros__parameters") if isinstance(wildcard, Mapping) else None
    if not isinstance(params, Mapping):
        raise T5NvbloxContractError("Nvblox params must use the /** ROS wildcard")
    required_nvblox = {
        "use_sim_time": True,
        "global_frame": "map",
        "num_cameras": 1,
        "use_tf_transforms": True,
        "use_depth": True,
        "use_lidar": True,
        "publish_esdf_distance_slice": True,
        "map_clearing_frame_id": "base_link",
    }
    for key, expected in required_nvblox.items():
        if params.get(key) != expected:
            raise T5NvbloxContractError(
                "Nvblox {} must equal {!r}".format(key, expected)
            )
    for key in ("integrate_depth_rate_hz", "integrate_lidar_rate_hz"):
        value = params.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise T5NvbloxContractError("Nvblox {} must be positive".format(key))

    base_local = _nav_params(base, "local_costmap")
    base_global = _nav_params(base, "global_costmap")
    if "voxel_layer" not in base_local.get("plugins", []):
        raise T5NvbloxContractError("Nav2 base must retain LiDAR VoxelLayer")
    if "static_layer" not in base_global.get("plugins", []):
        raise T5NvbloxContractError("Nav2 base must retain the static global map")
    active = _deep_merge(base, overlay)
    active_local = _nav_params(active, "local_costmap")
    active_global = _nav_params(active, "global_costmap")
    if active_local.get("plugins") != [
        "voxel_layer",
        "nvblox_layer",
        "inflation_layer",
    ]:
        raise T5NvbloxContractError("active-local plugin order is invalid")
    layer = active_local.get("nvblox_layer")
    if not isinstance(layer, Mapping):
        raise T5NvbloxContractError("active-local Nvblox layer is missing")
    if layer.get("enabled") is not False:
        raise T5NvbloxContractError("active-local Nvblox must start disabled")
    if layer.get("nvblox_map_slice_topic") != "nvblox_node/static_map_slice":
        raise T5NvbloxContractError("Nvblox slice topic must be lane-relative")
    if "nvblox_layer" in active_global.get("plugins", []):
        raise T5NvbloxContractError("Nvblox is forbidden in the global costmap")
    if active_global != base_global:
        raise T5NvbloxContractError("active-local overlay changed the global costmap")
    if active_local.get("voxel_layer") != base_local.get("voxel_layer"):
        raise T5NvbloxContractError("active-local overlay changed LiDAR VoxelLayer")

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "nvblox_path": nvblox_path,
        "nvblox": nvblox,
        "overlay_path": overlay_path,
        "overlay": overlay,
        "nav2_base_path": base_path,
        "nav2_base": base,
        "nav2_active": active,
    }


def materialize_profile(
    root: Path,
    output_dir: Path,
    mode: str,
    nav2_base: Optional[Path] = None,
    lane_namespace: Optional[str] = None,
) -> Mapping[str, Any]:
    """Write a fresh, deterministic T5 Nvblox runtime bundle."""

    if mode not in VALID_MODES:
        raise T5NvbloxContractError("mode must be off, shadow, or active_local_gt")
    if lane_namespace is not None and lane_namespace not in {
        "/t5/lane_a",
        "/t5/lane_b",
    }:
        raise T5NvbloxContractError("lane_namespace must be /t5/lane_a or /t5/lane_b")
    if mode == "active_local_gt" and lane_namespace is None:
        raise T5NvbloxContractError("active_local_gt requires lane_namespace")
    sources = load_profile_sources(root, nav2_base)
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected_nav = copy.deepcopy(
        sources["nav2_active"] if mode == "active_local_gt" else sources["nav2_base"]
    )
    slice_topic_absolute: Optional[str] = None
    if mode == "active_local_gt":
        slice_relative = sources["manifest"]["output"][
            "slice_relative_to_lane_namespace"
        ]
        slice_topic_absolute = "{}/{}".format(
            lane_namespace, str(slice_relative).lstrip("/")
        )
        active_local = _nav_params(selected_nav, "local_costmap")
        layer = active_local.get("nvblox_layer")
        if not isinstance(layer, MutableMapping):
            raise T5NvbloxContractError("materialized Nvblox layer is not mutable")
        layer["nvblox_map_slice_topic"] = slice_topic_absolute
    nav2_path = output / "nav2_params.yaml"
    nvblox_path = output / "nvblox_params.yaml"
    nav2_path.write_text(
        yaml.safe_dump(selected_nav, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    nvblox_path.write_text(
        yaml.safe_dump(sources["nvblox"], allow_unicode=False, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    manifest = sources["manifest"]
    mode_contract = manifest["modes"][mode]
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": "READY",
        "runtime_policy": "completion_sim",
        "target": "isaac_simulation_only",
        "mode": mode,
        "lane_namespace": lane_namespace,
        "starts_real_nvblox_node": mode_contract["starts_real_nvblox_node"],
        "feeds_navigation": mode_contract["feeds_navigation"],
        "pose_source": mode_contract.get("pose_source"),
        "required_inputs": manifest["inputs"],
        "slice_topic_relative_to_lane_namespace": manifest["output"][
            "slice_relative_to_lane_namespace"
        ],
        "slice_topic_absolute": slice_topic_absolute,
        "readiness": manifest["readiness"],
        "fallback": manifest["fallback"],
        "navigation": {
            "static_global_map_retained": True,
            "lidar_voxel_local_retained": True,
            "nvblox_global_layer_present": False,
            "nvblox_local_layer_loaded": mode == "active_local_gt",
            "nvblox_local_layer_initially_enabled": False,
        },
        "claims": manifest["claims"],
        "source_sha256": {
            "contract": _sha256_bytes(sources["manifest_path"].read_bytes()),
            "nvblox": _sha256_bytes(sources["nvblox_path"].read_bytes()),
            "active_local_overlay": _sha256_bytes(sources["overlay_path"].read_bytes()),
            "nav2_base": _sha256_bytes(sources["nav2_base_path"].read_bytes()),
        },
        "materialized_sha256": {
            "nav2": _sha256_bytes(nav2_path.read_bytes()),
            "nvblox": _sha256_bytes(nvblox_path.read_bytes()),
        },
        "supervisor": {
            "package": "internvla_t4_sensors",
            "executable": "internvla_t5_nvblox_supervisor",
            "mode_parameter": mode,
            "has_cmd_vel_authority": False,
            "has_terminal_stop_authority": False,
        },
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    contract_path = output / "runtime_contract.json"
    contract_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


@dataclass
class NvbloxReadinessState:
    """Pure sim-time gate for shadow evidence and active-local authority."""

    mode: str
    minimum_consecutive_slices: int = 10
    stale_after_ns: int = 2_500_000_000
    generation: int = -1
    process_epoch_sim_ns: int = 0
    sensor_stamps_ns: Dict[str, int] = field(default_factory=dict)
    last_slice_stamp_ns: int = 0
    consecutive_valid_slices: int = 0
    observed_classes: Set[str] = field(default_factory=set)
    layer_enabled: bool = False
    readiness_achieved: bool = False
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "active_local_gt"}:
            raise T5NvbloxContractError("runtime state requires shadow or active_local_gt")
        if self.minimum_consecutive_slices < 1:
            raise T5NvbloxContractError("minimum_consecutive_slices must be positive")
        if self.stale_after_ns <= 0:
            raise T5NvbloxContractError("stale_after_ns must be positive")

    def reset(self, generation: int, process_epoch_sim_ns: int) -> bool:
        if generation <= self.generation:
            return False
        was_enabled = self.layer_enabled
        self.generation = generation
        self.process_epoch_sim_ns = max(0, process_epoch_sim_ns)
        self.sensor_stamps_ns.clear()
        self.last_slice_stamp_ns = 0
        self.consecutive_valid_slices = 0
        self.observed_classes.clear()
        self.layer_enabled = False
        self.readiness_achieved = False
        self.fallback_reason = "episode_reset"
        return was_enabled

    def child_restarted(self, process_epoch_sim_ns: int) -> None:
        self.process_epoch_sim_ns = max(0, process_epoch_sim_ns)
        self.sensor_stamps_ns.clear()
        self.last_slice_stamp_ns = 0
        self.consecutive_valid_slices = 0
        self.observed_classes.clear()
        self.layer_enabled = False
        self.readiness_achieved = False

    def observe_sensor(self, name: str, stamp_ns: int) -> None:
        if name not in FUSED_INPUTS:
            raise T5NvbloxContractError("unknown Nvblox input: {}".format(name))
        if stamp_ns <= 0:
            return
        if stamp_ns >= self.process_epoch_sim_ns:
            self.sensor_stamps_ns[name] = max(
                stamp_ns, self.sensor_stamps_ns.get(name, 0)
            )

    def fused_inputs_fresh(self, now_sim_ns: int) -> bool:
        if now_sim_ns <= 0:
            return False
        for name in FUSED_INPUTS:
            stamp = self.sensor_stamps_ns.get(name, 0)
            age = now_sim_ns - stamp
            if stamp <= 0 or age < 0 or age > self.stale_after_ns:
                return False
        return True

    def observe_slice(
        self,
        stamp_ns: int,
        now_sim_ns: int,
        unknown_count: int,
        free_count: int,
        occupied_count: int,
    ) -> str:
        if (
            stamp_ns <= 0
            or stamp_ns < self.process_epoch_sim_ns
            or any(value < 0 for value in (unknown_count, free_count, occupied_count))
            or unknown_count + free_count + occupied_count <= 0
            or stamp_ns < self.last_slice_stamp_ns
        ):
            self.consecutive_valid_slices = 0
            return "REJECTED_SLICE"
        if stamp_ns == self.last_slice_stamp_ns:
            # Nvblox may republish the current slice between integration
            # updates.  A duplicate is neither a new valid slice nor a break
            # in the run of unique, monotonically stamped valid slices.
            return "DUPLICATE_SLICE"
        self.last_slice_stamp_ns = stamp_ns
        self.consecutive_valid_slices += 1
        if unknown_count:
            self.observed_classes.add("unknown")
        if free_count:
            self.observed_classes.add("free")
        if occupied_count:
            self.observed_classes.add("occupied")
        if not self.ready(now_sim_ns):
            return "OBSERVED"
        self.readiness_achieved = True
        self.fallback_reason = None
        if self.mode == "shadow":
            return "SHADOW_READY"
        if not self.layer_enabled:
            return "ENABLE_LAYER"
        return "ACTIVE_READY"

    def ready(self, now_sim_ns: int) -> bool:
        return (
            self.fused_inputs_fresh(now_sim_ns)
            and self.consecutive_valid_slices >= self.minimum_consecutive_slices
            and self.observed_classes == {"unknown", "free", "occupied"}
            and self.last_slice_stamp_ns > 0
            and 0 <= now_sim_ns - self.last_slice_stamp_ns <= self.stale_after_ns
        )

    def mark_layer_enabled(self) -> None:
        if self.mode != "active_local_gt":
            raise T5NvbloxContractError("shadow mode cannot enable a Nav2 layer")
        self.layer_enabled = True
        self.fallback_reason = None

    def tick(self, now_sim_ns: int, child_alive: bool) -> Optional[str]:
        reason: Optional[str] = None
        if not child_alive:
            reason = "nvblox_child_exit"
        elif self.layer_enabled and not self.ready(now_sim_ns):
            reason = "fused_input_or_slice_stale"
        elif self.mode == "shadow" and self.readiness_achieved and not self.ready(
            now_sim_ns
        ):
            reason = "fused_input_or_slice_stale"
        if reason is None:
            return None
        self.fallback_reason = reason
        if self.layer_enabled:
            self.consecutive_valid_slices = 0
            self.observed_classes.clear()
            self.readiness_achieved = False
            return "DISABLE_LAYER"
        self.consecutive_valid_slices = 0
        self.observed_classes.clear()
        self.readiness_achieved = False
        return "DEGRADED"

    def mark_layer_disabled(self, reason: str) -> None:
        self.layer_enabled = False
        self.fallback_reason = reason

    def snapshot(self, now_sim_ns: int, child_alive: bool) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "generation": self.generation,
            "child_alive": child_alive,
            "fused_inputs_fresh": self.fused_inputs_fresh(now_sim_ns),
            "sensor_stamps_ns": dict(self.sensor_stamps_ns),
            "last_slice_stamp_ns": self.last_slice_stamp_ns or None,
            "consecutive_valid_slices": self.consecutive_valid_slices,
            "observed_classes": sorted(self.observed_classes),
            "ready": self.ready(now_sim_ns),
            "readiness_achieved": self.readiness_achieved,
            "layer_enabled": self.layer_enabled,
            "fallback_reason": self.fallback_reason,
        }
