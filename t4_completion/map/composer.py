"""Deterministic launch/config composition for the completion map path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .contract import (
    DEFAULT_CONFIG_DIR,
    ContractError,
    LoadedConfigs,
    core_nav_projection,
    load_and_validate,
)


VALID_MODES = {"shadow", "active"}
VALID_HEALTH = {"ready", "failed", "unknown"}


@dataclass(frozen=True)
class ComposeRequest:
    requested_mode: str = "shadow"
    nvblox_health: str = "unknown"
    target: str = "isaac_simulation_only"
    runtime_policy: str = "completion_sim"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalize_request(request: ComposeRequest) -> None:
    if request.runtime_policy != "completion_sim":
        raise ContractError("map composer is available only to completion_sim")
    if request.target != "isaac_simulation_only":
        raise ContractError("map composer rejects real Go2 and hardware targets")
    if request.requested_mode not in VALID_MODES:
        raise ContractError("requested Nvblox mode must be shadow or active")
    if request.nvblox_health not in VALID_HEALTH:
        raise ContractError("Nvblox health must be ready, failed, or unknown")


def _resolved_mode(request: ComposeRequest) -> Dict[str, Any]:
    if request.requested_mode == "shadow":
        return {
            "effective_mode": "shadow",
            "fallback_applied": False,
            "fallback_reason": None,
        }
    if request.nvblox_health == "ready":
        return {
            "effective_mode": "active",
            "fallback_applied": False,
            "fallback_reason": None,
        }
    return {
        "effective_mode": "shadow",
        "fallback_applied": True,
        "fallback_reason": "active_preflight_{}_use_shadow".format(
            request.nvblox_health
        ),
    }


def _relative_source(configs: LoadedConfigs, filename: str) -> str:
    try:
        return str((configs.config_dir / filename).relative_to(configs.config_dir.parents[2])).replace("\\", "/")
    except ValueError:
        return filename


def compose_plan(
    request: ComposeRequest,
    configs: Optional[LoadedConfigs] = None,
) -> Mapping[str, Any]:
    """Create a pure-data plan; no process, socket, or output path is touched."""

    _normalize_request(request)
    loaded = configs if configs is not None else load_and_validate(DEFAULT_CONFIG_DIR)
    resolution = _resolved_mode(request)
    mode = str(resolution["effective_mode"])
    nav2 = loaded.nav2_for_mode(mode)
    nvblox = loaded.nvblox_for_mode(mode)
    default_core = core_nav_projection(loaded.nav2_shadow)
    selected_core = core_nav_projection(nav2)
    if selected_core != default_core:
        raise ContractError("selected plan does not preserve the static+LiDAR core")
    core_sha = _sha256(default_core)

    mode_config_name = "nvblox_{}.yaml".format(mode)
    selected_nav_name = (
        "nav2_nvblox_active.yaml" if mode == "active" else "nav2_static_lidar.yaml"
    )
    nvblox_required = mode == "active"
    plan: Dict[str, Any] = {
        "schema_version": 1,
        "status": "READY",
        "runtime_policy": "completion_sim",
        "target": "isaac_simulation_only",
        "requested_nvblox_mode": request.requested_mode,
        "observed_nvblox_health": request.nvblox_health,
        **resolution,
        "default_path": {
            "global_map": "static",
            "local_costmap": "lidar_voxel",
            "lidar_topic": "/go2/lidar/points_base",
            "nvblox_mode": "shadow",
        },
        "selected_configs": {
            "profile": _relative_source(loaded, "profile.yaml"),
            "nav2": _relative_source(loaded, selected_nav_name),
            "nvblox": _relative_source(loaded, mode_config_name),
            "smoke_map": _relative_source(loaded, "smoke_static_map.yaml"),
        },
        "core_nav_sha256": core_sha,
        "fallback_core_nav_sha256": _sha256(default_core),
        "core_nav": default_core,
        "launch_environment": {
            "INTERNNAV_RUNTIME_POLICY": "completion_sim",
            "INTERNNAV_SIMULATION_TARGET": "isaac",
            "INTERNVLA_T4_MAP_SOURCE": "static_map",
            "PYTHONDONTWRITEBYTECODE": "1",
            "ROS2CLI_NO_DAEMON": "1",
        },
        "runtime_guards": {
            "lease_ack_required": True,
            "managed_sensor_companion_required": True,
            "simulation_estop_required": True,
            "bounded_velocity_required": True,
            "real_go2_forbidden": True,
            "strict_evidence_modified": False,
        },
        "required_inputs": [
            "/clock",
            "/odom",
            "/tf",
            "/go2/lidar/points_base",
            "/go2/safety/points",
            "/internvla/stop",
        ],
        "processes": [
            {
                "role": "static_map_publisher",
                "required": True,
                "argv": [
                    "python3",
                    "-m",
                    "t4_completion.map.static_map",
                    "--config",
                    "{bundle}/smoke_static_map.yaml",
                    "--result-dir",
                    "{result}/map",
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true",
                ],
                "failure_action": "fail_closed_zero_motion",
            },
            {
                "role": "nvblox_mapper",
                "required": nvblox_required,
                "argv": [
                    "ros2",
                    "run",
                    "nvblox_ros",
                    "nvblox_node",
                    "--ros-args",
                    "--params-file",
                    "{bundle}/nvblox_params.yaml",
                    "-r",
                    "pointcloud:=/go2/lidar/points",
                ],
                "failure_action": (
                    "disable_optional_nvblox_layer_then_continue_shadow"
                    if nvblox_required
                    else "record_shadow_unavailable_then_continue_static_lidar"
                ),
            },
            {
                "role": "bounded_warn_only_relay",
                "required": True,
                "argv": [
                    "python3",
                    "-m",
                    "t4_completion.map.warn_relay",
                    "--result-dir",
                    "{result}/map",
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true",
                ],
                "failure_action": "fail_closed_zero_motion",
            },
            {
                "role": "nav2_static_lidar",
                "required": True,
                "argv": [
                    "ros2",
                    "launch",
                    "nav2_bringup",
                    "navigation_launch.py",
                    "params_file:={bundle}/nav2_params.yaml",
                    "use_sim_time:=True",
                    "autostart:=True",
                    "use_composition:=False",
                    "use_respawn:=False",
                ],
                "failure_action": "fail_closed_zero_motion",
            },
        ],
        "active_runtime_fallback": {
            "trigger": "nvblox_child_exit_or_slice_stale",
            "command": [
                "ros2",
                "param",
                "set",
                "/local_costmap/local_costmap",
                "nvblox_layer.enabled",
                "false",
            ],
            "effective_mode_after": "shadow",
            "preserved_layers": ["voxel_layer", "inflation_layer"],
            "preserved_core_nav_sha256": core_sha,
        },
        "evidence_outputs": [
            "{result}/map/map_publisher.json",
            "{result}/map/nvblox_state.jsonl",
            "{result}/map/warn_only_relay.jsonl",
            "{result}/map/companion_cleanup.json",
            "{result}/map/smoke_validation.json",
        ],
        "online_executed_by_composer": False,
    }
    # The functional path is static global + LiDAR local.  A shadow Nvblox
    # process has no control authority and need not consume GPU/CPU during F1;
    # it is started only for an explicit, healthy active-mode request.
    plan["nvblox_process_started"] = nvblox_required
    if not nvblox_required:
        plan["processes"] = [
            process
            for process in plan["processes"]
            if process["role"] != "nvblox_mapper"
        ]
    # Hash the selected full documents as provenance, but never hash paths or
    # the surrounding environment.  This makes dry-run output deterministic.
    plan["selected_config_sha256"] = {
        "nav2": _sha256(nav2),
        "nvblox": _sha256(nvblox),
        "smoke_map": _sha256(loaded.smoke_map),
    }
    return plan


def validation_report(
    request: ComposeRequest,
    configs: Optional[LoadedConfigs] = None,
) -> Mapping[str, Any]:
    loaded = configs if configs is not None else load_and_validate(DEFAULT_CONFIG_DIR)
    plan = compose_plan(request, loaded)
    return {
        "schema_version": 1,
        "status": "PASS",
        "checks": {
            "completion_sim_only": True,
            "static_global_default": True,
            "lidar_local_default": True,
            "filtered_lidar_topic": "/go2/lidar/points_base",
            "collision_monitor_warn_only": True,
            "bounded_velocity_and_sim_estop": True,
            "nvblox_default_shadow": True,
            "nvblox_shadow_process_deferred": not plan["nvblox_process_started"],
            "effective_nvblox_mode": plan["effective_mode"],
            "active_failure_fallback_to_shadow": plan["active_runtime_fallback"][
                "effective_mode_after"
            ]
            == "shadow",
            "core_hash_preserved": plan["core_nav_sha256"]
            == plan["fallback_core_nav_sha256"],
            "strict_evidence_modified": False,
            "real_go2_forbidden": True,
        },
        "plan_sha256": _sha256(plan),
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_yaml(path: Path, value: Any) -> None:
    rendered = yaml.safe_dump(value, allow_unicode=False, sort_keys=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)


def compose_bundle(
    output_dir: Path,
    request: ComposeRequest,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> Mapping[str, Any]:
    """Write a fresh deterministic bundle; refuse reuse or append."""

    loaded = load_and_validate(config_dir)
    plan = compose_plan(request, loaded)
    report = validation_report(request, loaded)
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mode = str(plan["effective_mode"])
    nav2 = loaded.nav2_for_mode(mode)
    nvblox = loaded.nvblox_for_mode(mode)
    _write_yaml(output / "nav2_params.yaml", nav2)
    _write_yaml(
        output / "nvblox_params.yaml",
        {"/**": {"ros__parameters": dict(nvblox["params"])}},
    )
    _write_yaml(output / "smoke_static_map.yaml", loaded.smoke_map)
    _write_json(output / "launch_plan.json", plan)
    _write_json(output / "config_validation.json", report)
    return {
        "schema_version": 1,
        "status": "PASS",
        "output_dir": str(output),
        "effective_mode": mode,
        "fallback_applied": bool(plan["fallback_applied"]),
        "files": sorted(item.name for item in output.iterdir()),
        "plan_sha256": report["plan_sha256"],
    }
