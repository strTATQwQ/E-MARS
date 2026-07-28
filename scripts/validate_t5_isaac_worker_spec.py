#!/usr/bin/env python3
"""Fail-closed validation for a prepared T5 Isaac worker container."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def _env_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if separator:
            result[key] = item
    return result


def _mount_map(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(value.get("Destination")): {
            "source": value.get("Source"),
            "rw": bool(value.get("RW")),
            "type": value.get("Type"),
        }
        for value in values
    }


def validate(expected: dict[str, Any], inspected: dict[str, Any]) -> dict[str, Any]:
    config = inspected.get("Config") or {}
    host = inspected.get("HostConfig") or {}
    labels = config.get("Labels") or {}
    environment = _env_map(config.get("Env") or [])
    mounts = _mount_map(inspected.get("Mounts") or [])

    expected_labels = {
        "internnav.t5.spec_version": str(expected["spec_version"]),
        "internnav.t5.role": "isaac_ros_worker",
        "internnav.t5.lane": expected["lane"],
        "internnav.t5.gpu": str(expected["gpu"]),
        "internnav.t5.ros_domain_id": str(expected["ros_domain_id"]),
        "internnav.t5.control_root": expected["control_root"],
        "internnav.t5.deployment_root": expected["lane_deployment_root"],
        "internnav.t5.worker_profile_root": expected["lane_profile_root"],
    }
    expected_environment = {
        "ROS_DOMAIN_ID": str(expected["ros_domain_id"]),
        "ROS_LOCALHOST_ONLY": "0",
        "ROS_NAMESPACE": expected["namespace"],
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
        "ROS_STATIC_PEERS": expected["static_peer"],
        "INTERNNAV_T5_LANE": expected["lane"],
        "INTERNNAV_T5_ID_PREFIX": expected["identity_prefix"],
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": str(expected["gpu"]),
        "XDG_CACHE_HOME": f'{expected["lane_profile_root"]}/cache/xdg',
        "XDG_CONFIG_HOME": f'{expected["lane_profile_root"]}/config',
        "XDG_DATA_HOME": f'{expected["lane_profile_root"]}/data',
        "OV_CACHE_ROOT": f'{expected["lane_profile_root"]}/cache/ov',
        "NVIDIA_SHADER_CACHE_PATH": f'{expected["lane_profile_root"]}/cache/nvidia',
        "TMPDIR": f'{expected["lane_profile_root"]}/tmp',
        "USERNAME": "admin",
        "HOST_USER_UID": str(expected["host_user_uid"]),
        "HOST_USER_GID": str(expected["host_user_gid"]),
    }
    expected_mounts = {
        expected["control_root"]: {
            "source": expected["control_root"], "rw": False, "type": "bind"
        },
        expected["lane_deployment_root"]: {
            "source": expected["lane_deployment_root"], "rw": True, "type": "bind"
        },
        expected["lane_profile_root"]: {
            "source": expected["lane_profile_root"], "rw": True, "type": "bind"
        },
        "/workspaces/isaac": {
            "source": expected["ros_workspace"], "rw": False, "type": "bind"
        },
    }
    t5_labels = {key: value for key, value in labels.items() if key.startswith("internnav.t5.")}
    requests = host.get("DeviceRequests") or []
    device_request_matches = len(requests) == 1
    if device_request_matches:
        request = requests[0]
        device_request_matches = (
            request.get("Driver") in ("", "nvidia")
            and sorted(request.get("DeviceIDs") or []) == [str(expected["gpu"])]
            and request.get("Capabilities") == [["gpu"]]
            and request.get("Count") == 0
            and (request.get("Options") or {}) == {}
        )

    checks = {
        "container_name": inspected.get("Name") == f'/{expected["container"]}',
        "labels_exact": t5_labels == expected_labels,
        "environment_contract": all(environment.get(key) == value for key, value in expected_environment.items()),
        "lane_identity_environment_exact": {
            key: environment.get(key)
            for key in (
                "ROS_DOMAIN_ID",
                "ROS_LOCALHOST_ONLY",
                "ROS_NAMESPACE",
                "ROS_AUTOMATIC_DISCOVERY_RANGE",
                "ROS_STATIC_PEERS",
                "INTERNNAV_T5_LANE",
                "INTERNNAV_T5_ID_PREFIX",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
            )
        }
        == {
            key: expected_environment[key]
            for key in (
                "ROS_DOMAIN_ID",
                "ROS_LOCALHOST_ONLY",
                "ROS_NAMESPACE",
                "ROS_AUTOMATIC_DISCOVERY_RANGE",
                "ROS_STATIC_PEERS",
                "INTERNNAV_T5_LANE",
                "INTERNNAV_T5_ID_PREFIX",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
            )
        },
        "image_id": inspected.get("Image") == expected["image_id"],
        "image_reference": config.get("Image") == expected["image_reference"],
        "entrypoint": config.get("Entrypoint") == ["/usr/local/bin/scripts/workspace-entrypoint.sh"],
        "command": config.get("Cmd") == ["sleep", "infinity"],
        "cpuset": host.get("CpusetCpus") == expected["cpuset"],
        # Docker represents its default private PID namespace as an empty mode.
        "pid_private": host.get("PidMode") == "",
        "ipc_private": host.get("IpcMode") == "private",
        "network_host": host.get("NetworkMode") == "host",
        "not_privileged": host.get("Privileged") is False,
        "restart_disabled": (host.get("RestartPolicy") or {}).get("Name", "no") == "no",
        "gpu_device_request": device_request_matches,
        "mounts_exact": mounts == expected_mounts,
        "control_root_read_only": mounts.get(expected["control_root"], {}).get("rw") is False,
        "own_deployment_read_write": mounts.get(expected["lane_deployment_root"], {}).get("rw") is True,
        "own_profile_read_write": mounts.get(expected["lane_profile_root"], {}).get("rw") is True,
        "other_lane_deployment_not_writable": not mounts.get(expected["other_lane_deployment_root"], {}).get("rw", False),
        "other_lane_profile_not_writable": not mounts.get(expected["other_lane_profile_root"], {}).get("rw", False),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "container": expected["container"],
        "lane": expected["lane"],
        "container_id": inspected.get("Id"),
        "image_id": inspected.get("Image"),
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "recorded_unix": time.time(),
    }


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: validate_t5_isaac_worker_spec.py EXPECTED_JSON INSPECT_JSON RESULT_JSON", file=sys.stderr)
        return 64
    expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    raw = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    inspected = raw[0] if isinstance(raw, list) else raw
    result = validate(expected, inspected)
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        print("container specification drift: " + ", ".join(result["failed_checks"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
