#!/usr/bin/env python3
"""Audit T5 host processes by executable identity, not arbitrary argv substrings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath


PYTHON_NAMES = {"python", "python3"}
SHELL_NAMES = {"bash", "dash", "sh", "zsh"}
RUNTIME_IDENTITIES = {
    "run_internnav_go2_entrypoint.py",
    "internnav_go2_runtime.py",
    "run_go2_continuous_phase.sh",
    "t5_clock_publisher.py",
    "internvla_t4_sensor_bridge",
    "internvla_t4_client",
    "internvla_nav2_oracle_bridge",
}


def _basename(value: str) -> str:
    return Path(value).name.lower()


def _is_python(executable: str) -> bool:
    return executable in PYTHON_NAMES or executable.startswith("python3.")


def _native_ros_package(executable: str) -> str | None:
    """Return the package for an installed ROS native executable.

    ROS 2 launch commonly replaces its Python ``ros2 run`` wrapper with the
    package executable.  In that state ``/proc/*/cmdline`` contains only a
    path such as ``/opt/ros/jazzy/lib/nav2_lifecycle_manager/lifecycle_manager``.
    Restrict recognition to argv[0] and the canonical ROS installation layout
    so diagnostic arguments and log paths cannot masquerade as live compute.
    """

    path = PurePosixPath(executable)
    parts = path.parts
    if (
        not path.is_absolute()
        or len(parts) < 7
        or parts[1:3] != ("opt", "ros")
        or not parts[3]
        or parts[4] != "lib"
    ):
        return None
    package = parts[5].lower()
    if package.startswith(("nav2_", "opennav_")):
        return package
    return None


def process_identities(argv: list[str]) -> set[str]:
    if not argv:
        return set()
    identities = {_basename(argv[0])}
    executable = _basename(argv[0])
    native_ros_package = _native_ros_package(argv[0])
    if native_ros_package:
        identities.update(
            {
                f"ros-native-package:{native_ros_package}",
                f"ros-native-node:{executable}",
                f"ros-native:{native_ros_package}/{executable}",
            }
        )

    if _is_python(executable):
        index = 1
        while index < len(argv):
            token = argv[index]
            if token == "--":
                if index + 1 < len(argv):
                    identities.add(_basename(argv[index + 1]))
                break
            # Inline source is intentionally opaque.  Treating its text as a
            # script identity makes harmless diagnostics such as
            # ``python -c \"print('nvblox')\"`` look like deployed compute.
            if token == "-c" or token.startswith("-c"):
                break
            if token == "-m" and index + 1 < len(argv):
                module = argv[index + 1].lower()
                identities.update({module, module.rsplit(".", 1)[-1]})
                break
            if token.startswith("-m") and len(token) > 2:
                module = token[2:].lower()
                identities.update({module, module.rsplit(".", 1)[-1]})
                break
            if token in {"-W", "-X", "--check-hash-based-pycs"}:
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            identities.add(_basename(token))
            break

    if executable in SHELL_NAMES:
        for index, token in enumerate(argv[1:], start=1):
            if token == "--":
                if index + 1 < len(argv):
                    identities.add(_basename(argv[index + 1]))
                break
            if token.startswith("-"):
                short_flags = token[1:] if not token.startswith("--") else ""
                if token == "--noexec" or "n" in short_flags:
                    break
                if token in {"-c", "--command"} or "c" in short_flags:
                    break
                continue
            identities.add(_basename(token))
            break

    # Only interpret ROS verbs when ROS 2 is the actual executable (or the
    # Python script/module identity already resolved to ros2).  Searching all
    # argv tokens would misclassify diagnostics such as ``rg ros2 run ...``.
    ros2_index = None
    if executable == "ros2" or (_is_python(executable) and "ros2" in identities):
        ros2_index = next(
            (index for index, token in enumerate(argv) if _basename(token) == "ros2"),
            None,
        )
    if ros2_index is not None:
        verb_index = ros2_index + 1
        while (
            verb_index < len(argv)
            and argv[verb_index] == "--use-python-default-buffering"
        ):
            verb_index += 1
        if verb_index < len(argv) and not argv[verb_index].startswith("-"):
            verb = argv[verb_index].lower()
            if verb == "run" and verb_index + 2 < len(argv):
                package = argv[verb_index + 1].lower()
                node = _basename(argv[verb_index + 2])
                identities.update(
                    {
                        package,
                        node,
                        f"{package}/{node}",
                        f"ros2-run-package:{package}",
                        f"ros2-run-node:{node}",
                    }
                )
            elif verb == "launch" and verb_index + 2 < len(argv):
                package = argv[verb_index + 1].lower()
                launch_file = _basename(argv[verb_index + 2])
                identities.update(
                    {
                        f"ros2-launch-package:{package}",
                        f"ros2-launch-file:{launch_file}",
                    }
                )
    return identities


def forbidden_reason(identities: set[str]) -> str | None:
    exact = {
        "internvla_t4_recovery.model_node",
        "internvla_ros2.model_node",
        "internvla_ros2.client_node",
        "internvla_nav2_adapter.shadow_node",
        "internvla_nav2_adapter.active_node",
        "internvla_go2_controller.bridge_node",
        "internvla_t4_sensors.nvblox_supervisor_node",
        "internvla_t4_sensors.odometry_supervisor_node",
        "internvla_t4_recovery.adapter_node",
        "internvla_t4_recovery.recovery_node",
        "internvla_t4_recovery/model_node",
        "internvla_ros2/model_node",
        "internvla_t4_model",
        "internvla_model_node",
        "internvla_t4_client",
        "internvla_t4_sensor_bridge",
        "internvla_t4_sensors.client_node",
        "internvla_client_node",
        "go2_sensor_bridge",
        "warn_relay.py",
        "internvla_nav2_shadow",
        "internvla_nav2_active",
        "internvla_go2_controller_bridge",
        "internvla_t4_nvblox_supervisor",
        "internvla_t4_odometry_supervisor",
        "internvla_t4_recovery",
        "run_t4_model_server.sh",
        "run_slow_model_service.py",
        "run_t4_dgx_onboard.sh",
        "run_t5_dgx_lane.sh",
        "slow_planner.serve",
        "omninav_cosmos.serve",
        "internvla_t4_adapter",
        "internvla_nav2_oracle_bridge",
        "controller_server",
        "planner_server",
        "bt_navigator",
        "behavior_server",
        "smoother_server",
        "waypoint_follower",
        "velocity_smoother",
        "collision_monitor",
        "map_server",
        "amcl",
        "slam_toolbox",
        "ekf_node",
        "ukf_node",
        "visual_slam_node",
        "rtabmap",
        "navsat_transform_node",
        "robot_localization_listener_node",
        "nvblox_node",
        "nvblox_human_node",
        "component_container",
        "component_container_mt",
        "component_container_isolated",
        "lidar_imu_odometry",
        "lidar_odometry",
        "kiss_icp_node",
        "point_lio",
        "fast_lio",
        "lio_sam",
        "hdl_localization",
        "internnav_t5_watchdog",
        "internnav_t5_velocity_adapter",
        "omninav_model_client_node",
        "omninav_step_scheduler.omninav_model_client_node",
        "safe_cmd_mux_node",
        "omninav_step_scheduler.safe_cmd_mux_node",
        "primitive_executor_node",
        "omninav_step_scheduler.primitive_executor_node",
        "sensor_only_planner_node",
        "omninav_step_scheduler.sensor_only_planner_node",
    }
    for identity in sorted(identities):
        if identity in exact:
            return identity
        if identity.startswith(("nav2_", "ros2-run-package:nav2_")):
            return identity
        if identity.startswith(
            ("ros-native-package:nav2_", "ros-native-package:opennav_")
        ):
            return identity
        if identity.startswith(
            (
                "ros2-launch-package:nav2_",
                "ros2-launch-package:nvblox",
                "ros2-run-package:nvblox",
                "ros2-launch-package:robot_localization",
                "ros2-run-package:robot_localization",
                "ros2-launch-package:slam_toolbox",
                "ros2-run-package:slam_toolbox",
                "ros2-launch-package:isaac_ros_visual_slam",
                "ros2-run-package:isaac_ros_visual_slam",
            )
        ):
            return identity
        if identity.startswith("ros2-launch-file:") and any(
            token in identity
            for token in ("nav2", "nvblox", "visual_slam", "cuvslam")
        ):
            return identity
    return None


def deployment_associated(argv: list[str], deployment_root: str) -> bool:
    prefix = deployment_root.rstrip("/") + "/"
    return any(token == deployment_root or token.startswith(prefix) for token in argv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("forbidden-compute", "lane-runtime-residual"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--deployment-root")
    args = parser.parse_args()
    if args.mode == "lane-runtime-residual" and not args.deployment_root:
        raise SystemExit("--deployment-root is required for lane-runtime-residual")

    proc_root = Path(args.proc_root)
    matches: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    scanned = 0
    if not proc_root.is_dir():
        errors.append({"pid": None, "error": "proc_root_missing"})
    else:
        for entry in sorted(proc_root.iterdir(), key=lambda item: item.name):
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except FileNotFoundError:
                continue
            except OSError as exc:
                errors.append({"pid": int(entry.name), "error": type(exc).__name__})
                continue
            if not raw:
                continue
            argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
            identities = process_identities(argv)
            scanned += 1
            reason: str | None
            if args.mode == "forbidden-compute":
                reason = forbidden_reason(identities)
            else:
                reason = next(
                    (item for item in sorted(identities) if item in RUNTIME_IDENTITIES),
                    None,
                )
                if reason and not deployment_associated(argv, str(args.deployment_root)):
                    reason = None
            if reason:
                matches.append(
                    {
                        "pid": int(entry.name),
                        "executable": _basename(argv[0]),
                        "identities": sorted(identities),
                        "reason": reason,
                    }
                )

    status = "ERROR" if errors else ("FAIL" if matches else "PASS")
    payload = {
        "schema_version": 1,
        "status": status,
        "mode": args.mode,
        "proc_root": str(proc_root),
        "deployment_root": args.deployment_root,
        "scanned_process_count": scanned,
        "match_count": len(matches),
        "matches": matches,
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    raise SystemExit(74 if errors else (73 if matches else 0))


if __name__ == "__main__":
    main()
