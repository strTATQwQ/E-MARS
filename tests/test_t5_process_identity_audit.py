from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts" / "t5_process_identity_audit.py"


def write_process(proc_root: Path, pid: int, argv: list[str]) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")


def run_audit(
    tmp_path: Path, proc_root: Path, mode: str, deployment_root: str | None = None
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    output = tmp_path / f"{mode}.json"
    command = [
        sys.executable,
        str(AUDITOR),
        "--mode",
        mode,
        "--proc-root",
        str(proc_root),
        "--output",
        str(output),
    ]
    if deployment_root is not None:
        command.extend(["--deployment-root", deployment_root])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_auditor_compiles_and_ignores_diagnostic_argv_substrings(tmp_path: Path) -> None:
    compile(AUDITOR.read_text(encoding="utf-8"), str(AUDITOR), "exec")
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    write_process(
        proc_root,
        100,
        ["codex", "exec", "rg", "/tmp/logs/nav2_amcl_nvblox_watchdog.txt"],
    )
    write_process(proc_root, 101, ["tail", "-f", "/tmp/nav2_controller.log"])
    write_process(proc_root, 102, ["python3", "-c", "print('nvblox')"])
    write_process(
        proc_root,
        103,
        ["python3", "scripts/trace_nvblox_costmap_stages.py", "--input", "/tmp/log"],
    )
    write_process(
        proc_root,
        104,
        ["rg", "ros2", "run", "nav2_map_server", "map_server"],
    )
    write_process(
        proc_root,
        105,
        ["codex", "exec", "ros2", "launch", "nav2_bringup", "navigation_launch.py"],
    )
    write_process(
        proc_root,
        106,
        ["bash", "-n", "scripts/run_t4_model_server.sh"],
    )
    write_process(
        proc_root,
        107,
        ["sh", "--noexec", "scripts/run_t5_dgx_lane.sh"],
    )
    write_process(
        proc_root,
        108,
        ["ros2", "--help", "run", "nav2_map_server", "map_server"],
    )
    native_lifecycle = (
        "/opt/ros/jazzy/lib/nav2_lifecycle_manager/lifecycle_manager"
    )
    write_process(proc_root, 109, ["codex", "exec", native_lifecycle])
    write_process(proc_root, 110, ["rg", "lifecycle_manager", native_lifecycle])
    write_process(proc_root, 111, ["tail", "-f", f"{native_lifecycle}.log"])
    write_process(
        proc_root,
        112,
        ["python3", "scripts/trace_nav2_lifecycle_manager.py", native_lifecycle],
    )
    write_process(proc_root, 113, ["bash", "-n", native_lifecycle])

    result, payload = run_audit(tmp_path, proc_root, "forbidden-compute")
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "PASS"
    assert payload["match_count"] == 0


def test_auditor_finds_native_nav2_and_opennav_executables(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    native_processes = {
        150: (
            "/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager",
            "ros-native-package:nav2_lifecycle_manager",
        ),
        151: (
            "/opt/ros/jazzy/lib/nav2_route/route_server",
            "ros-native-package:nav2_route",
        ),
        152: (
            "/opt/ros/rolling/lib/opennav_docking/opennav_docking",
            "ros-native-package:opennav_docking",
        ),
        153: (
            "/opt/ros/jazzy/lib/opennav_docking/docking_server",
            "ros-native-package:opennav_docking",
        ),
    }
    for pid, (executable, _) in native_processes.items():
        write_process(proc_root, pid, [executable, "--ros-args"])

    result, payload = run_audit(tmp_path, proc_root, "forbidden-compute")
    assert result.returncode == 73
    assert payload["status"] == "FAIL"
    matches = {row["pid"]: row for row in payload["matches"]}
    assert set(matches) == set(native_processes)
    for pid, (executable, reason) in native_processes.items():
        assert matches[pid]["executable"] == Path(executable).name
        assert matches[pid]["reason"] == reason
        assert f"ros-native-node:{Path(executable).name}" in matches[pid]["identities"]


def test_auditor_finds_forbidden_module_and_lane_runtime_identity(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    write_process(
        proc_root,
        200,
        ["/usr/bin/python3", "-m", "internvla_ros2.model_node", "--ros-args"],
    )
    result, payload = run_audit(tmp_path, proc_root, "forbidden-compute")
    assert result.returncode == 73
    assert payload["status"] == "FAIL"
    assert payload["matches"][0]["pid"] == 200

    (proc_root / "200" / "cmdline").unlink()
    write_process(
        proc_root,
        202,
        ["python3", "-W", "ignore", "-X", "dev", "-m", "internvla_ros2.model_node"],
    )
    result, payload = run_audit(tmp_path, proc_root, "forbidden-compute")
    assert result.returncode == 73
    assert payload["matches"][0]["pid"] == 202

    (proc_root / "202" / "cmdline").unlink()
    deployment = "/home/song/.t5-deployments/lane-a"
    write_process(
        proc_root,
        201,
        ["python3", f"{deployment}/scripts/t5_clock_publisher.py", "--port", "25141"],
    )
    result, payload = run_audit(
        tmp_path, proc_root, "lane-runtime-residual", deployment
    )
    assert result.returncode == 73
    assert payload["status"] == "FAIL"
    assert payload["matches"][0]["reason"] == "t5_clock_publisher.py"


def test_auditor_finds_console_entrypoints_and_composable_launches(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    write_process(proc_root, 300, ["/opt/ros/jazzy/lib/internvla_ros2/internvla_model_node"])
    write_process(
        proc_root,
        301,
        ["ros2", "run", "nvblox_ros", "nvblox_node", "--ros-args"],
    )
    write_process(
        proc_root,
        302,
        ["ros2", "launch", "nav2_bringup", "navigation_launch.py"],
    )
    write_process(
        proc_root,
        303,
        ["/opt/ros/jazzy/lib/rclcpp_components/component_container_mt", "--ros-args"],
    )
    write_process(proc_root, 304, ["/work/install/bin/internvla_nav2_active"])
    write_process(proc_root, 305, ["/work/install/bin/internvla_go2_controller_bridge"])
    write_process(proc_root, 306, ["/work/install/bin/internvla_t4_odometry_supervisor"])
    write_process(proc_root, 307, ["/work/install/bin/internvla_t4_nvblox_supervisor"])
    write_process(
        proc_root,
        308,
        ["ros2", "launch", "slam_toolbox", "online_async_launch.py"],
    )
    write_process(
        proc_root,
        309,
        ["ros2", "launch", "internvla_t4_sensors", "t4_cuvslam.launch.py"],
    )
    write_process(proc_root, 310, ["bash", "/work/scripts/run_t5_dgx_lane.sh", "a"])
    write_process(proc_root, 311, ["/work/install/bin/omninav_model_client_node"])
    write_process(proc_root, 312, ["/work/install/bin/safe_cmd_mux_node"])
    write_process(proc_root, 313, ["/work/install/bin/primitive_executor_node"])
    write_process(proc_root, 314, ["/work/install/bin/sensor_only_planner_node"])
    write_process(proc_root, 315, ["python3", "-m", "slow_planner.serve"])
    write_process(proc_root, 316, ["python3", "-m", "omninav_cosmos.serve"])
    write_process(proc_root, 317, ["python3", "scripts/run_slow_model_service.py"])
    write_process(proc_root, 318, ["python3", "-m", "internvla_ros2.client_node"])
    write_process(
        proc_root,
        319,
        ["python3", "-m", "internvla_nav2_adapter.active_node"],
    )
    write_process(
        proc_root,
        320,
        ["python3", "-m", "internvla_go2_controller.bridge_node"],
    )
    write_process(proc_root, 321, ["/work/install/bin/internvla_t4_client"])
    write_process(
        proc_root,
        322,
        ["python3", "-m", "internvla_t4_sensors.client_node"],
    )
    write_process(proc_root, 323, ["/work/install/bin/internvla_client_node"])
    write_process(proc_root, 324, ["/work/install/bin/internvla_t4_sensor_bridge"])
    write_process(proc_root, 325, ["/work/install/bin/go2_sensor_bridge"])
    write_process(proc_root, 326, ["python3", "/work/t4_completion/map/warn_relay.py"])

    result, payload = run_audit(tmp_path, proc_root, "forbidden-compute")
    assert result.returncode == 73
    assert payload["status"] == "FAIL"
    assert {row["pid"] for row in payload["matches"]} == {
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        326,
    }


def test_auditor_fails_closed_when_proc_root_is_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "missing-proc"
    result, payload = run_audit(tmp_path, missing, "forbidden-compute")
    assert result.returncode == 74
    assert payload["status"] == "ERROR"
    assert payload["errors"] == [{"error": "proc_root_missing", "pid": None}]
