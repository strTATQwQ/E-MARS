from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def remote_output_path(local_output: Path) -> str:
    try:
        rel = local_output.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        rel = Path("runs") / local_output.name
    return f"/home/song/dgx-unitree/isaac_vln_benchmark/{rel.as_posix()}"


def putty_args(binary: str, args: argparse.Namespace) -> list[str]:
    cmd = [binary, "-batch"]
    if getattr(args, "isaac_hostkey", ""):
        cmd += ["-hostkey", args.isaac_hostkey]
    cmd += ["-pw", args.isaac_password]
    return cmd


def run_remote_probe(
    args: argparse.Namespace,
    *,
    script_name: str,
    config_path: Path,
    mock_omninav: bool = False,
    real_step: bool = False,
) -> int:
    plink = shutil.which("plink")
    pscp = shutil.which("pscp")
    if not plink or not pscp:
        raise SystemExit("plink and pscp are required for remote live v4 probes on Windows")
    if not args.isaac_host:
        raise SystemExit("--isaac-host is required unless --remote-live or --mock-models is used")
    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    remote_output = remote_output_path(args.output)
    remote_config = f"{remote_output}/config.yaml"
    subprocess.check_call(putty_args(plink, args) + [f"{args.isaac_user}@{args.isaac_host}", f"mkdir -p {remote_output}"])
    subprocess.check_call(putty_args(pscp, args) + [str(config_path), f"{args.isaac_user}@{args.isaac_host}:{remote_config}"])

    remote_script = f"/home/song/dgx-unitree/isaac_vln_benchmark/scripts/{script_name}"
    command = _remote_command(
        remote_output,
        remote_config,
        remote_script,
        int(args.ros_domain_id),
        mock_omninav=mock_omninav,
        real_step=real_step,
    )
    result = subprocess.run(putty_args(plink, args) + [f"{args.isaac_user}@{args.isaac_host}", command], check=False)
    _fetch_remote_artifacts(args, remote_output)
    return result.returncode


def _remote_command(
    remote_output: str,
    remote_config: str,
    remote_script: str,
    ros_domain_id: int,
    *,
    mock_omninav: bool = False,
    real_step: bool = False,
) -> str:
    mock_omninav_launch = 'start_bg mock_omninav ros2 run omninav_step_scheduler mock_omninav_client_node' if mock_omninav else ':'
    real_step_launch = (
        'start_bg step_http ros2 run omninav_step_scheduler step_http_client_node '
        '--ros-args -p config_file:="$SCHEDULER_CONFIG"'
        if real_step
        else ':'
    )
    enable_local_step_verifiers = "false" if real_step else "true"
    return f"""bash -lc 'set -eo pipefail
RUN_DIR="{remote_output}"
mkdir -p "$RUN_DIR"
source /opt/ros/humble/setup.bash
source /home/song/dgx-unitree/ros2_ws/install/setup.bash
source /home/song/dgx-unitree/isaac_vln_benchmark/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID="{ros_domain_id}"
export RMW_IMPLEMENTATION="${{RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}}"
export GO2_ENABLE_MOTION=1
SCHEDULER_CONFIG="/home/song/dgx-unitree/ros2_ws/install/omninav_step_scheduler/share/omninav_step_scheduler/config/scheduler_isaac_real_models.yaml"
pids=()
start_bg() {{
  local name="$1"
  shift
  "$@" > "$RUN_DIR/${{name}}.log" 2>&1 &
  local pid="$!"
  pids+=("$pid")
  echo "$pid" > "$RUN_DIR/${{name}}.pid"
}}
cleanup() {{
  set +e
  for pid in "${{pids[@]}}"; do kill "$pid" 2>/dev/null || true; done
  pkill -f "[/]lib/isaac_vln_benchmark/go2_benchmark_adapter_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/mission_manager_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_supervisor_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_role_router_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/route_choice_verifier_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/semantic_stop_verifier_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/visible_to_stop_monitor_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/route_stop_primitive_bridge_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/omninav_scheduler_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/internnav_scheduler_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_pending_policy_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/primitive_executor_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/safe_cmd_mux_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/metrics_logger_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/mock_omninav_client_node" 2>/dev/null || true
  pkill -f "[/]lib/omninav_step_scheduler/step_http_client_node" 2>/dev/null || true
  timeout 4s ros2 topic pub --once /safe_cmd_vel geometry_msgs/msg/Twist "{{linear: {{x: 0.0, y: 0.0, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: 0.0}}}}" > "$RUN_DIR/final_zero_safe_cmd.log" 2>&1 || true
  timeout 4s ros2 topic pub --once /go2/cmd_vel geometry_msgs/msg/Twist "{{linear: {{x: 0.0, y: 0.0, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: 0.0}}}}" > "$RUN_DIR/final_zero_go2_cmd.log" 2>&1 || true
}}
trap cleanup EXIT
cleanup
set -e
start_bg scheduler_launch ros2 launch omninav_step_scheduler scheduler_isaac.launch.py \
  config_file:="$SCHEDULER_CONFIG" enable_local_step_verifiers:={enable_local_step_verifiers}
start_bg adapter ros2 run isaac_vln_benchmark go2_benchmark_adapter_node
{mock_omninav_launch}
{real_step_launch}
sleep 6
set +e
python3 "{remote_script}" --remote-live --config "{remote_config}" --output "$RUN_DIR/out" > "$RUN_DIR/runner.log" 2>&1
status="$?"
set -e
pid_files=()
for pid_file in "$RUN_DIR"/*.pid; do
  [ -e "$pid_file" ] && pid_files+=("${{pid_file##*/}}")
done
artifact_paths=(runner.log)
for path in out scheduler_launch.log adapter.log step_http.log final_zero_safe_cmd.log final_zero_go2_cmd.log; do
  [ -e "$RUN_DIR/$path" ] && artifact_paths+=("$path")
done
artifact_paths+=("${{pid_files[@]}}")
tar -czf "$RUN_DIR/report_artifacts.tgz" -C "$RUN_DIR" "${{artifact_paths[@]}}"
exit "$status"
'"""


def _fetch_remote_artifacts(args: argparse.Namespace, remote_output: str) -> None:
    pscp = shutil.which("pscp")
    if not pscp:
        return
    local_tar = args.output / "remote_report_artifacts.tgz"
    remote_tar = f"{remote_output.rstrip('/')}/report_artifacts.tgz"
    subprocess.run(putty_args(pscp, args) + [f"{args.isaac_user}@{args.isaac_host}:{remote_tar}", str(local_tar)], check=False)
    if not local_tar.exists():
        return
    extract_dir = args.output / "_remote"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as archive:
        archive.extractall(extract_dir)
    remote_out = extract_dir / "out"
    if remote_out.exists():
        for child in remote_out.iterdir():
            target = args.output / child.name
            if child.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
