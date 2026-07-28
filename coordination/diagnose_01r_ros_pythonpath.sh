#!/usr/bin/env bash
set -euo pipefail

readonly REMOTE_ROOT="/home/song/internnav-t1-t2"
readonly CONTAINER="${INTERNVLA_T4_CONTAINER_NAME:-internnav_t4_isaac_ros}"

die() {
  printf '01R ROS PYTHONPATH diagnostic: %s\n' "$*" >&2
  exit 64
}

sshpass_prefix() {
  SSH_AUTH=()
  if [[ -n "${ISAAC_PASSWORD_FILE:-}" ]]; then
    [[ -r "$ISAAC_PASSWORD_FILE" ]] || die "ISAAC_PASSWORD_FILE is unreadable"
    SSH_AUTH=(sshpass -f "$ISAAC_PASSWORD_FILE")
  elif [[ -n "${ISAAC_PASSWORD:-}" ]]; then
    export SSHPASS="$ISAAC_PASSWORD"
    SSH_AUTH=(sshpass -e)
  fi
}

run_under_lease() {
  local host="${ISAAC_HOST:-10.100.120.111}"
  local user="${ISAAC_USER:-song}"
  local port="${ISAAC_PORT:-22}"
  local target="${user}@${host}"
  local python_b64 remote_b64
  local -a ssh_options

  sshpass_prefix
  ssh_options=(-T -p "$port" -o ConnectTimeout=8 -o ServerAliveInterval=5
    -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)

  read -r -d '' python_program <<'PY' || true
from __future__ import annotations

import atexit
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import traceback

root = Path(os.environ["CONTROL_ROOT"]).resolve()
raw_pythonpath = os.environ.get("PYTHONPATH", "")
expected_pythonpath = os.pathsep.join(
    (str(root), "/opt/ros/jazzy/lib/python3.12/site-packages")
)
if raw_pythonpath != expected_pythonpath:
    raise SystemExit(
        f"controlled PYTHONPATH mismatch: {raw_pythonpath!r} != {expected_pythonpath!r}"
    )

modules = (
    "numpy",
    "rclpy",
    "tf2_ros",
    "geometry_msgs.msg",
    "nav_msgs.msg",
    "rosgraph_msgs.msg",
    "sensor_msgs.msg",
    "std_msgs.msg",
    "tf2_msgs.msg",
    "sensor_runtime.ros_inner_supervisor",
    "sensor_runtime.ros_sidecar",
    "sensor_runtime.downstream_recorder",
)
origins: dict[str, str] = {}
for name in modules:
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    origins[name] = str(Path(origin).resolve()) if origin else "<namespace-or-builtin>"

bridge_path = root / "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py"
spec = importlib.util.spec_from_file_location("internnav_diag_bridge_node", bridge_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load bridge module from {bridge_path}")
bridge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge_module)
origins["go2_sensor_bridge.bridge_node"] = str(bridge_path.resolve())

for name, origin in origins.items():
    if origin == "<namespace-or-builtin>":
        continue
    resolved = Path(origin)
    if resolved.is_relative_to(Path("/workspaces/isaac/build")):
        raise SystemExit(f"{name} resolved from forbidden build path: {resolved}")
    if resolved.is_relative_to(Path("/workspaces/isaac/install")):
        raise SystemExit(f"{name} resolved from forbidden project install path: {resolved}")

for name in (
    "sensor_runtime.ros_inner_supervisor",
    "sensor_runtime.ros_sidecar",
    "sensor_runtime.downstream_recorder",
    "go2_sensor_bridge.bridge_node",
):
    origin = Path(origins[name])
    if not origin.is_relative_to(root):
        raise SystemExit(f"{name} did not resolve from control root: {origin}")

import rclpy
from sensor_runtime.downstream_recorder import DownstreamRecorder
from sensor_runtime.graph_handshake import (
    BRIDGE_GRAPH_REQUIREMENTS,
    SIDECAR_GRAPH_REQUIREMENTS,
    observe_publishers,
    validate_graph_observation,
)
from sensor_runtime.ros_sidecar import ModelFreeRosSidecar

constructor_probes: dict[str, dict[str, str]] = {}
probe_root = Path(tempfile.mkdtemp(prefix="internnav-01r-ros-constructors-"))
atexit.register(shutil.rmtree, probe_root, ignore_errors=True)
try:
    for name in ("sidecar", "downstream_recorder"):
        node = None
        cleanup_errors: list[str] = []
        rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
        try:
            if name == "sidecar":
                result = probe_root / "sidecar"
                result.mkdir()
                node = ModelFreeRosSidecar(probe_root / "sensor.sock", result)
            else:
                node = DownstreamRecorder(probe_root / "downstream")
            constructor_probes[name] = {"status": "PASS"}
        except BaseException as exc:
            constructor_probes[name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        finally:
            for action_name, action in (
                ("close", None if node is None else getattr(node, "close", None)),
                ("destroy_node", None if node is None else node.destroy_node),
                ("rclpy_shutdown", (lambda: rclpy.shutdown()) if rclpy.ok() else None),
            ):
                if action is None:
                    continue
                try:
                    action()
                except BaseException as exc:
                    cleanup_errors.append(
                        f"{action_name}: {type(exc).__name__}: {exc}"
                    )
            if cleanup_errors:
                constructor_probes[name]["cleanup_errors"] = "; ".join(
                    cleanup_errors
                )
finally:
    # The atexit handler covers constructor failures and all graph-probe exits.
    pass

if any(item["status"] != "PASS" for item in constructor_probes.values()):
    print(json.dumps({"constructor_probes": constructor_probes}, indent=2))
    raise SystemExit("ROS constructor probe failed")

graph_root = probe_root / "graph"
graph_root.mkdir()
sidecar = bridge = recorder = executor = None
spin_thread = None
graph_cleanup_errors: list[str] = []
graph_observations: dict[str, object] = {}
rclpy.init(
    args=[
        "--ros-args",
        "-p",
        "use_sim_time:=true",
        "-p",
        f"result_dir:={graph_root / 'bridge'}",
        "-p",
        "sensor_timeout_sec:=0.35",
    ]
)
try:
    sidecar = ModelFreeRosSidecar(graph_root / "sensor.sock", graph_root)
    bridge = bridge_module.Go2SensorBridge()
    recorder = DownstreamRecorder(graph_root / "downstream")
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor(num_threads=4)
    for node in (sidecar, bridge, recorder):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        graph_observations = {
            "sidecar": validate_graph_observation(
                SIDECAR_GRAPH_REQUIREMENTS,
                observe_publishers(sidecar, sidecar._graph_publishers),
            ),
            "bridge": validate_graph_observation(
                BRIDGE_GRAPH_REQUIREMENTS,
                observe_publishers(bridge, bridge._graph_publishers),
            ),
        }
        if all(
            isinstance(item, dict) and item.get("ready") is True
            for item in graph_observations.values()
        ):
            break
        time.sleep(0.1)
finally:
    for action_name, action in (
        ("executor_shutdown", None if executor is None else executor.shutdown),
        ("sidecar_close", None if sidecar is None else sidecar.close),
        ("bridge_close", None if bridge is None else bridge.close),
        ("recorder_close", None if recorder is None else recorder.close),
        ("sidecar_destroy", None if sidecar is None else sidecar.destroy_node),
        ("bridge_destroy", None if bridge is None else bridge.destroy_node),
        ("recorder_destroy", None if recorder is None else recorder.destroy_node),
        ("rclpy_shutdown", (lambda: rclpy.shutdown()) if rclpy.ok() else None),
    ):
        if action is None:
            continue
        try:
            action()
        except BaseException as exc:
            graph_cleanup_errors.append(
                f"{action_name}: {type(exc).__name__}: {exc}"
            )
    if spin_thread is not None:
        spin_thread.join(2.0)

if graph_cleanup_errors:
    graph_observations["cleanup_errors"] = graph_cleanup_errors
if not all(
    isinstance(graph_observations.get(name), dict)
    and graph_observations[name].get("ready") is True
    for name in ("sidecar", "bridge")
):
    print(json.dumps({"graph_observations": graph_observations}, indent=2))
    raise SystemExit("ROS graph observation probe failed")

print(
    json.dumps(
        {
            "schema_version": 1,
            "status": "PASS",
            "controlled_pythonpath": raw_pythonpath.split(os.pathsep),
            "constructor_probes": constructor_probes,
            "graph_observations": graph_observations,
            "module_origins": origins,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  python_b64="$(printf '%s' "$python_program" | base64 | tr -d '\r\n')"

  read -r -d '' remote_program <<'REMOTE' || true
set -euo pipefail
root="$1"
container="$2"
python_b64="$3"
test -d "$root"
test "$(docker inspect -f '{{.State.Running}}' "$container")" = true
printf 'remote_control_head=%s\n' "$(git -C "$root" rev-parse HEAD)"
docker exec \
  --user admin \
  --workdir "$root" \
  -e "CONTROL_ROOT=$root" \
  -e "DIAG_PY_B64=$python_b64" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$container" \
  bash -lc 'for executable in /usr/bin/env /usr/bin/python3 /usr/bin/setsid /usr/bin/sha256sum; do
  test -x "$executable"
  printf "required_container_executable=%s\n" "$executable"
done
unset PYTHONPATH
source /opt/ros/jazzy/setup.bash
source /workspaces/isaac/install/setup.bash
printf "setup_pythonpath=%s\n" "${PYTHONPATH:-}"
export PYTHONPATH="$CONTROL_ROOT:/opt/ros/jazzy/lib/python3.12/site-packages"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
printf "%s" "$DIAG_PY_B64" | base64 -d | python3 -'
REMOTE
  remote_b64="$(printf '%s' "$remote_program" | base64 | tr -d '\r\n')"

  "${SSH_AUTH[@]}" ssh "${ssh_options[@]}" "$target" \
    "bash -c \"\$(printf '%s' '$remote_b64' | base64 -d)\" diag '$REMOTE_ROOT' '$CONTAINER' '$python_b64'"
}

if [[ "${1:-}" == "_under_lease" ]]; then
  [[ $# -eq 1 ]] || die "invalid internal invocation"
  run_under_lease
  exit $?
fi

[[ $# -eq 1 ]] || die "usage: diagnose_01r_ros_pythonpath.sh DIAGNOSTIC_ID"
diag_id="$1"
[[ "$diag_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe diagnostic id"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
log_dir="$root/results/parallel/sensor_producer/lease-ros-pythonpath-import-${diag_id}"
[[ ! -e "$log_dir" ]] || die "diagnostic result path already exists"

bash "$root/scripts/with_resource_lease.sh" isaac \
  --owner codex-00 \
  --task "01r-ros-pythonpath-import-${diag_id}" \
  --log-dir "$log_dir" \
  -- bash "$script_path" _under_lease
