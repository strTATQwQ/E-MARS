#!/usr/bin/env bash
set -euo pipefail

HOST="${ISAAC_HOST:-}"
USER_NAME="${ISAAC_USER:-}"
OUTPUT="${1:-isaac_probe.json}"
CONNECT_TIMEOUT="${ISAAC_SSH_CONNECT_TIMEOUT_SEC:-8}"

if [[ -z "$HOST" || -z "$USER_NAME" ]]; then
  echo "ISAAC_HOST and ISAAC_USER are required" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
SSH=(ssh -o "ConnectTimeout=$CONNECT_TIMEOUT" -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)
if [[ -n "${ISAAC_SSH_PASSWORD:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    python3 - "$OUTPUT" "$HOST" <<'PY'
import json, sys, time
path, host = sys.argv[1:]
json.dump({
    "status": "BLOCKED",
    "host": host,
    "timestamp": time.time(),
    "error": "ISAAC_SSH_PASSWORD was provided but sshpass is not installed",
    "minimum_manual_action": "Install sshpass or configure SSH key authentication.",
}, open(path, "w"), indent=2)
PY
    exit 3
  fi
  export SSHPASS="$ISAAC_SSH_PASSWORD"
  SSH=(sshpass -e "${SSH[@]}")
else
  SSH+=(-o BatchMode=yes)
fi

TARGET="$USER_NAME@$HOST"
RAW_FILE="${OUTPUT%.json}.raw.log"
ERROR_FILE="${OUTPUT%.json}.error.log"

set +e
"${SSH[@]}" "$TARGET" python3 - >"$RAW_FILE" 2>"$ERROR_FILE" <<'PY'
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import time


def command(args, timeout=15):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "returncode": None, "stdout": "", "stderr": repr(exc)}


version_candidates = []
for pattern in (
    "/home/*/isaacsim/VERSION",
    "/home/*/.local/share/ov/pkg/isaac-sim*/VERSION",
    "/opt/isaac-sim*/VERSION",
    "/isaac-sim/VERSION",
):
    version_candidates.extend(glob.glob(pattern))

launch_candidates = []
for pattern in (
    "/home/*/isaacsim/isaac-sim.sh",
    "/home/*/.local/share/ov/pkg/isaac-sim*/isaac-sim.sh",
    "/opt/isaac-sim*/isaac-sim.sh",
    "/isaac-sim/isaac-sim.sh",
    "/home/*/IsaacLab/isaaclab.sh",
):
    launch_candidates.extend(glob.glob(pattern))

isaac_python_candidates = sorted(set(glob.glob("/home/*/env_isaacsim/bin/python3")))
isaac_python_packages = {}
package_probe = (
    "import importlib.metadata as m, json; "
    "names=['isaacsim','isaaclab','isaacsim-core','isaacsim-ros2','isaacsim-app']; "
    "print(json.dumps({n:(m.version(n) if n in {d.metadata.get('Name') for d in m.distributions()} else None) for n in names}))"
)
for executable in isaac_python_candidates:
    result = command([executable, "-c", package_probe])
    try:
        isaac_python_packages[executable] = json.loads(result["stdout"])
    except Exception:
        isaac_python_packages[executable] = result

versions = {}
for path in sorted(set(version_candidates)):
    try:
        versions[path] = open(path, encoding="utf-8", errors="replace").read().strip()
    except Exception as exc:
        versions[path] = repr(exc)

payload = {
    "status": "OK",
    "timestamp": time.time(),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "os_release": command(["bash", "-lc", "cat /etc/os-release"]),
    "gpu": command(["nvidia-smi", "-L"]),
    "gpu_detail": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    "cuda": command(["bash", "-lc", "nvcc --version || true"]),
    "python": command(["python3", "--version"]),
    "disk": command(["df", "-h", "/"]),
    "isaac_versions": versions,
    "isaac_python_packages": isaac_python_packages,
    "isaac_launch_candidates": sorted(set(launch_candidates)),
    "running_isaac": command(["bash", "-lc", "ps -eo pid,lstart,args | grep -Ei '[i]saac-sim|[k]it/kit|[i]saaclab' || true"]),
    "ros2": command(["bash", "-lc", "command -v ros2 && ros2 --help >/dev/null && echo available || true"]),
    "pyzmq": command(["bash", "-lc", "python3 -c 'import zmq; print(zmq.__version__)' || true"]),
    "ports": {
        "ssh_22": command(["bash", "-lc", "ss -ltn | grep -E ':22\\s' || true"]),
        "listening": command(["bash", "-lc", "ss -ltn"]),
    },
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
SSH_RC=$?
set -e

if [[ $SSH_RC -ne 0 ]]; then
  python3 - "$OUTPUT" "$HOST" "$USER_NAME" "$SSH_RC" "$ERROR_FILE" <<'PY'
import json, pathlib, sys, time
path, host, user, rc, error_path = sys.argv[1:]
error = pathlib.Path(error_path).read_text(encoding="utf-8", errors="replace")
json.dump({
    "status": "BLOCKED",
    "host": host,
    "user": user,
    "timestamp": time.time(),
    "ssh_returncode": int(rc),
    "error": error,
    "minimum_manual_action": "Restore SSH reachability/authentication, then rerun this exact probe.",
}, open(path, "w"), ensure_ascii=False, indent=2)
PY
  cat "$OUTPUT"
  exit "$SSH_RC"
fi

python3 - "$RAW_FILE" "$OUTPUT" "$HOST" "$USER_NAME" <<'PY'
import json, sys
raw_path, output_path, host, user = sys.argv[1:]
payload = json.load(open(raw_path, encoding="utf-8"))
payload["host"] = host
payload["user"] = user
json.dump(payload, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY

cat "$OUTPUT"
