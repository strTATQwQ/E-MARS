#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <a|b> [output.json]\n' "${0##*/}" >&2
  exit 64
}

[[ $# -ge 1 && $# -le 2 ]] || usage
lane="$1"
output="${2:-}"
case "$lane" in
  a)
    target="${DGX_A_USER:-railgun}@${DGX_A_HOST:-10.100.100.128}"
    step3_root="/home/railgun/ai-stack"
    ;;
  b)
    target="${DGX_B_USER:-rail}@${DGX_B_HOST:-10.100.120.122}"
    step3_root="/home/rail/ai-stack"
    ;;
  *) usage ;;
esac

[[ "$target" =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$ ]] || usage
if [[ -n "$output" ]]; then
  mkdir -p -- "$(dirname -- "$output")"
fi

probe() {
  ssh -T -o BatchMode=yes -o ConnectTimeout=8 "$target" \
    python3 - "$lane" "$step3_root" <<'PY'
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


lane, step3_root = sys.argv[1:]


def command(argv, *, shell=False):
    completed = subprocess.run(
        argv,
        shell=shell,
        executable="/bin/bash" if shell else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    return completed.returncode, completed.stdout.strip()


def ros_prefix(package):
    rc, value = command(
        f"source /opt/ros/jazzy/setup.bash && ros2 pkg prefix {package}",
        shell=True,
    )
    return value if rc == 0 else None


def apt_candidate(package):
    rc, value = command(["apt-cache", "policy", package])
    if rc != 0:
        return None
    for line in value.splitlines():
        if line.strip().startswith("Candidate:"):
            candidate = line.split(":", 1)[1].strip()
            return None if candidate == "(none)" else candidate
    return None


def version_from_venv(venv, package):
    python = Path(venv) / "bin/python"
    if not python.is_file():
        return None
    rc, value = command(
        [str(python), "-c", f"import {package};print({package}.__version__)"]
    )
    return value if rc == 0 else None


_, disk = command(["df", "-B1", "/home"])
disk_line = disk.splitlines()[-1].split() if disk else []
meminfo = {}
for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.split()
    if fields:
        meminfo[key] = int(fields[0]) * 1024
model = Path(step3_root) / "models/Step3-VL-10B"
venv_candidates = [
    Path(step3_root) / "venvs/step3-vl-10b",
    Path(step3_root) / "venvs/step-vllm",
    Path(step3_root) / "venvs/step-vllm-sglang",
]
payload = {
    "schema_version": 1,
    "lane": lane,
    "host": platform.node(),
    "user": os.environ.get("USER"),
    "architecture": platform.machine(),
    "ros_distro": "jazzy" if Path("/opt/ros/jazzy/setup.bash").is_file() else None,
    "installed": {
        "nvblox_ros_prefix": ros_prefix("nvblox_ros"),
        "nvblox_nav2_prefix": ros_prefix("nvblox_nav2"),
        "isaac_ros_visual_slam_prefix": ros_prefix("isaac_ros_visual_slam"),
    },
    "apt_candidate": {
        "ros-jazzy-isaac-ros-nvblox": apt_candidate(
            "ros-jazzy-isaac-ros-nvblox"
        ),
        "ros-jazzy-isaac-ros-visual-slam": apt_candidate(
            "ros-jazzy-isaac-ros-visual-slam"
        ),
    },
    "container": {
        "docker": shutil.which("docker"),
        "nvidia_container_runtime": shutil.which("nvidia-container-runtime"),
    },
    "capacity": {
        "home_available_bytes": int(disk_line[3]) if len(disk_line) >= 4 else None,
        "memory_available_bytes": meminfo.get("MemAvailable"),
    },
    "step3": {
        "model_path": str(model),
        "model_present": model.is_dir(),
        "transformers_versions": {
            str(venv): version_from_venv(venv, "transformers")
            for venv in venv_candidates
            if venv.is_dir()
        },
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

if [[ -n "$output" ]]; then
  probe | tee "$output"
else
  probe
fi
