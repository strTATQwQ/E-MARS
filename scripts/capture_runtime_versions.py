#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _command(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=20, check=False).stdout.strip()
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}:{exc}"


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    payload = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "packages": {
            name: _version(name)
            for name in ("torch", "transformers", "pyzmq", "numpy", "Pillow", "opencv-python", "nvidia-modelopt")
        },
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "git_commit": _command(["git", "-C", str(workspace), "rev-parse", "HEAD"]),
        "git_status": _command(["git", "-C", str(workspace), "status", "--short"]),
        "ros_distro": os.environ.get("ROS_DISTRO", ""),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
