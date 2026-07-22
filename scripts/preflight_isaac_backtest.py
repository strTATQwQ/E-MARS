#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from omninav_cosmos.backtest import ExpectedModel, validate_seed_manifest, validate_service_health
from omninav_cosmos.transports.zmq_client import ZmqNavigationClient


def _isaac_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return processes
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except (OSError, PermissionError):
            continue
        if "go2_warehouse_waypoint_nav.py" not in command:
            continue
        processes.append({"pid": int(entry.name), "command": command})
    return sorted(processes, key=lambda value: value["pid"])


def _validate_headless_processes(processes: list[dict[str, Any]]) -> None:
    if not processes:
        raise RuntimeError("no live go2_warehouse_waypoint_nav.py Isaac process found")
    errors: list[str] = []
    for process in processes:
        command = str(process["command"])
        required = ("--headless", "--enable_cameras", "--ideal_kinematic_base", "--control_mode ros_twist")
        missing = [value for value in required if value not in command]
        if missing:
            errors.append(f"pid {process['pid']} missing {missing}")
        if "--viz" in command:
            errors.append(f"pid {process['pid']} contains --viz")
    if errors:
        raise RuntimeError("; ".join(errors))


def _probe_camera(topic: str, timeout_sec: float) -> dict[str, Any]:
    command = ["ros2", "topic", "echo", "--once", topic, "sensor_msgs/msg/Image"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
    if result.returncode != 0 or "width:" not in result.stdout or "height:" not in result.stdout:
        raise RuntimeError(
            f"real camera topic probe failed rc={result.returncode}: {(result.stderr or result.stdout)[-500:]}"
        )
    width = height = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("width:"):
            width = int(stripped.split(":", 1)[1])
        elif stripped.startswith("height:"):
            height = int(stripped.split(":", 1)[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"camera dimensions invalid: {width}x{height}")
    return {"topic": topic, "width": width, "height": height}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict preflight for a real OmniNav Isaac headless backtest.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--expected-model-variant", required=True)
    parser.add_argument("--expected-precision", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--camera-topic", default="/camera/front/isaac_image")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result: dict[str, Any] = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "endpoint": args.endpoint,
        "passed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = ZmqNavigationClient(args.endpoint, timeout_ms=max(1, int(args.timeout_sec * 1000.0)))
        try:
            health = client.health()
        finally:
            client.close()
        expected = ExpectedModel(args.expected_model_variant, args.expected_precision, True)
        result["model_health"] = validate_service_health(health, expected)
        manifest = json.loads(Path(args.seed_manifest).read_text(encoding="utf-8"))
        result["seed_manifest"] = validate_seed_manifest(manifest)
        processes = _isaac_processes()
        _validate_headless_processes(processes)
        result["isaac_processes"] = processes
        result["camera"] = _probe_camera(args.camera_topic, args.timeout_sec)
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
