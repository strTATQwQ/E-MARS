#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.config_loader import dump_data
from isaac_vln_benchmark.transport_stress_utils import evaluate_transport_profile, evaluate_transport_stress, write_transport_stress


PROFILES = [
    {
        "name": "latency",
        "delay": 0.20,
        "drop": 0.0,
        "reset": False,
        "seed": 201,
        "tasks": {
            "turn_choice": {"ids": ["turn_001"], "count": 1, "timeout_sec": 180},
            "semantic_target": {"ids": ["semantic_001"], "count": 1, "timeout_sec": 180},
        },
    },
    {
        "name": "packet_loss",
        "delay": 0.0,
        "drop": 1.0,
        "reset": False,
        "seed": 202,
        "tasks": {"turn_choice": {"ids": ["turn_001"], "count": 1, "timeout_sec": 120}},
    },
    {
        "name": "reset",
        "delay": 1.0,
        "drop": 0.0,
        "reset": True,
        "seed": 203,
        "tasks": {"turn_choice": {"ids": ["turn_002"], "count": 1, "timeout_sec": 120}},
    },
]


def session_runner_module():
    path = ROOT / "scripts" / "run_sim2real_low_speed_sessions.py"
    spec = importlib.util.spec_from_file_location("low_speed_session_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def config_for(profile: dict) -> dict:
    return {
        "benchmark": {
            "name": f"sim2real_vnext_transport_stress_{profile['name']}",
            "timeout_sec_default": 180,
            "seeds": [profile["seed"]],
            "modes": ["omninav_step_route_stop_sim2real_low_speed"],
        },
        "tasks": profile["tasks"],
        "policy": {
            "real_step_required": True,
            "forced_oracle": "forbidden",
            "max_linear_x_mps": 0.20,
            "max_yaw_rate_radps": 0.30,
            "real_robot": "disabled",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--isaac-host", default="10.100.120.111")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="ssh-ed25519 255 2e:c0:7c:18:c0:ad:e6:e6:9b:04:f8:2f:e1:2a:0e:60")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="spark")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT.parent / output if output.parts and output.parts[0] == ROOT.name else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    restart = session_runner_module().restart_isaac
    results = []
    for index, profile in enumerate(PROFILES, start=1):
        profile_dir = output / profile["name"]
        profile_dir.mkdir(parents=True, exist_ok=True)
        config_path = profile_dir / "requested_config.yaml"
        dump_data(config_for(profile), config_path)
        launch = restart(args, index)
        (profile_dir / "isaac_launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config", str(config_path),
            "--modes", "omninav_step_route_stop_sim2real_low_speed",
            "--max-episodes", str(sum(int(row.get("count", 1)) for row in profile["tasks"].values())),
            "--output", str(profile_dir),
            "--real-omninav", "--real-step", "--decision-stress", "--qualification-bag",
            "--decision-stress-delay-sec", str(profile["delay"]),
            "--decision-stress-drop-rate", str(profile["drop"]),
            "--decision-stress-profile", profile["name"],
            "--scheduler-config", str(ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_sim2real.yaml"),
            "--isaac-host", args.isaac_host,
            "--isaac-user", args.isaac_user,
            "--isaac-password", args.isaac_password,
            "--isaac-hostkey", args.isaac_hostkey,
            "--dgx-user", args.dgx_user,
            "--dgx-password", args.dgx_password,
        ]
        if profile["reset"]:
            command.append("--decision-stress-reset-on-raw")
        subprocess.run(command, check=False)
        result = evaluate_transport_profile(
            profile_dir,
            profile=profile["name"],
            requested_delay_sec=profile["delay"],
            requested_drop_rate=profile["drop"],
            reset_expected=profile["reset"],
        )
        (profile_dir / "stress_profile.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(result)
    overall = evaluate_transport_stress(results)
    write_transport_stress(output, overall)
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    return 0 if overall["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
