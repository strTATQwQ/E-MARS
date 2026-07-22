#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.config_loader import dump_data, load_data
from isaac_vln_benchmark.sim2real_session_utils import evaluate_sim2real_session, write_sim2real_session


ISAAC_FLAGS = [
    "--min_stable_linear_command", "0.0",
    "--min_stable_yaw_command", "0.0",
    "--linear_speed", "0.20",
    "--max_yaw_rate", "0.30",
    "--low_speed_gait_servo",
    "--low_speed_motion_guard",
    "--low_speed_guard_speed_trigger", "0.18",
    "--low_speed_guard_yaw_trigger", "0.30",
    "--low_speed_guard_speed_release", "0.08",
    "--low_speed_guard_yaw_release", "0.12",
    "--low_speed_guard_release_steps", "5",
    "--low_speed_gait_command", "0.45",
    "--low_speed_gait_on_below", "0.06",
    "--low_speed_gait_off_above", "0.115",
    "--low_speed_actual_hard_limit", "0.160",
    "--low_speed_total_hard_limit", "0.165",
    "--low_speed_velocity_filter_alpha", "0.20",
]
ISAAC_RESTART_ATTEMPTS = 2
ISAAC_EMPTY_LOG_HUNG_SEC = 60.0
ISAAC_STARTUP_TIMEOUT_SEC = 180.0


def build_isaac_flags(checkpoint: str = "") -> list[str]:
    flags = list(ISAAC_FLAGS)
    if checkpoint:
        flags.extend(["--checkpoint", checkpoint])
    return flags


def plink_base(args: argparse.Namespace) -> list[str]:
    return [args.plink, "-batch", "-pw", args.isaac_password, "-hostkey", args.isaac_hostkey, f"{args.isaac_user}@{args.isaac_host}"]


def restart_isaac(args: argparse.Namespace, session_index: int) -> dict:
    flag_list = build_isaac_flags(args.isaac_checkpoint)
    flags = shlex.join(flag_list)
    attempts: list[dict] = []
    for attempt in range(1, ISAAC_RESTART_ATTEMPTS + 1):
        log = (
            f"/home/song/isaac_projects/logs/go2_sim2real_session_{session_index}_attempt{attempt}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.out"
        )
        remote = (
            "pkill -KILL -f '[g]o2_warehouse_waypoint_nav.py' 2>/dev/null || true; "
            f"sleep 20; nohup /home/song/bin/launch_go2_warehouse_teleop_desktop.sh {flags} "
            f"> '{log}' 2>&1 < /dev/null & echo $!"
        )
        completed = subprocess.run(plink_base(args) + [remote], text=True, capture_output=True, check=True)
        pid = int(completed.stdout.strip().splitlines()[-1])
        started = time.monotonic()
        reason = "startup_timeout"
        while time.monotonic() - started < ISAAC_STARTUP_TIMEOUT_SEC:
            probe = subprocess.run(
                plink_base(args) + [f"grep -q BENCHMARK_CONTROL_READY '{log}' && grep -q TELEOP_READY '{log}'"],
                check=False,
            )
            if probe.returncode == 0:
                attempts.append({"attempt": attempt, "pid": pid, "log": log, "result": "ready"})
                return {
                    "pid": pid,
                    "log": log,
                    "flags": flag_list,
                    "ready": True,
                    "attempt": attempt,
                    "attempts": attempts,
                }
            age = time.monotonic() - started
            if age >= ISAAC_EMPTY_LOG_HUNG_SEC:
                nonempty = subprocess.run(plink_base(args) + [f"test -s '{log}'"], check=False)
                if nonempty.returncode != 0:
                    reason = "empty_log_startup_hung"
                    break
            time.sleep(5.0)
        attempts.append({"attempt": attempt, "pid": pid, "log": log, "result": reason})
        subprocess.run(
            plink_base(args) + ["pkill -KILL -f '[g]o2_warehouse_waypoint_nav.py' 2>/dev/null || true"],
            check=False,
        )
    raise TimeoutError(f"Isaac session {session_index} did not become ready; attempts={attempts}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run three independent low-speed Isaac qualification sessions.")
    parser.add_argument("--session-count", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=101)
    parser.add_argument("--config", default="configs/sim2real_vnext_low_speed_session.yaml")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaac-host", default="10.100.120.111")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="ssh-ed25519 255 2e:c0:7c:18:c0:ad:e6:e6:9b:04:f8:2f:e1:2a:0e:60")
    parser.add_argument("--isaac-checkpoint", default="", help="Optional remote TorchScript policy path for controlled Isaac A/B probes.")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "runs" / f"sim2real_vnext_low_speed_sessions_{stamp}"
    if not output_root.is_absolute():
        output_root = ROOT.parent / output_root if output_root.parts and output_root.parts[0] == ROOT.name else ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    template_path = Path(args.config)
    if not template_path.is_absolute():
        template_path = ROOT / template_path
    template = load_data(template_path)
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "sessions": args.session_count, "seeds": [args.seed_base + i for i in range(args.session_count)], "isaac_flags": build_isaac_flags(args.isaac_checkpoint)}, indent=2))
        return 0

    session_results = []
    for offset in range(args.session_count):
        session_index = offset + 1
        seed = args.seed_base + offset
        session_dir = output_root / f"session_{session_index}_seed{seed}"
        session_dir.mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(template))
        config["benchmark"]["seeds"] = [seed]
        requested_config = session_dir / "requested_config.yaml"
        dump_data(config, requested_config)
        launch = restart_isaac(args, session_index)
        (session_dir / "isaac_launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_live_success_benchmark.py"),
            "--config", str(requested_config),
            "--modes", "omninav_step_route_stop_sim2real_low_speed",
            "--max-episodes", str(max(1, int(args.max_episodes))),
            "--output", str(session_dir),
            "--real-omninav", "--real-step",
            "--scheduler-config", str(ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler" / "config" / "scheduler_isaac_sim2real.yaml"),
            "--isaac-host", args.isaac_host,
            "--isaac-user", args.isaac_user,
            "--isaac-password", args.isaac_password,
            "--isaac-hostkey", args.isaac_hostkey,
            "--dgx-user", args.dgx_user,
            "--dgx-password", args.dgx_password,
        ]
        launcher_code = subprocess.run(command, check=False).returncode
        if not (session_dir / "metrics.json").is_file():
            raise RuntimeError(f"session {session_index} metrics missing after launcher exit {launcher_code}")
        result = evaluate_sim2real_session(session_dir)
        write_sim2real_session(session_dir, result)
        artifact_names = ["requested_config.yaml", "config.yaml", "scheduler_config.yaml", "selected_tasks.yaml", "metrics.json", "events.jsonl", "session_qualification.json", "isaac_launch.json"]
        manifest = {
            "schema_version": 1,
            "session_index": session_index,
            "seed": seed,
            "launcher_return_code": launcher_code,
            "qualification": result,
            "artifacts": [
                {"path": name, "bytes": (session_dir / name).stat().st_size, "sha256": sha256(session_dir / name)}
                for name in artifact_names if (session_dir / name).is_file()
            ],
        }
        (session_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        session_results.append(result)
        if not result["pass"]:
            break

    overall = {
        "schema_version": 1,
        "pass": len(session_results) == args.session_count and all(row["pass"] for row in session_results),
        "required_sessions": args.session_count,
        "completed_sessions": len(session_results),
        "sessions": session_results,
        "real_robot_motion_enabled": False,
    }
    (output_root / "sessions_gate.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "gate": overall}, indent=2, ensure_ascii=False))
    return 0 if overall["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
