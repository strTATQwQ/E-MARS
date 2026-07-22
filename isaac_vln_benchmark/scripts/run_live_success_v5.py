#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v5_timebase_utils import analyze_stale_events_v5


MODE_ALIASES = {
    "step_route_stop_v5": "step_route_stop_v4",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def ensure_v5_artifacts(output: Path) -> None:
    events = load_jsonl(output / "events.jsonl")
    analysis = analyze_stale_events_v5(events)
    _write_csv(
        output / "stale_attribution.csv",
        ["attribution", "count"],
        [{"attribution": key, "count": value} for key, value in analysis["counts"].items()],
    )
    if not (output / "timebase_table.csv").exists():
        _write_csv(
            output / "timebase_table.csv",
            ["episode_id", "request_id", "clock_domain", "computed_age_sec", "discard", "discard_reason"],
            [],
        )
    if not (output / "trajectory.csv").exists():
        _write_csv(output / "trajectory.csv", ["episode_id", "t", "x", "y", "yaw", "source"], [])
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics["v5_stale_attribution"] = analysis
    metrics.setdefault("timebase_error_count", analysis["counts"].get("timebase_error", 0))
    metrics.setdefault("episode_mismatch_count", analysis["counts"].get("episode_mismatch", 0))
    metrics.setdefault("missing_timestamp_count", analysis["counts"].get("missing_timestamp", 0))
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "live_success_v5.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=15)
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"v5_{args.mode}_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_mode = MODE_ALIASES.get(args.mode, args.mode)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_live_success_benchmark.py"),
        "--config",
        str(args.config),
        "--modes",
        bench_mode,
        "--max-episodes",
        str(args.max_episodes),
        "--output",
        str(out_dir),
    ]
    if args.mock_models or args.dry_run:
        cmd.append("--mock-models")
    if args.real_omninav:
        cmd.append("--real-omninav")
    if args.isaac_host and not (args.mock_models or args.dry_run):
        cmd += ["--isaac-host", args.isaac_host, "--isaac-user", args.isaac_user, "--isaac-password", args.isaac_password]
    if args.isaac_hostkey:
        cmd += ["--isaac-hostkey", args.isaac_hostkey]
    if args.internnav_server:
        cmd += ["--internnav-server", args.internnav_server]
    if args.dgx_user:
        cmd += ["--dgx-user", args.dgx_user]
    if args.dgx_password:
        cmd += ["--dgx-password", args.dgx_password]
    result = subprocess.run(cmd, check=False)
    ensure_v5_artifacts(out_dir)
    (out_dir / "v5_mode_alias.json").write_text(json.dumps({"requested_mode": args.mode, "benchmark_mode": bench_mode}, indent=2) + "\n", encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
