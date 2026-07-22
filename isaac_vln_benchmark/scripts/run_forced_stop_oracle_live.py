#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v4_live_probe import mock_stop_run, run_stop_live
from isaac_vln_benchmark.v4_remote import run_remote_probe


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "forced_stop_oracle_live.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--remote-live", action="store_true")
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--ros-domain-id", type=int, default=42)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = Path(args.output) if args.output else ROOT / "runs" / f"forced_stop_oracle_live_{stamp}"
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)

    if args.remote_live:
        cfg = load_yaml(config_path)
        metrics = run_stop_live(cfg, args.output, project_root=ROOT)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0 if metrics.get("pass") else 2
    if args.mock_models:
        cfg = load_yaml(config_path)
        metrics = mock_stop_run(cfg, args.output)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0 if metrics.get("pass") else 2
    return run_remote_probe(args, script_name="run_forced_stop_oracle_live.py", config_path=config_path)


if __name__ == "__main__":
    raise SystemExit(main())
