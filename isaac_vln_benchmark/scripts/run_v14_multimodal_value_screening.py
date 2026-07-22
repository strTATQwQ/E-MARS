#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
for path in (PKG_ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

from isaac_vln_benchmark.multimodal_value_evidence import evaluate_multimodal_value_run
from isaac_vln_benchmark.dgx_model_fingerprint import capture_dgx_model_fingerprint, write_fingerprint
from run_v12_step_value_screening import main as run_screening


def require_perception_suite(path: Path) -> dict:
    gate = json.loads((path / "gate.json").read_text(encoding="utf-8"))
    if not bool(gate.get("pass")) or not bool(gate.get("screening_allowed")):
        raise SystemExit("perception/tracking/planning suite has not allowed multimodal screening")
    if bool(gate.get("qualification_evidence")):
        raise SystemExit("ideal-kinematic suite must not be marked as qualification evidence")
    return gate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gated 15-episode-per-mode true-multimodal value screen.")
    parser.add_argument("--perception-suite", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--scheduler-config", default=str(ROOT.parent / "ros2_ws/src/omninav_step_scheduler/config/scheduler_isaac_v14_multimodal.yaml"))
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--dgx-host", default="10.100.100.128")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="spark")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = Path(args.perception_suite)
    require_perception_suite(suite)
    output = Path(args.output) if args.output else ROOT / "runs" / f"v14_multimodal_screening_{datetime.now():%Y%m%d_%H%M%S}"
    command = [
        "--output", str(output),
        "--rerun-v11-pair",
        "--scheduler-config", str(Path(args.scheduler_config)),
    ]
    if args.isaac_hostkey:
        command += ["--isaac-hostkey", args.isaac_hostkey]
    if args.dry_run:
        command += ["--dry-run"]
    else:
        output.mkdir(parents=True, exist_ok=True)
        fingerprint = capture_dgx_model_fingerprint(
            plink=args.plink,
            host=args.dgx_host,
            user=args.dgx_user,
            password=args.dgx_password,
        )
        write_fingerprint(output / "dgx_model_fingerprint.json", fingerprint)
    result = run_screening(command)
    if args.dry_run:
        return result
    old_gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    multimodal = evaluate_multimodal_value_run(output, existing_gate=old_gate)
    multimodal["perception_suite"] = str(suite)
    multimodal["frozen_inputs"] = {
        "scheduler_config": str(Path(args.scheduler_config)),
        "scheduler_config_sha256": sha256(Path(args.scheduler_config)),
        "dgx_model_fingerprint": "dgx_model_fingerprint.json",
    }
    (output / "multimodal_value_gate.json").write_text(json.dumps(multimodal, indent=2) + "\n", encoding="utf-8")
    (output / "multimodal_http_evidence.json").write_text(
        json.dumps(multimodal["multimodal_evidence"], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(multimodal, indent=2))
    return 0 if result == 0 and multimodal["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
