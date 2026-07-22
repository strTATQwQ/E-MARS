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
from run_v13_paired_confirmation import main as run_confirmation


def require_multimodal_screen(path: Path) -> dict:
    gate = json.loads((path / "multimodal_value_gate.json").read_text(encoding="utf-8"))
    if not bool(gate.get("pass")):
        raise SystemExit("fresh-image multimodal screening gate has not allowed paired confirmation")
    return gate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gated randomized 45-pair true-multimodal confirmation.")
    parser.add_argument("--screen-run", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--scheduler-config", default=str(ROOT.parent / "ros2_ws/src/omninav_step_scheduler/config/scheduler_isaac_v14_multimodal.yaml"))
    parser.add_argument("--dgx-host", default="10.100.100.128")
    parser.add_argument("--dgx-user", default="railgun")
    parser.add_argument("--dgx-password", default="spark")
    parser.add_argument("--plink", default=r"C:\Program Files\PuTTY\plink.exe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    screen = Path(args.screen_run)
    require_multimodal_screen(screen)
    output = Path(args.output) if args.output else ROOT / "runs" / f"v14_multimodal_paired_{datetime.now():%Y%m%d_%H%M%S}"
    command = [
        "--screen-run", str(screen),
        "--output", str(output),
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
    result = run_confirmation(command)
    if args.dry_run:
        return result
    old_gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    multimodal = evaluate_multimodal_value_run(output, existing_gate=old_gate)
    multimodal["screen_run"] = str(screen)
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
