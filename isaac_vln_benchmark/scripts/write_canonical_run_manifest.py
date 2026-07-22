#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.perception_planning_suite import write_artifact_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a SHA256 manifest for a canonical perception/planning run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--metadata-json", default="")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_dir():
        raise SystemExit(f"run directory does not exist: {run}")
    metadata = {
        "qualification_evidence": False,
        "locomotion_fidelity": "ideal_kinematic",
        "internnav_identity": "CmaAgent/system1/fallback_static_cma_tokens",
        "sim2real_gate": "NOT READY FOR REAL ROBOT AUTONOMY",
    }
    if args.metadata_json:
        metadata.update(json.loads(Path(args.metadata_json).read_text(encoding="utf-8")))
    manifest = write_artifact_manifest(run, run_id=run.name, metadata=metadata)
    print(json.dumps({"run": str(run), "artifacts": len(manifest["artifacts"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
