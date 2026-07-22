#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v10_geometry_utils import target_pose_audit, write_common_v10_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit V10 target pose and frame consistency.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "v10_target_pose_audit.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "v10_target_pose_audit"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT.parent / out if out.parts and out.parts[0] == ROOT.name else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    audit = target_pose_audit()
    metrics = {"episodes": [], "v10_target_pose_audit": audit}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "events.jsonl").write_text("", encoding="utf-8")
    write_common_v10_artifacts(out, metrics, target_audit=audit)
    (out / "summary.md").write_text(
        "# V10 Target Pose Frame Audit\n\n"
        f"- pass: {audit.get('pass')}\n"
        f"- target_id: {audit.get('target_id')}\n"
        f"- distance_2d_m: {audit.get('distance_2d_m')}\n"
        f"- line_of_sight: {audit.get('line_of_sight')}\n"
        "- Step: FROZEN\n- Sim2Real: NOT READY FOR REAL ROBOT AUTONOMY\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), "target_pose_audit": audit}, indent=2, ensure_ascii=False))
    return 0 if audit.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
