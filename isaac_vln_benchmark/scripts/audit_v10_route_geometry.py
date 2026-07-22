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

from isaac_vln_benchmark.v10_geometry_utils import BRANCH_MATRIX_COLUMNS, route_geometry_audit, write_common_v10_artifacts, write_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit V10 route geometry frames and direction mapping.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "v10_route_geometry_audit.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "v10_route_geometry_audit"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT.parent / out if out.parts and out.parts[0] == ROOT.name else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    audit = route_geometry_audit(Path(args.config))
    metrics = {"episodes": [], "v10_route_geometry_audit": {"pass": bool(audit.get("direction_mapping_valid")), "audit": audit}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "events.jsonl").write_text("", encoding="utf-8")
    write_common_v10_artifacts(out, metrics, geometry_audit=audit)
    write_csv(
        out / "branch_confusion_matrix.csv",
        [
            {"instruction_branch": "left", "entered_left": 0, "entered_right": 0, "remained_intersection": 0, "left_workspace": 0},
            {"instruction_branch": "right", "entered_left": 0, "entered_right": 0, "remained_intersection": 0, "left_workspace": 0},
        ],
        BRANCH_MATRIX_COLUMNS,
    )
    (out / "summary.md").write_text(
        "# V10 Route Geometry Audit\n\n"
        f"- direction_mapping_valid: {audit.get('direction_mapping_valid')}\n"
        f"- intersection_center: {audit.get('intersection_center')}\n"
        f"- left_branch_heading_deg: {audit.get('left_branch_heading_deg')}\n"
        f"- right_branch_heading_deg: {audit.get('right_branch_heading_deg')}\n"
        "- Step: FROZEN\n- Sim2Real: NOT READY FOR REAL ROBOT AUTONOMY\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), "route_geometry_audit": audit}, indent=2, ensure_ascii=False))
    return 0 if audit.get("direction_mapping_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
