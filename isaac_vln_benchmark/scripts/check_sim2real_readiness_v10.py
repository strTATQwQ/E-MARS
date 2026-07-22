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

from isaac_vln_benchmark.v10_geometry_utils import write_sim2real_gate_v10


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT.parent / path if path.parts and path.parts[0] == ROOT.name else ROOT / path
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check V10 Sim2Real/V11 gate.")
    parser.add_argument("--route", default="")
    parser.add_argument("--semantic", default="")
    parser.add_argument("--output", default=str(ROOT / "runs" / "sim2real_gate_v10"))
    args = parser.parse_args(argv)
    out = resolve_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    route_metrics = json.loads(resolve_path(args.route).read_text(encoding="utf-8")) if args.route else {}
    semantic_metrics = json.loads(resolve_path(args.semantic).read_text(encoding="utf-8")) if args.semantic else {}
    result = write_sim2real_gate_v10(
        out,
        route=route_metrics.get("v10_branch_entry_controller"),
        semantic=semantic_metrics.get("v10_target_relative_approach"),
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "gate": result}, indent=2, ensure_ascii=False))
    return 0 if result["ready_for_v11"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
