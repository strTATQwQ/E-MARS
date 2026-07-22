#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_visual_route_stop_demo import SCENES, run_scene


V8_SCENES = ["intersection_forced_route", "semantic_forced_stop"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate V8 route/stop coverage visual overlay demo.")
    parser.add_argument("--output", default="")
    parser.add_argument("--duration-sec", type=int, default=24)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = Path(args.output) if args.output else ROOT / "runs" / f"visual_route_stop_coverage_v8_{stamp}"
    if not root_out.is_absolute():
        root_out = ROOT / root_out
    root_out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for scene_name in V8_SCENES:
        outputs.append(run_scene(scene_name, SCENES[scene_name], root_out / scene_name, args.duration_sec, args.record))
    (root_out / "summary.md").write_text(
        "# V8 Route/Stop Coverage Visual Demo\n\n"
        f"- output: {root_out}\n"
        "- scenes: intersection_forced_route, semantic_forced_stop\n"
        "- overlay includes route/stop JSON, primitive, safe_mux, stale_gate, target visibility, distance, latency, and branch state.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(root_out), "episodes": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
