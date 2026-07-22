#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
SCHEDULER = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
for path in (PKG, SCHEDULER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_vln_benchmark.compositional_micro_suite import build_v19_blocker_micro_suite
from isaac_vln_benchmark.config_loader import dump_data, load_data
from isaac_vln_benchmark.semantic_navigation_benchmark import validate_task_set


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize targeted V19 compositional micro suites.")
    parser.add_argument("--stage", choices=("single", "chain"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    tasks = load_data(ROOT / "configs" / "generated" / "natural_language_navigation_v19_tasks.yaml")
    scenes = load_data(ROOT / "configs" / "generated" / "natural_language_navigation_v19_scenes.yaml")
    micro_tasks, micro_scenes = build_v19_blocker_micro_suite(
        tasks,
        scenes,
        stage=args.stage,
        seeds=tuple(args.seeds),
    )
    validate_task_set(micro_tasks, require_full=False)
    prefix = ROOT / "configs" / "generated" / f"v19_compositional_{args.stage}"
    dump_data(micro_tasks, f"{prefix}_tasks.yaml")
    dump_data(micro_scenes, f"{prefix}_scenes.yaml")
    audit = {
        "schema_version": 1,
        "stage": args.stage,
        "tasks": len(micro_tasks["tasks"]),
        "scenes": len(micro_scenes["scenes"]),
        "seeds": list(args.seeds),
        "parent_tasks": sorted({task["semantic_runtime"]["parent_task_id"] for task in micro_tasks["tasks"]}),
        "runtime_profile": "v19_local_marker_v2",
        "controller_oracle_fields": False,
        "qualification_evidence": False,
        "pass": len(micro_tasks["tasks"]) == 18 and len(micro_scenes["scenes"]) == 18,
    }
    Path(f"{prefix}_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
