#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
SCHEDULER_ROOT = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
for path in (PKG_ROOT, SCHEDULER_ROOT):
    sys.path.insert(0, str(path))

from isaac_vln_benchmark.config_loader import dump_data
from isaac_vln_benchmark.semantic_navigation_benchmark import load_task_set
from isaac_vln_benchmark.semantic_navigation_runtime import materialize_semantic_navigation_runtime, runtime_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the V18 semantic benchmark into executable marker scenes.")
    parser.add_argument("--source", default=str(ROOT / "configs" / "natural_language_navigation_v18.yaml"))
    parser.add_argument(
        "--tasks-output",
        default=str(ROOT / "configs" / "generated" / "natural_language_navigation_v18_tasks.yaml"),
    )
    parser.add_argument(
        "--scenes-output",
        default=str(ROOT / "configs" / "generated" / "natural_language_navigation_v18_scenes.yaml"),
    )
    parser.add_argument("--audit-output", default="")
    args = parser.parse_args()

    source = load_task_set(args.source)
    tasks, scenes = materialize_semantic_navigation_runtime(source)
    audit = runtime_audit(tasks, scenes)
    if not audit["pass"]:
        raise SystemExit(json.dumps(audit, indent=2))
    tasks_path = Path(args.tasks_output)
    scenes_path = Path(args.scenes_output)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    scenes_path.parent.mkdir(parents=True, exist_ok=True)
    dump_data(tasks, tasks_path)
    dump_data(scenes, scenes_path)
    audit_path = Path(args.audit_output) if args.audit_output else tasks_path.parent / "natural_language_navigation_v18_runtime_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": str(tasks_path), "scenes": str(scenes_path), "audit": audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
