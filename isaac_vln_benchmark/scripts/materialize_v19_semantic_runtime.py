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

from isaac_vln_benchmark.config_loader import dump_data
from isaac_vln_benchmark.semantic_navigation_benchmark import load_task_set
from isaac_vln_benchmark.semantic_navigation_runtime import (
    V19_PROFILE,
    materialize_semantic_navigation_runtime,
    runtime_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the V19 local-marker semantic runtime.")
    parser.add_argument(
        "--source",
        default=str(ROOT / "configs" / "natural_language_navigation_v18.yaml"),
    )
    parser.add_argument(
        "--tasks-output",
        default=str(ROOT / "configs" / "generated" / "natural_language_navigation_v19_tasks.yaml"),
    )
    parser.add_argument(
        "--scenes-output",
        default=str(ROOT / "configs" / "generated" / "natural_language_navigation_v19_scenes.yaml"),
    )
    parser.add_argument(
        "--audit-output",
        default=str(ROOT / "configs" / "generated" / "natural_language_navigation_v19_runtime_audit.json"),
    )
    args = parser.parse_args()

    source = load_task_set(args.source)
    tasks, scenes = materialize_semantic_navigation_runtime(source, profile=V19_PROFILE)
    audit = runtime_audit(tasks, scenes)
    audit["runtime_profile"] = V19_PROFILE
    dump_data(tasks, args.tasks_output)
    dump_data(scenes, args.scenes_output)
    Path(args.audit_output).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
