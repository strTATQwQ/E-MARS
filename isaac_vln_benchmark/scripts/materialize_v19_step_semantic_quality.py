#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from isaac_vln_benchmark.step_semantic_quality import materialize_public_marker_tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(ROOT / "configs" / "generated" / "natural_language_navigation_v19_tasks.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "configs" / "generated" / "v19_step_semantic_quality_tasks.yaml"),
    )
    args = parser.parse_args()
    result = materialize_public_marker_tasks(args.source, args.output)
    print(f"materialized {len(result['tasks'])} tasks at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
