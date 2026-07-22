#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from isaac_vln_benchmark.semantic_value_screening import materialize_v20_screen_tasks


def main() -> int:
    result = materialize_v20_screen_tasks(
        ROOT / "configs" / "generated" / "natural_language_navigation_v19_tasks.yaml",
        ROOT / "configs" / "generated" / "v20_semantic_value_screen_tasks.yaml",
    )
    print(f"materialized {len(result['tasks'])} V20 screening tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
