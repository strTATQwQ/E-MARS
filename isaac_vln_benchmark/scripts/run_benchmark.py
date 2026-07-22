#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.benchmark_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
