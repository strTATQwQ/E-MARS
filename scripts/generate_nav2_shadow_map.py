#!/usr/bin/env python3
"""Generate the deterministic free-space map used only by the T1.2 shadow gate."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1280)
    arguments = parser.parse_args()
    if arguments.width <= 0 or arguments.height <= 0:
        raise SystemExit("width and height must be positive")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{arguments.width} {arguments.height}\n255\n".encode("ascii")
    # 254 is free under Nav2 trinary mode. This map validates only coordinate,
    # timestamp, lifecycle, and goal-reachability plumbing; active gates must
    # use scene-aware maps and never reuse this artifact.
    arguments.output.write_bytes(header + bytes([254]) * (arguments.width * arguments.height))
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
