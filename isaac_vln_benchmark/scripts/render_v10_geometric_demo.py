#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render or verify V10 geometric demo artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    visual = Path(args.output) if args.output else run_dir / "visual" / "viewport.png"
    if not visual.exists():
        raise SystemExit(f"missing visual artifact: {visual}")
    print(visual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
