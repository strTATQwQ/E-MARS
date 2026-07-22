#!/usr/bin/env python3
"""Freeze a D435i held-out set excluding every legacy T4.1 camera episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_t4_camera_splits import identity, read, write_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", required=True, type=Path)
    parser.add_argument("--legacy-heldout", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--heldout-output", required=True, type=Path)
    args = parser.parse_args()
    development = read(args.canary)["episodes"]
    legacy_heldout = read(args.legacy_heldout)["episodes"]
    pilot = read(args.pilot)["episodes"]
    excluded = {identity(item) for item in development + legacy_heldout}
    selected = [item for item in pilot if identity(item) not in excluded][:5]
    if len(selected) != 5:
        raise RuntimeError("pilot source lacks five untouched D435i held-out episodes")
    selected_ids = {identity(item) for item in selected}
    if selected_ids & excluded:
        raise RuntimeError("D435i held-out overlaps a previously exposed camera split")
    manifest = write_split(args.heldout_output, selected, args.pilot)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "heldout": manifest,
                "excluded_episode_count": len(excluded),
                "overlap_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
