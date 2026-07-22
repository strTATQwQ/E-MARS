#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print an episode event timeline.")
    parser.add_argument("episode_dir")
    args = parser.parse_args()
    path = Path(args.episode_dir) / "events.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        print(f"{event.get('t', 0):8.3f} {event.get('event')} {event.get('state', '')} {event.get('details', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
