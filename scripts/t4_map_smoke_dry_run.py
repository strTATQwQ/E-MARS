#!/usr/bin/env python3
"""Run the deterministic map smoke without starting a process or network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.map.smoke import run_dry_smoke


def main() -> int:
    try:
        payload = run_dry_smoke()
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 2
    except Exception as exc:
        print(
            json.dumps(
                {"schema_version": 1, "status": "FAIL", "error": str(exc)},
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
