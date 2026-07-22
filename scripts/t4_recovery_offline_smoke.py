#!/usr/bin/env python3
"""Repository entry point for the T4.5 offline-only recovery smoke."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from t4_completion.recovery.offline_smoke import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
