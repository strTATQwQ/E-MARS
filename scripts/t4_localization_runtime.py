#!/usr/bin/env python3
"""Repository entrypoint for the completion T4.3 localization selector."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from t4_completion.localization.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
