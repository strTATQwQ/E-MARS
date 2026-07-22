from __future__ import annotations

import sys
from pathlib import Path


FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "frontend"
if not (FRONTEND_ROOT / "pyproject.toml").is_file():
    raise RuntimeError(
        "frontend submodule is not initialized; run "
        "`git submodule update --init --recursive`"
    )
sys.path.insert(0, str(FRONTEND_ROOT))
