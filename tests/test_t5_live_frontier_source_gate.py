from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_missing_source_gate_writes_fail_closed_blocker(tmp_path) -> None:
    output = tmp_path / "gate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_t5_live_frontier_source.py"),
            "--snapshot",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
            "--sim-now-s",
            "10.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 75
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "BLOCKED"
    assert value["blocker_code"] == "MISSING_LIVE_NAV2_FRONTIER_SOURCE"
    assert value["bounded_advisor_request_admission"] is False
    assert value["live_navigation_promotion_eligible"] is False
    assert value["motion_authority"] == "none"
    assert value["terminal_stop_authority"] == "none"
