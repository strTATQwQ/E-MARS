"""Shared hard-gate helpers used by the T0 launch scripts."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


PASS_STATES = {"PASS", "PASSED"}


class GateBlocked(RuntimeError):
    pass


def read_gate_status(path: str | Path) -> dict[str, Any]:
    gate_path = Path(path)
    if not gate_path.is_file():
        raise GateBlocked(f"gate status file is missing: {gate_path}")
    value = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("gates"), dict):
        raise GateBlocked(f"invalid gate status schema: {gate_path}")
    return value


def require_passed_gate(path: str | Path, gate: str) -> dict[str, Any]:
    status = read_gate_status(path)
    row = status["gates"].get(gate)
    if not isinstance(row, dict):
        raise GateBlocked(f"required gate {gate} has no recorded result")
    if str(row.get("status", "")).upper() not in PASS_STATES:
        reason = str(row.get("reason") or "gate did not pass")
        raise GateBlocked(f"required gate {gate} is {row.get('status')}: {reason}")
    return row


def write_blocked_result(
    output: str | Path,
    *,
    gate: str,
    required_gate: str,
    reason: str,
) -> Path:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "blocked.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": gate,
                "status": "NOT_RUN",
                "required_gate": required_gate,
                "reason": reason,
                "timestamp_s": time.time(),
                "real_isaac_episodes": 0,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result_path
