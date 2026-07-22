from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ALLOWED_WHILE_NOT_READY = [
    "sensor-only dry-run",
    "bag replay",
    "offline scoring",
    "stationary camera validation",
    "manual-triggered primitive test",
]


def evaluate_sim2real_vnext(
    *,
    value_confirmation: dict[str, Any],
    route_micro: dict[str, Any],
    stop_micro: dict[str, Any],
    independent_sessions: list[dict[str, Any]],
    stress: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "value_confirmation_pass": bool(value_confirmation.get("pass")),
        "max_linear_x_lte_0p20": all(float(session.get("max_linear_x_mps", 999.0)) <= 0.20 for session in independent_sessions) and len(independent_sessions) >= 3,
        "route_correct_gte_0p80": float(route_micro.get("correct_branch_rate", 0.0)) >= 0.80,
        "semantic_stop_gte_0p90": float(stop_micro.get("stop_decision_accuracy", 0.0)) >= 0.90,
        "stop_latency_p95_lte_2s": float(stop_micro.get("visible_to_stop_latency_p95_sec", 999.0)) <= 2.0,
        "three_independent_sessions": len(independent_sessions) >= 3 and all(bool(session.get("pass")) for session in independent_sessions[:3]),
        "runtime_stale_zero": len(independent_sessions) >= 3 and all(int(session.get("runtime_stale_discards", 1)) == 0 for session in independent_sessions[:3]),
        "timebase_zero": len(independent_sessions) >= 3 and all(int(session.get("timebase_error", 1)) == 0 for session in independent_sessions[:3]),
        "parse_zero": len(independent_sessions) >= 3 and all(int(session.get("parse_error", 1)) == 0 for session in independent_sessions[:3]),
        "collision_zero": len(independent_sessions) >= 3 and all(int(session.get("collision_count", 1)) == 0 for session in independent_sessions[:3]),
        "fall_zero": len(independent_sessions) >= 3 and all(int(session.get("fall_count", 1)) == 0 for session in independent_sessions[:3]),
        "stale_action_zero": len(independent_sessions) >= 3 and all(int(session.get("stale_action_executed", 1)) == 0 for session in independent_sessions[:3]),
        "latency_packet_loss_reset_stress": bool(stress.get("pass")),
        "bag_replay_pass": bool(replay.get("pass")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    ready = not failures
    return {
        "schema_version": 1,
        "ready": ready,
        "status": "READY FOR GUARDED REAL-BRIDGE QUALIFICATION" if ready else "NOT READY FOR REAL ROBOT AUTONOMY",
        "checks": checks,
        "failures": failures,
        "allowed_only": [] if ready else ALLOWED_WHILE_NOT_READY,
        "internnav_identity": "CmaAgent/system1/fallback_static_cma_tokens",
        "full_internvla_n1": False,
        "real_go2_autonomy_enabled": False,
    }


def write_sim2real_vnext(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# Sim2Real Gate: {result['status']}",
        "",
        f"- ready: {result['ready']}",
        f"- InternNav: `{result['internnav_identity']}`",
        "- full InternVLA-N1: false",
        "- real Go2 autonomy enabled: false",
        "",
        "## Checks",
        "",
    ]
    lines += [f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in result["checks"].items()]
    if result["allowed_only"]:
        lines += ["", "## Allowed Only", ""] + [f"- {item}" for item in result["allowed_only"]]
    (output / "sim2real_gate_vnext.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
