#!/usr/bin/env python3
"""Audit the authoritative T3 gate evidence before packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASES = (
    ("t3_0_flash", "t3_0_go2_flash_nav2_sampled_baseline_attempt_007", 20, 0.01, True),
    ("t3_2_diagnostics", "t3_diagnostics_attempt_030", 5, 1.0, False),
    ("t3_2_oracle", "t3_continuous_oracle_attempt_007", 10, 0.9, False),
    ("t3_3_obstacle_oracle", "t3_obstacle_oracle_attempt_008", 10, 0.9, False),
    ("t3_4_canary_no_obstacle", "t3_4_canary_no_obstacle_attempt_007", 5, 0.4, False),
    ("t3_4_pilot_no_obstacle", "t3_4_pilot_no_obstacle_attempt_002", 20, 0.3, False),
    ("t3_4_canary_obstacle", "t3_4_canary_obstacle_aware_attempt_004", 5, 0.4, False),
    ("t3_4_pilot_obstacle", "t3_4_pilot_obstacle_aware_attempt_002", 20, 0.3, False),
    ("t3_4_stress", "t3_4_obstacle_stress_attempt_009", 10, 0.01, False),
)
MODEL_AUDITS = (
    "t3_4_model_attempt_008",
    "t3_4_model_attempt_010",
    "t3_4_model_attempt_017",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    result_root = workspace / "results" / "internnav_t3"
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    for name, directory, expected_count, minimum_sr, flash in PHASES:
        root = result_root / directory
        phase = load(root / "phase_status.json")
        validation = load(root / "validation.json")
        result = load(root / "result.json")
        metrics = result.get("val_unseen", result)
        controller = load(root / "controller_summary.json") if not flash else None
        passed = (
            phase.get("status") == "PASS"
            and all(int(value) == 0 for value in phase.get("exit_codes", {}).values())
            and validation.get("status") == "PASS"
            and int(metrics.get("Count", 0)) == expected_count
            and float(metrics.get("SR", 0.0)) >= minimum_sr
        )
        if controller is not None:
            passed = passed and all(
                int(controller.get(field, -1)) == 0
                for field in (
                    "flash_call_count",
                    "cmd_vel_quantization_count",
                    "direct_motion_bypass_count",
                    "physical_collision_count",
                    "fall_count",
                    "nan_count",
                    "stale_or_identity_reject_count",
                )
            )
            passed = passed and 20.0 <= float(controller.get("measured_control_hz", 0.0)) <= 50.0
        check(
            name,
            passed,
            {
                "directory": directory,
                "Count": metrics.get("Count"),
                "SR": metrics.get("SR"),
                "phase": phase.get("status"),
                "validation": validation.get("status"),
            },
        )

    fault_root = result_root / "t3_1_controller_faults_attempt_017"
    fault = load(fault_root / "fault_validation.json")
    fault_validation = load(fault_root / "validation.json")
    check(
        "t3_1_faults",
        fault.get("status") == "PASS" and fault_validation.get("status") == "PASS",
        {"fault_validation": fault.get("status"), "validation": fault_validation.get("status")},
    )

    stress = load(result_root / "t3_4_obstacle_stress_attempt_009" / "validation.json")
    stress_controller = stress["controller"]
    check(
        "stress_contract",
        stress.get("obstacle_acquired_episode_count") == 10
        and float(stress.get("obstacle_acquisition_rate", 0.0)) == 1.0
        and float(stress_controller.get("obstacle_detection_rate", 0.0)) >= 0.95
        and float(stress_controller.get("costmap_detection_rate", 0.0)) >= 0.95
        and int(stress_controller.get("collision_monitor_stop_count", 0)) > 0
        and int(stress_controller.get("collision_monitor_recovery_count", 0)) > 0,
        {
            "acquired": stress.get("obstacle_acquired_episode_count"),
            "detection_rate": stress_controller.get("obstacle_detection_rate"),
            "costmap_rate": stress_controller.get("costmap_detection_rate"),
            "stop_count": stress_controller.get("collision_monitor_stop_count"),
            "recovery_count": stress_controller.get("collision_monitor_recovery_count"),
        },
    )

    audits = [load(result_root / directory / "model_weight_audit.json") for directory in MODEL_AUDITS]
    inventory = {item.get("inventory_sha256") for item in audits}
    revisions = {(item.get("model_revision"), item.get("checkpoint_revision")) for item in audits}
    check(
        "model_weight_audits",
        all(item.get("status") == "PASS" for item in audits)
        and all(item.get("meta_parameter_count") == item.get("meta_buffer_count") == 0 for item in audits)
        and len(inventory) == 1
        and len(revisions) == 1,
        {"audit_count": len(audits), "inventory_count": len(inventory), "revision_count": len(revisions)},
    )

    comparison = load(result_root / "t3_4_comparison.json")
    check(
        "comparison",
        len(comparison.get("modes", [])) == 3
        and all(item.get("validation_status") == "PASS" for item in comparison.get("modes", [])),
        {"mode_count": len(comparison.get("modes", []))},
    )

    docs = [
        workspace / "WORKLOG_INTERNNAV_T3.md",
        workspace / "reports" / "internnav_t3.md",
        workspace / "docs" / "internnav_io_contract.md",
    ]
    pending_tokens = ("[STRESS_", "Status: `BLOCKED`", "PROVISIONAL")
    doc_text = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    check(
        "documents_frozen",
        not any(token in doc_text for token in pending_tokens),
        {"documents": [path.relative_to(workspace).as_posix() for path in docs]},
    )

    payload = {
        "schema_version": 1,
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
