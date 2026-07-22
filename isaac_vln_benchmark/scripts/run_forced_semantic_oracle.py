#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PACKAGE = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
SCHEDULER_PACKAGE = ROOT.parent / "ros2_ws" / "src" / "omninav_step_scheduler"
for path in (BENCH_PACKAGE, SCHEDULER_PACKAGE):
    sys.path.insert(0, str(path))

from isaac_vln_benchmark.semantic_navigation_benchmark import (
    evaluate_semantic_navigation_run,
    load_task_set,
    oracle_plan_payload,
    public_task_payload,
    read_jsonl,
    task_set_fingerprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or score the forced semantic oracle upper-bound run.")
    parser.add_argument(
        "--tasks",
        default=str(ROOT / "configs" / "natural_language_navigation_v18.yaml"),
    )
    parser.add_argument("--records", default="", help="Live episode result JSONL; omit for contract validation only.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    task_path = Path(args.tasks)
    task_set = load_task_set(task_path)
    output = Path(args.output) if args.output else ROOT / "runs" / f"v18_forced_semantic_oracle_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)

    public_lines = []
    oracle_lines = []
    for task in task_set["tasks"]:
        public_lines.append(json.dumps(public_task_payload(task), ensure_ascii=False))
        for subgoal in oracle_plan_payload(task):
            oracle_lines.append(json.dumps(subgoal, ensure_ascii=False))
    (output / "public_tasks.jsonl").write_text("\n".join(public_lines) + "\n", encoding="utf-8")
    (output / "oracle_subgoals.jsonl").write_text("\n".join(oracle_lines) + "\n", encoding="utf-8")

    records = read_jsonl(args.records) if args.records else []
    gate = evaluate_semantic_navigation_run(records, task_set, mode="forced_semantic_oracle")
    gate.update(
        {
            "contract_validation_pass": True,
            "live_execution_status": "scored" if args.records else "not_run",
            "screening_allowed": bool(args.records and gate["pass"]),
            "value_claim": "unproven",
            "sim2real": "NOT READY FOR REAL ROBOT AUTONOMY",
        }
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "forced_semantic_oracle",
        "task_set": task_set_fingerprint(task_path),
        "local_navigation_owner": "OmniNav",
        "semantic_decision_owner": "forced semantic oracle",
        "normal_chain_required": ["stale_gate", "semantic_executive", "OmniNav", "primitive_executor", "safe_mux"],
        "direct_motion_output_allowed": False,
        "qualification_evidence": False,
    }
    (output / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# V18 Forced Semantic Oracle",
        "",
        f"- contract validation: PASS",
        f"- live execution: {gate['live_execution_status']}",
        f"- episodes scored: {gate['episodes_scored']}/30",
        f"- forced semantic oracle gate: {'PASS' if gate['forced_semantic_oracle_gate'] else 'FAIL'}",
        f"- value screening allowed: {gate['screening_allowed']}",
        "- local navigation and obstacle avoidance owner: OmniNav",
        "- direct Step/oracle motion output: forbidden",
        "- qualification evidence: false",
        "- true multimodal OmniNav+Step value: unproven",
        "- Sim2Real: NOT READY FOR REAL ROBOT AUTONOMY",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": gate}, indent=2))
    return 0 if gate["contract_validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
