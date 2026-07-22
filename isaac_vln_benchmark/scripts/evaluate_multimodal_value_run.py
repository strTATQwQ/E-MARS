#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.multimodal_value_evidence import evaluate_multimodal_value_run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the fresh-image multimodal evidence gate to a screening or paired run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--scheduler-config", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    scheduler_config = Path(args.scheduler_config)
    existing_gate = json.loads((run / "gate.json").read_text(encoding="utf-8"))
    result = evaluate_multimodal_value_run(run, existing_gate=existing_gate)
    result["frozen_inputs"] = {
        "scheduler_config": str(scheduler_config),
        "scheduler_config_sha256": sha256(scheduler_config),
        "task_config_sha256": sha256(run / "config.yaml") if (run / "config.yaml").is_file() else None,
        "selected_tasks_sha256": sha256(run / "selected_tasks.yaml") if (run / "selected_tasks.yaml").is_file() else None,
    }
    (run / "multimodal_value_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run / "multimodal_http_evidence.json").write_text(
        json.dumps(result["multimodal_evidence"], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
