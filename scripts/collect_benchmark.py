#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect only fully validated benchmark runs into one index.")
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_ID=RUN_DIR")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configurations = {}
    for value in args.run:
        name, run_id, directory = value.split("=", 2)
        root = Path(directory).resolve()
        required = {
            key: root / filename
            for key, filename in {
                "validation": "validation.json",
                "summary": "summary.json",
                "precision_manifest": "precision_manifest.json",
                "input_lock": "inputs.lock.json",
                "episodes": "episodes.jsonl",
                "latency": "latency.jsonl",
                "hardware": "hardware.json",
                "batch_state": "batch_state.json",
                "commands": "commands.jsonl",
            }.items()
        }
        missing = [key for key, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name} lacks required artifacts: {missing}")
        validation = json.loads(required["validation"].read_text(encoding="utf-8"))
        if validation.get("run_id") != run_id or not validation.get("valid") or not validation.get("require_complete"):
            raise ValueError(f"{name} is not a complete validated run")
        precision = json.loads(required["precision_manifest"].read_text(encoding="utf-8"))
        if precision.get("run_id") != run_id:
            raise ValueError(f"{name} precision manifest run_id mismatch")
        configurations[name] = {
            "run_id": run_id,
            "run_dir": str(root),
            "validation": validation,
            "summary": json.loads(required["summary"].read_text(encoding="utf-8")),
            "precision_manifest": precision,
            "input_lock": json.loads(required["input_lock"].read_text(encoding="utf-8")),
            "artifact_sha256": {key: sha256_file(path) for key, path in required.items()},
            "video_count": len(list((root / "videos").glob("*.mp4"))),
        }
    payload = {"schema_version": 1, "configuration_count": len(configurations), "configurations": configurations}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "configuration_count": len(configurations)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
