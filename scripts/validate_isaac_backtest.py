#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omninav_cosmos.backtest import (
    ExpectedModel,
    load_jsonl,
    manifest_episodes,
    validate_artifact_tree,
    validate_episode_records,
    validate_seed_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate formal real-Isaac OmniNav result artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--expected-model-variant", required=True)
    parser.add_argument("--expected-precision", required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "seeds.json").read_text(encoding="utf-8"))
    manifest_summary = validate_seed_manifest(manifest)
    expected_rows = manifest_episodes(manifest, args.phase)
    records = load_jsonl(run_dir / "episodes.jsonl")
    expected_model = ExpectedModel(args.expected_model_variant, args.expected_precision, True)
    episode_summary = validate_episode_records(records, expected_rows, expected_model)
    artifacts = validate_artifact_tree(run_dir, require_video=not args.no_video)
    summary = {
        "passed": True,
        "phase": args.phase,
        "manifest": manifest_summary,
        "episodes": episode_summary,
        "artifacts": artifacts,
    }
    (run_dir / "validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
