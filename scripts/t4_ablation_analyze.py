#!/usr/bin/env python3
"""Build deterministic episode-level and aggregate T4.6 statistics offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.ablation.contract import (  # noqa: E402
    ContractError,
    load_matrix,
    write_json,
)
from t4_completion.ablation.records import load_episode_jsonl  # noqa: E402
from t4_completion.ablation.statistics import (  # noqa: E402
    analysis_manifest,
    build_analysis_outputs,
)


DEFAULT_MATRIX = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze paired T4.6 episode JSONL without running a model or simulator."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        matrix = load_matrix(args.matrix)
        records = load_episode_jsonl(args.episodes)
        episode_level, aggregate = build_analysis_outputs(matrix, records)
        manifest = analysis_manifest(episode_level, aggregate)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "episode_level.json", episode_level)
        write_json(args.output_dir / "aggregate.json", aggregate)
        write_json(args.output_dir / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": aggregate["analysis_status"],
                    "episode_count_per_variant": aggregate[
                        "episode_count_per_variant"
                    ],
                    "comparison_count": len(aggregate["comparisons"]),
                    "resource_use": "none",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (ContractError, FileExistsError, OSError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": str(exc), "resource_use": "none"},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
