#!/usr/bin/env python3
"""Validate frozen matrix, generated configs, episode JSONL, or analysis output."""

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
    VARIANT_IDS,
    load_json,
    load_matrix,
    matrix_summary,
    json_values_equal,
    validate_variant_config,
    variant_config_manifest,
)
from t4_completion.ablation.records import (  # noqa: E402
    load_episode_jsonl,
    validate_episode_collection,
)
from t4_completion.ablation.statistics import validate_analysis_outputs  # noqa: E402
from t4_completion.ablation.statistics import analysis_manifest  # noqa: E402


DEFAULT_MATRIX = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed offline validator for T4.6 ablation artifacts."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    subparsers = parser.add_subparsers(dest="artifact", required=True)
    subparsers.add_parser("matrix", help="validate only the frozen matrix")
    configs = subparsers.add_parser("configs", help="validate a generated config directory")
    configs.add_argument("--config-dir", type=Path, required=True)
    episodes = subparsers.add_parser("episodes", help="validate paired episode JSONL")
    episodes.add_argument("--episodes", type=Path, required=True)
    analysis = subparsers.add_parser("analysis", help="recompute and validate analysis files")
    analysis.add_argument("--analysis-dir", type=Path, required=True)
    return parser


def _validate_configs(matrix: dict[str, object], config_dir: Path) -> dict[str, object]:
    manifest = load_json(config_dir / "manifest.json")
    expected_manifest = variant_config_manifest(matrix)
    if not json_values_equal(manifest, expected_manifest):
        raise ContractError("generated config manifest is not deterministic")
    for variant_id in VARIANT_IDS:
        config = load_json(config_dir / "variants" / f"{variant_id}.json")
        validate_variant_config(matrix, variant_id, config)
    actual_files = sorted(
        path.relative_to(config_dir).as_posix()
        for path in config_dir.rglob("*")
        if path.is_file()
    )
    expected_files = ["manifest.json"] + [
        f"variants/{variant_id}.json" for variant_id in VARIANT_IDS
    ]
    if actual_files != sorted(expected_files):
        raise ContractError("generated config directory has missing or extra files")
    return {
        "status": "VALID",
        "artifact": "configs",
        "variant_count": len(VARIANT_IDS),
        "resource_use": "none",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        matrix = load_matrix(args.matrix)
        if args.artifact == "matrix":
            result = {**matrix_summary(matrix), "artifact": "matrix"}
        elif args.artifact == "configs":
            result = _validate_configs(matrix, args.config_dir)
        elif args.artifact == "episodes":
            collection = validate_episode_collection(
                load_episode_jsonl(args.episodes), matrix
            )
            result = {
                "status": "VALID",
                "artifact": "episodes",
                "evidence_kind": collection.evidence_kind,
                "episode_count_per_variant": collection.episode_count_per_variant,
                "resource_use": "none",
            }
        else:
            analysis_dir = args.analysis_dir
            actual_files = sorted(
                path.relative_to(analysis_dir).as_posix()
                for path in analysis_dir.rglob("*")
                if path.is_file()
            )
            if actual_files != ["aggregate.json", "episode_level.json", "manifest.json"]:
                raise ContractError("analysis directory has missing or extra files")
            episode_level = load_json(analysis_dir / "episode_level.json")
            aggregate = load_json(analysis_dir / "aggregate.json")
            manifest = load_json(analysis_dir / "manifest.json")
            validate_analysis_outputs(matrix, episode_level, aggregate)
            if not json_values_equal(manifest, analysis_manifest(episode_level, aggregate)):
                raise ContractError("analysis manifest is not a deterministic recomputation")
            result = {
                "status": "VALID",
                "artifact": "analysis",
                "analysis_status": aggregate["analysis_status"],
                "resource_use": "none",
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ContractError, OSError) as exc:
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
