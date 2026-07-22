#!/usr/bin/env python3
"""Generate deterministic, offline-only configs for all frozen T4.6 arms."""

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
    matrix_summary,
    resolve_variant_configs,
    variant_config_manifest,
    write_json,
)


DEFAULT_MATRIX = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or generate the frozen offline T4.6 ablation configs."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate without writing any generated files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        matrix = load_matrix(args.matrix)
        if args.check:
            if args.output_dir is not None:
                raise ContractError("--check cannot be combined with --output-dir")
            print(json.dumps(matrix_summary(matrix), indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise ContractError("--output-dir is required unless --check is used")
        configs = resolve_variant_configs(matrix)
        manifest = variant_config_manifest(matrix)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        variants_dir = args.output_dir / "variants"
        variants_dir.mkdir()
        for variant_id, config in configs.items():
            write_json(variants_dir / f"{variant_id}.json", config)
        write_json(args.output_dir / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": "GENERATED",
                    "matrix_sha256": manifest["matrix_sha256"],
                    "variant_count": len(configs),
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
