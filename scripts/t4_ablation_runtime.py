#!/usr/bin/env python3
"""Validate one generated T4.6 config for a coordinator-owned runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.ablation.contract import ContractError  # noqa: E402
from t4_completion.ablation.runtime import (  # noqa: E402
    DEFAULT_MATRIX,
    load_runtime_variant,
    ordered_fields,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("mode", choices=("validate", "fields"))
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = load_runtime_variant(args.config, matrix_path=args.matrix)
        if args.mode == "fields":
            for value in ordered_fields(summary):
                print(value)
        else:
            print(json.dumps(summary, indent=2, sort_keys=True))
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
