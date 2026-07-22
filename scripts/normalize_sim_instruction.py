#!/usr/bin/env python3
"""Resolve one simulation instruction through the optional normalizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from slow_planner.sim_mission import (
    SimulationInstructionNormalizer,
    SimulationNormalizationConfig,
    config_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--mission-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--reset-generation", type=int, default=0)
    parser.add_argument("--sequence-id", type=int, default=0)
    args = parser.parse_args()
    path = Path(args.config).resolve()
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("simulation mission-normalization config must be an object")
    layer = value.get("mission_normalization")
    if not isinstance(layer, dict):
        raise ValueError("mission_normalization config section is required")
    normalizer = SimulationInstructionNormalizer(
        SimulationNormalizationConfig.from_mapping(layer),
        config_sha256=config_sha256(raw),
    )
    resolution = normalizer.normalize(
        args.instruction,
        mission_id=args.mission_id,
        episode_id=args.episode_id,
        reset_generation=args.reset_generation,
        sequence_id=args.sequence_id,
    )
    json.dump(resolution.to_mapping(), sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
