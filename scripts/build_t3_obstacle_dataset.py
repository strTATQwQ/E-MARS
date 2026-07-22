#!/usr/bin/env python3
"""Freeze ten episodes covering five deterministic injected-obstacle types."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SCENARIOS = (
    "static_box",
    "doorway",
    "narrow_corridor",
    "sudden_blockage",
    "dynamic_crossing",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(instruction_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(instruction_text(item) for item in value)
    return ""


def evaluator_instruction_string(value: object) -> str:
    """Match the VLN task observation's ``obs['instruction']`` exactly.

    ``VLNEvalTask`` exposes only ``instruction.instruction_text`` to the agent;
    token metadata remains in a separate observation field.  Retain a scalar
    fallback for defensive compatibility with already-flattened datasets.
    """
    if isinstance(value, dict) and "instruction_text" in value:
        return str(value["instruction_text"])
    if isinstance(value, str):
        return value
    return instruction_text(value)


def write_dataset(path: Path, episodes: list[dict[str, object]]) -> None:
    encoded = (
        json.dumps({"episodes": episodes}, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--scenario-plan",
        default=",".join(SCENARIOS * 2),
        help="comma-separated ten-entry plan; every frozen scenario must occur twice",
    )
    args = parser.parse_args()
    excluded_episode_keys: set[str] = set()
    excluded_pairs: set[str] = set()
    exclusion_hashes: list[str] = []
    for path in args.exclude_manifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded_episode_keys.update(str(value) for value in payload.get("episode_keys", []))
        excluded_pairs.update(
            str(value) for value in payload.get("scenario_episode_pairs", [])
        )
        exclusion_hashes.append(sha256(path))
    scenario_plan = tuple(
        value.strip() for value in args.scenario_plan.split(",") if value.strip()
    )
    if len(scenario_plan) != 10 or Counter(scenario_plan) != Counter(
        {name: 2 for name in SCENARIOS}
    ):
        raise RuntimeError("scenario plan must contain all five scenarios exactly twice")
    with gzip.open(args.source, "rt", encoding="utf-8") as stream:
        source = json.load(stream)
    episodes = source.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 10:
        raise RuntimeError("obstacle stress source must contain at least ten episodes")
    selected = episodes[:10]
    output = args.output_root / "val_unseen" / "val_unseen.json.gz"
    write_dataset(output, selected)
    evidence = []
    scenario_occurrence: defaultdict[str, int] = defaultdict(int)
    for index, episode in enumerate(selected):
        text = instruction_text(episode.get("instruction", {}))
        scenario = scenario_plan[index]
        key = f"{episode.get('trajectory_id', '')}_{episode.get('episode_id', '')}"
        if key in excluded_episode_keys or f"{key}:{scenario}" in excluded_pairs:
            raise RuntimeError(f"excluded obstacle route/scenario selected: {key}:{scenario}")
        scenario_occurrence[scenario] += 1
        evidence.append(
            {
                "source_index": index,
                "scenario": scenario,
                "repeat": scenario_occurrence[scenario],
                "trajectory_id": str(episode.get("trajectory_id", "")),
                "episode_id": str(episode.get("episode_id", "")),
                "instruction_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "instruction_runtime_sha256": hashlib.sha256(
                    evaluator_instruction_string(episode.get("instruction", "")).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "selection_policy": "first_ten_frozen_source_episodes_explicit_five_scenarios_twice",
        "scenario_plan": list(scenario_plan),
        "source_episode_count": len(episodes),
        "selected_episode_count": len(selected),
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(output),
        "exclusion_manifest_sha256": exclusion_hashes,
        "excluded_pair_violation_count": 0,
        "scenarios": evidence,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
