#!/usr/bin/env python3
"""Materialize the frozen, disjoint A10/B10 T5 final-pilot datasets.

The command-line interface deliberately has no source-SHA override.  The
twenty-episode source gzip and both authored lane lists are frozen here as an
independent guard against two manifests drifting together.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION_MANIFEST = (
    ROOT / "configs" / "internnav_t5" / "t5_24h_execution_manifest.json"
)
DEFAULT_PILOT_MANIFEST = (
    ROOT / "configs" / "internnav_t3" / "model_pilot_episode_manifest.json"
)
EXPECTED_SOURCE_GZIP_SHA256 = (
    "f09b11004d15d579620ba2b2769dcbf863efe4cf04d9e404f071f66244463f8b"
)
FROZEN_LANE_A_KEYS = (
    "6898_1741",
    "6292_1573",
    "5840_1474",
    "6842_1720",
    "1420_364",
    "4084_1003",
    "583_145",
    "1803_448",
    "6623_1657",
    "2613_625",
)
FROZEN_LANE_B_KEYS = (
    "5627_1417",
    "4182_1027",
    "4009_976",
    "654_157",
    "4943_1255",
    "6982_1765",
    "3542_877",
    "6157_1561",
    "2564_610",
    "2853_703",
)
FROZEN_KEYS = FROZEN_LANE_A_KEYS + FROZEN_LANE_B_KEYS
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _episode_key(episode: dict[str, Any]) -> str:
    trajectory_id = episode.get("trajectory_id")
    episode_id = episode.get("episode_id")
    if trajectory_id is None or episode_id is None:
        raise ValueError("every frozen episode needs trajectory_id and episode_id")
    key = f"{trajectory_id}_{episode_id}"
    declared = episode.get("episode_key")
    if declared is not None and str(declared) != key:
        raise ValueError(f"episode identity fields disagree for {key}")
    return key


def _manifest_contract(
    execution_manifest_path: Path,
    pilot_manifest_path: Path,
    expected_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    if SHA256.fullmatch(expected_source_sha256) is None:
        raise ValueError("expected source SHA-256 is malformed")
    execution = _load_object(execution_manifest_path, "T5 execution manifest")
    pilot_manifest = _load_object(pilot_manifest_path, "frozen pilot manifest")
    final = execution.get("final_evaluation")
    authored = final.get("pilot") if isinstance(final, dict) else None
    authored = authored if isinstance(authored, dict) else {}
    checks = {
        "execution_lane_a_keys_exact": authored.get("lane_a_episode_keys")
        == list(FROZEN_LANE_A_KEYS),
        "execution_lane_b_keys_exact": authored.get("lane_b_episode_keys")
        == list(FROZEN_LANE_B_KEYS),
        "execution_aggregate_count_exact": authored.get("aggregate_episode_count")
        == len(FROZEN_KEYS),
        "held_out_tuning_forbidden": authored.get("held_out_tuning_forbidden")
        is True,
        "pilot_episode_count_exact": pilot_manifest.get("episode_count")
        == len(FROZEN_KEYS),
        "pilot_episode_keys_exact": pilot_manifest.get("episode_keys")
        == list(FROZEN_KEYS),
        "pilot_overlay_sha_exact": pilot_manifest.get("overlay_sha256")
        == expected_source_sha256,
        "authored_lanes_disjoint": set(FROZEN_LANE_A_KEYS).isdisjoint(
            FROZEN_LANE_B_KEYS
        ),
        "authored_lanes_union_exact": set(FROZEN_LANE_A_KEYS)
        | set(FROZEN_LANE_B_KEYS)
        == set(FROZEN_KEYS),
        "authored_lane_concatenation_exact": FROZEN_LANE_A_KEYS
        + FROZEN_LANE_B_KEYS
        == FROZEN_KEYS,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"frozen manifest contract failed: {failed}")
    return execution, pilot_manifest, checks


def _write_gzip_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as stream:
                stream.write(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"audit output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_split(
    source: Path,
    lane_a_output_root: Path,
    lane_b_output_root: Path,
    *,
    execution_manifest_path: Path = DEFAULT_EXECUTION_MANIFEST,
    pilot_manifest_path: Path = DEFAULT_PILOT_MANIFEST,
    audit_output: Path | None = None,
    expected_source_sha256: str = EXPECTED_SOURCE_GZIP_SHA256,
) -> dict[str, Any]:
    """Validate and split the exact source; the SHA parameter is test-only.

    The CLI never exposes ``expected_source_sha256`` and therefore always uses
    the production constant above.
    """

    if source.is_symlink():
        raise ValueError("frozen source dataset must not be a symlink")
    if execution_manifest_path.is_symlink() or pilot_manifest_path.is_symlink():
        raise ValueError("frozen manifests must not be symlinks")
    source = source.resolve()
    execution_manifest_path = execution_manifest_path.resolve()
    pilot_manifest_path = pilot_manifest_path.resolve()
    output_roots = {
        "a": lane_a_output_root.resolve(strict=False),
        "b": lane_b_output_root.resolve(strict=False),
    }
    if output_roots["a"] == output_roots["b"]:
        raise ValueError("Lane A and Lane B output roots must be distinct")
    if (
        output_roots["a"] in output_roots["b"].parents
        or output_roots["b"] in output_roots["a"].parents
    ):
        raise ValueError("one lane output root must not contain the other")
    for lane, root in output_roots.items():
        if root.exists() or root.is_symlink():
            raise ValueError(f"Lane {lane.upper()} output root already exists")
        if source == root or root in source.parents:
            raise ValueError("source dataset must not be inside an output root")
    resolved_audit = audit_output.resolve(strict=False) if audit_output else None
    if resolved_audit is not None and (
        resolved_audit.exists() or resolved_audit.is_symlink()
    ):
        raise ValueError(f"audit output already exists: {resolved_audit}")

    _execution, _pilot, manifest_checks = _manifest_contract(
        execution_manifest_path, pilot_manifest_path, expected_source_sha256
    )
    if not source.is_file() or source.is_symlink():
        raise ValueError("frozen source dataset is missing or is a symlink")
    source_sha = _sha256(source)
    if source_sha != expected_source_sha256:
        raise ValueError(
            "frozen source gzip SHA-256 mismatch: "
            f"expected {expected_source_sha256}, observed {source_sha}"
        )
    try:
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"frozen source gzip is invalid: {error}") from error
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not all(
        isinstance(episode, dict) for episode in episodes
    ):
        raise ValueError("frozen source must contain an episodes object array")
    source_keys = [_episode_key(episode) for episode in episodes]
    if source_keys != list(FROZEN_KEYS):
        raise ValueError("source episode identities/order differ from frozen 20 keys")

    lane_keys = {
        "a": list(FROZEN_LANE_A_KEYS),
        "b": list(FROZEN_LANE_B_KEYS),
    }
    source_by_key = dict(zip(source_keys, episodes))
    lane_records: dict[str, dict[str, Any]] = {}
    created_roots: list[Path] = []
    try:
        for lane in ("a", "b"):
            selected = dict(payload)
            selected["episodes"] = [source_by_key[key] for key in lane_keys[lane]]
            if "episode_count" in selected:
                selected["episode_count"] = len(lane_keys[lane])
            output = output_roots[lane] / "val_unseen" / "val_unseen.json.gz"
            created_roots.append(output_roots[lane])
            _write_gzip_json(output, selected)
            lane_records[lane] = {
                "output_dataset": str(output),
                "output_sha256": _sha256(output),
                "episode_count": len(lane_keys[lane]),
                "episode_keys": lane_keys[lane],
            }

        checks = dict(manifest_checks)
        checks.update(
            {
                "source_gzip_sha_exact": source_sha == expected_source_sha256,
                "source_episode_count_exact": len(episodes) == len(FROZEN_KEYS),
                "source_episode_keys_exact": source_keys == list(FROZEN_KEYS),
                "lane_a_count_exact": lane_records["a"]["episode_count"] == 10,
                "lane_b_count_exact": lane_records["b"]["episode_count"] == 10,
                "materialized_lanes_disjoint": set(lane_records["a"]["episode_keys"])
                .isdisjoint(lane_records["b"]["episode_keys"]),
                "materialized_union_exact": set(lane_records["a"]["episode_keys"])
                | set(lane_records["b"]["episode_keys"])
                == set(FROZEN_KEYS),
                "materialized_concatenation_exact": lane_records["a"][
                    "episode_keys"
                ]
                + lane_records["b"]["episode_keys"]
                == list(FROZEN_KEYS),
                "lane_output_shas_valid": all(
                    SHA256.fullmatch(record["output_sha256"]) is not None
                    for record in lane_records.values()
                ),
            }
        )
        if not all(checks.values()):
            raise ValueError("internal final-pilot split checks failed")
        audit = {
            "schema_version": 1,
            "status": "PASS",
            "stage": "t5_final_pilot_a10_b10_split",
            "selection": "exact_frozen_disjoint_lane_keys",
            "source": {
                "dataset": str(source),
                "sha256": source_sha,
                "episode_count": len(episodes),
                "episode_keys": source_keys,
            },
            "manifests": {
                "execution": {
                    "path": str(execution_manifest_path),
                    "sha256": _sha256(execution_manifest_path),
                },
                "pilot": {
                    "path": str(pilot_manifest_path),
                    "sha256": _sha256(pilot_manifest_path),
                    "declared_overlay_sha256": expected_source_sha256,
                },
            },
            "lanes": lane_records,
            "checks": checks,
        }
        if resolved_audit is not None:
            _write_new_json(resolved_audit, audit)
        return audit
    except BaseException:
        for root in reversed(created_roots):
            if root.is_dir() and not root.is_symlink():
                shutil.rmtree(root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--lane-a-output-root", required=True, type=Path)
    parser.add_argument("--lane-b-output-root", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument(
        "--execution-manifest", type=Path, default=DEFAULT_EXECUTION_MANIFEST
    )
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    arguments = parser.parse_args()
    try:
        audit = materialize_split(
            arguments.source,
            arguments.lane_a_output_root,
            arguments.lane_b_output_root,
            execution_manifest_path=arguments.execution_manifest,
            pilot_manifest_path=arguments.pilot_manifest,
            audit_output=arguments.audit_output,
        )
    except (OSError, ValueError) as error:
        print(f"final-pilot split failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
