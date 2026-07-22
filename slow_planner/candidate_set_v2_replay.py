"""Offline-only replay audit for Lane-B private candidate-set v2 artifacts.

The audit never invents candidates and never grants navigation authority.  It
accepts archived JSON/JSONL, validates exact source/decision bindings, and
returns ``BLOCKED_DATA`` until enough *real* multi-candidate snapshots and
paired ranking decisions exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_set_v2 import (
    ABSTAIN,
    PRIVATE_DECISION_KIND,
    SELECT_VIEWPOINT,
    materialize_candidate_set_v2_shadow,
    parse_candidate_set_v2_decision,
)
from .lane_b import LaneBSnapshotIdentity
from .reachable_viewpoint import (
    REACHABLE_VIEWPOINT_KIND,
    validate_reachable_viewpoint_mapping,
)


REPLAY_KIND = "t5_lane_b_candidate_set_v2_replay_status"
_JSON_SUFFIXES = {".json", ".jsonl"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(paths: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in paths:
        candidates = root.rglob("*") if root.is_dir() else (root,)
        for candidate in candidates:
            if (
                candidate.is_file()
                and candidate.suffix.lower() in _JSON_SUFFIXES
                and candidate.resolve() not in seen
            ):
                seen.add(candidate.resolve())
                yield candidate


def _objects(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    if path.suffix.lower() == ".jsonl":
        raw_values = (line for line in text.splitlines() if line.strip())
    else:
        raw_values = (text,)
    for raw in raw_values:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            yield value


def audit_candidate_set_v2_replay(
    paths: Sequence[str | Path], *, minimum_multi_candidate_snapshots: int = 5
) -> dict[str, Any]:
    """Audit archived artifacts without contacting ROS, Isaac, or a model."""

    if minimum_multi_candidate_snapshots <= 0:
        raise ValueError("minimum_multi_candidate_snapshots must be positive")
    roots = tuple(Path(path) for path in paths)
    files = tuple(_iter_files(roots))
    provenance = [
        {"artifact": path.as_posix(), "sha256": _sha256(path)} for path in files
    ]

    sources_by_digest: dict[str, tuple[str, Mapping[str, Any]]] = {}
    snapshot_digests: dict[str, str] = {}
    decisions: list[tuple[str, Mapping[str, Any]]] = []
    blockers: list[dict[str, Any]] = []
    invalid_sources: list[dict[str, str]] = []
    identity_conflicts: list[str] = []

    for path in files:
        for value in _objects(path):
            kind = value.get("kind")
            if kind == REACHABLE_VIEWPOINT_KIND:
                try:
                    validate_reachable_viewpoint_mapping(value)
                except Exception as exc:
                    invalid_sources.append(
                        {"artifact": path.as_posix(), "error": type(exc).__name__}
                    )
                    continue
                digest = str(value["candidate_set_sha256"])
                snapshot_id = str(value["snapshot_id"])
                prior = snapshot_digests.get(snapshot_id)
                if prior is not None and prior != digest:
                    identity_conflicts.append(snapshot_id)
                    continue
                snapshot_digests[snapshot_id] = digest
                sources_by_digest.setdefault(digest, (path.as_posix(), value))
            elif kind == PRIVATE_DECISION_KIND:
                decisions.append((path.as_posix(), value))
            elif kind == "t5_live_frontier_capture_runtime_status" and value.get(
                "status"
            ) in {"BLOCKED", "WAITING_FOR_IDENTITY"}:
                blockers.append(
                    {
                        "artifact": path.as_posix(),
                        "status": value.get("status"),
                        "blocker_code": value.get("blocker_code"),
                        "snapshot_id": value.get("snapshot_id"),
                        "snapshot_ready_count": value.get("snapshot_ready_count", 0),
                    }
                )

    multi_sources = {
        digest: pair
        for digest, pair in sources_by_digest.items()
        if len(pair[1].get("candidates", ())) >= 2
    }
    paired: list[dict[str, Any]] = []
    invalid_decisions = 0
    for decision_artifact, raw_decision in decisions:
        try:
            parsed = parse_candidate_set_v2_decision(
                json.dumps(raw_decision, sort_keys=True, separators=(",", ":"))
            )
        except Exception:
            invalid_decisions += 1
            continue
        source_pair = multi_sources.get(parsed.source_candidate_set_sha256)
        if source_pair is None:
            continue
        source_artifact, source = source_pair
        identity = LaneBSnapshotIdentity(
            str(source["episode_id"]), source["reset_id"], source["sequence_id"]
        )
        result = materialize_candidate_set_v2_shadow(
            candidate_set=source,
            decision_text=json.dumps(
                raw_decision, sort_keys=True, separators=(",", ":")
            ),
            expected_identity=identity,
            expected_source_artifact=parsed.source_artifact,
            expected_candidate_set_sha256=str(source["candidate_set_sha256"]),
            expected_grid_frame_id=str(source["grid_frame_id"]),
            now_sim_time_s=float(source["captured_sim_time_s"]),
        )
        candidates = list(source["candidates"])
        geometry_baseline = min(
            candidates,
            key=lambda item: (
                float(item["geodesic_distance_m"]),
                float(item["distance_m"]),
                int(item["candidate_id"]),
            ),
        )["candidate_id"]
        paired.append(
            {
                "snapshot_id": identity.snapshot_id,
                "source_artifact": source_artifact,
                "decision_artifact": decision_artifact,
                "candidate_count": len(candidates),
                "decision": result["decision"],
                "candidate_id": result["candidate_id"],
                "differs_from_geometry_baseline": (
                    result["decision"] == SELECT_VIEWPOINT
                    and result["candidate_id"] != geometry_baseline
                ),
                "proposal_only": result["authority"]["mode"] == "proposal_only",
            }
        )

    unique_paired = {item["snapshot_id"] for item in paired}
    reasons: list[str] = []
    if len(multi_sources) < minimum_multi_candidate_snapshots:
        reasons.append("INSUFFICIENT_REAL_MULTI_CANDIDATE_SNAPSHOTS")
    if len(unique_paired) < minimum_multi_candidate_snapshots:
        reasons.append("INSUFFICIENT_IDENTITY_BOUND_RANKING_DECISIONS")
    if identity_conflicts:
        reasons.append("CONFLICTING_SNAPSHOT_ID_REUSE")
    status = "BLOCKED_DATA" if reasons else "OFFLINE_SHADOW_REPLAY_PASS"
    return {
        "schema_version": 1,
        "kind": REPLAY_KIND,
        "status": status,
        "minimum_multi_candidate_snapshots": minimum_multi_candidate_snapshots,
        "counts": {
            "files_scanned": len(files),
            "valid_candidate_sets": len(sources_by_digest),
            "valid_multi_candidate_snapshots": len(multi_sources),
            "decision_artifacts": len(decisions),
            "identity_bound_replays": len(unique_paired),
            "selected": sum(item["decision"] == SELECT_VIEWPOINT for item in paired),
            "abstained": sum(item["decision"] == ABSTAIN for item in paired),
            "nontrivial_vs_geometry_baseline": sum(
                bool(item["differs_from_geometry_baseline"]) for item in paired
            ),
            "invalid_sources": len(invalid_sources),
            "invalid_decisions": invalid_decisions,
        },
        "blocker_codes": reasons,
        "archived_runtime_blockers": blockers,
        "identity_conflicts": sorted(set(identity_conflicts)),
        "replays": paired,
        "provenance": provenance,
        "authority": {
            "online_resources_used": False,
            "publish_navigation_goal": False,
            "publish_cmd_vel": False,
            "publish_terminal_stop": False,
        },
        "interpretation": (
            "This validates only identity-bound proposal-only replay plumbing; "
            "it does not prove semantic ranking quality or navigation gain."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--minimum", type=int, default=5)
    arguments = parser.parse_args(argv)
    result = audit_candidate_set_v2_replay(
        arguments.paths, minimum_multi_candidate_snapshots=arguments.minimum
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OFFLINE_SHADOW_REPLAY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
