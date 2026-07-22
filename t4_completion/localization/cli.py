"""Dependency-free CLI plus explicit optional ROS entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import Pose, PoseSample
from .evidence import LocalizationEvidenceWriter
from .ros_runtime import run_ros
from .selector import LocalizationSelector


INTEGRATION_REF = "refs/heads/codex/parallel-integration"
ORACLE_TRACKED_PATHS = (
    "scripts/run_t4_completion_localization_oracle.sh",
    "scripts/t4_localization_oracle.sh",
    "scripts/t4_localization_runtime.py",
    "scripts/t4_localization_validate.py",
    "configs/completion_sim/localization/selector.json",
    "configs/completion_sim/localization/oracle_gate.json",
    "t4_completion/localization/cli.py",
    "t4_completion/localization/ros_runtime.py",
    "t4_completion/localization/selector.py",
    "t4_completion/localization/validation.py",
)


def _config(arguments: argparse.Namespace):
    return load_config(
        arguments.config,
        expected_runtime_policy=arguments.runtime_policy,
        expected_runtime_target=arguments.runtime_target,
    )


def _validate_config(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "runtime_policy": config.policy.runtime_policy,
                "runtime_target": config.policy.runtime_target,
                "sources": sorted(config.sources),
                "allow_isaac_gt_fallback": config.policy.allow_isaac_gt_fallback,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _event_integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}_must_be_integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{label}_out_of_range")
    return value


def _event_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}_must_be_nonempty_string")
    return value


def _event_vector(value: Any, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}_must_be_length_{length}_array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label}_must_contain_numbers")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{label}_must_be_finite")
    return converted


def _sample(payload: dict[str, Any]) -> PoseSample:
    required = {
        "event",
        "source",
        "generation",
        "sequence_id",
        "stamp_ns",
        "received_monotonic_ns",
        "parent_frame",
        "child_frame",
        "translation_xyz",
        "quaternion_xyzw",
    }
    optional = {
        "linear_velocity_xyz",
        "angular_velocity_xyz",
        "backend_ready",
        "tracking",
        "health_reason",
    }
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("sample_event_keys_mismatch")
    backend_ready = payload.get("backend_ready", True)
    tracking = payload.get("tracking", True)
    if not isinstance(backend_ready, bool) or not isinstance(tracking, bool):
        raise ValueError("sample_event_health_must_be_boolean")
    return PoseSample(
        source=_event_string(payload["source"], "sample_source"),
        generation=_event_integer(payload["generation"], "sample_generation"),
        sequence_id=_event_integer(payload["sequence_id"], "sample_sequence_id"),
        stamp_ns=_event_integer(payload["stamp_ns"], "sample_stamp_ns", positive=True),
        received_monotonic_ns=_event_integer(
            payload["received_monotonic_ns"], "sample_received_monotonic_ns"
        ),
        parent_frame=_event_string(payload["parent_frame"], "sample_parent_frame"),
        child_frame=_event_string(payload["child_frame"], "sample_child_frame"),
        pose=Pose(
            _event_vector(
                payload["translation_xyz"], "sample_translation_xyz", 3
            ),  # type: ignore[arg-type]
            _event_vector(
                payload["quaternion_xyzw"], "sample_quaternion_xyzw", 4
            ),  # type: ignore[arg-type]
        ),
        linear_velocity_xyz=_event_vector(
            payload.get("linear_velocity_xyz", [0, 0, 0]),
            "sample_linear_velocity_xyz",
            3,
        ),  # type: ignore[arg-type]
        angular_velocity_xyz=_event_vector(
            payload.get("angular_velocity_xyz", [0, 0, 0]),
            "sample_angular_velocity_xyz",
            3,
        ),  # type: ignore[arg-type]
        backend_ready=backend_ready,
        tracking=tracking,
        health_reason=_event_string(
            payload.get("health_reason", "tracking"), "sample_health_reason"
        ),
    )


def _replay(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    selector = LocalizationSelector(policy=config.policy, sources=config.sources)
    writer = LocalizationEvidenceWriter(
        arguments.result_dir,
        record_filename=config.evidence_record_filename,
        summary_filename=config.evidence_summary_filename,
    )
    decisions = 0
    try:
        for line_number, line in enumerate(
            arguments.events.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise ValueError(f"invalid_event_line_{line_number}")
            if event["event"] == "reset":
                if set(event) != {"event", "generation"}:
                    raise ValueError(f"reset_event_keys_mismatch_line_{line_number}")
                selector.reset_generation(
                    _event_integer(event["generation"], "reset_generation")
                )
            elif event["event"] == "sample":
                accepted, reason = selector.ingest(_sample(event))
                if not accepted:
                    print(
                        json.dumps(
                            {"line": line_number, "accepted": False, "reason": reason}
                        ),
                        file=sys.stderr,
                    )
            elif event["event"] == "tick":
                if set(event) != {"event", "monotonic_ns", "sim_time_ns"}:
                    raise ValueError(f"tick_event_keys_mismatch_line_{line_number}")
                decision = selector.decide(
                    _event_integer(event["monotonic_ns"], "tick_monotonic_ns"),
                    _event_integer(event["sim_time_ns"], "tick_sim_time_ns"),
                )
                writer.append(decision)
                decisions += 1
                print(json.dumps(decision.to_dict(), sort_keys=True, allow_nan=False))
            else:
                raise ValueError(f"unknown_event_line_{line_number}")
        summary = selector.summary()
        status = (
            "PASS_WITH_DEVIATION"
            if summary["current_output_available"] and summary["deviation_count"]
            else ("PASS" if summary["current_output_available"] else "FAIL")
        )
        writer.close(summary, status=status)
        return 0 if decisions and status != "FAIL" else 2
    except BaseException:
        # The fresh directory and partial append-only record are retained as
        # failure evidence; a caller must never reuse it.  Best-effort sealing
        # must not replace the original exception.
        try:
            writer.close(selector.summary(), status="FAIL")
        except BaseException:
            pass
        raise


def _git(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git_command_failed:{arguments[0]}") from exc
    return completed.stdout.strip()


def _normalize_oracle_result_dir(
    repository_root: Path, result_dir: Path, grant_id: str
) -> tuple[str, Path]:
    if not grant_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", grant_id):
        raise ValueError("grant_id_is_not_a_safe_path_token")
    if result_dir.is_absolute() or any(part in {"", ".", ".."} for part in result_dir.parts):
        raise ValueError("oracle_result_dir_must_be_safe_repository_relative")
    normalized = result_dir.as_posix()
    expected = f"results/parallel/localization/oracle-{grant_id}"
    if normalized != expected:
        raise ValueError("oracle_result_dir_does_not_match_frozen_grant_path")
    root = repository_root.resolve()
    resolved = (root / result_dir).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("oracle_result_dir_escaped_repository") from exc
    return normalized, resolved


def _parse_authoritative_grant(board: str) -> dict[str, Any]:
    start_token = "<!-- INTERNAV_ONLINE_GRANT_V1"
    end_token = "INTERNAV_ONLINE_GRANT_V1 -->"
    if board.count(start_token) != 1 or board.count(end_token) != 1:
        raise RuntimeError("authoritative_grant_block_must_be_unique")
    block = board.split(start_token, 1)[1].split(end_token, 1)[0].strip()
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"authoritative_grant_duplicate_key:{key}")
            value[key] = item
        return value

    grant = json.loads(block, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(grant, dict):
        raise RuntimeError("authoritative_grant_must_be_object")
    return grant


def _oracle_preflight(arguments: argparse.Namespace) -> int:
    repository_root = arguments.repository_root.resolve()
    actual_root = Path(_git(repository_root, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != repository_root:
        raise RuntimeError("repository_root_does_not_match_git_toplevel")
    normalized_result, resolved_result = _normalize_oracle_result_dir(
        repository_root, arguments.result_dir, arguments.grant_id
    )
    ref_sha = _git(repository_root, "rev-parse", "--verify", INTEGRATION_REF)
    if not re.fullmatch(r"[0-9a-f]{40}", ref_sha):
        raise RuntimeError("authoritative_ref_sha_invalid")
    board = _git(repository_root, "show", f"{ref_sha}:coordination/TASK_BOARD.md")
    grant = _parse_authoritative_grant(board)
    expected = {
        "schema_version": 1,
        "status": "GRANTED",
        "worker": "20",
        "resource": "isaac",
        "profile": "completion_sim_localization_oracle",
        "result_dir": normalized_result,
        "grant_id": arguments.grant_id,
    }
    if any(type(grant.get(key)) is not type(value) for key, value in expected.items()):
        raise RuntimeError("authoritative_grant_field_types_invalid")
    if grant != expected:
        raise RuntimeError("authoritative_grant_does_not_exactly_match_oracle_request")
    execution_head = _git(repository_root, "rev-parse", "HEAD")
    if execution_head != ref_sha:
        raise RuntimeError("oracle_execution_head_does_not_match_authoritative_ref")
    tracked_status = _git(
        repository_root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_status:
        raise RuntimeError("oracle_execution_worktree_has_tracked_changes")
    for path in ORACLE_TRACKED_PATHS:
        _git(repository_root, "cat-file", "-e", f"{ref_sha}:{path}")
    if resolved_result.exists():
        raise FileExistsError("oracle_result_dir_must_be_fresh")
    resolved_result.parent.mkdir(parents=True, exist_ok=True)
    resolved_result.mkdir(exist_ok=False)
    ref_after_claim = _git(repository_root, "rev-parse", "--verify", INTEGRATION_REF)
    claim = {
        "schema_version": 1,
        "status": "PASS" if ref_after_claim == ref_sha else "FAIL",
        "integration_ref": INTEGRATION_REF,
        "integration_ref_sha": ref_sha,
        "execution_head_sha": execution_head,
        "tracked_worktree_clean": True,
        "tracked_runtime_paths": list(ORACLE_TRACKED_PATHS),
        "worker": "20",
        "resource": "isaac",
        "profile": "completion_sim_localization_oracle",
        "result_dir": normalized_result,
        "grant_id": arguments.grant_id,
        "ref_unchanged_after_claim": ref_after_claim == ref_sha,
    }
    claim_path = resolved_result / "localization_grant_claim.json"
    with claim_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(claim, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    if ref_after_claim != ref_sha:
        raise RuntimeError("authoritative_ref_changed_during_result_claim")
    print(json.dumps(claim, sort_keys=True))
    return 0


def _oracle_ref_check(arguments: argparse.Namespace) -> int:
    repository_root = arguments.repository_root.resolve()
    actual_root = Path(_git(repository_root, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != repository_root:
        raise RuntimeError("repository_root_does_not_match_git_toplevel")
    if arguments.result_dir.is_absolute() or any(
        part in {"", ".", ".."} for part in arguments.result_dir.parts
    ):
        raise ValueError("oracle_result_dir_must_be_safe_repository_relative")
    claim_path = (
        repository_root / arguments.result_dir / "localization_grant_claim.json"
    ).resolve()
    try:
        claim_path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("grant_claim_escaped_repository") from exc
    def reject_duplicate_claim_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"grant_claim_duplicate_key:{key}")
            value[key] = item
        return value

    claim = json.loads(
        claim_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_claim_keys,
    )
    expected_keys = {
        "schema_version",
        "status",
        "integration_ref",
        "integration_ref_sha",
        "execution_head_sha",
        "tracked_worktree_clean",
        "tracked_runtime_paths",
        "worker",
        "resource",
        "profile",
        "result_dir",
        "grant_id",
        "ref_unchanged_after_claim",
    }
    if not isinstance(claim, dict) or set(claim) != expected_keys:
        raise RuntimeError("grant_claim_contract_invalid")
    if (
        isinstance(claim["schema_version"], bool)
        or not isinstance(claim["schema_version"], int)
        or not isinstance(claim["grant_id"], str)
        or not isinstance(claim["result_dir"], str)
        or not isinstance(claim["integration_ref_sha"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", claim["integration_ref_sha"])
        or not isinstance(claim["execution_head_sha"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", claim["execution_head_sha"])
        or claim["tracked_runtime_paths"] != list(ORACLE_TRACKED_PATHS)
    ):
        raise RuntimeError("grant_claim_field_types_invalid")
    normalized, resolved = _normalize_oracle_result_dir(
        repository_root, arguments.result_dir, claim["grant_id"]
    )
    if resolved != (repository_root / arguments.result_dir).resolve() or claim[
        "result_dir"
    ] != normalized:
        raise RuntimeError("grant_claim_result_identity_mismatch")
    if (
        claim["schema_version"] != 1
        or claim["status"] != "PASS"
        or claim["integration_ref"] != INTEGRATION_REF
        or claim["execution_head_sha"] != claim["integration_ref_sha"]
        or claim["tracked_worktree_clean"] is not True
        or claim["worker"] != "20"
        or claim["resource"] != "isaac"
        or claim["profile"] != "completion_sim_localization_oracle"
        or claim["ref_unchanged_after_claim"] is not True
    ):
        raise RuntimeError("grant_claim_not_approved")
    current_ref = _git(repository_root, "rev-parse", "--verify", INTEGRATION_REF)
    current_head = _git(repository_root, "rev-parse", "HEAD")
    tracked_status = _git(
        repository_root, "status", "--porcelain", "--untracked-files=no"
    )
    if (
        not re.fullmatch(r"[0-9a-f]{40}", current_ref)
        or current_ref != claim["integration_ref_sha"]
        or current_head != claim["execution_head_sha"]
        or tracked_status
    ):
        raise RuntimeError("oracle_execution_provenance_changed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "integration_ref_sha": current_ref,
                "execution_head_sha": current_head,
                "tracked_worktree_clean": True,
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_policy(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", type=Path, required=True)
        command.add_argument(
            "--runtime-policy",
            choices=("strict_evidence", "completion_sim"),
            required=True,
        )
        command.add_argument("--runtime-target", required=True)

    validate = subparsers.add_parser("validate-config")
    add_policy(validate)
    validate.set_defaults(handler=_validate_config)

    replay = subparsers.add_parser("replay")
    add_policy(replay)
    replay.add_argument("--events", type=Path, required=True)
    replay.add_argument("--result-dir", type=Path, required=True)
    replay.set_defaults(handler=_replay)

    ros = subparsers.add_parser("run-ros")
    add_policy(ros)
    ros.add_argument("--result-dir", type=Path, required=True)
    ros.set_defaults(handler=lambda arguments: run_ros(_config(arguments), arguments.result_dir))

    preflight = subparsers.add_parser("oracle-preflight")
    preflight.add_argument("--repository-root", type=Path, required=True)
    preflight.add_argument("--result-dir", type=Path, required=True)
    preflight.add_argument("--grant-id", required=True)
    preflight.set_defaults(handler=_oracle_preflight)

    ref_check = subparsers.add_parser("oracle-ref-check")
    ref_check.add_argument("--repository-root", type=Path, required=True)
    ref_check.add_argument("--result-dir", type=Path, required=True)
    ref_check.set_defaults(handler=_oracle_ref_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
