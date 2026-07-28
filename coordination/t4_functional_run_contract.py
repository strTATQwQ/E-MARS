#!/usr/bin/env python3
"""Validate ref-bound functional staging and online T4 run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from t4_functional_payload import FUNCTIONAL_ROOTS


SHA1 = re.compile(r"^[0-9a-f]{40}$")
GRANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$")
ALLOWED_AUTHORIZATION_DIFFS = {"coordination/TASK_BOARD.md"}
ISAAC_STABLE_ROOT = "/home/song/internnav-t1-t2"
DGX_STABLE_ROOT = "/home/railgun/internnav-t1-t2"
ISAAC_ROS_WORKSPACE = "/home/song/internnav-t4/isaac_ros_ws_45"
ISAAC_OVERLAY_PACKAGES = (
    "internvla_ros2_msgs",
    "internvla_ros2",
    "internvla_nav2_adapter",
    "internvla_go2_controller",
    "internvla_t4_sensors",
    "internvla_t4_recovery",
)


class ContractError(RuntimeError):
    """One immutable functional-run contract was violated."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"required JSON is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"required JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"required JSON is not an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _git(git_dir: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", f"--git-dir={git_dir}", *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _verify_ref_compatibility(
    git_dir: Path, deployment_ref: str, authorization_ref: str
) -> list[str]:
    _require(git_dir.is_dir(), "Git directory does not exist")
    _require(SHA1.fullmatch(deployment_ref) is not None, "invalid deployment ref")
    _require(SHA1.fullmatch(authorization_ref) is not None, "invalid authorization ref")
    for ref in (deployment_ref, authorization_ref):
        if _git(git_dir, "cat-file", "-e", f"{ref}^{{commit}}", check=False).returncode:
            raise ContractError(f"Git commit does not exist: {ref}")
    if _git(
        git_dir,
        "merge-base",
        "--is-ancestor",
        deployment_ref,
        authorization_ref,
        check=False,
    ).returncode:
        raise ContractError("functional deployment is not an ancestor of authorization")
    completed = _git(
        git_dir,
        "diff",
        "--name-only",
        "--diff-filter=ACDMRTUXB",
        deployment_ref,
        authorization_ref,
        "--",
        *FUNCTIONAL_ROOTS,
    )
    changed = [line for line in completed.stdout.splitlines() if line]
    disallowed = sorted(set(changed) - ALLOWED_AUTHORIZATION_DIFFS)
    if disallowed:
        raise ContractError(
            f"functional code changed after deployment; restaging is required: {disallowed}"
        )
    return changed


def validate_prepare_result(
    result_dir: Path, git_dir: Path, authorization_ref: str
) -> dict[str, Any]:
    if not result_dir.is_dir() or result_dir.is_symlink():
        raise ContractError("functional prepare result directory is missing or unsafe")
    summary = _load(result_dir / "functional_prepare_summary.json")
    grant_id = str(summary.get("grant_id", ""))
    deployment_ref = str(summary.get("ref_sha", ""))
    _require(GRANT.fullmatch(grant_id) is not None, "invalid prepare grant ID")
    _require(SHA1.fullmatch(deployment_ref) is not None, "invalid prepare ref")
    _require(result_dir.name == f"prepare-00-{grant_id}", "prepare result path mismatch")
    _require(summary.get("status") == "PASS", "functional prepare did not pass")
    _require(int(summary.get("exit_code", -1)) == 0, "functional prepare exit was nonzero")
    _require(summary.get("resource_order") == ["dgx", "isaac"], "resource order changed")
    _require(summary.get("runtime_policy") == "completion_sim", "runtime policy changed")
    _require(summary.get("runtime_target") == "isaac_simulation", "runtime target changed")
    _require(summary.get("model_host") == "dgx_spark_only", "model host changed")
    _require(summary.get("strict_evidence_modified") is False, "strict evidence was modified")
    _require(summary.get("real_go2_targeted") is False, "real Go2 was targeted")

    receipts: dict[str, dict[str, Any]] = {}
    for role, stable_root, workspace in (
        ("dgx", DGX_STABLE_ROOT, None),
        ("isaac", ISAAC_STABLE_ROOT, ISAAC_ROS_WORKSPACE),
    ):
        receipt = _load(result_dir / "remote-receipts" / f"{role}_deployment_receipt.json")
        receipts[role] = receipt
        expected_root = (
            f"{stable_root}/.t4-deployments/"
            f"{grant_id}-{deployment_ref[:12]}-{role}"
        )
        _require(receipt.get("schema_version") == 1, f"{role} receipt schema changed")
        _require(receipt.get("role") == role, f"{role} receipt role changed")
        _require(receipt.get("status") == "PASS", f"{role} deployment did not pass")
        _require(receipt.get("phase") == "ready", f"{role} deployment is not ready")
        _require(int(receipt.get("exit_code", -1)) == 0, f"{role} deployment exit was nonzero")
        _require(receipt.get("grant_id") == grant_id, f"{role} grant mismatch")
        _require(receipt.get("ref_sha") == deployment_ref, f"{role} ref mismatch")
        _require(receipt.get("deployment_root") == expected_root, f"{role} root mismatch")
        _require(receipt.get("runtime_policy") == "completion_sim", f"{role} policy changed")
        _require(receipt.get("runtime_target") == "isaac_simulation", f"{role} target changed")
        _require(receipt.get("resource_lease_ack") == "dgx+isaac", f"{role} lease changed")
        _require(receipt.get("strict_evidence_modified") is False, f"{role} strict mutation")
        _require(receipt.get("real_go2_targeted") is False, f"{role} real Go2 target")
        if workspace is not None:
            _require(receipt.get("ros_workspace") == workspace, "Isaac ROS workspace changed")
    _require(summary.get("receipts") == receipts, "prepare summary receipt copies differ")
    changed = _verify_ref_compatibility(git_dir, deployment_ref, authorization_ref)
    return {
        "schema_version": 1,
        "status": "PASS",
        "prepare_grant_id": grant_id,
        "deployment_ref_sha": deployment_ref,
        "authorization_ref_sha": authorization_ref,
        "isaac_deployment_root": receipts["isaac"]["deployment_root"],
        "isaac_ros_workspace": ISAAC_ROS_WORKSPACE,
        "dgx_deployment_root": receipts["dgx"]["deployment_root"],
        "dgx_ros_workspace": receipts["dgx"]["ros_workspace"],
        "allowed_post_deployment_changes": changed,
    }


def validate_deployment(
    deployment_root: Path, deployment_ref: str, role: str = "isaac"
) -> dict[str, Any]:
    _require(role in {"dgx", "isaac"}, "invalid deployment role")
    root_text = deployment_root.as_posix()
    _require(SHA1.fullmatch(deployment_ref) is not None, "invalid deployment ref")
    _require(deployment_root.is_dir() and not deployment_root.is_symlink(), "unsafe deployment root")
    _require(deployment_root.resolve().as_posix() == root_text, "deployment root is not canonical")
    receipt = _load(deployment_root / "deployment_receipt.json")
    manifest = _load(deployment_root / "payload_manifest.json")
    grant_id = str(receipt.get("grant_id", ""))
    parallel_lane_clone = receipt.get("parallel_lane_clone")
    _require(parallel_lane_clone in {None, "b"}, "invalid parallel lane clone marker")
    stable_root = ISAAC_STABLE_ROOT if role == "isaac" else DGX_STABLE_ROOT
    role_suffix = f"{role}-b" if parallel_lane_clone == "b" else role
    expected_root = (
        f"{stable_root}/.t4-deployments/"
        f"{grant_id}-{deployment_ref[:12]}-{role_suffix}"
    )
    expected_workspace = (
        ISAAC_ROS_WORKSPACE if role == "isaac" else f"{expected_root}/ros_ws"
    )
    checks = {
        "root": root_text == expected_root,
        "receipt_status": receipt.get("status") == "PASS",
        "receipt_role": receipt.get("role") == role,
        "receipt_parallel_lane": (
            receipt.get("parallel_lane_clone") == parallel_lane_clone
        ),
        "receipt_phase": receipt.get("phase") == "ready",
        "receipt_exit": int(receipt.get("exit_code", -1)) == 0,
        "receipt_ref": receipt.get("ref_sha") == deployment_ref,
        "receipt_root": receipt.get("deployment_root") == expected_root,
        "receipt_workspace": receipt.get("ros_workspace") == expected_workspace,
        "receipt_policy": receipt.get("runtime_policy") == "completion_sim",
        "receipt_target": receipt.get("runtime_target") == "isaac_simulation",
        "receipt_lease": receipt.get("resource_lease_ack") == "dgx+isaac",
        "receipt_strict": receipt.get("strict_evidence_modified") is False,
        "receipt_go2": receipt.get("real_go2_targeted") is False,
        "manifest_status": manifest.get("status") == "FUNCTIONAL_PAYLOAD_READY",
        "manifest_ref": manifest.get("ref_sha") == deployment_ref,
        "manifest_policy": manifest.get("runtime_policy") == "completion_sim",
        "manifest_target": manifest.get("runtime_target") == "isaac_simulation",
        "manifest_model_host": manifest.get("model_host") == "dgx_spark_only",
        "manifest_strict": manifest.get("strict_evidence_modified") is False,
        "manifest_go2": manifest.get("real_go2_targeted") is False,
    }
    _require(GRANT.fullmatch(grant_id) is not None, "invalid deployment grant")
    _require(all(checks.values()), f"deployment contract failed: {checks}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "checks": checks,
        "grant_id": grant_id,
        "parallel_lane_clone": parallel_lane_clone,
    }


def validate_dgx_workspace(deployment_root: Path, workspace: Path) -> dict[str, Any]:
    expected_root_prefix = f"{DGX_STABLE_ROOT}/.t4-deployments/"
    _require(
        deployment_root.resolve(strict=True).as_posix().startswith(expected_root_prefix),
        "unsafe DGX deployment root",
    )
    expected_workspace = deployment_root / "ros_ws"
    _require(workspace.resolve(strict=True) == expected_workspace.resolve(strict=True), "DGX workspace changed")
    _require((workspace / "install" / "setup.bash").is_file(), "DGX ROS install is missing")
    source_root = workspace / "src"
    _require(source_root.is_dir() and not source_root.is_symlink(), "DGX source root is unsafe")
    observed = sorted(item.name for item in source_root.iterdir())
    expected = sorted(
        [
            "go2_sensor_bridge",
            "internvla_go2_controller",
            "internvla_nav2_adapter",
            "internvla_ros2",
            "internvla_ros2_msgs",
            "internvla_t4_recovery",
            "internvla_t4_sensors",
        ]
    )
    _require(observed == expected, f"DGX workspace package set changed: {observed}")
    targets: dict[str, str] = {}
    for package in expected:
        candidate = source_root / package
        _require(candidate.is_symlink(), f"DGX workspace package is not ref-bound: {package}")
        target = candidate.resolve(strict=True)
        _require(target == (deployment_root / package).resolve(strict=True), f"DGX package target changed: {package}")
        targets[package] = target.as_posix()
    return {
        "schema_version": 1,
        "status": "PASS",
        "deployment_root": deployment_root.as_posix(),
        "workspace": workspace.as_posix(),
        "package_targets": targets,
    }


def _source_tree(root: Path) -> dict[str, tuple[str, bool]]:
    _require(root.is_dir() and not root.is_symlink(), f"source root is unsafe: {root}")
    observed: dict[str, tuple[str, bool]] = {}
    for directory, directory_names, file_names in os.walk(root):
        directory_path = Path(directory)
        for name in tuple(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ContractError(f"source tree contains a symlink: {candidate}")
        directory_names[:] = sorted(
            name for name in directory_names if name != "__pycache__"
        )
        for name in sorted(file_names):
            if name.endswith(".pyc"):
                continue
            candidate = directory_path / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f"source tree contains a non-regular file: {candidate}")
            relative = candidate.relative_to(root).as_posix()
            observed[relative] = (
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
                bool(metadata.st_mode & 0o111),
            )
    return observed


def validate_overlay_source(deployment_root: Path, workspace: Path) -> dict[str, Any]:
    _require(
        deployment_root.resolve(strict=True).as_posix().startswith(
            f"{ISAAC_STABLE_ROOT}/.t4-deployments/"
        ),
        "unsafe deployment root for overlay validation",
    )
    _require(workspace.resolve(strict=True).as_posix() == ISAAC_ROS_WORKSPACE, "overlay workspace changed")
    package_digests: dict[str, str] = {}
    for package in ISAAC_OVERLAY_PACKAGES:
        source = _source_tree(deployment_root / package)
        destination = _source_tree(workspace / "src" / package)
        _require(source == destination, f"Isaac overlay source drifted: {package}")
        digest = hashlib.sha256()
        for path, (file_hash, executable) in sorted(source.items()):
            digest.update(path.encode("utf-8") + b"\0")
            digest.update(file_hash.encode("ascii") + b"\0")
            digest.update(b"x" if executable else b"-")
        package_digests[package] = digest.hexdigest()
    return {
        "schema_version": 1,
        "status": "PASS",
        "deployment_root": deployment_root.as_posix(),
        "workspace": workspace.as_posix(),
        "packages": package_digests,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"required JSONL is missing or unsafe: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"JSONL record is not an object: {path}")
            records.append(value)
    return records


def validate_oracle_result(
    result_dir: Path,
    grant_id: str,
    authorization_ref: str,
    deployment_ref: str,
) -> dict[str, Any]:
    _require(result_dir.is_dir() and not result_dir.is_symlink(), "Oracle result is missing or unsafe")
    _require(GRANT.fullmatch(grant_id) is not None, "invalid Oracle grant")
    _require(SHA1.fullmatch(authorization_ref) is not None, "invalid authorization ref")
    _require(SHA1.fullmatch(deployment_ref) is not None, "invalid deployment ref")
    validation = _load(result_dir / "validation.json")
    phase = _load(result_dir / "phase_status.json")
    contract = _load(result_dir / "t4_run_contract.json")
    metrics_root = _load(result_dir / "result.json")
    active = _load(result_dir / "active_summary.json")
    controller = _load(result_dir / "controller_summary.json")
    per_episode = _load(result_dir / "per_episode.json")
    invocation = _load(result_dir / "oracle_invocation.json")
    active_records = _jsonl(result_dir / "active_records.jsonl")
    metrics = metrics_root.get("val_unseen", metrics_root)
    required = validation.get("required", {})
    safety = validation.get("safety", {})
    t4_summaries = [
        item for item in active_records if item.get("event") == "ablation_runtime_summary"
    ]
    expected_factors = {
        "system_mode": "full_system1_system2",
        "trajectory_mode": "full_trajectory",
        "termination_mode": "model_stop",
        "history_mode": "on",
        "recovery_mode": "off",
        "view_mode": "go2_view",
    }
    collisions = int(safety.get("physical_collision_count", -1))
    checks = {
        "validation": validation.get("status") == "PASS",
        "validation_schema": validation.get("schema_version") == 2,
        "validation_policy": validation.get("runtime_policy") == "completion_sim",
        "validation_quality": validation.get("quality_status") in {"PASS", "WARN"},
        "episode_requirement": int(required.get("episode_count", -1)) == 10,
        "sr_requirement": abs(float(required.get("minimum_sr", -1.0)) - 0.8) <= 1e-12,
        "map_requirement": required.get("map_source") == "static_map",
        "pose_requirement": required.get("pose_source") == "ground_truth",
        "collision_policy": required.get("physical_collision_policy") == "metric_only_warn",
        "fatal_safety_names": set(required.get("fatal_safety_zero", []))
        == {"nan_count", "fall_count", "stale_motion_execution_count"},
        "fatal_safety_zero": all(
            int(safety.get(name, -1)) == 0
            for name in ("nan_count", "fall_count", "stale_motion_execution_count")
        ),
        "collision_recorded": collisions >= 0,
        "collision_quality": validation.get("quality_status")
        == ("WARN" if collisions else "PASS"),
        "phase": phase.get("status") == "PASS" and phase.get("phase") == "obstacle_oracle",
        "phase_exits": phase.get("exit_codes")
        == {"preflight": 0, "evaluator": 0, "validation": 0},
        "run_policy": contract.get("runtime_policy") == "completion_sim",
        "run_target": contract.get("runtime_target") == "isaac",
        "run_map": contract.get("map_source") == "static_map",
        "run_pose": contract.get("pose_source") == "ground_truth",
        "run_recovery": contract.get("recovery_enabled") is False,
        "run_geometry": contract.get("mapping_and_safety_geometry") == "go2_frozen",
        "metrics_count": int(metrics.get("Count", metrics.get("length", -1))) == 10,
        "metrics_sr": float(metrics.get("SR", -1.0)) >= 0.8,
        "active": active.get("status") == "FINISHED"
        and int(active.get("failure_count", -1)) == 0,
        "controller": controller.get("status") == "FINISHED"
        and controller.get("map_source") == "static_map"
        and controller.get("pose_source") == "ground_truth"
        and controller.get("ground_truth_pose_used_for_nav") is True,
        "episodes": int(per_episode.get("completed_episode_count", -1)) == 10,
        "t4_adapter": bool(t4_summaries)
        and t4_summaries[-1].get("variant_id") == "none"
        and t4_summaries[-1].get("factors") == expected_factors,
        "invocation_grant": invocation.get("grant_id") == grant_id,
        "invocation_auth_ref": invocation.get("authorization_ref_sha") == authorization_ref,
        "invocation_deployment_ref": invocation.get("deployment_ref_sha") == deployment_ref,
        "invocation_policy": invocation.get("runtime_policy") == "completion_sim",
        "invocation_target": invocation.get("runtime_target") == "isaac_simulation",
        "invocation_lease": invocation.get("resource_lease_ack") == "isaac",
        "invocation_model_host": invocation.get("model_host") == "dgx_spark_only",
        "invocation_no_model": invocation.get("model_process_started") is False,
        "invocation_adapter": invocation.get("use_t4_adapter") is True,
        "invocation_count": int(invocation.get("expected_episode_count", -1)) == 10,
        "invocation_sr": abs(float(invocation.get("minimum_sr", -1.0)) - 0.8) <= 1e-12,
        "invocation_strict": invocation.get("strict_evidence_modified") is False,
        "invocation_go2": invocation.get("real_go2_targeted") is False,
    }
    _require(all(checks.values()), f"Oracle result contract failed: {checks}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "quality_status": validation["quality_status"],
        "physical_collision_count": collisions,
        "grant_id": grant_id,
        "authorization_ref_sha": authorization_ref,
        "deployment_ref_sha": deployment_ref,
        "checks": checks,
    }


def validate_pilot_result(
    result_dir: Path,
    grant_id: str,
    authorization_ref: str,
    deployment_ref: str,
) -> dict[str, Any]:
    _require(result_dir.is_dir() and not result_dir.is_symlink(), "pilot result is missing or unsafe")
    _require(GRANT.fullmatch(grant_id) is not None, "invalid pilot grant")
    _require(SHA1.fullmatch(authorization_ref) is not None, "invalid authorization ref")
    _require(SHA1.fullmatch(deployment_ref) is not None, "invalid deployment ref")
    validation = _load(result_dir / "validation.json")
    phase = _load(result_dir / "phase_status.json")
    contract = _load(result_dir / "t4_run_contract.json")
    metrics_root = _load(result_dir / "result.json")
    active = _load(result_dir / "active_summary.json")
    controller = _load(result_dir / "controller_summary.json")
    per_episode = _load(result_dir / "per_episode.json")
    client = _load(result_dir / "client_summary.json")
    invocation = _load(result_dir / "pilot_invocation.json")
    active_records = _jsonl(result_dir / "active_records.jsonl")
    metrics = metrics_root.get("val_unseen", metrics_root)
    required = validation.get("required", {})
    safety = validation.get("safety", {})
    model_health = client.get("model_health_start") or {}
    t4_summaries = [
        item for item in active_records if item.get("event") == "ablation_runtime_summary"
    ]
    expected_factors = {
        "system_mode": "full_system1_system2",
        "trajectory_mode": "full_trajectory",
        "termination_mode": "model_stop",
        "history_mode": "on",
        "recovery_mode": "off",
        "view_mode": "go2_view",
    }
    collisions = int(safety.get("physical_collision_count", -1))
    sr = float(metrics.get("SR", 0.0))
    checks = {
        "validation": validation.get("status") == "PASS",
        "validation_schema": validation.get("schema_version") == 2,
        "validation_policy": validation.get("runtime_policy") == "completion_sim",
        "validation_quality": validation.get("quality_status") in {"PASS", "WARN"},
        "episode_requirement": int(required.get("episode_count", -1)) == 20,
        "sr_requirement": abs(float(required.get("minimum_sr", -1.0)) - 0.01) <= 1e-12,
        "map_requirement": required.get("map_source") == "static_map",
        "pose_requirement": required.get("pose_source") == "ground_truth",
        "collision_policy": required.get("physical_collision_policy") == "metric_only_warn",
        "fatal_safety_names": set(required.get("fatal_safety_zero", []))
        == {"nan_count", "fall_count", "stale_motion_execution_count"},
        "fatal_safety_zero": all(
            int(safety.get(name, -1)) == 0
            for name in ("nan_count", "fall_count", "stale_motion_execution_count")
        ),
        "collision_recorded": collisions >= 0,
        "collision_quality": validation.get("quality_status")
        == ("WARN" if collisions else "PASS"),
        "phase": phase.get("status") == "PASS" and phase.get("phase") == "pilot",
        "phase_exits": phase.get("exit_codes")
        == {"preflight": 0, "evaluator": 0, "validation": 0},
        "run_policy": contract.get("runtime_policy") == "completion_sim",
        "run_target": contract.get("runtime_target") == "isaac",
        "run_map": contract.get("map_source") == "static_map",
        "run_pose": contract.get("pose_source") == "ground_truth",
        "run_recovery": contract.get("recovery_enabled") is False,
        "run_geometry": contract.get("mapping_and_safety_geometry") == "go2_frozen",
        "metrics_count": int(metrics.get("Count", metrics.get("length", -1))) == 20,
        "metrics_sr_nonzero": sr > 0.0,
        "active": active.get("status") == "FINISHED"
        and int(active.get("failure_count", -1)) == 0,
        "controller": controller.get("status") == "FINISHED"
        and controller.get("map_source") == "static_map"
        and controller.get("pose_source") == "ground_truth"
        and controller.get("ground_truth_pose_used_for_nav") is True,
        "episodes": int(per_episode.get("completed_episode_count", -1)) == 20,
        "client": client.get("status") == "FINISHED"
        and int(client.get("step_count", 0)) > 0,
        "client_pose": validation.get("client_pose_ok") is True,
        "model_health": model_health.get("status") == "PASS"
        and model_health.get("status_code") == 0
        and model_health.get("initialized") is False
        and model_health.get("lifecycle_state") == 0
        and model_health.get("model_revision") == "1d8d078aa9031a4a02a1ae05844d49a1768a10e4"
        and model_health.get("checkpoint_revision") == "a698a9e898b4001621a319e1bc89f02ec715cc86",
        "t4_adapter": bool(t4_summaries)
        and t4_summaries[-1].get("variant_id") == "none"
        and t4_summaries[-1].get("factors") == expected_factors,
        "invocation_grant": invocation.get("grant_id") == grant_id,
        "invocation_auth_ref": invocation.get("authorization_ref_sha") == authorization_ref,
        "invocation_deployment_ref": invocation.get("deployment_ref_sha") == deployment_ref,
        "invocation_policy": invocation.get("runtime_policy") == "completion_sim",
        "invocation_target": invocation.get("runtime_target") == "isaac_simulation",
        "invocation_lease": invocation.get("resource_lease_ack") == "dgx+isaac",
        "invocation_model_host": invocation.get("model_host") == "dgx_spark_only",
        "invocation_model": invocation.get("model_process_started") is True,
        "invocation_adapter": invocation.get("use_t4_adapter") is True,
        "invocation_count": int(invocation.get("expected_episode_count", -1)) == 20,
        "invocation_sr": abs(float(invocation.get("minimum_sr", -1.0)) - 0.01) <= 1e-12,
        "invocation_strict": invocation.get("strict_evidence_modified") is False,
        "invocation_go2": invocation.get("real_go2_targeted") is False,
    }
    _require(all(checks.values()), f"pilot result contract failed: {checks}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "quality_status": validation["quality_status"],
        "physical_collision_count": collisions,
        "success_rate": sr,
        "grant_id": grant_id,
        "authorization_ref_sha": authorization_ref,
        "deployment_ref_sha": deployment_ref,
        "checks": checks,
    }


def residual_probe(
    deployment_root: Path, socket_root: Path | None, role: str = "isaac"
) -> dict[str, Any]:
    _require(role in {"dgx", "isaac"}, "invalid residual role")
    root = deployment_root.resolve(strict=True)
    stable_root = ISAAC_STABLE_ROOT if role == "isaac" else DGX_STABLE_ROOT
    _require(root.as_posix().startswith(f"{stable_root}/.t4-deployments/"), "unsafe residual root")
    if role == "isaac":
        _require(socket_root == deployment_root / "runtime" / "t4_ipc", "socket root changed")
    else:
        _require(socket_root is None, "DGX residual probe cannot inspect Isaac sockets")
    excluded: set[int] = set()
    pid = os.getpid()
    while pid > 1:
        excluded.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except (FileNotFoundError, IndexError, ValueError):
            break
    process_ids: list[int] = []
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) in excluded:
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", "replace"
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if root.as_posix() in command:
                process_ids.append(int(entry.name))
    sockets: list[str] = []
    if socket_root is not None and socket_root.exists():
        _require(socket_root.is_dir() and not socket_root.is_symlink(), "unsafe socket root")
        for candidate in socket_root.rglob("*"):
            try:
                mode = candidate.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISSOCK(mode):
                sockets.append(candidate.relative_to(deployment_root).as_posix())
    payload = {
        "schema_version": 1,
        "status": "PASS" if not process_ids and not sockets else "FAIL",
        "deployment_root": root.as_posix(),
        "role": role,
        "pid_count": len(process_ids),
        "pids": sorted(process_ids),
        "socket_count": len(sockets),
        "sockets": sorted(sockets),
        "recorded_unix": time.time(),
    }
    return payload


def validate_collected(
    result_dir: Path,
    receipt_path: Path,
    output: Path,
    grant_id: str,
    authorization_ref: str,
    deployment_ref: str,
    remote_stage_cleanup: bool,
) -> dict[str, Any]:
    receipt = _load(receipt_path)
    oracle = validate_oracle_result(
        result_dir, grant_id, authorization_ref, deployment_ref
    )
    checks = {
        "receipt_status": receipt.get("status") == "PASS",
        "receipt_grant": receipt.get("grant_id") == grant_id,
        "receipt_auth_ref": receipt.get("authorization_ref_sha") == authorization_ref,
        "receipt_deployment_ref": receipt.get("deployment_ref_sha") == deployment_ref,
        "receipt_run": int(receipt.get("run_exit_code", -1)) == 0,
        "receipt_validation": int(receipt.get("validation_exit_code", -1)) == 0,
        "receipt_cleanup": int(receipt.get("cleanup_probe_exit_code", -1)) == 0,
        "receipt_residual": int(receipt.get("residual_probe_exit_code", -1)) == 0,
        "receipt_archive": int(receipt.get("archive_exit_code", -1)) == 0,
        "receipt_policy": receipt.get("runtime_policy") == "completion_sim",
        "receipt_target": receipt.get("runtime_target") == "isaac_simulation",
        "receipt_lease": receipt.get("resource_lease_ack") == "isaac",
        "receipt_model_host": receipt.get("model_host") == "dgx_spark_only",
        "receipt_no_model": receipt.get("model_process_started") is False,
        "receipt_strict": receipt.get("strict_evidence_modified") is False,
        "receipt_go2": receipt.get("real_go2_targeted") is False,
        "remote_stage_cleanup": remote_stage_cleanup,
    }
    _require(all(checks.values()), f"collected Oracle receipt failed: {checks}")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "quality_status": oracle["quality_status"],
        "physical_collision_count": oracle["physical_collision_count"],
        "grant_id": grant_id,
        "authorization_ref_sha": authorization_ref,
        "deployment_ref_sha": deployment_ref,
        "runtime_policy": "completion_sim",
        "runtime_target": "isaac_simulation",
        "map_source": "static_map",
        "pose_source": "ground_truth",
        "model_host": "dgx_spark_only",
        "model_process_started": False,
        "strict_evidence_modified": False,
        "real_go2_targeted": False,
        "remote_receipt_checks": checks,
        "oracle_validation": oracle,
        "recorded_unix": time.time(),
    }
    if output.exists():
        raise ContractError("collected summary output already exists")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_pilot_collected(
    pilot_result: Path,
    model_result: Path,
    model_ready_receipt_path: Path,
    model_stop_receipt_path: Path,
    pilot_receipt_path: Path,
    output: Path,
    grant_id: str,
    authorization_ref: str,
    deployment_ref: str,
    remote_stages_cleaned: bool,
) -> dict[str, Any]:
    pilot = validate_pilot_result(
        pilot_result, grant_id, authorization_ref, deployment_ref
    )
    ready = _load(model_ready_receipt_path)
    stop = _load(model_stop_receipt_path)
    pilot_receipt = _load(pilot_receipt_path)
    weight = _load(model_result / "model_weight_audit.json")
    health = _load(model_result / "model_health_ready.json")
    model_invocation = _load(model_result / "model_invocation.json")
    model_residual = _load(model_result / "model_residual_probe.json")
    pilot_invocation = _load(pilot_result / "pilot_invocation.json")
    ready_sha256 = hashlib.sha256(model_ready_receipt_path.read_bytes()).hexdigest()
    checks = {
        "model_ready": ready.get("status") == "PASS",
        "model_ready_grant": ready.get("grant_id") == grant_id,
        "model_ready_refs": ready.get("authorization_ref_sha") == authorization_ref
        and ready.get("deployment_ref_sha") == deployment_ref,
        "model_ready_lease": ready.get("resource_lease_ack") == "dgx+isaac",
        "model_ready_backend": ready.get("backend") == "real",
        "model_stop": stop.get("status") == "PASS"
        and stop.get("model_was_alive") is True
        and int(stop.get("process_group_remaining", -1)) == 0,
        "model_stop_refs": stop.get("authorization_ref_sha") == authorization_ref
        and stop.get("deployment_ref_sha") == deployment_ref,
        "model_stop_residual": int(stop.get("residual_probe_exit_code", -1)) == 0
        and model_residual.get("status") == "PASS"
        and int(model_residual.get("pid_count", -1)) == 0,
        "model_archive": int(stop.get("archive_exit_code", -1)) == 0,
        "model_weight": weight.get("status") == "PASS"
        and weight.get("backend") == "real"
        and weight.get("meta_parameter_count") == 0
        and weight.get("meta_buffer_count") == 0,
        "model_health": health.get("status") == "PASS"
        and health.get("model_host") == "dgx_spark"
        and health.get("step_action_ready") is True,
        "model_identity": weight.get("model_revision") == health.get("model_revision")
        and weight.get("checkpoint_revision") == health.get("checkpoint_revision"),
        "model_invocation": model_invocation.get("grant_id") == grant_id
        and model_invocation.get("backend") == "real"
        and model_invocation.get("preload_model") is True,
        "pilot_receipt": pilot_receipt.get("status") == "PASS",
        "pilot_receipt_refs": pilot_receipt.get("authorization_ref_sha") == authorization_ref
        and pilot_receipt.get("deployment_ref_sha") == deployment_ref,
        "pilot_receipt_run": int(pilot_receipt.get("run_exit_code", -1)) == 0
        and int(pilot_receipt.get("validation_exit_code", -1)) == 0,
        "pilot_receipt_cleanup": int(pilot_receipt.get("cleanup_probe_exit_code", -1)) == 0
        and int(pilot_receipt.get("residual_probe_exit_code", -1)) == 0,
        "pilot_receipt_archive": int(pilot_receipt.get("archive_exit_code", -1)) == 0,
        "model_ready_digest": pilot_invocation.get("model_ready_receipt_sha256")
        == ready_sha256
        and pilot_receipt.get("model_ready_receipt_sha256") == ready_sha256,
        "remote_stages_cleaned": remote_stages_cleaned,
    }
    _require(all(checks.values()), f"collected pilot contract failed: {checks}")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "quality_status": pilot["quality_status"],
        "physical_collision_count": pilot["physical_collision_count"],
        "success_rate": pilot["success_rate"],
        "episode_count": 20,
        "grant_id": grant_id,
        "authorization_ref_sha": authorization_ref,
        "deployment_ref_sha": deployment_ref,
        "runtime_policy": "completion_sim",
        "runtime_target": "isaac_simulation",
        "map_source": "static_map",
        "pose_source": "ground_truth",
        "model_host": "dgx_spark_only",
        "model_process_started": True,
        "strict_evidence_modified": False,
        "real_go2_targeted": False,
        "checks": checks,
        "pilot_validation": pilot,
        "recorded_unix": time.time(),
    }
    if output.exists():
        raise ContractError("pilot summary output already exists")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def extract_oracle_archive(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(archive_path.is_file() and not archive_path.is_symlink(), "Oracle archive is missing or unsafe")
    _require(output_dir.is_dir() and not output_dir.is_symlink(), "Oracle extraction root is unsafe")
    _require(not any(output_dir.iterdir()), "Oracle extraction root is not empty")
    observed: set[str] = set()
    extracted = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            raw = member.name
            while raw.startswith("./"):
                raw = raw[2:]
            if raw in {"", "."} and member.isdir():
                continue
            path = PurePosixPath(raw)
            if (
                not raw
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in raw
                or raw in observed
                or not (member.isdir() or member.isfile())
            ):
                raise ContractError(f"unsafe Oracle archive member: {member.name}")
            observed.add(raw)
        for member in members:
            raw = member.name
            while raw.startswith("./"):
                raw = raw[2:]
            if raw in {"", "."} and member.isdir():
                continue
            destination = output_dir / Path(*PurePosixPath(raw).parts)
            resolved_parent = destination.parent.resolve()
            if os.path.commonpath((str(output_dir.resolve()), str(resolved_parent))) != str(
                output_dir.resolve()
            ):
                raise ContractError(f"Oracle archive escaped extraction root: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ContractError(f"Oracle archive member is unreadable: {member.name}")
            with destination.open("xb") as output:
                output.write(stream.read())
            extracted += 1
    _require(extracted > 0, "Oracle archive was empty")
    return {
        "schema_version": 1,
        "status": "PASS",
        "archive": str(archive_path),
        "output_dir": str(output_dir),
        "member_count": len(observed),
        "file_count": extracted,
    }


def _write_fresh(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ContractError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-fields")
    prepare.add_argument("--result-dir", type=Path, required=True)
    prepare.add_argument("--git-dir", type=Path, required=True)
    prepare.add_argument("--authorization-ref", required=True)
    prepare.add_argument(
        "--format", choices=("json", "lines", "pilot-lines"), default="json"
    )

    deployment = subparsers.add_parser("validate-deployment")
    deployment.add_argument("--deployment-root", type=Path, required=True)
    deployment.add_argument("--deployment-ref", required=True)
    deployment.add_argument("--role", choices=("dgx", "isaac"), default="isaac")

    dgx_workspace = subparsers.add_parser("validate-dgx-workspace")
    dgx_workspace.add_argument("--deployment-root", type=Path, required=True)
    dgx_workspace.add_argument("--workspace", type=Path, required=True)

    overlay = subparsers.add_parser("validate-overlay-source")
    overlay.add_argument("--deployment-root", type=Path, required=True)
    overlay.add_argument("--workspace", type=Path, required=True)

    oracle = subparsers.add_parser("validate-oracle-result")
    oracle.add_argument("--result-dir", type=Path, required=True)
    oracle.add_argument("--grant-id", required=True)
    oracle.add_argument("--authorization-ref", required=True)
    oracle.add_argument("--deployment-ref", required=True)
    oracle.add_argument("--output", type=Path)

    pilot = subparsers.add_parser("validate-pilot-result")
    pilot.add_argument("--result-dir", type=Path, required=True)
    pilot.add_argument("--grant-id", required=True)
    pilot.add_argument("--authorization-ref", required=True)
    pilot.add_argument("--deployment-ref", required=True)
    pilot.add_argument("--output", type=Path)

    residual = subparsers.add_parser("residual-probe")
    residual.add_argument("--deployment-root", type=Path, required=True)
    residual.add_argument("--socket-root", type=Path)
    residual.add_argument("--role", choices=("dgx", "isaac"), default="isaac")
    residual.add_argument("--output", type=Path, required=True)

    collected = subparsers.add_parser("validate-collected")
    collected.add_argument("--result-dir", type=Path, required=True)
    collected.add_argument("--receipt", type=Path, required=True)
    collected.add_argument("--output", type=Path, required=True)
    collected.add_argument("--grant-id", required=True)
    collected.add_argument("--authorization-ref", required=True)
    collected.add_argument("--deployment-ref", required=True)
    collected.add_argument("--remote-stage-cleanup", choices=("PASS", "FAIL"), required=True)

    pilot_collected = subparsers.add_parser("validate-pilot-collected")
    pilot_collected.add_argument("--pilot-result", type=Path, required=True)
    pilot_collected.add_argument("--model-result", type=Path, required=True)
    pilot_collected.add_argument("--model-ready-receipt", type=Path, required=True)
    pilot_collected.add_argument("--model-stop-receipt", type=Path, required=True)
    pilot_collected.add_argument("--pilot-receipt", type=Path, required=True)
    pilot_collected.add_argument("--output", type=Path, required=True)
    pilot_collected.add_argument("--grant-id", required=True)
    pilot_collected.add_argument("--authorization-ref", required=True)
    pilot_collected.add_argument("--deployment-ref", required=True)
    pilot_collected.add_argument("--remote-stages-cleanup", choices=("PASS", "FAIL"), required=True)

    extract = subparsers.add_parser("extract-oracle-archive")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "prepare-fields":
            payload = validate_prepare_result(
                args.result_dir, args.git_dir, args.authorization_ref
            )
            if args.format == "lines":
                print(payload["prepare_grant_id"])
                print(payload["deployment_ref_sha"])
                print(payload["isaac_deployment_root"])
                print(payload["isaac_ros_workspace"])
            elif args.format == "pilot-lines":
                print(payload["prepare_grant_id"])
                print(payload["deployment_ref_sha"])
                print(payload["dgx_deployment_root"])
                print(payload["dgx_ros_workspace"])
                print(payload["isaac_deployment_root"])
                print(payload["isaac_ros_workspace"])
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "validate-deployment":
            print(
                json.dumps(
                    validate_deployment(
                        args.deployment_root, args.deployment_ref, args.role
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "validate-dgx-workspace":
            print(
                json.dumps(
                    validate_dgx_workspace(args.deployment_root, args.workspace),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "validate-overlay-source":
            print(
                json.dumps(
                    validate_overlay_source(args.deployment_root, args.workspace),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "validate-oracle-result":
            payload = validate_oracle_result(
                args.result_dir,
                args.grant_id,
                args.authorization_ref,
                args.deployment_ref,
            )
            if args.output is not None:
                _write_fresh(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "validate-pilot-result":
            payload = validate_pilot_result(
                args.result_dir,
                args.grant_id,
                args.authorization_ref,
                args.deployment_ref,
            )
            if args.output is not None:
                _write_fresh(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "residual-probe":
            payload = residual_probe(
                args.deployment_root, args.socket_root, args.role
            )
            _write_fresh(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            if payload["status"] != "PASS":
                raise SystemExit(1)
        elif args.command == "validate-collected":
            payload = validate_collected(
                args.result_dir,
                args.receipt,
                args.output,
                args.grant_id,
                args.authorization_ref,
                args.deployment_ref,
                args.remote_stage_cleanup == "PASS",
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "validate-pilot-collected":
            payload = validate_pilot_collected(
                args.pilot_result,
                args.model_result,
                args.model_ready_receipt,
                args.model_stop_receipt,
                args.pilot_receipt,
                args.output,
                args.grant_id,
                args.authorization_ref,
                args.deployment_ref,
                args.remote_stages_cleanup == "PASS",
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                json.dumps(
                    extract_oracle_archive(args.archive, args.output_dir),
                    indent=2,
                    sort_keys=True,
                )
            )
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"functional run contract: {exc}\n")


if __name__ == "__main__":
    main()
