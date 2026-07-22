#!/usr/bin/env python3
"""Outer lease-internal supervisor for one fresh model-free sensor result."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .atomic import atomic_write_json
from .contract import (
    CONTRACT_SHA256,
    DIAGNOSTIC_GEOMETRY,
    DIAGNOSTIC_LIGHT,
    PROFILES,
    contract_payload,
)
from .isaac_eula import (
    FROZEN_ISAAC_EULA_POLICY,
    ISAAC_RUNTIME_PREFLIGHT_FILENAME,
    apply_frozen_isaac_eula_environment,
    require_frozen_runtime_preflight,
)
from .isaac_experience import (
    ISAAC_STARTUP_READY_FILENAME,
    require_frozen_startup_ready,
)
from .outer_liveness import OuterAliveLock
from .processes import ProcessRegistry, socket_listener_count
from .pythonpath_policy import (
    require_frozen_setup_sha256,
    validate_child_pythonpath,
)
from .runtime_policy import RuntimePolicy, policy_for_session_profile
from .validate import validate_result


GRANT_REF = "refs/heads/codex/parallel-integration"
GRANT_MARKER = "INTERNAV_ONLINE_GRANT_V1"
CALIBRATION_OVERRIDE_PREFIXES = (
    "INTERNVLA_T4_CAMERA_",
    "INTERNVLA_T4_DEPTH_",
    "INTERNVLA_T4_STEREO_",
    "INTERNVLA_T4_R3_FRONT_RGB_",
)
MANAGED_OUTER_ROLES = frozenset(
    {
        "asset_builder",
        "ros_container_client",
        "isaac_model_free_workload",
        "inner_cleanup_recovery",
    }
)
COMPLETION_MAP_POST_PRODUCER_WAIT_SEC = 20.0


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            return sum(1 for line in stream if line.rstrip())
    except FileNotFoundError:
        return 0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} is not a JSON object")
    return value


def _profile_worker(profile: str) -> str:
    return "10" if profile == "completion_sim_map" else "01R"


def _profile_result_root(control_root: Path, profile: str) -> Path:
    relative = (
        "results/parallel/t4_map"
        if profile == "completion_sim_map"
        else "results/parallel/sensor_producer"
    )
    return (control_root / relative).resolve()


def _normal_result(
    control_root: Path, value: Path, profile: str = "bootstrap"
) -> Path:
    allowed = _profile_result_root(control_root, profile)
    result = value.expanduser()
    if not result.is_absolute():
        result = control_root / result
    result = result.resolve(strict=False)
    try:
        result.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(
            f"result directory is outside the {_profile_worker(profile)} exclusive path"
        ) from exc
    return result


def _resolve_fresh_result(
    control_root: Path, value: Path, profile: str = "bootstrap"
) -> Path:
    allowed = _profile_result_root(control_root, profile)
    allowed.mkdir(parents=True, exist_ok=True)
    result = _normal_result(control_root, value, profile)
    # mkdir is the exclusive single-use claim.  Incomplete attempts are never
    # reusable or promotable to PASS.
    result.mkdir(parents=False, exist_ok=False)
    return result


def parse_grant_document(
    document: str,
    *,
    control_root: Path,
    profile: str,
    result_value: Path,
    grant_id: str,
) -> dict[str, Any]:
    pattern = re.compile(
        rf"<!--\s*{GRANT_MARKER}\s*\r?\n(?P<payload>.*?)\r?\n{GRANT_MARKER}\s*-->",
        re.DOTALL,
    )
    matches = list(pattern.finditer(document))
    if len(matches) != 1:
        raise RuntimeError("authoritative grant document must contain exactly one V1 JSON block")
    try:
        payload = json.loads(matches[0].group("payload"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("authoritative grant JSON is malformed") from exc
    required = {
        "schema_version",
        "status",
        "worker",
        "resource",
        "profile",
        "result_dir",
        "grant_id",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("authoritative grant has missing or unexpected fields")
    if payload["schema_version"] != 1 or payload["status"] != "GRANTED":
        raise RuntimeError("authoritative grant status is not exactly GRANTED")
    if payload["worker"] != _profile_worker(profile) or payload["resource"] != "isaac":
        raise RuntimeError("authoritative grant worker/resource mismatch")
    if payload["profile"] != profile:
        raise RuntimeError("authoritative grant profile mismatch")
    if not isinstance(grant_id, str) or not grant_id or payload["grant_id"] != grant_id:
        raise RuntimeError("authoritative grant_id is absent or mismatched")
    if not isinstance(payload["result_dir"], str):
        raise RuntimeError("authoritative result_dir is not a string")
    granted = _normal_result(control_root, Path(payload["result_dir"]), profile)
    requested = _normal_result(control_root, result_value, profile)
    if granted != requested:
        raise RuntimeError("authoritative grant result_dir mismatch")
    return payload


def _git_ref_sha(control_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", GRANT_REF],
        cwd=control_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_authoritative_grant(
    control_root: Path, profile: str, result_value: Path, grant_id: str
) -> tuple[str, dict[str, Any]]:
    ref_sha = _git_ref_sha(control_root)
    document = subprocess.run(
        ["git", "show", f"{ref_sha}:coordination/TASK_BOARD.md"],
        cwd=control_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return ref_sha, parse_grant_document(
        document,
        control_root=control_root,
        profile=profile,
        result_value=result_value,
        grant_id=grant_id,
    )


def _git_tree_blobs(
    control_root: Path, ref_sha: str, *, include_map: bool = False
) -> dict[str, str]:
    source_roots = [
        "sensor_runtime",
        "go2_sensor_bridge/go2_sensor_bridge",
        "scripts",
    ]
    if include_map:
        source_roots.extend(["t4_completion/map", "configs/completion_sim/map"])
    completed = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "-z",
            ref_sha,
            "--",
            *source_roots,
        ],
        cwd=control_root,
        check=True,
        capture_output=True,
    )
    blobs: dict[str, str] = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, raw_path = raw.split(b"\t", 1)
        _mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type == "blob":
            blobs[raw_path.decode("utf-8")] = object_id
    return blobs


def _git_blob_bytes(control_root: Path, object_id: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=control_root,
        check=True,
        capture_output=True,
    ).stdout


def _builder_dependency_closure(
    control_root: Path, blobs: Mapping[str, str], root: str
) -> list[str]:
    pending: list[tuple[str, int]] = [(root, 0)]
    observed: set[str] = set()
    while pending:
        path, depth = pending.pop(0)
        if path in observed:
            continue
        if path not in blobs:
            raise RuntimeError(f"coordination ref lacks required builder source: {path}")
        observed.add(path)
        if depth >= 2:
            continue
        try:
            tree = ast.parse(_git_blob_bytes(control_root, blobs[path]), filename=path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"cannot parse frozen builder dependency {path}") from exc
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Import):
                modules.update(item.name.split(".", 1)[0] for item in node.names)
        for module in sorted(modules):
            candidate = f"scripts/{module}.py"
            if candidate in blobs and module.startswith("build_t4"):
                pending.append((candidate, depth + 1))
    return sorted(observed)


def verify_source_provenance(
    control_root: Path, ref_sha: str, *, include_map: bool = False
) -> dict[str, Any]:
    """Bind every online executable source byte to the frozen coordination ref."""

    if not re.fullmatch(r"[0-9a-f]{40,64}", ref_sha):
        raise RuntimeError("source provenance ref SHA is malformed")
    for shadow in ("sitecustomize.py", "usercustomize.py"):
        if (control_root / shadow).exists():
            raise RuntimeError(f"root Python startup shadow is forbidden: {shadow}")
    root_shadows = sorted(
        path.name
        for path in control_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".py", ".pyc", ".pyo", ".so", ".pyd"}
    )
    if root_shadows:
        raise RuntimeError(f"root Python import shadow is forbidden: {root_shadows}")
    blobs = _git_tree_blobs(control_root, ref_sha, include_map=include_map)
    builder_root = "scripts/build_t4_r3_go2_usd.py"
    bridge_root = "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py"
    builder_closure = _builder_dependency_closure(control_root, blobs, builder_root)
    selected = {
        path
        for path in blobs
        if path.startswith("sensor_runtime/")
        or (
            path.startswith("go2_sensor_bridge/go2_sensor_bridge/")
            and path.endswith((".py", ".sh"))
        )
        or (path.startswith("scripts/") and path.endswith((".py", ".sh")))
    }
    selected.update(builder_closure)
    selected.add(bridge_root)
    if include_map:
        selected.update(
            path
            for path in blobs
            if path.startswith("t4_completion/map/")
            or path.startswith("configs/completion_sim/map/")
        )
    missing_ref = sorted(path for path in (builder_root, bridge_root) if path not in blobs)
    if missing_ref or not any(path.startswith("sensor_runtime/") for path in selected):
        raise RuntimeError(f"coordination ref lacks critical execution source: {missing_ref}")

    local_executables: set[str] = set()
    scan_roots = [
        control_root / "sensor_runtime",
        control_root / "go2_sensor_bridge/go2_sensor_bridge",
        control_root / "scripts",
    ]
    if include_map:
        scan_roots.append(control_root / "t4_completion/map")
    forbidden_runtime_artifacts: list[str] = []
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            raise RuntimeError(f"critical source directory is absent: {scan_root}")
        for path in scan_root.rglob("*"):
            relative = path.relative_to(control_root).as_posix()
            if (
                path.name == "__pycache__"
                or path.suffix.lower() in {".pyc", ".pyo", ".so", ".pyd"}
            ):
                forbidden_runtime_artifacts.append(relative)
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            local_executables.add(relative)
    if forbidden_runtime_artifacts:
        raise RuntimeError(
            "cached/extension execution shadow is forbidden: "
            f"{sorted(forbidden_runtime_artifacts)}"
        )
    expected_executables = {
        path for path in selected if path.endswith((".py", ".sh"))
    }
    extras = sorted(local_executables - expected_executables)
    missing = sorted(expected_executables - local_executables)
    if extras or missing:
        raise RuntimeError(
            f"critical executable source set differs from coordination ref: extras={extras} missing={missing}"
        )

    verified: dict[str, dict[str, str]] = {}
    for relative in sorted(selected):
        path = control_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"critical source is missing, non-regular, or symlinked: {relative}")
        observed = subprocess.run(
            ["git", "hash-object", "--no-filters", "--", str(path)],
            cwd=control_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        expected = blobs[relative]
        if observed != expected:
            raise RuntimeError(
                f"critical source blob differs from coordination ref: {relative}"
            )
        verified[relative] = {"ref_blob": expected, "local_blob": observed}
    return {
        "schema_version": 1,
        "verified": True,
        "ref_sha": ref_sha,
        "critical_paths": verified,
        "builder_dependency_closure": builder_closure,
        "extra_executable_paths": [],
        "python_startup_shadows": [],
        "root_import_shadows": [],
        "cached_or_extension_shadows": [],
    }


def pre_spawn_source_gate(
    control_root: Path, ref_sha: str, *, include_map: bool = False
) -> dict[str, Any]:
    provenance = (
        verify_source_provenance(control_root, ref_sha, include_map=True)
        if include_map
        else verify_source_provenance(control_root, ref_sha)
    )
    if _git_ref_sha(control_root) != ref_sha:
        raise RuntimeError("coordination ref changed after source provenance verification")
    return provenance


def _reject_calibration_overrides(environment: Mapping[str, str]) -> None:
    bad = sorted(
        key for key in environment if any(key.startswith(prefix) for prefix in CALIBRATION_OVERRIDE_PREFIXES)
    )
    if bad:
        raise RuntimeError(f"frozen calibration environment override(s) present: {', '.join(bad)}")


def _expect_mapping(payload: Mapping[str, Any], name: str, expected: Mapping[str, Any]) -> None:
    observed = payload.get(name)
    if not isinstance(observed, dict):
        raise RuntimeError(f"asset manifest lacks {name}")
    for key, value in expected.items():
        actual = observed.get(key)
        if isinstance(value, float):
            if not isinstance(actual, (int, float)) or not math.isclose(float(actual), value, abs_tol=1e-9):
                raise RuntimeError(f"asset manifest {name}.{key} is not frozen")
        elif actual != value:
            raise RuntimeError(f"asset manifest {name}.{key} is not frozen")


def validate_asset_manifest(path: Path, wrapper_path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    _expect_mapping(
        payload,
        "semantic_camera",
        {
            "prim_path": "base/internvla_camera",
            "resolution": [640, 480],
            "translation_from_base_m": [0.2, 0.0, 0.2],
            "height_above_support_m": 0.62,
            "pitch_down_deg": 20.0,
            "hfov_deg": 69.4,
            "vfov_deg": 42.5,
        },
    )
    _expect_mapping(
        payload,
        "depth_camera",
        {
            "prim_path": "base/t4_d435i_depth",
            "resolution": [640, 480],
            "translation_from_base_m": [0.2, 0.0, 0.2],
            "height_above_support_m": 0.62,
            "pitch_down_deg": 20.0,
            "hfov_deg": 87.0,
            "vfov_deg": 58.0,
            "minimum_depth_m": 0.28,
        },
    )
    _expect_mapping(
        payload,
        "go2_front_rgb",
        {
            "prim_path": "base/go2_front_rgb",
            "resolution": [320, 240],
            "ros_publish_resolution": [160, 120],
            "translation_from_base_m": [0.29, 0.0, -0.06],
            "pitch_down_deg": 8.0,
            "hfov_deg": 120.0,
            "vfov_deg": 75.0,
            "clipping_range_m": [0.2, 1_000_000.0],
            "is_depth_source": False,
        },
    )
    _expect_mapping(
        payload,
        "go2_4d_lidar",
        {
            "prim_path": "base/go2_l1_lidar",
            "translation_from_base_m": [0.25, 0.0, 0.18],
            "azimuth_samples": 180,
            "elevation_channels": 8,
            "range_m": [0.1, 12.0],
        },
    )
    wrapper_hash = payload.get("wrapper_sha256")
    if not isinstance(wrapper_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", wrapper_hash):
        raise RuntimeError("asset manifest wrapper_sha256 is invalid")
    if not wrapper_path.is_file():
        raise RuntimeError("asset wrapper named by the session does not exist")
    observed_hash = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    if observed_hash != wrapper_hash:
        raise RuntimeError("asset manifest wrapper_sha256 does not match wrapper content")
    return payload


def _wait_file(
    path: Path,
    deadline: float,
    registry: ProcessRegistry,
    stage: str,
    interrupted: callable,
) -> None:
    while not path.is_file():
        registry.check(stage)
        if interrupted():
            raise InterruptedError("session received an external signal")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{path.name} missed the {stage} deadline")
        time.sleep(0.02)


def remaining_ready_budget(deadline: float, stage: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError(f"{stage} consumed the session-to-ready budget")
    return remaining


def _managed_command(role: str, argv: list[str]) -> list[str]:
    if role not in MANAGED_OUTER_ROLES:
        raise ValueError(f"outer role is not in the managed launch contract: {role}")
    if not argv:
        raise ValueError("managed command argv is empty")
    return [sys.executable, "-m", "sensor_runtime.managed_command", "--", *argv]


def _validate_snapshot_acks(
    result_dir: Path, snapshot_id: str, runtime_policy: RuntimePolicy
) -> dict[str, Any]:
    producer = _load_json(result_dir / "producer_snapshot_ack.json")
    sidecar = _load_json(result_dir / "sidecar_snapshot_ack.json")
    bridge = _load_json(result_dir / "bridge_snapshot_ack.json")
    downstream = _load_json(result_dir / "downstream_snapshot_ack.json")
    identity = (int(producer["generation"]), int(producer["sequence"]))
    if producer.get("status") != "PASS":
        raise RuntimeError("producer snapshot acknowledgement is not PASS")
    for name, payload in (("sidecar", sidecar), ("bridge", bridge), ("downstream", downstream)):
        if payload.get("status") != "PASS" or payload.get("snapshot_id") != snapshot_id:
            raise RuntimeError(f"{name} snapshot acknowledgement is invalid")
    if (int(sidecar["generation"]), int(sidecar["sequence"])) != identity:
        raise RuntimeError("sidecar snapshot identity differs from producer")
    emitter = producer.get("emitter", {})
    accepted, overwritten, reset_cleared, barrier_dropped, sent = (
        int(emitter.get("accepted_count", -1)),
        int(emitter.get("overwrite_count", -1)),
        int(emitter.get("reset_clear_count", -1)),
        int(emitter.get("barrier_drop_count", -1)),
        int(emitter.get("sent_count", -1)),
    )
    if (
        accepted != sent + overwritten + reset_cleared + barrier_dropped
        or int(producer.get("capture_count", -2)) != accepted
    ):
        raise RuntimeError("producer latest-only counts do not reconcile")
    for name in ("last_submitted", "last_sent"):
        value = emitter.get(name)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or (int(value[0]), int(value[1])) != identity
        ):
            raise RuntimeError(f"producer emitter {name} differs from snapshot target")
    if emitter.get("thread_alive") is not True or emitter.get("fault") not in (None, ""):
        raise RuntimeError("producer emitter was unhealthy at snapshot")
    received = int(sidecar.get("server_received_count", -1))
    receive_overwrite = int(sidecar.get("receive_overwrite_count", -1))
    receive_reset_clear = int(sidecar.get("receive_reset_clear_count", -1))
    receive_barrier_drop = int(sidecar.get("receive_barrier_drop_count", -1))
    counts = sidecar.get("writer_counts", {})
    sensor_count = int(counts.get("sensor", -1))
    if (
        received != sent
        or received
        != sensor_count + receive_overwrite + receive_reset_clear + receive_barrier_drop
    ):
        raise RuntimeError("producer/sidecar received counts do not reconcile")
    if int(counts.get("controller", -1)) != sensor_count:
        raise RuntimeError("sidecar controller safe-stop count does not cover every sensor frame")
    if not 1 <= int(counts.get("reset", -1)) <= sensor_count:
        raise RuntimeError("sidecar reset evidence count is invalid")
    if sidecar.get("writer_thread_alive") is not True or sidecar.get("writer_fault") not in (None, ""):
        raise RuntimeError("sidecar evidence writer was unhealthy at snapshot")
    bridge_count = int(bridge.get("frame_count", -1))
    downstream_count = int(downstream.get("frame_count", -1))
    if runtime_policy.name == "completion_sim":
        group_counts = downstream.get("consumer_group_counts")
        functional_groups = downstream.get("required_functional_groups")
        bridge_identity = (int(bridge["generation"]), int(bridge["sequence"]))
        downstream_identity = (
            int(downstream["generation"]),
            int(downstream["sequence"]),
        )
        if (
            bridge.get("runtime_policy") != "completion_sim"
            or int(bridge.get("pending_count", -1)) != 0
            or bridge.get("writer_thread_alive") is not True
            or bridge.get("writer_fault") not in (None, "")
            or downstream.get("runtime_policy") != "completion_sim"
            or downstream.get("recorder_mode") != "nonfatal_consumer_shadow"
            or not isinstance(group_counts, dict)
            or functional_groups != ["internvla", "nav2"]
            or any(int(group_counts.get(name, 0)) < 1 for name in functional_groups)
        ):
            raise RuntimeError("completion_sim consumer-group shadow is incomplete")
        if bridge_count < 1 or downstream_count < len(functional_groups):
            raise RuntimeError("completion_sim lacks actual bridge/consumer output")
        return {
            "producer": producer,
            "sidecar": sidecar,
            "bridge": bridge,
            "downstream": downstream,
            "identity": identity,
            "consumer_identities": {
                "bridge": bridge_identity,
                "downstream": downstream_identity,
            },
            "sensor_count": sensor_count,
        }
    for name, payload in (("bridge", bridge), ("downstream", downstream)):
        if (int(payload["generation"]), int(payload["sequence"])) != identity:
            raise RuntimeError(f"{name} snapshot identity differs from producer")
    if bridge_count != sensor_count or downstream_count != sensor_count:
        raise RuntimeError("sidecar/bridge/actual-downstream frozen prefix counts differ")
    if bridge.get("writer_thread_alive") is not True or bridge.get("writer_fault") not in (None, ""):
        raise RuntimeError("bridge evidence writer was unhealthy at snapshot")
    if (
        int(downstream.get("pending_count", -1)) != 0
        or int(downstream.get("generation_event_queue_count", -1)) != 0
        or int(downstream.get("tf_lookup_exception_count", -1)) != 0
        or downstream.get("recorder_fault") not in (None, "")
    ):
        raise RuntimeError("downstream recorder snapshot was not drained")
    return {
        "producer": producer,
        "sidecar": sidecar,
        "bridge": bridge,
        "downstream": downstream,
        "identity": identity,
        "sensor_count": sensor_count,
    }


def _wait_inner_cleanup(
    registry: ProcessRegistry,
    *,
    control_root: Path,
    result_dir: Path,
    socket_path: Path,
    logs: Path,
    environment: dict[str, str],
    runtime_policy: RuntimePolicy,
) -> dict[str, Any]:
    atomic_write_json(
        result_dir / "inner_stop.request",
        {"schema_version": 2, "requested_wall_unix": time.time(), "reason": "outer_cleanup"},
    )
    artifact = result_dir / "inner_cleanup.json"
    original_client = registry.process("ros_container_client")
    deadline = time.monotonic() + 8.0
    normal_exit = False
    while time.monotonic() < deadline:
        code = original_client.poll()
        if code is not None:
            child = _load_json(artifact) if artifact.is_file() else {}
            allowed_normal_codes = {0}
            if runtime_policy.sigterm_143_is_normal_with_zero_residuals:
                allowed_normal_codes.update({143, -signal.SIGTERM})
            normal_exit = (
                code in allowed_normal_codes
                and child.get("child_cleanup_confirmed") is True
            )
            break
        time.sleep(0.05)
    # Always use a separate container-side process to prove that the original
    # supervisor PID/PGID is zero.  In the timeout path this probe also safely
    # terminates it using the persisted Linux start identity.
    container = environment.get("INTERNVLA_T4_CONTAINER_NAME", "internnav_t4_isaac_ros")
    recovery_log = (logs / "inner_cleanup_recovery.log").open("xb")
    try:
        registry.start(
            "inner_cleanup_recovery",
            _managed_command("inner_cleanup_recovery", [
                "docker", "exec", "--user", "admin", "--workdir", str(control_root),
                "-e", f"CONTROL_ROOT={control_root}", "-e", f"RESULT_DIR={result_dir}",
                "-e", f"SENSOR_SOCKET={socket_path}",
                "-e", (
                    "INTERNNAV_SESSION_PROFILE="
                    f"{environment.get('INTERNNAV_SESSION_PROFILE', '')}"
                ),
                container, "bash", "-lc",
                "exec bash sensor_runtime/run_ros_inner_cleanup.sh",
            ]),
            stdout=recovery_log,
            stderr=subprocess.STDOUT,
            env=environment,
            required=False,
            expected_long_running=False,
        )
        registry.wait_preparation("inner_cleanup_recovery", 10.0)
    finally:
        try:
            recovery_log.close()
        except BaseException as exc:
            raise RuntimeError(f"inner recovery log close failed: {exc}") from exc
    try:
        client_code = original_client.wait(timeout=3.0)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("original ROS container client remained alive after supervisor probe") from exc
    payload = _load_json(artifact)
    mode = "normal_exit_probe" if normal_exit else "recovery_terminated"
    _validate_inner_cleanup_result(
        payload,
        client_code=client_code,
        mode=mode,
        allow_sigterm_143=runtime_policy.sigterm_143_is_normal_with_zero_residuals,
        expected_roles=(
            {
                "go2_sensor_bridge",
                "sensor_ros_sidecar",
                "downstream_recorder",
                "t4_map_companion",
            }
            if environment.get("INTERNNAV_SESSION_PROFILE") == "completion_sim_map"
            else None
        ),
    )
    payload["outer_observed_client_exit_code"] = client_code
    payload["client_exit_mode"] = mode
    payload["original_client_alive_after_probe"] = False
    atomic_write_json(artifact, payload)
    return payload


def _validate_inner_cleanup_result(
    payload: Mapping[str, Any],
    *,
    client_code: int | None,
    mode: str,
    allow_sigterm_143: bool = False,
    expected_roles: set[str] | None = None,
) -> None:
    """Reject fake-zero artifacts or an original docker exec client still alive."""

    if client_code is None:
        raise RuntimeError("original ROS container client is still alive")
    if mode == "normal_exit_probe":
        allowed_normal_codes = {0}
        if allow_sigterm_143:
            allowed_normal_codes.update({143, -signal.SIGTERM})
        if client_code not in allowed_normal_codes:
            raise RuntimeError("normal inner supervisor exit code is not zero")
    elif mode == "recovery_terminated":
        allowed_recovery_codes = {0, 2, 137, 143, -signal.SIGKILL, -signal.SIGTERM}
        if client_code not in allowed_recovery_codes:
            raise RuntimeError(f"recovery inner supervisor exit code is unexpected: {client_code}")
    else:
        raise RuntimeError("unknown inner cleanup client exit mode")
    zero_fields = (
        "bridge_sidecar_pid_count",
        "bridge_sidecar_pgid_count",
        "supervisor_pid_count",
        "supervisor_pgid_count",
        "pid_count",
        "pgid_count",
        "sensor_socket_count",
    )
    if (
        payload.get("status") != "PASS"
        or payload.get("cleanup_confirmed") is not True
        or any(int(payload.get(name, -1)) != 0 for name in zero_fields)
    ):
        raise RuntimeError("container child/supervisor PID/PGID/socket cleanup is not zero")
    supervisor = payload.get("supervisor_cleanup")
    if (
        not isinstance(supervisor, Mapping)
        or supervisor.get("identity_verified") is not True
        or supervisor.get("live_after_kill") != []
    ):
        raise RuntimeError("container-side supervisor identity/zero probe is invalid")
    exits = payload.get("required_role_exits")
    expected = expected_roles or {
        "go2_sensor_bridge",
        "sensor_ros_sidecar",
        "downstream_recorder",
    }
    if (
        not isinstance(exits, list)
        or {item.get("role") for item in exits if isinstance(item, Mapping)} != expected
        or len(exits) != len(expected)
        or any(
            not isinstance(item, Mapping)
            or item.get("clean_exit") is not True
            or int(item.get("exit_code", -1)) != 0
            for item in exits
        )
    ):
        raise RuntimeError("container required ROS roles did not all exit cleanly")


def _validate_isaac_cleanup_artifacts(
    result_dir: Path,
    outer_cleanup: Mapping[str, Any],
    *,
    allow_sigterm_143: bool = False,
) -> None:
    groups = outer_cleanup.get("groups")
    if not isinstance(groups, list):
        raise RuntimeError("outer cleanup lacks per-role exit evidence")
    matches = [
        item
        for item in groups
        if isinstance(item, Mapping) and item.get("role") == "isaac_model_free_workload"
    ]
    allowed_codes = {0}
    signal_codes = {143, -signal.SIGTERM}
    if allow_sigterm_143:
        allowed_codes.update(signal_codes)
    if (
        len(matches) != 1
        or matches[0].get("clean_exit") is not True
        or int(matches[0].get("exit_code", -1)) not in allowed_codes
        or matches[0].get("live_after_kill") != []
    ):
        raise RuntimeError("Isaac required role did not exit cleanly with zero residuals")
    artifact_path = result_dir / "isaac_worker_cleanup.json"
    if not artifact_path.is_file():
        if not (
            allow_sigterm_143
            and int(matches[0]["exit_code"]) in signal_codes
            and outer_cleanup.get("residual_cleanup_confirmed") is True
            and int(outer_cleanup.get("pid_count", -1)) == 0
            and int(outer_cleanup.get("pgid_count", -1)) == 0
            and int(outer_cleanup.get("socket_count", -1)) == 0
        ):
            raise RuntimeError("Isaac worker cleanup artifact is absent")
        return
    artifact = _load_json(artifact_path)
    if (
        artifact.get("status") != "PASS"
        or artifact.get("errors") != []
        or artifact.get("actions_attempted") != ["emitter", "backend", "simulation_app"]
    ):
        raise RuntimeError("Isaac worker cleanup artifact is not PASS with no errors")


def _attempt_cleanup_actions(
    *,
    inner_action: Callable[[], dict[str, Any]] | None,
    outer_action: Callable[[], dict[str, Any]],
    outer_evidence: Callable[[], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Run inner and outer cleanup independently and retain all failures.

    This small seam is deliberately behavior-tested: an inner artifact/probe
    failure must never prevent the outer process registry from closing every
    PGID and socket.
    """

    errors: list[str] = []
    inner_result: dict[str, Any] | None = None
    outer_result: dict[str, Any] | None = None
    if inner_action is not None:
        try:
            inner_result = inner_action()
        except BaseException as exc:
            errors.append(f"inner_cleanup: {type(exc).__name__}: {exc}")
    try:
        outer_result = outer_action()
    except BaseException as exc:
        errors.append(f"outer_cleanup: {type(exc).__name__}: {exc}")
        if outer_evidence is not None:
            try:
                outer_result = outer_evidence()
            except BaseException as evidence_exc:
                errors.append(
                    "outer_cleanup_evidence: "
                    f"{type(evidence_exc).__name__}: {evidence_exc}"
                )
    return inner_result, outer_result, errors


def _inner_cleanup_then_release(
    wait_action: Callable[[], dict[str, Any]],
    liveness: OuterAliveLock,
    evidence_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any], BaseException | None]:
    """Wait for the independent supervisor-zero probe before releasing flock."""

    payload: dict[str, Any] | None = None
    wait_error: BaseException | None = None
    try:
        payload = wait_action()
    except BaseException as exc:
        wait_error = exc
    probe_completed = bool(
        payload
        and payload.get("cleanup_confirmed") is True
        and int(payload.get("supervisor_pid_count", -1)) == 0
        and int(payload.get("supervisor_pgid_count", -1)) == 0
    )
    release = liveness.release(
        evidence_path,
        inner_probe_completed=probe_completed,
        reason=(
            "inner_cleanup_probe_completed"
            if probe_completed
            else "inner_cleanup_failed"
        ),
    )
    return payload, release, wait_error


def _is_final_session_pass(
    *,
    succeeded: bool,
    validation: Mapping[str, Any] | None,
    inner_cleanup: Mapping[str, Any] | None,
    outer_cleanup: Mapping[str, Any] | None,
    isaac_cleanup_confirmed: bool,
    outer_liveness_release: Mapping[str, Any] | None,
) -> bool:
    return bool(
        succeeded
        and validation
        and validation.get("status") == "PASS"
        and inner_cleanup
        and inner_cleanup.get("cleanup_confirmed") is True
        and outer_cleanup
        and outer_cleanup.get("cleanup_confirmed") is True
        and isaac_cleanup_confirmed
        and outer_liveness_release
        and outer_liveness_release.get("status") == "RELEASED"
        and outer_liveness_release.get("inner_supervisor_zero_probe_completed") is True
    )


def _zero_cleanup(result_dir: Path, socket_path: Path | None) -> dict[str, Any]:
    socket_count = 0
    path_exists = False
    if socket_path is not None:
        try:
            socket_count = socket_listener_count(socket_path)
            path_exists = socket_path.exists()
        except BaseException:
            socket_count = 1
    payload = {
        "schema_version": 2,
        "status": "PASS" if socket_count == 0 and not path_exists else "FAIL",
        "cleanup_confirmed": socket_count == 0 and not path_exists,
        "pid_count": 0,
        "pgid_count": 0,
        "socket_count": socket_count,
        "groups": [],
        "sockets": [] if socket_path is None else [{"path": str(socket_path), "entries_after_unlink": socket_count, "path_exists_after": path_exists}],
        "no_process_registry_was_created": True,
    }
    atomic_write_json(result_dir / "cleanup.json", payload)
    return payload


def _validate_completion_sim_result(
    result_dir: Path,
    snapshot: Mapping[str, Any],
    deviations: list[dict[str, Any]],
    *,
    profile_name: str = "completion_sim",
    map_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    downstream = snapshot["downstream"]
    bridge = snapshot["bridge"]
    sidecar = snapshot["sidecar"]
    group_counts = downstream.get("consumer_group_counts", {})
    errors: list[str] = []
    if int(snapshot.get("sensor_count", 0)) < 1:
        errors.append("no real simulated sensor frame reached the sidecar")
    if int(bridge.get("frame_count", 0)) < 1:
        errors.append("no derived LiDAR/local-safety bridge frame was published")
    for group in ("nav2", "internvla"):
        if int(group_counts.get(group, 0)) < 1:
            errors.append(f"consumer group {group} never became usable")
    if sidecar.get("writer_thread_alive") is not True:
        errors.append("sidecar raw evidence writer was not alive at snapshot")
    if profile_name == "completion_sim_map" and not map_runtime:
        errors.append("map companion runtime validation is absent")
    recorded_deviations = list(deviations)
    backend_runtime = snapshot["producer"].get("backend_runtime", {})
    if int(backend_runtime.get("render_resync_event_count", 0)):
        recorded_deviations.append(
            {
                "kind": "camera_render_resync",
                "severity": "WARN",
                "count": int(backend_runtime["render_resync_event_count"]),
                "dropped_capture_count": int(
                    backend_runtime.get("render_resync_drop_count", 0)
                ),
                "extra_render_only_drain_count": int(
                    backend_runtime.get("render_resync_extra_drain_total", 0)
                ),
                "action": "bounded_render_only_drain_or_drop_then_continue",
            }
        )
    if int(bridge.get("partial_batches_dropped", 0)):
        recorded_deviations.append(
            {
                "kind": "bridge_partial_batches_dropped",
                "count": int(bridge["partial_batches_dropped"]),
                "action": "warn_and_continue",
            }
        )
    if int(downstream.get("warning_count", 0)):
        recorded_deviations.append(
            {
                "kind": "downstream_shadow_warnings",
                "count": int(downstream["warning_count"]),
                "action": "warn_and_continue",
            }
        )
    payload = {
        "schema_version": 3,
        "status": "PASS" if not errors else "FAIL",
        "profile": profile_name,
        "runtime_policy": "completion_sim",
        "functional_checks": {
            "real_sim_sensor_frames": int(snapshot.get("sensor_count", 0)),
            "bridge_frames": int(bridge.get("frame_count", 0)),
            "consumer_group_counts": group_counts,
            "required_groups": ["internvla", "nav2"],
            "map_runtime": None if map_runtime is None else dict(map_runtime),
        },
        "deviations": recorded_deviations,
        "strict_extensions_pending": [
            "exact_atomic_downstream_evidence",
            "strict_freshness",
            "active_nvblox",
            "sensor_odometry",
            "600_sec_stability_soak",
        ],
        "errors": errors,
    }
    atomic_write_json(result_dir / "session_validation.json", payload)
    atomic_write_json(
        result_dir / "completion_sim_deviations.json",
        {
            "schema_version": 1,
            "status": "RECORDED",
            "runtime_policy": "completion_sim",
            "items": recorded_deviations,
            "strict_extensions_pending": payload["strict_extensions_pending"],
        },
    )
    return payload


def _validate_completion_sim_map_runtime(
    result_dir: Path,
) -> dict[str, Any]:
    payload = _load_json(result_dir / "map/runtime_validation.json")
    plan = _load_json(result_dir / "runtime/map_bundle/launch_plan.json")
    required_topics = {
        "/map",
        "/go2/lidar/points_base",
        "/local_costmap/costmap",
        "/global_costmap/costmap",
        "/cmd_vel_safe",
        "/internvla/stop",
    }
    topic_checks = payload.get("topic_checks")
    if (
        payload.get("status") != "PASS"
        or payload.get("runtime_policy") != "completion_sim"
        or payload.get("profile") != "completion_sim_map"
        or payload.get("target") != "isaac_simulation_only"
        or float(payload.get("duration_sec", -1.0)) < 60.0
        or float(payload.get("requested_duration_sec", -1.0)) != 60.0
        or payload.get("effective_nvblox_mode") != "shadow"
        or payload.get("strict_evidence_modified") is not False
        or payload.get("continues_until_shared_stop") is not True
        or not isinstance(topic_checks, Mapping)
        or set(topic_checks) != required_topics
        or any(value is not True for value in topic_checks.values())
        or payload.get("core_nav_sha256") != plan.get("core_nav_sha256")
    ):
        raise RuntimeError("completion_sim_map runtime validation is invalid")
    return payload


def _validate_completion_sim_map_cleanup(
    result_dir: Path,
) -> dict[str, Any]:
    cleanup = _load_json(result_dir / "map/companion_cleanup.json")
    validation = _load_json(result_dir / "map/smoke_validation.json")
    runtime = _load_json(result_dir / "map/runtime_validation.json")
    if (
        cleanup.get("status") != "PASS"
        or cleanup.get("child_roles_alive") != []
        or cleanup.get("descendant_pids_alive") != []
        or int(cleanup.get("owned_socket_count", -1)) != 0
        or validation.get("status") != "PASS"
        or validation.get("runtime_validation_written") is not True
        or validation.get("residual_roles") != []
        or validation.get("residual_group_pids") != []
        or validation.get("core_nav_sha256") != runtime.get("core_nav_sha256")
    ):
        raise RuntimeError("completion_sim_map cleanup validation is invalid")
    return {"runtime": runtime, "cleanup": cleanup, "validation": validation}


def run(profile_name: str, result_value: Path, grant_id: str) -> int:
    if os.name != "posix":
        raise RuntimeError("online sensor sessions require a POSIX Isaac host")
    if os.environ.get("INTERNNAV_SENSOR_SESSION_LEASE_ACK") != "1":
        raise RuntimeError("sensor session must run inside scripts/with_resource_lease.sh")
    session_started_ns = time.monotonic_ns()
    control_root = Path(os.environ.get("INTERNNAV_T1_CONTROL_ROOT", Path(__file__).resolve().parents[1])).resolve()
    grant_ref_sha, grant = _assert_authoritative_grant(control_root, profile_name, result_value, grant_id)
    result_dir = _resolve_fresh_result(control_root, result_value, profile_name)
    claim_ns = time.monotonic_ns()
    # The ref must remain unchanged across the exclusive directory claim; the
    # grant is consumed by this unique path/grant-id pair.
    if _git_ref_sha(control_root) != grant_ref_sha:
        atomic_write_json(result_dir / "session_completion.json", {"schema_version": 2, "status": "FAIL", "failure": "grant ref changed during result claim"})
        _zero_cleanup(result_dir, None)
        return 2

    profile = PROFILES[profile_name]
    runtime_policy = policy_for_session_profile(profile_name)
    ready_deadline = session_started_ns / 1e9 + profile.ready_deadline_sec
    logs = result_dir / "logs"
    runtime = result_dir / "runtime"
    registry: ProcessRegistry | None = None
    socket_path: Path | None = None
    log_streams: list[Any] = []
    validation: dict[str, Any] | None = None
    map_runtime_validation: dict[str, Any] | None = None
    map_cleanup_validation: dict[str, Any] | None = None
    inner_cleanup: dict[str, Any] | None = None
    outer_cleanup: dict[str, Any] | None = None
    outer_liveness: OuterAliveLock | None = None
    isaac_cleanup_confirmed = False
    outer_liveness_release: dict[str, Any] | None = None
    failure = ""
    deviations: list[dict[str, Any]] = []
    succeeded = False
    interrupted: int | None = None
    environment = os.environ.copy()
    environment["INTERNNAV_RUNTIME_POLICY"] = runtime_policy.name
    environment["INTERNNAV_SESSION_PROFILE"] = profile_name
    map_environment_keys = {
        "INTERNNAV_T4_MAP_COMPANION_MODULE",
        "INTERNNAV_T4_MAP_CONFIG_DIR",
        "INTERNNAV_T4_MAP_NVBLOX_MODE",
        "INTERNNAV_T4_MAP_NVBLOX_HEALTH",
        "INTERNNAV_SIMULATION_TARGET",
    }
    if profile_name == "completion_sim_map":
        environment.update(
            {
                "INTERNNAV_T4_MAP_COMPANION_MODULE": "t4_completion.map.companion",
                "INTERNNAV_T4_MAP_CONFIG_DIR": str(
                    control_root / "configs/completion_sim/map"
                ),
                "INTERNNAV_T4_MAP_NVBLOX_MODE": "shadow",
                "INTERNNAV_T4_MAP_NVBLOX_HEALTH": "unknown",
                "INTERNNAV_SIMULATION_TARGET": "isaac",
            }
        )
    else:
        for key in map_environment_keys:
            environment.pop(key, None)

    def interrupted_now() -> bool:
        return interrupted is not None

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = signum
        if registry is not None:
            registry.record_failure(stage="signal", reason=f"received_signal_{signum}")

    for handled in tuple(
        dict.fromkeys((signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)))
    ):
        signal.signal(handled, on_signal)

    try:
        provenance = pre_spawn_source_gate(
            control_root,
            grant_ref_sha,
            include_map=profile_name == "completion_sim_map",
        )
        _reject_calibration_overrides(environment)
        isaac_eula_policy = apply_frozen_isaac_eula_environment(environment)
        logs.mkdir()
        runtime.mkdir()
        socket_tag = hashlib.sha256(str(result_dir).encode("utf-8")).hexdigest()[:16]
        socket_path = control_root / "runtime/sensor_producer" / f"01r-{socket_tag}.sock"
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            raise FileExistsError(f"refusing an existing sensor socket: {socket_path}")
        atomic_write_json(result_dir / "producer_contract.json", {**contract_payload(), "contract_sha256": CONTRACT_SHA256})
        atomic_write_json(
            result_dir / "diagnostic_geometry.json",
            {
                "schema_version": 2,
                "status": "FROZEN",
                "objects": [dict(item) for item in DIAGNOSTIC_GEOMETRY],
                "light": dict(DIAGNOSTIC_LIGHT),
                "created_by_backend_every_session": True,
                "model_free": True,
            },
        )
        liveness_owner = OuterAliveLock(result_dir / "outer_alive.lock")
        liveness_evidence = liveness_owner.acquire()
        outer_liveness = liveness_owner
        atomic_write_json(
            result_dir / "session_manifest.json",
            {
                "schema_version": 2,
                "status": "STARTING",
                "profile": profile_name,
                "runtime_policy": runtime_policy.as_dict(),
                "contract_sha256": CONTRACT_SHA256,
                "result_dir_fresh_claimed": True,
                "grant": {**grant, "ref_sha": grant_ref_sha, "consumed": True},
                "source_provenance": provenance,
                "outer_liveness": liveness_evidence,
                "session_started_monotonic_ns": session_started_ns,
                "result_claimed_monotonic_ns": claim_ns,
                "navigation_evaluator_used": False,
                "model_loaded": False,
                "thresholds_from_argv_or_env": False,
                "isaac_eula_policy": isaac_eula_policy,
                "isaac_runtime_preflight_verified": False,
                "isaac_startup_ready_verified": False,
            },
        )
        if _git_ref_sha(control_root) != grant_ref_sha:
            raise RuntimeError("coordination ref changed immediately before process registry creation")
        outer_liveness.check()
        allowed_required_exit_codes = {0}
        if runtime_policy.sigterm_143_is_normal_with_zero_residuals:
            allowed_required_exit_codes.update({143, -signal.SIGTERM})
        registry = ProcessRegistry(
            result_dir,
            socket_paths=[socket_path],
            term_timeout_sec=(
                10.0
                if runtime_policy.sigterm_143_is_normal_with_zero_residuals
                else 2.0
            ),
            allowed_required_exit_codes=allowed_required_exit_codes,
        )
        environment.update(
            {
                "CONTROL_ROOT": str(control_root),
                "RESULT_DIR": str(result_dir),
                "SENSOR_SOCKET": str(socket_path),
                "PYTHONPATH": str(control_root),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        isaac_python = environment.get("INTERNNAV_ISAAC_PYTHON", str(Path.home() / "env_isaacsim/bin/python"))
        source_usd = Path(environment.get("INTERNVLA_GO2_SOURCE_USD", str(Path.home() / "isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd")))

        build_log = (logs / "asset_builder.log").open("xb")
        log_streams.append(build_log)
        registry.start(
            "asset_builder",
            _managed_command("asset_builder", [
                isaac_python,
                str(control_root / "scripts/build_t4_r3_go2_usd.py"),
                "--source",
                str(source_usd),
                "--output",
                str(runtime / "go2_model_free.usda"),
                "--manifest",
                str(result_dir / "go2_asset_manifest.json"),
            ]),
            stdout=build_log,
            stderr=subprocess.STDOUT,
            env=environment,
            required=False,
            expected_long_running=False,
        )
        remaining = remaining_ready_budget(ready_deadline, "asset preparation")
        registry.wait_preparation("asset_builder", remaining)
        validate_asset_manifest(
            result_dir / "go2_asset_manifest.json", runtime / "go2_model_free.usda"
        )

        ros_log = (logs / "ros_container_client.log").open("xb")
        log_streams.append(ros_log)
        registry.start(
            "ros_container_client",
            _managed_command(
                "ros_container_client",
                ["bash", str(control_root / "sensor_runtime/run_ros_container.sh")]
            ),
            stdout=ros_log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        isaac_log = (logs / "isaac_worker.log").open("xb")
        log_streams.append(isaac_log)
        isaac_started_ns = time.monotonic_ns()
        registry.start(
            "isaac_model_free_workload",
            _managed_command(
                "isaac_model_free_workload",
                [
                    isaac_python,
                    "-m",
                    "sensor_runtime.isaac_worker",
                    "--profile",
                    profile_name,
                    "--socket",
                    str(socket_path),
                    "--wrapper-usd",
                    str(runtime / "go2_model_free.usda"),
                    "--result-dir",
                    str(result_dir),
                ]
            ),
            stdout=isaac_log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        registry.start_monitor()
        _wait_file(result_dir / "inner_ready.json", ready_deadline, registry, "outer_pre_ready_inner", interrupted_now)
        inner_identity = _load_json(result_dir / "inner_supervisor_identity.json")
        inner_outer_probe = inner_identity.get("outer_liveness")
        if (
            not isinstance(inner_outer_probe, Mapping)
            or inner_outer_probe.get("status") != "OUTER_ALIVE"
            or inner_outer_probe.get("mechanism")
            != "flock_exclusive_nonblocking"
            or int(inner_outer_probe.get("device", -1))
            != int(liveness_evidence["device"])
            or int(inner_outer_probe.get("inode", -1))
            != int(liveness_evidence["inode"])
        ):
            raise RuntimeError(
                "container did not prove shared bind-mounted outer-owner flock"
            )
        raw_child_pythonpath = inner_identity.get("child_pythonpath")
        if not isinstance(raw_child_pythonpath, list) or not all(
            isinstance(item, str) for item in raw_child_pythonpath
        ):
            raise RuntimeError("inner supervisor did not record a child PYTHONPATH")
        child_pythonpath = validate_child_pythonpath(
            control_root, os.pathsep.join(raw_child_pythonpath)
        )
        setup_pythonpath_sha256 = require_frozen_setup_sha256(
            inner_identity.get("setup_pythonpath_sha256")
        )
        if (
            inner_identity.get("python_no_user_site") is not True
            or inner_identity.get("python_dont_write_bytecode") is not True
        ):
            raise RuntimeError("inner Python isolation flags are absent")
        session_manifest = _load_json(result_dir / "session_manifest.json")
        session_manifest["child_pythonpath"] = child_pythonpath
        session_manifest["child_pythonpath_verified"] = True
        session_manifest["setup_pythonpath_sha256"] = setup_pythonpath_sha256
        session_manifest["setup_pythonpath_verified"] = True
        session_manifest["child_python_isolation"] = {
            "no_user_site": True,
            "dont_write_bytecode": True,
        }
        session_manifest["bind_mount_flock_verified"] = True
        session_manifest["inner_outer_liveness_probe"] = dict(inner_outer_probe)
        atomic_write_json(result_dir / "session_manifest.json", session_manifest)

        _wait_file(
            result_dir / ISAAC_RUNTIME_PREFLIGHT_FILENAME,
            ready_deadline,
            registry,
            "outer_pre_ready_isaac_preflight",
            interrupted_now,
        )
        isaac_runtime_preflight = require_frozen_runtime_preflight(
            _load_json(result_dir / ISAAC_RUNTIME_PREFLIGHT_FILENAME)
        )
        _wait_file(
            result_dir / ISAAC_STARTUP_READY_FILENAME,
            ready_deadline,
            registry,
            "outer_pre_ready_isaac_startup",
            interrupted_now,
        )
        isaac_startup_ready = require_frozen_startup_ready(
            _load_json(result_dir / ISAAC_STARTUP_READY_FILENAME)
        )
        session_manifest = _load_json(result_dir / "session_manifest.json")
        if session_manifest.get("isaac_eula_policy") != dict(FROZEN_ISAAC_EULA_POLICY):
            raise RuntimeError("session manifest Isaac EULA policy drifted before worker start")
        session_manifest["isaac_runtime_preflight"] = isaac_runtime_preflight
        session_manifest["isaac_runtime_preflight_verified"] = True
        session_manifest["isaac_startup_ready"] = isaac_startup_ready
        session_manifest["isaac_startup_ready_verified"] = True
        atomic_write_json(result_dir / "session_manifest.json", session_manifest)
        while True:
            registry.check("outer_pre_ready_capture")
            sensor_count = _line_count(result_dir / "sensor_producer_audit.jsonl")
            controller_count = _line_count(result_dir / "controller_stop_audit.jsonl")
            bridge_count = _line_count(result_dir / "bridge/go2_sensor_bridge_frames.jsonl")
            downstream_count = _line_count(result_dir / "downstream/downstream_frames.jsonl")
            if sensor_count >= 3 and controller_count >= 3 and bridge_count >= 1 and downstream_count >= 1:
                break
            if interrupted_now():
                raise InterruptedError("session received an external signal")
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("asset/container/Isaac/output chain missed session-to-ready deadline")
            time.sleep(0.02)
        ready_ns = time.monotonic_ns()
        atomic_write_json(
            result_dir / "session_ready.json",
            {
                "schema_version": 2,
                "status": "READY",
                "profile": profile_name,
                "session_ready_elapsed_sec": (ready_ns - session_started_ns) / 1e9,
                "claim_ready_elapsed_sec": (ready_ns - claim_ns) / 1e9,
                "isaac_process_ready_elapsed_sec": (ready_ns - isaac_started_ns) / 1e9,
                "sensor_record_count": sensor_count,
                "controller_record_count": controller_count,
                "bridge_record_count": bridge_count,
                "downstream_record_count": downstream_count,
                "required_roles": [record.role for record in registry.records if record.required],
                "inner_role_ledger": "inner_lifecycle/process_ledger.json",
            },
        )

        last_growth_count, last_growth_wall = downstream_count, time.monotonic()
        producer_marker = result_dir / "producer_completion.json"
        completion_deadline = isaac_started_ns / 1e9 + profile.duration_sec + 60.0
        while not producer_marker.is_file():
            registry.check("outer_bounded_workload")
            if interrupted_now():
                raise InterruptedError("session received an external signal")
            observed = _line_count(result_dir / "downstream/downstream_frames.jsonl")
            if observed > last_growth_count:
                last_growth_count, last_growth_wall = observed, time.monotonic()
            elif time.monotonic() - last_growth_wall >= runtime_policy.downstream_growth_timeout_sec:
                if runtime_policy.name == "completion_sim":
                    if not any(item.get("code") == "DOWNSTREAM_GROWTH_WARN" for item in deviations):
                        deviations.append(
                            {
                                "code": "DOWNSTREAM_GROWTH_WARN",
                                "severity": "WARN",
                                "message": (
                                    "consumer shadow stopped growing for "
                                    f"{runtime_policy.downstream_growth_timeout_sec:.1f} seconds"
                                ),
                            }
                        )
                    last_growth_wall = time.monotonic()
                else:
                    raise RuntimeError(
                        "actual downstream audit stopped growing for 0.35 seconds"
                    )
            if time.monotonic() >= completion_deadline:
                raise TimeoutError("independent monotonic workload did not complete")
            time.sleep(0.02)

        registry.check("outer_snapshot_freeze")
        atomic_write_json(result_dir / "snapshot_freeze.request", {"schema_version": 2, "requested_monotonic_ns": time.monotonic_ns()})
        snapshot_deadline = time.monotonic() + 5.0
        _wait_file(result_dir / "producer_snapshot_ack.json", snapshot_deadline, registry, "producer_snapshot_ack", interrupted_now)
        producer_ack = _load_json(result_dir / "producer_snapshot_ack.json")
        snapshot_id = hashlib.sha256(f"{grant_id}:{time.monotonic_ns()}".encode()).hexdigest()
        atomic_write_json(
            result_dir / "snapshot_request.json",
            {"schema_version": 2, "snapshot_id": snapshot_id, "generation": int(producer_ack["generation"]), "sequence": int(producer_ack["sequence"])},
        )
        for path in (
            result_dir / "sidecar_snapshot_ack.json",
            result_dir / "bridge_snapshot_ack.json",
            result_dir / "downstream_snapshot_ack.json",
        ):
            _wait_file(path, snapshot_deadline, registry, path.stem, interrupted_now)
        snapshot = _validate_snapshot_acks(result_dir, snapshot_id, runtime_policy)
        side_counts = snapshot["sidecar"]["writer_counts"]
        ready_payload = _load_json(result_dir / "session_ready.json")
        atomic_write_json(
            result_dir / "workload_completion.json",
            {
                "schema_version": 2,
                "status": "BOUNDED_DURATION_REACHED",
                "profile": profile_name,
                "runtime_policy": runtime_policy.name,
                "snapshot_id": snapshot_id,
                "generation": snapshot["identity"][0],
                "sequence": snapshot["identity"][1],
                "sensor_record_count": int(side_counts["sensor"]),
                "controller_record_count": int(side_counts["controller"]),
                "reset_record_count": int(side_counts["reset"]),
                "bridge_record_count": int(snapshot["bridge"]["frame_count"]),
                "downstream_record_count": int(snapshot["downstream"]["frame_count"]),
                "required_roles_alive": True,
                "producer_elapsed_sec": _load_json(producer_marker).get("elapsed_sec"),
                "ready_post_growth_count": int(snapshot["downstream"]["frame_count"]) - int(ready_payload["downstream_record_count"]),
                "frozen_prefix_reconciled": runtime_policy.exact_atomic_evidence_required,
            },
        )
        registry.check("outer_validator_pre")
        if profile_name == "completion_sim_map":
            _wait_file(
                result_dir / "map/runtime_validation.json",
                time.monotonic() + COMPLETION_MAP_POST_PRODUCER_WAIT_SEC,
                registry,
                "map_runtime_validation",
                interrupted_now,
            )
            map_runtime_validation = _validate_completion_sim_map_runtime(result_dir)
        validation = (
            _validate_completion_sim_result(
                result_dir,
                snapshot,
                deviations,
                profile_name=profile_name,
                map_runtime=map_runtime_validation,
            )
            if runtime_policy.name == "completion_sim"
            else validate_result(result_dir, profile_name)
        )
        registry.check("outer_validator_post")
        if validation["status"] != "PASS":
            raise RuntimeError("offline-frozen online validator rejected the session")
        registry.stop_boundary_snapshot(result_dir / "outer_stop_boundary.json", "outer_stop_boundary")
        succeeded = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if registry is not None:
            registry.record_failure(stage="outer_session", reason=failure)
    finally:
        if registry is not None:
            cleanup_errors: list[str] = []
            try:
                registry.begin_cleanup()
            except BaseException as cleanup_exc:
                cleanup_errors.append(
                    f"begin_cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            inner_action: Callable[[], dict[str, Any]] | None = None
            if (result_dir / "inner_lifecycle").exists() and socket_path is not None:
                def inner_action() -> dict[str, Any]:
                    nonlocal outer_liveness_release
                    if outer_liveness is None:
                        raise RuntimeError("outer liveness owner is absent")
                    payload, outer_liveness_release, wait_error = (
                        _inner_cleanup_then_release(
                            lambda: _wait_inner_cleanup(
                                registry,
                                control_root=control_root,
                                result_dir=result_dir,
                                socket_path=socket_path,
                                logs=logs,
                                environment=environment,
                                runtime_policy=runtime_policy,
                            ),
                            outer_liveness,
                            result_dir / "outer_liveness_release.json",
                        )
                    )
                    if wait_error is not None:
                        raise wait_error
                    if payload is None:
                        raise RuntimeError("inner cleanup returned no payload")
                    if any(
                        int(payload.get(key, 1)) != 0
                        for key in (
                            "bridge_sidecar_pid_count",
                            "bridge_sidecar_pgid_count",
                            "supervisor_pid_count",
                            "supervisor_pgid_count",
                            "pid_count",
                            "pgid_count",
                            "sensor_socket_count",
                        )
                    ):
                        raise RuntimeError("inner cleanup artifact is not PID/PGID/socket zero")
                    return payload
            elif succeeded:
                def inner_action() -> dict[str, Any]:
                    nonlocal outer_liveness_release
                    try:
                        raise RuntimeError("inner lifecycle was never created")
                    finally:
                        if outer_liveness is not None:
                            outer_liveness_release = outer_liveness.release(
                                result_dir / "outer_liveness_release.json",
                                inner_probe_completed=False,
                                reason="inner_lifecycle_missing",
                            )
            elif outer_liveness is not None:
                def inner_action() -> dict[str, Any]:
                    nonlocal outer_liveness_release
                    outer_liveness_release = outer_liveness.release(
                        result_dir / "outer_liveness_release.json",
                        inner_probe_completed=False,
                        reason="session_failed_before_inner_start",
                    )
                    return {"status": "NOT_STARTED", "cleanup_confirmed": False}

            inner_cleanup, outer_cleanup, action_errors = _attempt_cleanup_actions(
                inner_action=inner_action,
                outer_action=registry.cleanup,
                outer_evidence=lambda: _load_json(registry.cleanup_path),
            )
            cleanup_errors.extend(action_errors)
            if succeeded and outer_cleanup is not None:
                try:
                    _validate_isaac_cleanup_artifacts(
                        result_dir,
                        outer_cleanup,
                        allow_sigterm_143=(
                            runtime_policy.sigterm_143_is_normal_with_zero_residuals
                        ),
                    )
                    isaac_cleanup_confirmed = True
                except BaseException as cleanup_exc:
                    cleanup_errors.append(
                        f"isaac_cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
            if cleanup_errors:
                succeeded = False
                combined_cleanup_error = "; ".join(cleanup_errors)
                failure = failure or f"cleanup: {combined_cleanup_error}"
                registry.record_failure(stage="cleanup", reason=combined_cleanup_error)
        else:
            if outer_liveness is not None:
                try:
                    outer_liveness_release = outer_liveness.release(
                        result_dir / "outer_liveness_release.json",
                        inner_probe_completed=False,
                        reason="session_failed_before_registry",
                    )
                except BaseException as cleanup_exc:
                    failure = failure or (
                        "cleanup: outer liveness release: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
            try:
                outer_cleanup = _zero_cleanup(result_dir, socket_path)
            except BaseException as cleanup_exc:
                failure = failure or f"cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}"
        close_errors: list[str] = []
        for stream in log_streams:
            try:
                stream.close()
            except BaseException as exc:
                close_errors.append(f"{getattr(stream, 'name', 'log')}: {type(exc).__name__}: {exc}")
        if close_errors:
            succeeded = False
            failure = failure or "log close: " + "; ".join(close_errors)

    if profile_name == "completion_sim_map" and succeeded:
        try:
            map_cleanup_validation = _validate_completion_sim_map_cleanup(result_dir)
        except BaseException as exc:
            succeeded = False
            failure = failure or f"map cleanup: {type(exc).__name__}: {exc}"

    final_pass = _is_final_session_pass(
        succeeded=succeeded,
        validation=validation,
        inner_cleanup=inner_cleanup,
        outer_cleanup=outer_cleanup,
        isaac_cleanup_confirmed=isaac_cleanup_confirmed,
        outer_liveness_release=outer_liveness_release,
    ) and (profile_name != "completion_sim_map" or map_cleanup_validation is not None)
    atomic_write_json(
        result_dir / "session_completion.json",
        {
            "schema_version": 2,
            "status": "PASS" if final_pass else "FAIL",
            "profile": profile_name,
            "runtime_policy": runtime_policy.name,
            "validation_status": None if validation is None else validation.get("status"),
            "inner_cleanup_confirmed": bool(inner_cleanup and inner_cleanup.get("cleanup_confirmed") is True),
            "outer_cleanup_confirmed": bool(outer_cleanup and outer_cleanup.get("cleanup_confirmed") is True),
            "isaac_cleanup_confirmed": isaac_cleanup_confirmed,
            "outer_liveness_release": outer_liveness_release,
            "map_runtime_validation": map_runtime_validation,
            "map_cleanup_validation": map_cleanup_validation,
            "pid_count": None if outer_cleanup is None else outer_cleanup.get("pid_count"),
            "pgid_count": None if outer_cleanup is None else outer_cleanup.get("pgid_count"),
            "socket_count": None if outer_cleanup is None else outer_cleanup.get("socket_count"),
            "failure": failure or None,
        },
    )
    return 0 if final_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--grant-id", required=True)
    args = parser.parse_args()
    return run(args.profile, args.result_dir, args.grant_id)


if __name__ == "__main__":
    raise SystemExit(main())
