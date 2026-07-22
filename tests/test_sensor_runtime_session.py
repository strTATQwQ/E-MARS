from __future__ import annotations

import hashlib
import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sensor_runtime import session
from sensor_runtime import ros_inner_cleanup
from sensor_runtime.ros_inner_cleanup import (
    expected_roles_for_profile,
    validate_supervisor_identity,
)
from sensor_runtime.runtime_policy import POLICIES


def _grant(payload: dict) -> str:
    return (
        "ordinary table text may say GRANTED or NOT GRANTED\n"
        "<!-- INTERNAV_ONLINE_GRANT_V1\n"
        + json.dumps(payload, sort_keys=True)
        + "\nINTERNAV_ONLINE_GRANT_V1 -->\n"
    )


def _payload(result: Path, **updates: object) -> dict:
    value = {
        "schema_version": 1,
        "status": "GRANTED",
        "worker": "01R",
        "resource": "isaac",
        "profile": "bootstrap",
        "result_dir": str(result),
        "grant_id": "grant-123",
    }
    value.update(updates)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_completion_snapshot_accepts_independent_consumer_identities(
    tmp_path: Path,
) -> None:
    snapshot_id = "snapshot-independent-consumers"
    producer = {
        "status": "PASS",
        "generation": 0,
        "sequence": 10,
        "capture_count": 11,
        "emitter": {
            "accepted_count": 11,
            "sent_count": 11,
            "overwrite_count": 0,
            "reset_clear_count": 0,
            "barrier_drop_count": 0,
            "last_submitted": [0, 10],
            "last_sent": [0, 10],
            "thread_alive": True,
            "fault": None,
        },
        "backend_runtime": {
            "render_resync_event_count": 2,
            "render_resync_drop_count": 1,
            "render_resync_extra_drain_total": 4,
        },
    }
    sidecar = {
        "status": "PASS",
        "snapshot_id": snapshot_id,
        "generation": 0,
        "sequence": 10,
        "server_received_count": 11,
        "receive_overwrite_count": 0,
        "receive_reset_clear_count": 0,
        "receive_barrier_drop_count": 0,
        "writer_counts": {"sensor": 11, "controller": 11, "reset": 1},
        "writer_thread_alive": True,
        "writer_fault": None,
    }
    bridge = {
        "status": "PASS",
        "runtime_policy": "completion_sim",
        "snapshot_id": snapshot_id,
        "generation": 0,
        "sequence": 8,
        "frame_count": 8,
        "pending_count": 0,
        "partial_batches_dropped": 2,
        "writer_thread_alive": True,
        "writer_fault": None,
    }
    downstream = {
        "status": "PASS",
        "runtime_policy": "completion_sim",
        "recorder_mode": "nonfatal_consumer_shadow",
        "snapshot_id": snapshot_id,
        "generation": 0,
        "sequence": 9,
        "frame_count": 20,
        "consumer_group_counts": {"nav2": 10, "internvla": 10},
        "required_functional_groups": ["internvla", "nav2"],
    }
    for name, payload in (
        ("producer_snapshot_ack.json", producer),
        ("sidecar_snapshot_ack.json", sidecar),
        ("bridge_snapshot_ack.json", bridge),
        ("downstream_snapshot_ack.json", downstream),
    ):
        _write_json(tmp_path / name, payload)

    result = session._validate_snapshot_acks(
        tmp_path, snapshot_id, POLICIES["completion_sim"]
    )
    assert result["identity"] == (0, 10)
    assert result["consumer_identities"] == {
        "bridge": (0, 8),
        "downstream": (0, 9),
    }
    completion = session._validate_completion_sim_result(tmp_path, result, [])
    resync = next(
        item
        for item in completion["deviations"]
        if item.get("kind") == "camera_render_resync"
    )
    assert resync == {
        "kind": "camera_render_resync",
        "severity": "WARN",
        "count": 2,
        "dropped_capture_count": 1,
        "extra_render_only_drain_count": 4,
        "action": "bounded_render_only_drain_or_drop_then_continue",
    }

    with pytest.raises(RuntimeError, match="identity differs from producer"):
        session._validate_snapshot_acks(
            tmp_path, snapshot_id, POLICIES["strict_evidence"]
        )


def test_exact_grant_parser_ignores_unstructured_granted_text(tmp_path: Path) -> None:
    root = tmp_path
    result = root / "results/parallel/sensor_producer/fresh"
    assert session.parse_grant_document(
        _grant(_payload(result)),
        control_root=root,
        profile="bootstrap",
        result_value=result,
        grant_id="grant-123",
    )["status"] == "GRANTED"


def test_completion_map_grant_uses_worker_10_and_exclusive_result_root(
    tmp_path: Path,
) -> None:
    result = tmp_path / "results/parallel/t4_map/fresh"
    payload = _payload(
        result,
        worker="10",
        profile="completion_sim_map",
    )
    assert session.parse_grant_document(
        _grant(payload),
        control_root=tmp_path,
        profile="completion_sim_map",
        result_value=result,
        grant_id="grant-123",
    )["worker"] == "10"
    with pytest.raises(RuntimeError, match="worker/resource mismatch"):
        session.parse_grant_document(
            _grant({**payload, "worker": "01R"}),
            control_root=tmp_path,
            profile="completion_sim_map",
            result_value=result,
            grant_id="grant-123",
        )
    with pytest.raises(RuntimeError, match="outside the 10 exclusive path"):
        session.parse_grant_document(
            _grant(
                {
                    **payload,
                    "result_dir": str(
                        tmp_path / "results/parallel/sensor_producer/wrong"
                    ),
                }
            ),
            control_root=tmp_path,
            profile="completion_sim_map",
            result_value=result,
            grant_id="grant-123",
        )


def test_completion_map_post_producer_wait_preserves_full_runtime_gate() -> None:
    assert session.COMPLETION_MAP_POST_PRODUCER_WAIT_SEC == 20.0
    source = inspect.getsource(session.run)
    assert "time.monotonic() + COMPLETION_MAP_POST_PRODUCER_WAIT_SEC" in source
    assert "_validate_completion_sim_map_runtime" in source


@pytest.mark.parametrize(
    "updates,profile,result_suffix,grant_id",
    [
        ({"status": "NOT GRANTED"}, "bootstrap", "fresh", "grant-123"),
        ({"status": "NO_GRANT"}, "bootstrap", "fresh", "grant-123"),
        ({"profile": "soak"}, "bootstrap", "fresh", "grant-123"),
        ({"profile": "bootstrap"}, "soak", "fresh", "grant-123"),
        ({"resource": "dgx"}, "bootstrap", "fresh", "grant-123"),
        ({"worker": "01"}, "bootstrap", "fresh", "grant-123"),
        ({"grant_id": "other"}, "bootstrap", "fresh", "grant-123"),
        ({"result_dir": "results/parallel/sensor_producer/other"}, "bootstrap", "fresh", "grant-123"),
    ],
)
def test_grant_mismatch_is_rejected(
    tmp_path: Path,
    updates: dict,
    profile: str,
    result_suffix: str,
    grant_id: str,
) -> None:
    result = tmp_path / f"results/parallel/sensor_producer/{result_suffix}"
    with pytest.raises(RuntimeError):
        session.parse_grant_document(
            _grant(_payload(result, **updates)),
            control_root=tmp_path,
            profile=profile,
            result_value=result,
            grant_id=grant_id,
        )


def test_duplicate_and_malformed_grant_blocks_are_rejected(tmp_path: Path) -> None:
    result = tmp_path / "results/parallel/sensor_producer/fresh"
    block = _grant(_payload(result))
    with pytest.raises(RuntimeError, match="exactly one"):
        session.parse_grant_document(block + block, control_root=tmp_path, profile="bootstrap", result_value=result, grant_id="grant-123")
    malformed = "<!-- INTERNAV_ONLINE_GRANT_V1\n{bad json}\nINTERNAV_ONLINE_GRANT_V1 -->"
    with pytest.raises(RuntimeError, match="malformed"):
        session.parse_grant_document(malformed, control_root=tmp_path, profile="bootstrap", result_value=result, grant_id="grant-123")
    with pytest.raises(RuntimeError, match="exactly one"):
        session.parse_grant_document("| 01R | GRANTED |", control_root=tmp_path, profile="bootstrap", result_value=result, grant_id="grant-123")


def test_fresh_result_directory_is_single_use(tmp_path: Path) -> None:
    allowed = tmp_path / "results/parallel/sensor_producer"
    allowed.mkdir(parents=True)
    result = session._resolve_fresh_result(tmp_path, Path("results/parallel/sensor_producer/attempt"))
    assert result.is_dir()
    with pytest.raises(FileExistsError):
        session._resolve_fresh_result(tmp_path, result)


def test_ready_budget_includes_delayed_asset_and_inner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session.time, "monotonic", lambda: 21.0)
    with pytest.raises(TimeoutError, match="asset preparation"):
        session.remaining_ready_budget(20.0, "asset preparation")

    class Registry:
        def check(self, stage: str) -> None:
            assert stage == "delayed_inner"

    with pytest.raises(TimeoutError, match="delayed_inner"):
        session._wait_file(tmp_path / "missing", 20.0, Registry(), "delayed_inner", lambda: False)  # type: ignore[arg-type]


def _manifest(wrapper: Path) -> dict:
    return {
        "semantic_camera": {
            "prim_path": "base/internvla_camera", "resolution": [640, 480],
            "translation_from_base_m": [0.2, 0.0, 0.2], "height_above_support_m": 0.62,
            "pitch_down_deg": 20.0, "hfov_deg": 69.4, "vfov_deg": 42.5,
        },
        "depth_camera": {
            "prim_path": "base/t4_d435i_depth", "resolution": [640, 480],
            "translation_from_base_m": [0.2, 0.0, 0.2], "height_above_support_m": 0.62,
            "pitch_down_deg": 20.0, "hfov_deg": 87.0, "vfov_deg": 58.0,
            "minimum_depth_m": 0.28,
        },
        "go2_front_rgb": {
            "prim_path": "base/go2_front_rgb", "resolution": [320, 240],
            "ros_publish_resolution": [160, 120], "translation_from_base_m": [0.29, 0.0, -0.06],
            "pitch_down_deg": 8.0, "hfov_deg": 120.0, "vfov_deg": 75.0,
            "clipping_range_m": [0.2, 1_000_000.0],
            "is_depth_source": False,
        },
        "go2_4d_lidar": {
            "prim_path": "base/go2_l1_lidar", "translation_from_base_m": [0.25, 0.0, 0.18],
            "azimuth_samples": 180, "elevation_channels": 8, "range_m": [0.1, 12.0],
        },
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    }


def test_manifest_calibration_and_wrapper_content_are_verified(tmp_path: Path) -> None:
    wrapper = tmp_path / "go2.usda"
    wrapper.write_bytes(b"frozen wrapper")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(wrapper)), encoding="utf-8")
    session.validate_asset_manifest(path, wrapper)
    wrapper.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="does not match"):
        session.validate_asset_manifest(path, wrapper)


def test_calibration_environment_overrides_are_rejected() -> None:
    for key in (
        "INTERNVLA_T4_CAMERA_HFOV_DEG",
        "INTERNVLA_T4_DEPTH_VFOV_DEG",
        "INTERNVLA_T4_STEREO_BASELINE_M",
        "INTERNVLA_T4_R3_FRONT_RGB_PITCH_DEG",
    ):
        with pytest.raises(RuntimeError, match="override"):
            session._reject_calibration_overrides({key: "999"})


def test_ros_and_isaac_are_started_before_monitor_and_inner_ready_wait() -> None:
    source = Path(session.__file__).read_text(encoding="utf-8")
    asset_wait = source.index('registry.wait_preparation("asset_builder"')
    ros_start = source.index('"ros_container_client",', asset_wait)
    isaac_start = source.index('"isaac_model_free_workload",', ros_start)
    monitor = source.index("registry.start_monitor()", isaac_start)
    inner_wait = source.index(
        '_wait_file(result_dir / "inner_ready.json"', monitor
    )
    assert asset_wait < ros_start < isaac_start < monitor < inner_wait


def _source_fixture(root: Path) -> str:
    files = {
        "sensor_runtime/runtime.py": "VALUE = 1\n",
        "go2_sensor_bridge/go2_sensor_bridge/__init__.py": "\n",
        "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py": "BRIDGE = True\n",
        "scripts/build_t4_r3_go2_usd.py": "from build_t4_camera import VALUE\n",
        "scripts/build_t4_camera.py": "from build_t4_leaf import VALUE\n",
        "scripts/build_t4_leaf.py": "VALUE = 1\n",
        "t4_completion/map/runtime.py": "MAP = True\n",
        "configs/completion_sim/map/profile.yaml": "runtime_policy: completion_sim\n",
    }
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "offline@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Offline Fixture"], cwd=root, check=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_source_provenance_rejects_mutation_extra_executable_and_startup_shadow(tmp_path: Path) -> None:
    sha = _source_fixture(tmp_path)
    evidence = session.verify_source_provenance(tmp_path, sha)
    assert evidence["verified"] is True
    assert "scripts/build_t4_leaf.py" in evidence["builder_dependency_closure"]

    critical = tmp_path / "sensor_runtime/runtime.py"
    critical.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="blob differs"):
        session.verify_source_provenance(tmp_path, sha)
    critical.write_text("VALUE = 1\n", encoding="utf-8")

    extra = tmp_path / "sensor_runtime/extra.py"
    extra.write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source set differs"):
        session.verify_source_provenance(tmp_path, sha)
    extra.unlink()

    script_shadow = tmp_path / "scripts/json.py"
    script_shadow.write_bytes(b"raise RuntimeError\n")
    with pytest.raises(RuntimeError, match="source set differs"):
        session.verify_source_provenance(tmp_path, sha)
    script_shadow.unlink()

    cache = tmp_path / "sensor_runtime/__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-312.pyc").write_bytes(b"shadow")
    with pytest.raises(RuntimeError, match="execution shadow"):
        session.verify_source_provenance(tmp_path, sha)
    (cache / "runtime.cpython-312.pyc").unlink()
    cache.rmdir()

    extension = tmp_path / "sensor_runtime/runtime.pyd"
    extension.write_bytes(b"shadow")
    with pytest.raises(RuntimeError, match="execution shadow"):
        session.verify_source_provenance(tmp_path, sha)
    extension.unlink()

    root_shadow = tmp_path / "json.py"
    root_shadow.write_bytes(b"raise RuntimeError\n")
    with pytest.raises(RuntimeError, match="root Python import shadow"):
        session.verify_source_provenance(tmp_path, sha)
    root_shadow.unlink()

    shadow = tmp_path / "sitecustomize.py"
    shadow.write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="startup shadow"):
        session.verify_source_provenance(tmp_path, sha)


def test_source_provenance_includes_completion_map_runtime(tmp_path: Path) -> None:
    sha = _source_fixture(tmp_path)
    evidence = session.verify_source_provenance(tmp_path, sha, include_map=True)
    assert "t4_completion/map/runtime.py" in evidence["critical_paths"]
    assert "configs/completion_sim/map/profile.yaml" in evidence["critical_paths"]
    (tmp_path / "configs/completion_sim/map/profile.yaml").write_text(
        "runtime_policy: strict_evidence\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="blob differs"):
        session.verify_source_provenance(tmp_path, sha, include_map=True)


def test_pre_spawn_gate_rejects_ref_change_before_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = "a" * 40
    monkeypatch.setattr(
        session,
        "verify_source_provenance",
        lambda _root, ref: {"verified": True, "ref_sha": ref},
    )
    monkeypatch.setattr(session, "_git_ref_sha", lambda _root: "b" * 40)
    with pytest.raises(RuntimeError, match="changed after source"):
        session.pre_spawn_source_gate(tmp_path, expected)


def _inner_cleanup_payload(**updates: object) -> dict:
    value = {
        "status": "PASS",
        "cleanup_confirmed": True,
        "bridge_sidecar_pid_count": 0,
        "bridge_sidecar_pgid_count": 0,
        "supervisor_pid_count": 0,
        "supervisor_pgid_count": 0,
        "pid_count": 0,
        "pgid_count": 0,
        "sensor_socket_count": 0,
        "supervisor_cleanup": {"identity_verified": True, "live_after_kill": []},
        "required_role_exits": [
            {"role": role, "exit_code": 0, "clean_exit": True}
            for role in (
                "go2_sensor_bridge",
                "sensor_ros_sidecar",
                "downstream_recorder",
            )
        ],
    }
    value.update(updates)
    return value


def test_inner_cleanup_rejects_fake_pass_with_live_client_or_supervisor() -> None:
    session._validate_inner_cleanup_result(
        _inner_cleanup_payload(), client_code=0, mode="normal_exit_probe"
    )
    with pytest.raises(RuntimeError, match="still alive"):
        session._validate_inner_cleanup_result(
            _inner_cleanup_payload(), client_code=None, mode="normal_exit_probe"
        )
    with pytest.raises(RuntimeError, match="not zero"):
        session._validate_inner_cleanup_result(
            _inner_cleanup_payload(supervisor_pid_count=1),
            client_code=0,
            mode="normal_exit_probe",
        )
    with pytest.raises(RuntimeError, match="identity/zero"):
        session._validate_inner_cleanup_result(
            _inner_cleanup_payload(
                supervisor_cleanup={"identity_verified": True, "live_after_kill": [{"pid": 9}]}
            ),
            client_code=0,
            mode="normal_exit_probe",
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonzero"])
def test_inner_cleanup_rejects_invalid_required_role_exits(mutation: str) -> None:
    payload = _inner_cleanup_payload()
    exits = list(payload["required_role_exits"])
    if mutation == "missing":
        exits.pop()
    elif mutation == "duplicate":
        exits[-1] = dict(exits[0])
    else:
        exits[0] = {**exits[0], "exit_code": 2, "clean_exit": False}
    payload["required_role_exits"] = exits
    with pytest.raises(RuntimeError, match="did not all exit cleanly"):
        session._validate_inner_cleanup_result(
            payload, client_code=0, mode="normal_exit_probe"
        )


def test_inner_cleanup_accepts_managed_map_companion_only_when_expected() -> None:
    payload = _inner_cleanup_payload()
    payload["required_role_exits"].append(
        {"role": "t4_map_companion", "exit_code": 0, "clean_exit": True}
    )
    expected = {
        "go2_sensor_bridge",
        "sensor_ros_sidecar",
        "downstream_recorder",
        "t4_map_companion",
    }
    session._validate_inner_cleanup_result(
        payload,
        client_code=0,
        mode="normal_exit_probe",
        expected_roles=expected,
    )
    with pytest.raises(RuntimeError, match="did not all exit cleanly"):
        session._validate_inner_cleanup_result(
            payload, client_code=0, mode="normal_exit_probe"
        )


def test_completion_map_runtime_and_cleanup_validation(tmp_path: Path) -> None:
    result = tmp_path
    (result / "map").mkdir()
    (result / "runtime/map_bundle").mkdir(parents=True)
    core_sha = "a" * 64
    topics = {
        "/map": True,
        "/go2/lidar/points_base": True,
        "/local_costmap/costmap": True,
        "/global_costmap/costmap": True,
        "/cmd_vel_safe": True,
        "/internvla/stop": True,
    }
    runtime = {
        "status": "PASS",
        "runtime_policy": "completion_sim",
        "profile": "completion_sim_map",
        "target": "isaac_simulation_only",
        "duration_sec": 60.1,
        "requested_duration_sec": 60.0,
        "topic_checks": topics,
        "effective_nvblox_mode": "shadow",
        "core_nav_sha256": core_sha,
        "strict_evidence_modified": False,
        "continues_until_shared_stop": True,
    }
    _write_json(result / "map/runtime_validation.json", runtime)
    _write_json(
        result / "runtime/map_bundle/launch_plan.json",
        {"core_nav_sha256": core_sha},
    )
    assert session._validate_completion_sim_map_runtime(result) == runtime
    _write_json(
        result / "map/companion_cleanup.json",
        {
            "status": "PASS",
            "child_roles_alive": [],
            "descendant_pids_alive": [],
            "owned_socket_count": 0,
        },
    )
    _write_json(
        result / "map/smoke_validation.json",
        {
            "status": "PASS",
            "runtime_validation_written": True,
            "residual_roles": [],
            "residual_group_pids": [],
            "core_nav_sha256": core_sha,
        },
    )
    assert session._validate_completion_sim_map_cleanup(result)["runtime"] == runtime


def test_cleanup_actions_always_run_outer_after_inner_failure() -> None:
    called: list[str] = []

    def inner() -> dict:
        called.append("inner")
        raise RuntimeError("broken inner artifact")

    def outer() -> dict:
        called.append("outer")
        return {
            "status": "PASS",
            "cleanup_confirmed": True,
            "pid_count": 0,
            "pgid_count": 0,
            "socket_count": 0,
        }

    inner_result, outer_result, errors = session._attempt_cleanup_actions(
        inner_action=inner, outer_action=outer
    )
    assert called == ["inner", "outer"]
    assert inner_result is None
    assert outer_result is not None
    assert outer_result["pid_count"] == outer_result["pgid_count"] == 0
    assert outer_result["socket_count"] == 0
    assert errors and errors[0].startswith("inner_cleanup:")


class _FakeLiveness:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def release(
        self, _path: Path, *, inner_probe_completed: bool, reason: str
    ) -> dict:
        self.events.append("release")
        return {
            "status": "RELEASED",
            "inner_supervisor_zero_probe_completed": inner_probe_completed,
            "reason": reason,
        }


def test_inner_wait_precedes_release_and_proof_requires_supervisor_zero(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def wait() -> dict:
        events.append("wait_return")
        return {
            "cleanup_confirmed": True,
            "supervisor_pid_count": 0,
            "supervisor_pgid_count": 0,
        }

    payload, release, error = session._inner_cleanup_then_release(
        wait, _FakeLiveness(events), tmp_path / "release.json"  # type: ignore[arg-type]
    )
    assert events == ["wait_return", "release"]
    assert payload is not None and error is None
    assert release["inner_supervisor_zero_probe_completed"] is True


def test_inner_wait_failure_releases_false_and_outer_cleanup_still_once(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    outer_calls = 0

    def inner() -> dict:
        payload, release, error = session._inner_cleanup_then_release(
            lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
            _FakeLiveness(events),  # type: ignore[arg-type]
            tmp_path / "release.json",
        )
        assert payload is None
        assert release["inner_supervisor_zero_probe_completed"] is False
        assert error is not None
        raise error

    def outer() -> dict:
        nonlocal outer_calls
        outer_calls += 1
        return {"cleanup_confirmed": True, "pid_count": 0, "pgid_count": 0}

    _, outer_result, errors = session._attempt_cleanup_actions(
        inner_action=inner, outer_action=outer
    )
    assert events == ["release"]
    assert outer_calls == 1 and outer_result is not None
    assert errors and "probe failed" in errors[0]


def test_final_pass_requires_completed_outer_liveness_release() -> None:
    base = {
        "succeeded": True,
        "validation": {"status": "PASS"},
        "inner_cleanup": {"cleanup_confirmed": True},
        "outer_cleanup": {"cleanup_confirmed": True},
        "isaac_cleanup_confirmed": True,
    }
    assert session._is_final_session_pass(
        **base,
        outer_liveness_release={
            "status": "RELEASED",
            "inner_supervisor_zero_probe_completed": True,
        },
    )
    assert not session._is_final_session_pass(
        **base, outer_liveness_release=None
    )
    assert not session._is_final_session_pass(
        **base,
        outer_liveness_release={
            "status": "RELEASED",
            "inner_supervisor_zero_probe_completed": False,
        },
    )


def test_all_outer_launches_use_managed_process_group_wrapper() -> None:
    tree = ast.parse(inspect.getsource(session))
    observed: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        role = node.args[0].value
        if role not in session.MANAGED_OUTER_ROLES:
            continue
        command = node.args[1]
        assert isinstance(command, ast.Call)
        assert isinstance(command.func, ast.Name)
        assert command.func.id == "_managed_command"
        assert isinstance(command.args[0], ast.Constant)
        assert command.args[0].value == role
        observed.add(role)
    assert observed == set(session.MANAGED_OUTER_ROLES)


def test_isaac_cleanup_requires_pass_artifact_and_zero_exit(tmp_path: Path) -> None:
    artifact = {
        "status": "PASS",
        "errors": [],
        "actions_attempted": ["emitter", "backend", "simulation_app"],
    }
    (tmp_path / "isaac_worker_cleanup.json").write_text(json.dumps(artifact))
    group = {
        "role": "isaac_model_free_workload",
        "exit_code": 0,
        "clean_exit": True,
        "live_after_kill": [],
    }
    session._validate_isaac_cleanup_artifacts(tmp_path, {"groups": [group]})
    (tmp_path / "isaac_worker_cleanup.json").write_text(
        json.dumps({**artifact, "status": "FAIL", "errors": ["backend close"]})
    )
    with pytest.raises(RuntimeError, match="artifact"):
        session._validate_isaac_cleanup_artifacts(tmp_path, {"groups": [group]})
    (tmp_path / "isaac_worker_cleanup.json").write_text(json.dumps(artifact))
    with pytest.raises(RuntimeError, match="did not exit cleanly"):
        session._validate_isaac_cleanup_artifacts(
            tmp_path,
            {"groups": [{**group, "exit_code": 2, "clean_exit": False}]},
        )


def test_completion_accepts_sigterm_cleanup_only_with_measured_zero_residuals(
    tmp_path: Path,
) -> None:
    group = {
        "role": "isaac_model_free_workload",
        "exit_code": 143,
        "clean_exit": True,
        "live_after_kill": [],
    }
    cleanup = {
        "groups": [group],
        "residual_cleanup_confirmed": True,
        "pid_count": 0,
        "pgid_count": 0,
        "socket_count": 0,
    }
    session._validate_isaac_cleanup_artifacts(
        tmp_path, cleanup, allow_sigterm_143=True
    )
    with pytest.raises(RuntimeError, match="did not exit cleanly"):
        session._validate_isaac_cleanup_artifacts(tmp_path, cleanup)
    with pytest.raises(RuntimeError, match="artifact"):
        session._validate_isaac_cleanup_artifacts(
            tmp_path,
            {**cleanup, "pid_count": 1},
            allow_sigterm_143=True,
        )


def test_inner_recovery_attempts_supervisor_when_child_cleanup_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = tmp_path / "result"
    lifecycle = result / "inner_lifecycle"
    lifecycle.mkdir(parents=True)
    (result / "inner_supervisor_identity.json").write_text(
        json.dumps(
            {
                "pid": 123,
                "pgid": 123,
                "linux_start_ticks": 456,
                "isolated_process_group": True,
            }
        )
    )
    called: list[str] = []

    def broken_child(*_args: object, **_kwargs: object) -> dict:
        called.append("child")
        raise OSError("broken ledger artifact")

    def clean_supervisor(**_kwargs: object) -> dict:
        called.append("supervisor")
        return {
            "identity_verified": True,
            "identity_error": None,
            "live_before": [],
            "live_after_term": [],
            "kill_sent": False,
            "live_after_kill": [],
        }

    monkeypatch.setattr(ros_inner_cleanup, "cleanup_persisted_ledger", broken_child)
    monkeypatch.setattr(
        ros_inner_cleanup, "terminate_bound_process_group", clean_supervisor
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ros_inner_cleanup",
            "--result-dir",
            str(result),
            "--socket",
            str(tmp_path / "sensor.sock"),
        ],
    )
    monkeypatch.setenv("INTERNNAV_SESSION_PROFILE", "bootstrap")
    assert ros_inner_cleanup.main() == 2
    assert called == ["child", "supervisor"]
    payload = json.loads((result / "inner_cleanup.json").read_text())
    assert payload["status"] == "FAIL"
    assert payload["supervisor_pid_count"] == payload["supervisor_pgid_count"] == 0


def test_inner_cleanup_expected_roles_are_exactly_profile_bound() -> None:
    base = {"go2_sensor_bridge", "sensor_ros_sidecar", "downstream_recorder"}
    assert expected_roles_for_profile("bootstrap") == base
    assert expected_roles_for_profile("completion_sim") == base
    assert expected_roles_for_profile("completion_sim_map") == {
        *base,
        "t4_map_companion",
    }
    with pytest.raises(RuntimeError, match="frozen session profile"):
        expected_roles_for_profile("")
    with pytest.raises(RuntimeError, match="frozen session profile"):
        expected_roles_for_profile("completion_sim_map_typo")


def test_supervisor_identity_requires_isolated_pid_equal_pgid() -> None:
    assert validate_supervisor_identity(
        {"pid": 10, "pgid": 10, "linux_start_ticks": 20, "isolated_process_group": True}
    ) == (10, 10, 20)
    for identity in (
        {"pid": 10, "pgid": 9, "linux_start_ticks": 20, "isolated_process_group": True},
        {"pid": 10, "pgid": 10, "linux_start_ticks": 20, "isolated_process_group": False},
    ):
        with pytest.raises(RuntimeError):
            validate_supervisor_identity(identity)
