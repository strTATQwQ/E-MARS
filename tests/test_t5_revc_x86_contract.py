from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
import types
import zlib
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from t5_revc_sensor_contract import (  # noqa: E402
    camera_orientation_wxyz,
    camera_translation_base_m,
    load_revc_contract,
    t5_revc_feature_enabled,
)


def _build_runtime(tmp_path: Path, *, revc_enabled: bool) -> tuple[str, dict[str, Any]]:
    output = tmp_path / "runtime.py"
    manifest = tmp_path / "manifest.json"
    env = os.environ.copy()
    env["INTERNVLA_T5_REVC_ENABLE"] = "1" if revc_enabled else "0"
    for name in (
        "INTERNNAV_RUNTIME_POLICY",
        "INTERNNAV_SIMULATION_TARGET",
        "INTERNNAV_T5_LANE",
        "INTERNNAV_T5_ID_PREFIX",
        "INTERNVLA_T4_RESULT_ROOT",
        "INTERNVLA_T5_REVC_CAMERA_CONFIG",
        "INTERNVLA_T4_R3_ENABLE_D435I",
        "INTERNVLA_T4_R3_ENABLE_LIDAR",
        "INTERNVLA_T4_R3_LIDAR_RAY_COUNT",
        "INTERNVLA_T4_R3_ENABLE_RGB_IPC",
    ):
        env.pop(name, None)
    if revc_enabled:
        result_root = tmp_path / "lane-a-result"
        result_root.mkdir()
        env.update(
            {
                "INTERNNAV_RUNTIME_POLICY": "completion_sim",
                "INTERNNAV_SIMULATION_TARGET": "isaac",
                "INTERNNAV_T5_LANE": "a",
                "INTERNNAV_T5_ID_PREFIX": "a::",
                "INTERNVLA_T4_RESULT_ROOT": str(result_root.resolve()),
            }
        )
    else:
        # A disabled build must not even attempt to load T5-only contracts.
        env["INTERNVLA_T5_REVC_CAMERA_CONFIG"] = str(
            tmp_path / "must-not-be-loaded.json"
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_t4_r3_sensor_runtime_overlay.py"),
            "--source",
            str(SCRIPTS / "internnav_go2_runtime.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    generated = output.read_text(encoding="utf-8")
    ast.parse(generated)
    return generated, json.loads(manifest.read_text(encoding="utf-8"))


def _build_direct_runtime(tmp_path: Path) -> tuple[str, dict[str, Any]]:
    output = tmp_path / "direct_runtime.py"
    manifest = tmp_path / "direct_manifest.json"
    result_root = tmp_path / "lane-b-result"
    result_root.mkdir()
    env = {
        **os.environ,
        "INTERNVLA_T5_REVC_ENABLE": "1",
        "INTERNNAV_RUNTIME_POLICY": "completion_sim",
        "INTERNNAV_SIMULATION_TARGET": "isaac",
        "INTERNNAV_T5_LANE": "b",
        "INTERNNAV_T5_LANE_NAMESPACE": "/t5/lane_b",
        "INTERNNAV_T5_ID_PREFIX": "b::",
        "INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL": "1",
        "INTERNVLA_T4_RESULT_ROOT": str(result_root.resolve()),
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_t5_lane_b_step3_direct_runtime_overlay.py"),
            "--source",
            str(SCRIPTS / "internnav_go2_runtime.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    generated = output.read_text(encoding="utf-8")
    ast.parse(generated)
    return generated, json.loads(manifest.read_text(encoding="utf-8"))


def _function_source(generated: str, name: str) -> str:
    tree = ast.parse(generated)
    node = next(
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == name
    )
    source = ast.get_source_segment(generated, node)
    assert source is not None
    return source


def _runtime_probe_class(
    generated: str, names: tuple[str, ...], namespace: dict[str, Any]
) -> type:
    tree = ast.parse(generated)
    methods = []
    for name in names:
        methods.append(
            next(
                candidate
                for candidate in ast.walk(tree)
                if isinstance(candidate, ast.FunctionDef) and candidate.name == name
            )
        )
    probe = ast.ClassDef(
        name="RuntimeProbe",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=methods,
    )
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    exec(compile(module, "<generated-runtime-probe>", "exec"), namespace)
    return namespace["RuntimeProbe"]


class _Vector:
    def __init__(self, values: Any) -> None:
        self.values = tuple(float(value) for value in values)
        self.shape = (len(self.values),)

    def copy(self) -> "_Vector":
        return _Vector(self.values)

    def tolist(self) -> list[float]:
        return list(self.values)

    def _binary(self, other: Any, operation: Any) -> "_Vector":
        values = other.values if isinstance(other, _Vector) else (other,) * len(self.values)
        return _Vector(operation(left, right) for left, right in zip(self.values, values))

    def __add__(self, other: Any) -> "_Vector":
        return self._binary(other, lambda left, right: left + right)

    def __sub__(self, other: Any) -> "_Vector":
        return self._binary(other, lambda left, right: left - right)

    def __mul__(self, other: Any) -> "_Vector":
        return self._binary(other, lambda left, right: left * right)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "_Vector":
        return self._binary(other, lambda left, right: left / right)

    def __iter__(self):
        return iter(self.values)


class _FiniteResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def all(self) -> bool:
        values = self.value.values if isinstance(self.value, _Vector) else self.value
        return all(math.isfinite(float(item)) for item in values)


class _FakeNumpy:
    float64 = float

    class linalg:
        @staticmethod
        def norm(vector: _Vector) -> float:
            return math.sqrt(sum(value * value for value in vector.values))

    @staticmethod
    def asarray(value: Any, dtype: Any = None) -> _Vector:
        del dtype
        return _Vector(value)

    @staticmethod
    def isfinite(value: Any) -> _FiniteResult:
        return _FiniteResult(value)

    @staticmethod
    def cross(left: _Vector, right: _Vector) -> _Vector:
        lx, ly, lz = left.values
        rx, ry, rz = right.values
        return _Vector((ly * rz - lz * ry, lz * rx - lx * rz, lx * ry - ly * rx))


def test_revc_contract_freezes_geometry_optics_order_and_stereo_separation() -> None:
    contract = load_revc_contract()
    assert contract["camera_order"] == [
        "front_left",
        "front",
        "front_right",
        "rear",
    ]
    assert contract["optics"] == {
        "hfov_deg": 73.0,
        "pitch_down_deg": 10.0,
        "resolution": [640, 480],
    }
    assert contract["frames"]["temporary_T_base_link_F_M"] == {
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation_m": [0.14, 0.0, 0.18],
    }
    assert contract["snapshot"]["request_claim"] == "atomic_rename_then_atomic_ack"
    assert contract["snapshot"]["render_metadata_policy"] == (
        "strict_isaac_reference_time_required_and_pause_stable"
    )
    assert contract["snapshot"]["required_execution_identity"] == [
        "episode_id",
        "reset_generation",
        "sequence_id",
    ]
    expected_base_positions = {
        "front_left": (0.17, 0.051962, 0.20),
        "front": (0.20, 0.0, 0.20),
        "front_right": (0.17, -0.051962, 0.20),
        "rear": (0.08, 0.0, 0.20),
    }
    for camera in contract["cameras"]:
        translation = camera_translation_base_m(contract, camera)
        assert all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(
                translation, expected_base_positions[camera["identity"]], strict=True
            )
        )
        orientation = camera_orientation_wxyz(
            contract["optics"]["pitch_down_deg"], camera["yaw_deg"]
        )
        assert math.isclose(sum(value * value for value in orientation), 1.0)
    stereo = contract["cuvslam_stereo_separation"]
    assert stereo["resolution"] == [320, 240]
    assert stereo["baseline_m"] == 0.12
    assert stereo["must_not_reuse_revc_cameras"] is True
    assert set(stereo["sensor_names"]).isdisjoint(
        camera["sensor_name"] for camera in contract["cameras"]
    )
    usd_builder = (SCRIPTS / "build_t4_r3_go2_usd.py").read_text(encoding="utf-8")
    assert "revc_scope = t5_revc_feature_scope()" in usd_builder
    assert "revc_enabled = revc_scope is not None" in usd_builder
    assert usd_builder.index("if revc_enabled:") < usd_builder.index(
        "revc_contract = load_revc_contract(revc_config_path)"
    )
    assert '"schema_version": 2' in usd_builder
    assert usd_builder.index('payload["schema_version"] = 3') > usd_builder.index(
        "if revc_enabled:"
    )
    assert "translation = camera_translation_base_m(revc_contract, camera)" in usd_builder
    assert "orientation = camera_orientation_wxyz(" in usd_builder
    assert 'f"/go2_description/{camera[\'prim_path\']}"' in usd_builder


def test_t5_sensor_extensions_require_exact_scope_and_fail_closed(tmp_path: Path) -> None:
    result_root = (tmp_path / "lane-a-result").resolve()
    exact = {
        "INTERNVLA_T5_REVC_ENABLE": "1",
        "INTERNNAV_RUNTIME_POLICY": "completion_sim",
        "INTERNNAV_SIMULATION_TARGET": "isaac",
        "INTERNNAV_T5_LANE": "a",
        "INTERNNAV_T5_ID_PREFIX": "a::",
        "INTERNVLA_T4_RESULT_ROOT": str(result_root),
    }
    assert t5_revc_feature_enabled(exact)
    assert t5_revc_feature_enabled(
        {
            **exact,
            "INTERNNAV_T5_LANE": "b",
            "INTERNNAV_T5_ID_PREFIX": "b::",
        }
    )
    assert not t5_revc_feature_enabled(
        {**exact, "INTERNVLA_T5_REVC_ENABLE": "0"}
    )
    for drift in (
        {"INTERNNAV_RUNTIME_POLICY": "strict_evidence"},
        {"INTERNNAV_SIMULATION_TARGET": "real_go2"},
        {"INTERNNAV_T5_LANE": "c"},
        {"INTERNNAV_T5_ID_PREFIX": "b::"},
        {"INTERNVLA_T4_RESULT_ROOT": "relative/result"},
    ):
        try:
            t5_revc_feature_enabled({**exact, **drift})
        except RuntimeError as error:
            assert "exact T5 completion_sim/isaac" in str(error)
        else:
            raise AssertionError(f"scope drift did not fail closed: {drift}")


def test_runtime_builder_rejects_feature_flag_outside_exact_scope(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "INTERNVLA_T5_REVC_ENABLE": "1",
            "INTERNNAV_RUNTIME_POLICY": "strict_evidence",
            "INTERNNAV_SIMULATION_TARGET": "isaac",
            "INTERNNAV_T5_LANE": "a",
            "INTERNNAV_T5_ID_PREFIX": "a::",
            "INTERNVLA_T4_RESULT_ROOT": str((tmp_path / "lane-a-result").resolve()),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_t4_r3_sensor_runtime_overlay.py"),
            "--source",
            str(SCRIPTS / "internnav_go2_runtime.py"),
            "--output",
            str(tmp_path / "must-not-exist.py"),
            "--manifest",
            str(tmp_path / "must-not-exist.json"),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "requires exact T5 completion_sim/isaac" in completed.stderr
    assert not (tmp_path / "must-not-exist.py").exists()
    assert not (tmp_path / "must-not-exist.json").exists()


def test_generated_runtime_and_bridge_reject_cross_lane_or_outside_root(
    tmp_path: Path, monkeypatch: Any
) -> None:
    generated, manifest = _build_runtime(tmp_path, revc_enabled=True)
    result_root = Path(
        manifest["revc_four_camera"]["frozen_scope"]["result_root"]
    )
    probe_type = _runtime_probe_class(
        generated,
        ("_sample_revc_snapshot",),
        {
            "Any": Any,
            "Path": Path,
            "hashlib": hashlib,
            "json": json,
            "math": math,
            "os": os,
            "time": time,
            "_T5_SIM_CLOCK_NS": 1,
        },
    )
    probe = probe_type()
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "a::")
    monkeypatch.setenv("INTERNVLA_T4_RESULT_ROOT", str(result_root))
    cross_lane = probe._sample_revc_snapshot(
        episode_id="b::foreign",
        reset_generation=0,
        sequence_id=1,
        sim_stamp_ns=1,
    )
    assert cross_lane["revc_snapshot_status"] == "WARN_LANE_SCOPE_MISMATCH"

    outside = tmp_path / "outside" / "revc_snapshot.request.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH", str(outside))
    escaped = probe._sample_revc_snapshot(
        episode_id="a::episode",
        reset_generation=0,
        sequence_id=1,
        sim_stamp_ns=1,
    )
    assert escaped["revc_snapshot_status"] == "WARN_PATH_SCOPE_MISMATCH"
    assert outside.is_file()

    bridge_source = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
    ).read_text(encoding="utf-8")
    bridge_tree = ast.parse(bridge_source)
    bridge_gate = next(
        node
        for node in bridge_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_t5_revc_sensor_extensions_enabled"
    )
    bridge_namespace = {"os": os}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[bridge_gate], type_ignores=[])),
            "<bridge-gate>",
            "exec",
        ),
        bridge_namespace,
    )
    gate = bridge_namespace["_t5_revc_sensor_extensions_enabled"]
    monkeypatch.setenv("INTERNVLA_T5_REVC_ENABLE", "1")
    monkeypatch.setenv("INTERNNAV_RUNTIME_POLICY", "completion_sim")
    monkeypatch.setenv("INTERNNAV_SIMULATION_TARGET", "isaac")
    assert gate() is True
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "b::")
    with pytest.raises(RuntimeError, match="lane/prefix"):
        gate()


def test_feature_disabled_preserves_frozen_r3_output_and_manifest_bytes(
    tmp_path: Path,
) -> None:
    generated, manifest = _build_runtime(tmp_path, revc_enabled=False)
    # T5 camera-source metadata and the navigation-only planar motion profile
    # were integrated after the Rev-C worker branch. Freeze the pre-Rev-C
    # output at the current T5 source while still proving that disabling Rev-C
    # injects no camera/IMU extension. This does not change frozen T4 results.
    pre_revc_integrated_output_sha256 = (
        "c7f26886e1bcc0b5642dc76ae996bdd04f763e36a7c14b9a81673c9423f427d5"
        if os.name == "nt"
        else "c9e3461bee6091a0e7a17b1ec3a7ba797d107bbf364d3fc8171f6e59433844fb"
    )
    assert "_sample_revc_snapshot" not in generated
    assert "_sample_simulated_imu_linear_acceleration" not in generated
    assert "_revc_render_barrier_serial" not in generated
    assert manifest == {
        "schema_version": 2,
        "status": "PASS",
        "source_sha256": hashlib.sha256(
            (SCRIPTS / "internnav_go2_runtime.py").read_bytes()
        ).hexdigest(),
        "output_sha256": pre_revc_integrated_output_sha256,
        "inherited_overlay_sha256": manifest["inherited_overlay_sha256"],
        "changes": [
            "parameterized_depth_stride",
            "dedicated_d435i_depth_sensor",
            "parameterized_depth_extrinsics",
            "parameterized_d435i_depth_frustum",
            "parameterized_runtime_audit",
            "optional_calibrated_stereo_payload",
            "bounded_stereo_sensor_diagnostic_and_rgb_rgba_acceptance",
            "bounded_uint16_depth_ipc",
            "bounded_completion_controller_ipc_timeout",
            "state_only_sensor_fast_path",
            "dgx_onboard_tcp_controller_endpoint",
            "physx_scene_query_go2_4d_lidar",
            "standard_ros_payload_for_go2_front_rgb",
            "standard_ros_payload_for_d435i_rgb",
            "articulation_imu_payload",
            "independent_d435i_lidar_enable_switches",
            "independent_bounded_rgb_ipc_switch",
        ],
        "map_or_obstacle_truth_used_for_lidar": False,
        "frozen_geometry_sources": {
            "d435i": True,
            "lidar": True,
            "lidar_ray_count": 1440,
            "lidar_width": 180,
            "rgb_ipc": True,
        },
        "motion_control_semantics_changed": False,
    }
    # Assert the generated runtime itself, not merely the absence of new
    # manifest keys.
    assert manifest["output_sha256"] == pre_revc_integrated_output_sha256


def test_generated_runtime_enables_four_dedicated_cameras_without_stereo_alias(
    tmp_path: Path,
) -> None:
    generated, manifest = _build_runtime(tmp_path, revc_enabled=True)
    contract = load_revc_contract()
    assert manifest["revc_four_camera"] == {
        "camera_order": contract["camera_order"],
        "contract_id": contract["contract_id"],
        "contract_sha256": hashlib.sha256(
            (ROOT / "configs/internnav_t5/revc_four_camera_snapshot.json").read_bytes()
        ).hexdigest(),
        "cuvslam_stereo_is_separate": True,
        "enabled": True,
        "external_preview_max_hz": 1.0,
        "frozen_scope": {
            "lane": "a",
            "identity_prefix": "a::",
            "result_root": str((tmp_path / "lane-a-result").resolve()),
        },
        "resolution": [640, 480],
        "render_barrier": "replicator_step_pause_timeline_wait_for_render",
        "snapshot_capture": "on_demand_same_render_tick",
    }
    for camera in contract["cameras"]:
        assert f"name={camera['sensor_name']!r}" in generated
        assert f"prim_path={camera['prim_path']!r}" in generated
    assert 'name=f"t4_stereo_{side}"' in generated
    assert "resolution=[320, 240]" in generated
    assert "INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH" in generated
    assert '"same_render_tick": same_render_tick' in generated
    assert '"external_preview_max_hz": preview_hz' in generated
    assert '"revc_snapshot_status": "RATE_LIMITED"' in generated


def test_revc_snapshot_aggregator_has_one_tick_identity_sidecar_and_rate_limit(
    tmp_path: Path,
) -> None:
    generated, _ = _build_runtime(tmp_path, revc_enabled=True)
    sampler = _function_source(generated, "_sample_revc_snapshot")
    assert "rep.orchestrator.step(" in sampler
    assert "pause_timeline=True" in sampler
    assert "wait_for_render=True" in sampler
    assert "_advance_t5_sim_clock" not in sampler
    assert 'before_frame = sensor.get_data()' in sampler
    assert "frame = sensor.get_data()" in sampler
    assert 'rgba.shape != (480, 640, 4)' in sampler
    assert "self._revc_reference_time_metadata(" in sampler
    assert "self._revc_require_pause_stable_render(" in sampler
    assert sampler.index("captured_frames.append") < sampler.index(
        "snapshot_dir.mkdir"
    )
    assert '"episode_id": str(episode_id)' in sampler
    assert '"reset_generation": int(reset_generation)' in sampler
    assert '"sequence_id": int(sequence_id)' in sampler
    assert '"same_render_tick": same_render_tick' in sampler
    assert 'raise RuntimeError("Rev-C cameras are from different render frames")' in sampler
    assert "replicator_orchestrator_step_wait_for_render" not in sampler
    assert '"render_barrier_id": render_barrier_id' in sampler
    assert '"camera_order": expected_order' in sampler
    assert 'snapshot_dir / "snapshot.json"' in sampler
    assert 'hashlib.sha256(png).hexdigest()' in sampler
    assert '"revc_snapshot_status": "RATE_LIMITED"' in sampler
    assert "1.0 / preview_hz" in sampler
    assert "def scoped_path(candidate: Path, label: str) -> Path:" in sampler
    assert "resolved = candidate.expanduser().resolve()" in sampler
    assert "not resolved.is_relative_to(result_root)" in sampler
    assert 'result_root / "revc_snapshots", "snapshot root"' in sampler
    assert '"snapshot image path"' in sampler
    assert '"snapshot sidecar path"' in sampler
    assert 'not request_id.startswith(expected_prefix)' in sampler
    assert 'not str(episode_id).startswith(expected_prefix)' in sampler
    assert "sim_stamp_before_ns != sim_stamp_ns" in sampler
    assert 'snapshot_request.get("schema_version") != 1' in sampler
    assert 'required_identity = {"episode_id", "reset_generation", "sequence_id"}' in sampler
    assert "type(requested_value) is not type(active_value)" in sampler
    assert "requested_value != active_value" in sampler
    assert 'request_path.rename(claim_path)' in sampler
    assert 'temporary_ack.replace(ack_path)' in sampler
    assert 'claim_path.unlink(missing_ok=True)' in sampler
    assert '"revc_snapshot_status": "WARN_MISSING_EXECUTION_IDENTITY"' in sampler
    assert '"revc_snapshot_status": "WARN_CAPTURE_FAILED"' in sampler
    assert (
        '"replicator_annotator:ReferenceTime+SimulationManager"' in sampler
    )
    assert 'frame.get("rendering_frame")' not in sampler

    assert (
        "from internnav.env.utils.internutopia_extension.sensors.vln_camera "
        "import VLNCamera"
    ) in generated
    assert 'rep.AnnotatorRegistry.get_annotator("ReferenceTime")' in generated
    assert 'reference_time.attach(render_product)' in generated
    assert 'annotators["_t5_revc_reference_time"] = reference_time' in generated


def test_lane_b_direct_overlay_captures_current_observation_before_agent_step(
    tmp_path: Path,
) -> None:
    generated, manifest = _build_direct_runtime(tmp_path)
    observation = _function_source(generated, "get_rgb_depth_with_source_metadata")
    controller_update = _function_source(generated, "_update_command")
    assert '_revc_observation_capture_hook.get("callable")' in observation
    assert "decision_sequence = identity.sequence_id + (0 if identity.stop else 1)" in observation
    assert 'result.get("revc_snapshot_status") != "CAPTURED"' in observation
    assert observation.index("original_get_rgb_depth(self)") < observation.index(
        "result = capture("
    )
    assert observation.index("result = capture(") < observation.index(
        "return observation"
    )
    assert "_sample_revc_snapshot(" not in controller_update
    assert manifest["step3_direct_high_level"]["capture_phase"] == (
        "observation_sampling_before_agent_step"
    )
    assert manifest["step3_direct_high_level"]["internvla_loaded"] is False


def test_lane_b_direct_overlay_rejects_non_lane_b_scope(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.py"
    manifest = tmp_path / "must-not-exist.json"
    env = {
        **os.environ,
        "INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL": "1",
        "INTERNNAV_T5_LANE": "a",
        "INTERNNAV_T5_LANE_NAMESPACE": "/t5/lane_a",
        "INTERNNAV_T5_ID_PREFIX": "a::",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_t5_lane_b_step3_direct_runtime_overlay.py"),
            "--source",
            str(SCRIPTS / "internnav_go2_runtime.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "restricted to T5 Lane B" in completed.stderr
    assert not output.exists() and not manifest.exists()


def test_revc_snapshot_fails_without_strict_pause_stable_render_metadata(
    tmp_path: Path, monkeypatch: Any
) -> None:
    generated, manifest = _build_runtime(tmp_path, revc_enabled=True)
    result_root = Path(
        manifest["revc_four_camera"]["frozen_scope"]["result_root"]
    )
    namespace = {
        "Any": Any,
        "Integral": Integral,
        "Real": Real,
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "np": object(),
        "os": os,
        "struct": struct,
        "time": time,
        "zlib": zlib,
        "_T5_SIM_CLOCK_NS": 10_000_000_000,
    }
    probe_type = _runtime_probe_class(
        generated,
        (
            "_revc_render_metadata",
            "_revc_reference_time_metadata",
            "_revc_require_pause_stable_render",
            "_encode_revc_png",
            "_sample_revc_snapshot",
        ),
        namespace,
    )

    class Orchestrator:
        calls = 0

        @classmethod
        def step(cls, **kwargs: Any) -> None:
            assert kwargs == {
                "rt_subframes": 0,
                "pause_timeline": True,
                "wait_for_render": True,
            }
            cls.calls += 1

    omni = types.ModuleType("omni")
    replicator = types.ModuleType("omni.replicator")
    core = types.ModuleType("omni.replicator.core")
    core.orchestrator = Orchestrator
    omni.replicator = replicator
    replicator.core = core
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core)

    class SimulationInterface:
        @staticmethod
        def get_simulation_time_at_time(identity: tuple[int, int]) -> float:
            return identity[0] / identity[1]

    class SimulationManager:
        _simulation_manager_interface = SimulationInterface()

    isaacsim = types.ModuleType("isaacsim")
    isaacsim_core = types.ModuleType("isaacsim.core")
    simulation_manager = types.ModuleType("isaacsim.core.simulation_manager")
    simulation_manager.SimulationManager = SimulationManager
    isaacsim.core = isaacsim_core
    isaacsim_core.simulation_manager = simulation_manager
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", isaacsim_core)
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.simulation_manager", simulation_manager
    )
    monkeypatch.setenv("INTERNNAV_T5_LANE", "a")
    monkeypatch.setenv("INTERNNAV_T5_ID_PREFIX", "a::")
    monkeypatch.setenv("INTERNVLA_T4_RESULT_ROOT", str(result_root))

    strict_identity = {
        "referenceTimeNumerator": 10,
        "referenceTimeDenominator": 10,
    }
    probe_type._revc_require_pause_stable_render(
        (10, 10, 1.0), (10, 10, 1.0), "t5_revc_front"
    )
    with pytest.raises(RuntimeError, match="changed at paused render barrier"):
        probe_type._revc_require_pause_stable_render(
            (10, 10, 1.0), (20, 10, 2.0), "t5_revc_front"
        )
    scenarios = (
        ("missing", [{"rgba": "unused"}], None, 0),
        (
            "advanced",
            [{"rgba": "unused"}, {"rgba": "unused"}],
            [
                strict_identity,
                {
                    "referenceTimeNumerator": 20,
                    "referenceTimeDenominator": 10,
                },
            ],
            1,
        ),
    )
    for index, (label, frames, reference_times, expected_steps) in enumerate(scenarios):
        class ReferenceTime:
            def __init__(self) -> None:
                self.values = list(reference_times or [])

            def get_data(self) -> dict[str, Any]:
                return self.values.pop(0)

        class Sensor:
            def __init__(self) -> None:
                self.frames = list(frames)
                annotators = {}
                if reference_times is not None:
                    annotators["_t5_revc_reference_time"] = ReferenceTime()
                self._camera = types.SimpleNamespace(rp_annotators=annotators)

            def get_data(self) -> dict[str, Any]:
                return self.frames.pop(0)

        probe = probe_type()
        probe.robot = types.SimpleNamespace(
            sensors={
                f"t5_revc_{view}": Sensor()
                for view in ("front_left", "front", "front_right", "rear")
            }
        )
        probe.control_update_index = index
        probe._revc_last_request_id = ""
        probe._revc_last_preview_monotonic = -math.inf
        request = {
            "schema_version": 1,
            "request_id": f"a::{label}",
            "episode_id": "a::episode",
            "reset_generation": 2,
            "sequence_id": 3,
        }
        (result_root / "revc_snapshot.request.json").write_text(
            json.dumps(request), encoding="utf-8"
        )
        before_steps = Orchestrator.calls
        result = probe._sample_revc_snapshot(
            episode_id="a::episode",
            reset_generation=2,
            sequence_id=3,
            sim_stamp_ns=10_000_000_000,
        )
        assert result["revc_snapshot_status"] == "WARN_CAPTURE_FAILED"
        assert Orchestrator.calls - before_steps == expected_steps
        ack = json.loads((result_root / "revc_snapshot.ack.json").read_text())
        assert ack["status"] == "WARN_CAPTURE_FAILED"
        assert not (result_root / "revc_snapshots").exists()

    class MixedSensor:
        def __init__(self, post_numerator: int) -> None:
            self.frames = [
                {"rgba": "unused"},
                {"rgba": "must-not-be-decoded-before-identity-check"},
            ]
            identity = {
                "referenceTimeNumerator": post_numerator,
                "referenceTimeDenominator": 10,
            }
            reference_times = [identity, dict(identity)]
            self._camera = types.SimpleNamespace(
                rp_annotators={
                    "_t5_revc_reference_time": types.SimpleNamespace(
                        get_data=lambda: reference_times.pop(0)
                    )
                }
            )

        def get_data(self) -> dict[str, Any]:
            return self.frames.pop(0)

    probe = probe_type()
    probe.robot = types.SimpleNamespace(
        sensors={
            f"t5_revc_{view}": MixedSensor(21 if view == "rear" else 20)
            for view in ("front_left", "front", "front_right", "rear")
        }
    )
    probe.control_update_index = 3
    probe._revc_last_request_id = ""
    probe._revc_last_preview_monotonic = -math.inf
    (result_root / "revc_snapshot.request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_id": "a::mixed",
                "episode_id": "a::episode",
                "reset_generation": 2,
                "sequence_id": 3,
            }
        ),
        encoding="utf-8",
    )
    result = probe._sample_revc_snapshot(
        episode_id="a::episode",
        reset_generation=2,
        sequence_id=3,
        sim_stamp_ns=10_000_000_000,
    )
    assert result["revc_snapshot_status"] == "WARN_CAPTURE_FAILED"
    assert "different render frames" in result["revc_snapshot_error"]
    assert not (result_root / "revc_snapshots").exists()


def test_simulated_imu_overlay_uses_sim_dt_and_bridge_marks_availability(
    tmp_path: Path,
) -> None:
    generated, manifest = _build_runtime(tmp_path, revc_enabled=True)
    contract = json.loads(
        (ROOT / "configs/internnav_t5/simulated_imu_linear_acceleration.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["derivative_clock"] == "consecutive_normal_sample_sim_stamps"
    assert contract["invalid_sample_behavior"] == (
        "clear_baseline_and_require_two_new_consecutive_normal_samples"
    )
    assert contract["published_sample_stamp_key"] == (
        "go2_imu_linear_acceleration_sample_sim_stamp_ns"
    )
    assert contract["lio_backend_implemented"] is False
    assert manifest["simulated_imu_linear_acceleration"]["purpose"] == (
        "sensor_contract_precondition_only"
    )
    sampler = _function_source(
        generated, "_sample_simulated_imu_linear_acceleration"
    )
    assert "previous = self._imu_previous_normal_sample" in sampler
    assert "if state_only:" in sampler
    assert "previous_stamp_ns, previous_velocity = previous" in sampler
    assert "int(sim_stamp_ns) - int(previous_stamp_ns)" in sampler
    assert "acceleration_world = (current - previous_velocity) / dt" in sampler
    assert "[0.0, 0.0, -9.80665]" in sampler
    assert '"go2_imu_linear_acceleration_available": True' in sampler
    assert '"go2_imu_linear_acceleration_sample_sim_stamp_ns": int(sim_stamp_ns)' in sampler
    assert "specific_force_body.tolist()" in sampler
    assert sampler.index("if state_only:") < sampler.index(
        "previous = self._imu_previous_normal_sample"
    )
    assert sampler.count("self._imu_previous_normal_sample = None") >= 4
    assert sampler.rindex(
        "self._imu_previous_normal_sample = (int(sim_stamp_ns), current.copy())"
    ) > sampler.index("if not np.isfinite(specific_force_body).all()")
    bridge = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
    ).read_text(encoding="utf-8")
    assert 'request.get("go2_imu_linear_acceleration_mps2")' in bridge
    assert '"go2_imu_linear_acceleration_available"' in bridge
    assert '"go2_imu_linear_acceleration_sample_sim_stamp_ns"' in bridge
    assert 'request.get("t5_revc_sim_sample_stamp_ns")' in bridge
    assert "acceleration_stamp_ns == request_sim_stamp_ns" in bridge
    assert "acceleration_stamp_ns == frame_stamp_ns" in bridge
    assert "imu.header.stamp.sec = acceleration_stamp_ns // 1_000_000_000" in bridge
    assert "_t5_revc_sensor_extensions_enabled()" in bridge
    assert "imu.linear_acceleration_covariance[0] = -1.0" in bridge
    assert "no LIO backend is enabled" in bridge


def test_simulated_imu_invalid_sample_breaks_normal_pair(tmp_path: Path) -> None:
    generated, _ = _build_runtime(tmp_path, revc_enabled=True)
    probe_type = _runtime_probe_class(
        generated,
        ("_sample_simulated_imu_linear_acceleration",),
        {"Any": Any, "math": math, "np": _FakeNumpy},
    )
    probe = probe_type()
    probe._imu_previous_normal_sample = None
    pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    first = probe._sample_simulated_imu_linear_acceleration(
        [0.0, 0.0, 0.0], pose, 10_000_000_000
    )
    regressed = probe._sample_simulated_imu_linear_acceleration(
        [100.0, 0.0, 0.0], pose, 9_000_000_000
    )
    after_gap = probe._sample_simulated_imu_linear_acceleration(
        [2.0, 0.0, 0.0], pose, 11_000_000_000
    )
    paired = probe._sample_simulated_imu_linear_acceleration(
        [4.0, 0.0, 0.0], pose, 12_000_000_000
    )
    assert first["go2_imu_linear_acceleration_available"] is False
    assert regressed["go2_imu_linear_acceleration_available"] is False
    assert after_gap["go2_imu_linear_acceleration_available"] is False
    assert paired["go2_imu_linear_acceleration_available"] is True
    assert paired["go2_imu_linear_acceleration_sample_dt_sec"] == 1.0
    assert paired["go2_imu_linear_acceleration_mps2"][0] == 2.0

    invalid_vector = probe._sample_simulated_imu_linear_acceleration(
        None, pose, 12_500_000_000
    )
    after_invalid_vector = probe._sample_simulated_imu_linear_acceleration(
        [5.0, 0.0, 0.0], pose, 13_000_000_000
    )
    assert invalid_vector["go2_imu_linear_acceleration_available"] is False
    assert after_invalid_vector["go2_imu_linear_acceleration_available"] is False

    missing = probe._sample_simulated_imu_linear_acceleration(
        [6.0, 0.0, 0.0], pose, None
    )
    reseed = probe._sample_simulated_imu_linear_acceleration(
        [7.0, 0.0, 0.0], pose, 14_000_000_000
    )
    assert missing["go2_imu_linear_acceleration_available"] is False
    assert reseed["go2_imu_linear_acceleration_available"] is False


def test_x86_lane_resource_mapping_is_even_gpu0_and_odd_gpu1() -> None:
    prepare = (SCRIPTS / "prepare_t5_isaac_workers.sh").read_text(encoding="utf-8")
    run = (SCRIPTS / "run_t5_distributed_isaac.sh").read_text(encoding="utf-8")
    even = "0,2,4,6,8,10,12,14,16"
    odd = "1,3,5,7,9,11,13,15,17"
    assert f'prepare_lane a 0 75 "$lane_a_cpuset"' in prepare
    assert f'prepare_lane b 1 76 "$lane_b_cpuset"' in prepare
    assert even in prepare and odd in prepare
    assert "gpu=0" in run and "gpu=1" in run
    assert f'test "$cpuset" = {even}' in run
    assert f'test "$cpuset" = {odd}' in run
    assert "container_cuda_visible_devices=0" in run
