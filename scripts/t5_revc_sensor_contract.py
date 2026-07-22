#!/usr/bin/env python3
"""Load and validate the completion-sim Rev-C camera sensor contract."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/internnav_t5/revc_four_camera_snapshot.json"
EXPECTED_ORDER = ("front_left", "front", "front_right", "rear")
EXPECTED_F_M_MM = {
    "front_left": ((30.0, 51.962, 20.0), 60.0),
    "front": ((60.0, 0.0, 20.0), 0.0),
    "front_right": ((30.0, -51.962, 20.0), -60.0),
    "rear": ((-60.0, 0.0, 20.0), 180.0),
}


@dataclass(frozen=True)
class T5RevCFeatureScope:
    lane: str
    identity_prefix: str
    result_root: Path


def t5_revc_feature_scope(
    environ: Mapping[str, str] | None = None,
    *,
    fail_on_scope_mismatch: bool = True,
) -> T5RevCFeatureScope | None:
    """Freeze the exact Lane and result root for one enabled T5 build."""

    values = os.environ if environ is None else environ
    raw_flag = values.get("INTERNVLA_T5_REVC_ENABLE", "0")
    if raw_flag not in {"0", "1"}:
        if fail_on_scope_mismatch:
            raise RuntimeError("INTERNVLA_T5_REVC_ENABLE must be exactly 0 or 1")
        return None
    if raw_flag == "0":
        return None

    lane = values.get("INTERNNAV_T5_LANE", "")
    identity_prefix = values.get("INTERNNAV_T5_ID_PREFIX", "")
    result_root_text = values.get("INTERNVLA_T4_RESULT_ROOT", "")
    expected_prefix = f"{lane}::" if lane in {"a", "b"} else ""
    raw_result_root = Path(result_root_text).expanduser() if result_root_text else None
    exact_scope = (
        values.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        and values.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        and lane in {"a", "b"}
        and identity_prefix == expected_prefix
        and raw_result_root is not None
        and raw_result_root.is_absolute()
    )
    if not exact_scope:
        if fail_on_scope_mismatch:
            raise RuntimeError(
                "INTERNVLA_T5_REVC_ENABLE=1 requires exact T5 "
                "completion_sim/isaac/lane/prefix/absolute-result-root scope"
            )
        return None
    assert raw_result_root is not None
    result_root = raw_result_root.resolve()
    if result_root == result_root.parent:
        if fail_on_scope_mismatch:
            raise RuntimeError("T5 Rev-C result root must not be a filesystem root")
        return None
    return T5RevCFeatureScope(
        lane=lane,
        identity_prefix=identity_prefix,
        result_root=result_root,
    )


def t5_revc_feature_enabled(
    environ: Mapping[str, str] | None = None,
    *,
    fail_on_scope_mismatch: bool = True,
) -> bool:
    """Return the fail-closed Rev-C/sim-IMU build-time feature decision.

    The single feature flag deliberately gates both T5-only additions.  This
    keeps an unset/zero flag byte-compatible with the frozen T4 R3 builders and
    prevents either addition from leaking into strict-evidence or real-robot
    configurations.
    """

    return (
        t5_revc_feature_scope(
            environ,
            fail_on_scope_mismatch=fail_on_scope_mismatch,
        )
        is not None
    )


def _finite_vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _close_vector(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-9)
        for a, b in zip(left, right, strict=True)
    )


def load_revc_contract(path: Path | None = None) -> dict[str, Any]:
    """Return a validated contract; fail closed on geometry or identity drift."""

    contract_path = (path or DEFAULT_CONTRACT).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Rev-C camera contract schema")
    if payload.get("scope") != "completion_sim_only":
        raise ValueError("Rev-C concept geometry must remain completion-sim only")
    order = tuple(str(item) for item in payload.get("camera_order", []))
    if order != EXPECTED_ORDER:
        raise ValueError("Rev-C camera order drift")
    optics = payload.get("optics", {})
    if optics.get("resolution") != [640, 480]:
        raise ValueError("Rev-C snapshot resolution must remain 640x480")
    if not math.isclose(float(optics.get("hfov_deg", math.nan)), 73.0):
        raise ValueError("Rev-C HFOV drift")
    if not math.isclose(float(optics.get("pitch_down_deg", math.nan)), 10.0):
        raise ValueError("Rev-C pitch drift")
    transform = payload.get("frames", {}).get("temporary_T_base_link_F_M", {})
    if not _close_vector(
        _finite_vector(transform.get("translation_m"), 3, "temporary translation"),
        (0.14, 0.0, 0.18),
    ):
        raise ValueError("temporary T_base_link_F_M translation drift")
    if not _close_vector(
        _finite_vector(transform.get("rotation_wxyz"), 4, "temporary rotation"),
        (1.0, 0.0, 0.0, 0.0),
    ):
        raise ValueError("temporary T_base_link_F_M must remain identity rotation")
    cameras = payload.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 4:
        raise ValueError("Rev-C contract must contain four cameras")
    identities = tuple(str(item.get("identity", "")) for item in cameras)
    if identities != order:
        raise ValueError("Rev-C camera list does not match fixed order")
    sensor_names: set[str] = set()
    prim_paths: set[str] = set()
    for camera in cameras:
        identity = str(camera["identity"])
        expected_position, expected_yaw = EXPECTED_F_M_MM[identity]
        position_mm = _finite_vector(camera.get("position_F_M_mm"), 3, identity)
        position_m = _finite_vector(camera.get("position_F_M_m"), 3, identity)
        if not _close_vector(position_mm, expected_position):
            raise ValueError(f"Rev-C {identity} position drift")
        if not _close_vector(position_m, tuple(value / 1000.0 for value in position_mm)):
            raise ValueError(f"Rev-C {identity} metre/mm mismatch")
        if not math.isclose(float(camera.get("yaw_deg", math.nan)), expected_yaw):
            raise ValueError(f"Rev-C {identity} yaw drift")
        sensor_name = str(camera.get("sensor_name", ""))
        prim_path = str(camera.get("prim_path", ""))
        if not sensor_name.startswith("t5_revc_") or not prim_path.startswith("base/t5_revc_"):
            raise ValueError("Rev-C camera identities must use dedicated names")
        sensor_names.add(sensor_name)
        prim_paths.add(prim_path)
    if len(sensor_names) != 4 or len(prim_paths) != 4:
        raise ValueError("Rev-C sensor names and prim paths must be unique")
    snapshot = payload.get("snapshot", {})
    if snapshot.get("capture") != "on_demand_same_render_tick":
        raise ValueError("Rev-C capture must remain on-demand and same-render-tick")
    if (
        snapshot.get("render_barrier")
        != "replicator_step_pause_timeline_wait_for_render"
        or snapshot.get("request_claim") != "atomic_rename_then_atomic_ack"
        or snapshot.get("ack_file") != "revc_snapshot.ack.json"
        or snapshot.get("render_metadata_policy")
        != "strict_isaac_reference_time_required_and_pause_stable"
        or snapshot.get("required_render_metadata")
        != [
            "rendering_frame.referenceTimeNumerator",
            "rendering_frame.referenceTimeDenominator",
            "rendering_time",
        ]
        or snapshot.get("required_execution_identity")
        != ["episode_id", "reset_generation", "sequence_id"]
    ):
        raise ValueError("Rev-C snapshot synchronization contract drift")
    preview_hz = float(snapshot.get("external_preview_max_hz", math.nan))
    if not math.isfinite(preview_hz) or not 0.0 < preview_hz <= 1.0:
        raise ValueError("Rev-C external preview must be bounded to at most 1 Hz")
    stereo = payload.get("cuvslam_stereo_separation", {})
    if (
        stereo.get("resolution") != [320, 240]
        or not math.isclose(float(stereo.get("baseline_m", math.nan)), 0.12)
        or stereo.get("must_not_reuse_revc_cameras") is not True
    ):
        raise ValueError("cuVSLAM stereo separation contract drift")
    if sensor_names.intersection(str(item) for item in stereo.get("sensor_names", [])):
        raise ValueError("Rev-C cameras must not alias cuVSLAM stereo sensors")
    return payload


def camera_translation_base_m(
    contract: dict[str, Any], camera: dict[str, Any]
) -> tuple[float, float, float]:
    mast = _finite_vector(
        contract["frames"]["temporary_T_base_link_F_M"]["translation_m"],
        3,
        "temporary translation",
    )
    local = _finite_vector(camera["position_F_M_m"], 3, "camera translation")
    return tuple(a + b for a, b in zip(mast, local, strict=True))


def camera_orientation_wxyz(
    pitch_down_deg: float, yaw_deg: float
) -> tuple[float, float, float, float]:
    """Compose base-Z yaw with the existing Isaac camera optical convention."""

    alpha = math.radians((90.0 - float(pitch_down_deg)) / 2.0)
    scale = math.sqrt(0.5)
    forward = (
        scale * math.cos(alpha),
        scale * math.sin(alpha),
        -scale * math.sin(alpha),
        -scale * math.cos(alpha),
    )
    half_yaw = math.radians(float(yaw_deg)) / 2.0
    yaw = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    aw, ax, ay, az = yaw
    bw, bx, by, bz = forward
    result = (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
    norm = math.sqrt(sum(value * value for value in result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("computed Rev-C camera orientation is not normalized")
    return result


if __name__ == "__main__":
    print(json.dumps(load_revc_contract(), indent=2, sort_keys=True))
