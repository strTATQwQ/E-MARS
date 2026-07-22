#!/usr/bin/env python3
"""Lane-B-only Rev-C snapshot transport layered over the frozen agent client."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from internvla_go2_continuous_agent_client import ContinuousROS2IPCAgentClient
from internvla_ipc_agent_client import MAX_IPC_MESSAGE_BYTES, ROS2IPCAgentClient

MAX_ADVISOR_IMAGE_BYTES = 256 * 1024
MAX_ADVISOR_TOTAL_IMAGE_BYTES = 1024 * 1024
STEP3_SNAPSHOT_WAIT_SEC = 5.0
STEP3_SNAPSHOT_POLL_SEC = 0.02
STEP3_VIEW_ORDER = ("front_left", "front", "front_right", "rear")


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _step3_contract_values(path: Path) -> tuple[str, dict[str, str], dict[str, dict[str, Any]]]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if tuple(value.get("camera_order") or ()) != STEP3_VIEW_ORDER:
        raise RuntimeError("Rev-C Step3 camera order drifted")
    frames = value["frames"]
    mast = frames["temporary_T_base_link_F_M"]
    optics = value["optics"]
    cameras = {str(item["identity"]): dict(item) for item in value["cameras"]}
    if tuple(cameras) != STEP3_VIEW_ORDER:
        raise RuntimeError("Rev-C Step3 camera contract is incomplete")
    extrinsics = {
        view_id: _canonical_sha256(
            {
                "schema_version": 1,
                "base_frame": frames["base_frame"],
                "mast_frame": frames["mast_frame"],
                "temporary_T_base_link_F_M": mast,
                "camera": cameras[view_id],
                "pitch_down_deg": optics["pitch_down_deg"],
            }
        )
        for view_id in STEP3_VIEW_ORDER
    }
    return hashlib.sha256(raw).hexdigest(), extrinsics, cameras

def _encode_step3_jpegs(paths: list[Path]) -> list[bytes]:
    from PIL import Image

    for quality in (90, 85, 80, 75, 70, 65, 60, 55):
        output: list[bytes] = []
        for path in paths:
            with Image.open(path) as image:
                image.load()
                if image.size != (640, 480) or image.mode != "RGB":
                    raise RuntimeError("Rev-C Step3 source is not 640x480 RGB")
                buffer = io.BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    optimize=False,
                    progressive=False,
                    subsampling=2,
                )
                output.append(buffer.getvalue())
        if (
            all(0 < len(value) <= MAX_ADVISOR_IMAGE_BYTES for value in output)
            and sum(len(value) for value in output) <= MAX_ADVISOR_TOTAL_IMAGE_BYTES
        ):
            return output
    raise RuntimeError("Rev-C Step3 JPEG payload cannot satisfy frozen bounds")

class LaneBStep3ROS2IPCAgentClient(ROS2IPCAgentClient):
    """Arm/consume current Lane-B snapshots over the existing TCP step."""

    def __init__(self, config: Any):
        self._step3_live_enabled = False
        super().__init__(config)
        advisor = os.environ.get("INTERNNAV_T5_STEP3_LIVE_ADVISOR", "0") == "1"
        direct = os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL", "0") == "1"
        if advisor == direct:
            raise RuntimeError("exactly one private Step3 agent mode is required")
        self._step3_direct_high_level = direct
        self._step3_live_enabled = True
        self._step3_result_root: Path | None = None
        self._step3_request_path: Path | None = None
        self._step3_ack_path: Path | None = None
        self._step3_contract_sha256 = ""
        self._step3_extrinsics: dict[str, str] = {}
        self._step3_contract_cameras: dict[str, dict[str, Any]] = {}
        self._step3_episode_id = ""
        self._step3_reset_generation = -1
        self._step3_next_sequence = 0
        self._configure_step3_live()
        self._step3_update_identity(self.handshake_response, next_sequence=0)
        self._step3_arm_next_snapshot()

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = str(request.get("operation", ""))
        if self._step3_live_enabled and operation == "step":
            snapshot = self._step3_wait_snapshot()
            if snapshot is not None:
                request["advisor_snapshot"] = snapshot
                encoded = json.dumps(
                    request, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                if len(encoded) > MAX_IPC_MESSAGE_BYTES:
                    request.pop("advisor_snapshot", None)
                    print(
                        "INTERNVLA_STEP3_SNAPSHOT_FALLBACK "
                        + json.dumps({"schema_version": 1,
                            "episode_ordinal": self.episode_ordinal,
                            "error_class": "TotalIPCMessageBound"}, sort_keys=True),
                        file=sys.stderr, flush=True,
                    )
                    if self._step3_direct_high_level:
                        raise RuntimeError(
                            "Step3 direct snapshot exceeded total IPC message bound"
                        )
            elif self._step3_direct_high_level:
                raise RuntimeError(
                    "Step3 direct current observation snapshot was not captured"
                )
        response = super()._exchange(request)
        if self._step3_live_enabled and response.get("status") == "ok":
            try:
                if operation == "step":
                    self._step3_update_identity(
                        response, next_sequence=int(response["sequence_id"]) + 1
                    )
                    self._step3_arm_next_snapshot()
                elif operation == "reset":
                    self._step3_update_identity(response, next_sequence=0)
                    self._step3_arm_next_snapshot()
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                self._step3_live_enabled = False
                print(
                    "INTERNVLA_STEP3_ARM_DISABLED "
                    + json.dumps({"schema_version": 1,
                        "episode_ordinal": self.episode_ordinal,
                        "error_class": type(exc).__name__}, sort_keys=True),
                    file=sys.stderr, flush=True,
                )
                if self._step3_direct_high_level:
                    raise RuntimeError(
                        "Step3 direct failed to arm the next observation snapshot"
                    ) from exc
        return response

    def _step3_wait_snapshot(
        self,
        *,
        timeout_sec: float = STEP3_SNAPSHOT_WAIT_SEC,
        poll_sec: float = STEP3_SNAPSHOT_POLL_SEC,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any] | None:
        if timeout_sec < 0.0 or poll_sec <= 0.0:
            raise ValueError("Step3 snapshot wait bounds are invalid")
        deadline = clock() + timeout_sec
        while True:
            snapshot = self._step3_load_snapshot()
            if snapshot is not None:
                return snapshot
            remaining = deadline - clock()
            if remaining <= 0.0:
                return None
            sleeper(min(poll_sec, remaining))

    def _configure_step3_live(self) -> None:
            if not self._step3_live_enabled:
                return
            if not (
                self._t5_camera_sensor_identity
                and self.identity_prefix == "b::"
                and os.environ.get("INTERNNAV_T5_LANE", "") == "b"
                and self.tcp_endpoint == "tcp://10.100.120.116:25239"
            ):
                raise RuntimeError("Step3 live snapshot transport is restricted to T5 Lane B")
            result_root = Path(os.environ.get("INTERNVLA_T4_RESULT_ROOT", "")).resolve()
            if not result_root.is_dir():
                raise RuntimeError("Step3 live result root is unavailable")
            request_path = Path(
                os.environ.get(
                    "INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH",
                    str(result_root / "revc_snapshot.request.json"),
                )
            ).resolve()
            ack_path = Path(
                os.environ.get(
                    "INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH",
                    str(result_root / "revc_snapshot.ack.json"),
                )
            ).resolve()
            if (
                request_path.parent != result_root
                or ack_path.parent != result_root
                or request_path.name != "revc_snapshot.request.json"
                or ack_path.name != "revc_snapshot.ack.json"
            ):
                raise RuntimeError("Step3 live snapshot request paths escape Lane B")
            control_root = Path(os.environ.get("INTERNNAV_T1_CONTROL_ROOT", "")).resolve()
            contract_path = (
                control_root / "configs" / "internnav_t5" / "revc_four_camera_snapshot.json"
            )
            contract_sha, extrinsics, cameras = _step3_contract_values(contract_path)
            self._step3_result_root = result_root
            self._step3_request_path = request_path
            self._step3_ack_path = ack_path
            self._step3_contract_sha256 = contract_sha
            self._step3_extrinsics = extrinsics
            self._step3_contract_cameras = cameras

    def _step3_update_identity(
            self, response: dict[str, Any], *, next_sequence: int
        ) -> None:
            episode_id = str(response.get("episode_id") or "")
            reset_generation = response.get("reset_generation")
            if (
                not episode_id.startswith("b::")
                or isinstance(reset_generation, bool)
                or not isinstance(reset_generation, int)
                or reset_generation < 0
                or isinstance(next_sequence, bool)
                or next_sequence < 0
            ):
                raise RuntimeError("Step3 live runtime identity is invalid")
            self._step3_episode_id = episode_id
            self._step3_reset_generation = reset_generation
            self._step3_next_sequence = int(next_sequence)

    def _step3_arm_next_snapshot(self) -> None:
            if not self._step3_live_enabled:
                return
            assert self._step3_request_path is not None
            assert self._step3_ack_path is not None
            if self._step3_request_path.exists():
                raise RuntimeError("Step3 live snapshot request was not consumed")
            try:
                self._step3_ack_path.unlink()
            except FileNotFoundError:
                pass
            request_id = (
                f"{self._step3_episode_id}:step3:{self._step3_reset_generation}:"
                f"{self._step3_next_sequence}"
            )
            value = {
                "schema_version": 1,
                "request_id": request_id,
                "episode_id": self._step3_episode_id,
                "reset_generation": self._step3_reset_generation,
                "sequence_id": self._step3_next_sequence,
            }
            temporary = self._step3_request_path.with_name(
                f".{self._step3_request_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self._step3_request_path)

    def _step3_load_snapshot(self) -> dict[str, Any] | None:
            if not self._step3_live_enabled:
                return None
            assert self._step3_result_root is not None
            assert self._step3_ack_path is not None
            if not self._step3_ack_path.is_file():
                return None
            try:
                ack = json.loads(self._step3_ack_path.read_text(encoding="utf-8"))
                expected_request_id = (
                    f"{self._step3_episode_id}:step3:{self._step3_reset_generation}:"
                    f"{self._step3_next_sequence}"
                )
                if (
                    ack.get("schema_version") != 1
                    or ack.get("status") != "CAPTURED"
                    or ack.get("request_id") != expected_request_id
                    or ack.get("episode_id") != self._step3_episode_id
                    or ack.get("reset_generation") != self._step3_reset_generation
                    or ack.get("sequence_id") != self._step3_next_sequence
                ):
                    return None
                relative = Path(str(ack.get("sidecar") or ""))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Step3 snapshot sidecar path is unsafe")
                sidecar_path = (self._step3_result_root / relative).resolve()
                snapshot_root = (self._step3_result_root / "revc_snapshots").resolve()
                if (
                    sidecar_path.name != "snapshot.json"
                    or sidecar_path.parent.parent != snapshot_root
                    or not sidecar_path.is_file()
                    or sidecar_path.is_symlink()
                ):
                    raise RuntimeError("Step3 snapshot sidecar is outside Lane B")
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                raw_episode = self._step3_episode_id[3:]
                snapshot_id = (
                    f"b::{raw_episode}::{self._step3_reset_generation}::"
                    f"{self._step3_next_sequence}"
                )
                if (
                    sidecar.get("schema_version") != 1
                    or sidecar.get("contract_sha256") != self._step3_contract_sha256
                    or sidecar.get("same_render_tick") is not True
                    or sidecar.get("sim_stamp_before_ns")
                    != sidecar.get("sim_stamp_after_ns")
                    or sidecar.get("episode_id") != self._step3_episode_id
                    or sidecar.get("reset_generation") != self._step3_reset_generation
                    or sidecar.get("sequence_id") != self._step3_next_sequence
                    or tuple(sidecar.get("camera_order") or ()) != STEP3_VIEW_ORDER
                ):
                    raise RuntimeError("Step3 snapshot sidecar contract mismatch")
                cameras = sidecar.get("cameras")
                if not isinstance(cameras, list) or len(cameras) != len(STEP3_VIEW_ORDER):
                    raise RuntimeError("Step3 snapshot camera metadata is incomplete")
                source_paths: list[Path] = []
                for index, (view_id, row) in enumerate(zip(STEP3_VIEW_ORDER, cameras)):
                    expected = self._step3_contract_cameras[view_id]
                    if (
                        not isinstance(row, dict)
                        or row.get("identity") != view_id
                        or row.get("order_index") != index
                        or row.get("encoding") != "png_rgb8"
                        or row.get("frame_id") != expected["frame_id"]
                        or row.get("sensor_name") != expected["sensor_name"]
                        or row.get("prim_path") != expected["prim_path"]
                        or row.get("position_F_M_mm") != expected["position_F_M_mm"]
                        or float(row.get("yaw_deg")) != float(expected["yaw_deg"])
                    ):
                        raise RuntimeError("Step3 snapshot camera contract drifted")
                    image_relative = Path(str(row.get("path") or ""))
                    image_path = (self._step3_result_root / image_relative).resolve()
                    if (
                        image_relative.is_absolute()
                        or ".." in image_relative.parts
                        or image_path.parent != sidecar_path.parent
                        or not image_path.is_file()
                        or image_path.is_symlink()
                    ):
                        raise RuntimeError("Step3 snapshot image path is unsafe")
                    payload = image_path.read_bytes()
                    if (
                        hashlib.sha256(payload).hexdigest() != row.get("sha256")
                        or len(payload) != int(row.get("bytes", -1))
                    ):
                        raise RuntimeError("Step3 snapshot PNG digest mismatch")
                    source_paths.append(image_path)
                jpegs = _encode_step3_jpegs(source_paths)
                sim_stamp_ns = int(sidecar["sim_stamp_before_ns"])
                images = []
                for view_id, row, jpeg in zip(STEP3_VIEW_ORDER, cameras, jpegs):
                    images.append(
                        {
                            "view_id": view_id,
                            "width": 640,
                            "height": 480,
                            "source_frame_id": str(row["frame_id"]),
                            "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                            "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
                            "extrinsic_sha256": self._step3_extrinsics[view_id],
                        }
                    )
                return {
                    "schema_version": 1,
                    "runtime_episode_id": self._step3_episode_id,
                    "reset_generation": self._step3_reset_generation,
                    "sequence_id": self._step3_next_sequence,
                    "snapshot_id": snapshot_id,
                    "sim_stamp_ns": sim_stamp_ns,
                    "same_render_tick": True,
                    "config_sha256": self._step3_contract_sha256,
                    "images": images,
                }
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
                print(
                    (
                        "INTERNNAV_STEP3_DIRECT_SNAPSHOT_REJECTED "
                        if self._step3_direct_high_level
                        else "INTERNVLA_STEP3_SNAPSHOT_FALLBACK "
                    )
                    + json.dumps(
                        {
                            "schema_version": 1,
                            "episode_ordinal": self.episode_ordinal,
                            "error_class": type(exc).__name__,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return None

class LaneBStep3ContinuousROS2IPCAgentClient(
    LaneBStep3ROS2IPCAgentClient, ContinuousROS2IPCAgentClient
):
    """Continuous evaluator facade with the same private snapshot transport."""
