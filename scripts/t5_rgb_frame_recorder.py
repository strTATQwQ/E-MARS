#!/usr/bin/env python3
"""Default-off, observation-only RGB recorder for T5 completion_sim runs."""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    """Encode one uint8 RGB image without adding a runtime image dependency."""

    frame = np.ascontiguousarray(rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape HxWx3, got {frame.shape!r}")
    height, width, _ = frame.shape
    if height <= 0 or width <= 0:
        raise ValueError("RGB frame dimensions must be positive")
    scanlines = b"".join(b"\x00" + row.tobytes() for row in frame)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=1))
        + _png_chunk(b"IEND", b"")
    )


class T5RGBFrameRecorder:
    """Sample evaluator RGB observations at 5 Hz in simulation time."""

    interval_ns = 200_000_000

    def __init__(self, root: Path):
        if not root.is_absolute():
            raise ValueError("T5 RGB capture root must be absolute")
        self.root = root
        self.frames_root = root / "frames"
        self.events_path = root / "frames.jsonl"
        self.summary_path = root / "capture_summary.json"
        self.frame_count = 0
        self.source_frame_count = 0
        self.last_source_sequence = -1
        self.last_source_stamp_ns = -1
        self.last_recorded_stamp_ns = -1
        self.first_recorded_stamp_ns: int | None = None
        self.disabled_error: str | None = None
        self.frames_root.mkdir(parents=True, exist_ok=False)
        self._write_summary("RUNNING")

    @classmethod
    def from_environment(cls) -> T5RGBFrameRecorder | None:
        enabled = os.environ.get("INTERNVLA_T5_FULL_RGB_CAPTURE", "0")
        if enabled == "0":
            return None
        if enabled != "1":
            raise ValueError("INTERNVLA_T5_FULL_RGB_CAPTURE must be 0 or 1")
        root_text = os.environ.get("INTERNVLA_T5_FULL_RGB_CAPTURE_ROOT", "")
        if not root_text:
            raise ValueError("enabled T5 RGB capture requires an output root")
        return cls(Path(root_text))

    def _write_summary(self, status: str) -> None:
        duration_sec = None
        measured_hz = None
        if (
            self.first_recorded_stamp_ns is not None
            and self.last_recorded_stamp_ns > self.first_recorded_stamp_ns
        ):
            duration_sec = (
                self.last_recorded_stamp_ns - self.first_recorded_stamp_ns
            ) / 1e9
            measured_hz = (self.frame_count - 1) / duration_sec
        payload = {
            "schema_version": 1,
            "status": status,
            "target_hz": 5.0,
            "sampling_timebase": "x86_sim_stamp",
            "source": "evaluator_internvla_rgb",
            "frame_count": self.frame_count,
            "source_frame_count": self.source_frame_count,
            "first_sim_stamp_ns": self.first_recorded_stamp_ns,
            "last_sim_stamp_ns": (
                self.last_recorded_stamp_ns
                if self.last_recorded_stamp_ns >= 0
                else None
            ),
            "sim_duration_sec": duration_sec,
            "measured_capture_hz": measured_hz,
            "disabled_error": self.disabled_error,
        }
        temporary = self.summary_path.with_name(
            f".{self.summary_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.summary_path)

    def record(self, observation: dict[str, Any]) -> None:
        if self.disabled_error is not None:
            return
        metadata = observation.get("camera_sensor_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("T5 RGB capture observation has no camera metadata")
        source_sequence = metadata.get("sequence")
        source_stamp_ns = metadata.get("sim_stamp_ns")
        if (
            isinstance(source_sequence, bool)
            or not isinstance(source_sequence, int)
            or source_sequence <= self.last_source_sequence
        ):
            raise ValueError("T5 RGB capture source sequence did not advance")
        if (
            isinstance(source_stamp_ns, bool)
            or not isinstance(source_stamp_ns, int)
            or source_stamp_ns <= self.last_source_stamp_ns
        ):
            raise ValueError("T5 RGB capture simulation stamp did not advance")
        self.source_frame_count += 1
        self.last_source_sequence = source_sequence
        self.last_source_stamp_ns = source_stamp_ns
        if (
            self.last_recorded_stamp_ns >= 0
            and source_stamp_ns - self.last_recorded_stamp_ns < self.interval_ns
        ):
            return
        rgb = np.ascontiguousarray(observation["rgb"], dtype=np.uint8)
        frame_index = self.frame_count
        relative_path = Path("frames") / f"{frame_index:08d}.png"
        output = self.root / relative_path
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encode_rgb_png(rgb))
        os.replace(temporary, output)
        event = {
            "schema_version": 1,
            "frame_index": frame_index,
            "path": relative_path.as_posix(),
            "source_sequence": source_sequence,
            "sim_stamp_ns": source_stamp_ns,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "wall_monotonic_ns": time.monotonic_ns(),
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        self.frame_count += 1
        if self.first_recorded_stamp_ns is None:
            self.first_recorded_stamp_ns = source_stamp_ns
        self.last_recorded_stamp_ns = source_stamp_ns
        self._write_summary("RUNNING")

    def record_without_affecting_control(self, observation: dict[str, Any]) -> None:
        try:
            self.record(observation)
        except Exception as exc:  # capture is deliberately outside control safety
            self.disabled_error = f"{type(exc).__name__}: {exc}"
            try:
                self._write_summary("ERROR")
            finally:
                print(
                    f"WARN: T5 full RGB capture disabled: {self.disabled_error}",
                    file=sys.stderr,
                    flush=True,
                )
