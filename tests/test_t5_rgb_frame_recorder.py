from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.t5_rgb_frame_recorder import T5RGBFrameRecorder, encode_rgb_png


def _observation(sequence: int, stamp_ns: int) -> dict:
    return {
        "rgb": np.full((2, 3, 3), sequence, dtype=np.uint8),
        "camera_sensor_metadata": {
            "schema_version": 1,
            "source": "x86_isaac_pano_camera_0",
            "sequence": sequence,
            "sim_stamp_ns": stamp_ns,
        },
    }


def test_rgb_png_encoder_writes_truecolor_png() -> None:
    encoded = encode_rgb_png(np.zeros((2, 3, 3), dtype=np.uint8))
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in encoded
    assert encoded.endswith(b"IEND\xaeB`\x82")


def test_recorder_samples_unique_source_observations_at_five_hz(
    tmp_path: Path,
) -> None:
    recorder = T5RGBFrameRecorder(tmp_path / "capture")
    recorder.record(_observation(1, 1_000_000_000))
    recorder.record(_observation(2, 1_050_000_000))
    recorder.record(_observation(3, 1_200_000_000))
    recorder.record(_observation(4, 1_410_000_000))

    frames = sorted((tmp_path / "capture" / "frames").glob("*.png"))
    assert [frame.name for frame in frames] == [
        "00000000.png",
        "00000001.png",
        "00000002.png",
    ]
    events = [
        json.loads(line)
        for line in (tmp_path / "capture" / "frames.jsonl").read_text().splitlines()
    ]
    assert [event["source_sequence"] for event in events] == [1, 3, 4]
    summary = json.loads(
        (tmp_path / "capture" / "capture_summary.json").read_text()
    )
    assert summary["target_hz"] == 5.0
    assert summary["frame_count"] == 3
    assert summary["source_frame_count"] == 4
    assert summary["sim_duration_sec"] == 0.41


def test_capture_error_is_nonfatal_and_disables_later_writes(tmp_path: Path) -> None:
    recorder = T5RGBFrameRecorder(tmp_path / "capture")
    recorder.record_without_affecting_control(_observation(1, 1_000_000_000))
    recorder.record_without_affecting_control(_observation(1, 1_200_000_000))
    recorder.record_without_affecting_control(_observation(2, 1_400_000_000))

    summary = json.loads(
        (tmp_path / "capture" / "capture_summary.json").read_text()
    )
    assert summary["status"] == "ERROR"
    assert summary["frame_count"] == 1
    assert "did not advance" in summary["disabled_error"]
