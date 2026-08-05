from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import finalize_t5_d435_capture as finalizer


_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf\xc0\x00\x00\x04\x00\x01"
    b"\x9b\x0e\x1f\x8d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _capture(tmp_path: Path, *, count: int = 3) -> Path:
    root = tmp_path / "d435_rgb_5hz"
    frames = root / "frames"
    frames.mkdir(parents=True)
    with (root / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(count):
            payload = _PNG + bytes([index])
            relative = f"frames/{index:08d}.png"
            (root / relative).write_bytes(payload)
            event = {
                "schema_version": 1,
                "event_type": "d435_rgb_5hz",
                "frame_index": index,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "episode_id": "a::433_121",
                "reset_generation": 1,
                "sequence_id": index + 10,
                "sim_stamp_ns": 1_000_000_000 + index * 200_000_000,
                "height": 480,
                "width": 640,
                "independent_of_model_request_cadence": True,
            }
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    return root


def test_validate_only_does_not_require_ffmpeg_or_write_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _capture(tmp_path)

    assert (
        finalizer.main(
            [
                "--capture-root",
                str(root),
                "--validate-only",
                "--ffmpeg",
                "definitely-not-installed",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "VALIDATED"
    assert payload["frame_count"] == 3
    assert payload["capture_role"] == "independent_d435_rgb_review_stream"
    assert payload["independent_of_model_request_cadence"] is True
    assert not (root / "full_d435_rgb_5hz.mp4").exists()
    assert not (root / "video_sidecar.json").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda events: events[1].update(frame_index=7), "expected frame_index"),
        (lambda events: events[1].update(path="frames/evil.png"), "expected path"),
        (lambda events: events[1].update(sha256="0" * 64), "SHA-256 mismatch"),
        (
            lambda events: events[1].update(
                sim_stamp_ns=events[0]["sim_stamp_ns"]
            ),
            "sim_stamp_ns did not advance",
        ),
    ],
)
def test_validation_fails_closed_on_authority_mismatch(
    tmp_path: Path, mutate: object, message: str
) -> None:
    root = _capture(tmp_path)
    events = [
        json.loads(line)
        for line in (root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    mutate(events)  # type: ignore[operator]
    (root / "frames.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    with pytest.raises(finalizer.CaptureValidationError, match=message):
        finalizer.validate_capture(root)

    assert not (root / "full_d435_rgb_5hz.mp4").exists()
    assert not (root / "video_sidecar.json").exists()


def test_finalize_writes_atomic_video_and_alignment_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _capture(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs["cwd"] == root.resolve()
        Path(command[-1]).write_bytes(b"synthetic-h264-mp4")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(finalizer.subprocess, "run", fake_run)
    payload = finalizer.finalize_capture(root)

    video = root / "full_d435_rgb_5hz.mp4"
    sidecar = json.loads((root / "video_sidecar.json").read_text(encoding="utf-8"))
    assert video.read_bytes() == b"synthetic-h264-mp4"
    assert payload == sidecar
    assert sidecar["video"]["frame_count"] == 3
    assert sidecar["video"]["fps"] == 5.0
    assert sidecar["video"]["codec"] == "h264"
    assert sidecar["video"]["pixel_format"] == "yuv420p"
    assert sidecar["video"]["sha256"] == hashlib.sha256(
        b"synthetic-h264-mp4"
    ).hexdigest()
    assert sidecar["source_events"]["sha256"] == hashlib.sha256(
        (root / "frames.jsonl").read_bytes()
    ).hexdigest()
    assert "frame N maps exactly" in sidecar["alignment"]["mapping"]
    command = commands[0]
    assert command[command.index("-framerate") + 1] == "5"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"


def test_ffmpeg_failure_preserves_raw_capture_and_leaves_no_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _capture(tmp_path)
    original_events = (root / "frames.jsonl").read_bytes()
    original_frames = [(path.name, path.read_bytes()) for path in sorted((root / "frames").glob("*.png"))]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 1, "", "encoder failed")

    monkeypatch.setattr(finalizer.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="encoder failed"):
        finalizer.finalize_capture(root)

    assert (root / "frames.jsonl").read_bytes() == original_events
    assert [(path.name, path.read_bytes()) for path in sorted((root / "frames").glob("*.png"))] == original_frames
    assert not (root / "full_d435_rgb_5hz.mp4").exists()
    assert not (root / "video_sidecar.json").exists()
    assert not list(root.glob(".*.tmp*"))
