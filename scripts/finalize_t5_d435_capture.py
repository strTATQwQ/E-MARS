#!/usr/bin/env python3
"""Validate and package the independent T5 D435 5 Hz review stream.

``frames.jsonl`` and its referenced ``frames/%08d.png`` files are the
authoritative capture.  This utility never edits those inputs.  It either
validates them, or creates a fixed-rate review MP4 plus a sidecar that binds
the video back to the simulation timestamps in the source event stream.

This utility is deliberately specific to ``d435_rgb_5hz`` events.  The
evaluator's model-observation recorder has a different cadence and must not be
presented as the independent physical-sensor review stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_TARGET_FPS = 5.0
_DEFAULT_VIDEO_NAME = "full_d435_rgb_5hz.mp4"
_DEFAULT_SIDECAR_NAME = "video_sidecar.json"


class CaptureValidationError(ValueError):
    """The authoritative D435 capture is incomplete or inconsistent."""


@dataclass(frozen=True)
class ValidatedCapture:
    root: Path
    events_path: Path
    events_sha256: str
    events: tuple[dict[str, Any], ...]
    first_sim_stamp_ns: int
    last_sim_stamp_ns: int

    @property
    def frame_count(self) -> int:
        return len(self.events)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: object, *, field: str, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureValidationError(
            f"frames.jsonl line {line_number}: {field} must be an integer"
        )
    return value


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    try:
        raw_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CaptureValidationError(
            f"cannot read authoritative event stream {events_path}: {error}"
        ) from error
    if not raw_lines:
        raise CaptureValidationError("frames.jsonl contains no captured frames")

    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: blank lines are not allowed"
            )
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(payload, dict):
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: event must be an object"
            )
        events.append(payload)
    return events


def validate_capture(capture_root: Path) -> ValidatedCapture:
    """Validate one independent D435 capture without modifying it."""

    root = capture_root.expanduser().resolve()
    if not root.is_dir():
        raise CaptureValidationError(f"capture root is not a directory: {root}")
    events_path = root / "frames.jsonl"
    frames_root = root / "frames"
    if not events_path.is_file():
        raise CaptureValidationError(f"missing authoritative event stream: {events_path}")
    if not frames_root.is_dir():
        raise CaptureValidationError(f"missing authoritative frame directory: {frames_root}")

    events = _load_events(events_path)
    previous_stamp_ns: int | None = None
    expected_paths: list[Path] = []
    validated_events: list[dict[str, Any]] = []

    for expected_index, event in enumerate(events):
        line_number = expected_index + 1
        if event.get("event_type") != "d435_rgb_5hz":
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: event_type is not d435_rgb_5hz"
            )
        if event.get("independent_of_model_request_cadence") is not True:
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: event is not marked as an "
                "independent D435 capture"
            )
        frame_index = _require_int(
            event.get("frame_index"), field="frame_index", line_number=line_number
        )
        if frame_index != expected_index:
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: expected frame_index "
                f"{expected_index}, got {frame_index}"
            )

        expected_relative = Path("frames") / f"{expected_index:08d}.png"
        path_text = event.get("path")
        if path_text != expected_relative.as_posix():
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: expected path "
                f"{expected_relative.as_posix()!r}, got {path_text!r}"
            )
        frame_path = root / expected_relative
        if not frame_path.is_file() or frame_path.is_symlink():
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: missing regular frame {frame_path}"
            )
        try:
            if frame_path.read_bytes()[: len(_PNG_SIGNATURE)] != _PNG_SIGNATURE:
                raise CaptureValidationError(
                    f"frames.jsonl line {line_number}: frame is not a PNG"
                )
        except OSError as error:
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: cannot read frame: {error}"
            ) from error

        declared_sha256 = event.get("sha256")
        if not isinstance(declared_sha256, str) or not _SHA256_RE.fullmatch(
            declared_sha256
        ):
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: sha256 is not lowercase SHA-256"
            )
        actual_sha256 = _sha256_file(frame_path)
        if actual_sha256 != declared_sha256:
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: SHA-256 mismatch for {frame_path.name}"
            )

        sim_stamp_ns = _require_int(
            event.get("sim_stamp_ns"), field="sim_stamp_ns", line_number=line_number
        )
        if sim_stamp_ns <= 0:
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: sim_stamp_ns must be positive"
            )
        if previous_stamp_ns is not None and sim_stamp_ns <= previous_stamp_ns:
            raise CaptureValidationError(
                f"frames.jsonl line {line_number}: sim_stamp_ns did not advance"
            )
        previous_stamp_ns = sim_stamp_ns

        for identity_field in ("episode_id", "reset_generation", "sequence_id"):
            if identity_field not in event:
                raise CaptureValidationError(
                    f"frames.jsonl line {line_number}: missing {identity_field}"
                )
        expected_paths.append(frame_path)
        validated_events.append(event)

    actual_paths = sorted(frames_root.glob("*.png"))
    if actual_paths != expected_paths:
        expected_names = {path.name for path in expected_paths}
        actual_names = {path.name for path in actual_paths}
        extras = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        raise CaptureValidationError(
            "frames directory does not exactly match frames.jsonl "
            f"(missing={missing}, extra={extras})"
        )

    return ValidatedCapture(
        root=root,
        events_path=events_path,
        events_sha256=_sha256_file(events_path),
        events=tuple(validated_events),
        first_sim_stamp_ns=int(validated_events[0]["sim_stamp_ns"]),
        last_sim_stamp_ns=int(validated_events[-1]["sim_stamp_ns"]),
    )


def _validation_payload(capture: ValidatedCapture) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "VALIDATED",
        "capture_role": "independent_d435_rgb_review_stream",
        "independent_of_model_request_cadence": True,
        "frame_count": capture.frame_count,
        "fps": _TARGET_FPS,
        "first_sim_stamp_ns": capture.first_sim_stamp_ns,
        "last_sim_stamp_ns": capture.last_sim_stamp_ns,
        "source_events_path": "frames.jsonl",
        "source_events_sha256": capture.events_sha256,
    }


def _run_ffmpeg(
    capture: ValidatedCapture,
    *,
    ffmpeg: str,
    temporary_video: Path,
) -> None:
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "5",
        "-start_number",
        "0",
        "-i",
        "frames/%08d.png",
        "-frames:v",
        str(capture.frame_count),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-r",
        "5",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_video),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=capture.root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"failed to execute ffmpeg: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"ffmpeg failed with exit {result.returncode}: {detail}")
    if not temporary_video.is_file() or temporary_video.stat().st_size <= 0:
        raise RuntimeError("ffmpeg reported success but produced no non-empty MP4")


def finalize_capture(
    capture_root: Path,
    *,
    ffmpeg: str = "ffmpeg",
    video_name: str = _DEFAULT_VIDEO_NAME,
    sidecar_name: str = _DEFAULT_SIDECAR_NAME,
) -> dict[str, Any]:
    """Validate inputs, encode one fixed 5 Hz MP4, and atomically add metadata."""

    capture = validate_capture(capture_root)
    if Path(video_name).name != video_name or not video_name.endswith(".mp4"):
        raise ValueError("video name must be a plain .mp4 filename")
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".json"):
        raise ValueError("sidecar name must be a plain .json filename")

    video_path = capture.root / video_name
    sidecar_path = capture.root / sidecar_name
    if video_path.exists() or sidecar_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing finalized video or sidecar"
        )

    temporary_video = capture.root / f".{video_name}.{os.getpid()}.tmp.mp4"
    temporary_sidecar = capture.root / f".{sidecar_name}.{os.getpid()}.tmp"
    installed_video = False
    try:
        _run_ffmpeg(capture, ffmpeg=ffmpeg, temporary_video=temporary_video)
        video_sha256 = _sha256_file(temporary_video)
        video_bytes = temporary_video.stat().st_size
        payload = {
            "schema_version": 1,
            "status": "COMPLETE",
            "capture_role": "independent_d435_rgb_review_stream",
            "independent_of_model_request_cadence": True,
            "video": {
                "path": video_name,
                "container": "mp4",
                "codec": "h264",
                "pixel_format": "yuv420p",
                "frame_count": capture.frame_count,
                "fps": _TARGET_FPS,
                "sha256": video_sha256,
                "bytes": video_bytes,
            },
            "frame_count": capture.frame_count,
            "fps": _TARGET_FPS,
            "first_sim_stamp_ns": capture.first_sim_stamp_ns,
            "last_sim_stamp_ns": capture.last_sim_stamp_ns,
            "source_events": {
                "path": "frames.jsonl",
                "sha256": capture.events_sha256,
            },
            "alignment": {
                "semantic_timebase": "x86_sim_stamp",
                "mapping": (
                    "video frame N maps exactly to frames.jsonl event with "
                    "frame_index N; use that event's episode/reset/sequence and "
                    "sim_stamp_ns to join the unified replay timeline"
                ),
                "playback_note": (
                    "the MP4 uses a fixed 5 fps review clock; authoritative "
                    "simulation timing remains in frames.jsonl"
                ),
            },
        }
        temporary_sidecar.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_video, video_path)
        installed_video = True
        os.replace(temporary_sidecar, sidecar_path)
        return payload
    except Exception:
        if installed_video:
            video_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_video.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate authoritative inputs without requiring ffmpeg or writing files",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-name", default=_DEFAULT_VIDEO_NAME)
    parser.add_argument("--sidecar-name", default=_DEFAULT_SIDECAR_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_only:
            payload = _validation_payload(validate_capture(args.capture_root))
        else:
            payload = finalize_capture(
                args.capture_root,
                ffmpeg=args.ffmpeg,
                video_name=args.video_name,
                sidecar_name=args.sidecar_name,
            )
    except (CaptureValidationError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
