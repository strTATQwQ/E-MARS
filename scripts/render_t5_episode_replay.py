#!/usr/bin/env python3
"""Render one T5 episode into a review MP4 on the evidence host.

The normal view is the independent D435 RGB stream.  When a Step3 ADVISE
record is active, the renderer substitutes the four same-render-tick Rev-C
views and prints the public bounded primitive delivered to InternVLA.  Hidden
model reasoning is deliberately neither loaded nor rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap
from typing import Any, BinaryIO

from PIL import Image, ImageDraw, ImageFont


ACTION_LABELS = {
    1: "short_forward",
    2: "short_turn_left",
    3: "short_turn_right",
}
CAMERA_LABELS = {
    "front_left": "FRONT LEFT",
    "front": "FRONT",
    "front_right": "FRONT RIGHT",
    "rear": "REAR",
}
OUTPUT_SIZE = (1280, 720)
VIEW_SIZE = (960, 720)
PANEL_X = 960


class ReplayError(RuntimeError):
    pass


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"{label} is not a regular file: {path}")
    return path


def _inside(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    if path != root and root not in path.parents:
        raise ReplayError(f"{label} escaped evaluator root: {path}")
    return path


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(
        _regular_file(path, "JSONL").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReplayError(f"{path}:{ordinal} is not a JSON object")
        rows.append(value)
    return rows


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_label(
    image: Image.Image,
    text: str,
    xy: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    box = draw.textbbox(xy, text, font=font, stroke_width=1)
    draw.rectangle((box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3), fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 255), font=font, stroke_width=1)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, "black")
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def _load_snapshots(evaluator: Path, episode_id: str) -> dict[tuple[int, int], dict[str, Any]]:
    root = evaluator / "revc_snapshots"
    if not root.is_dir() or root.is_symlink():
        return {}
    snapshots: dict[tuple[int, int], dict[str, Any]] = {}
    for sidecar in sorted(root.glob("*/snapshot.json")):
        value = json.loads(_regular_file(sidecar, "snapshot sidecar").read_text(encoding="utf-8"))
        if value.get("episode_id") != episode_id:
            continue
        if value.get("same_render_tick") is not True:
            raise ReplayError(f"snapshot lacks same-render-tick evidence: {sidecar}")
        reset = value.get("reset_generation")
        sequence = value.get("sequence_id")
        if isinstance(reset, bool) or not isinstance(reset, int):
            raise ReplayError(f"invalid snapshot reset: {sidecar}")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ReplayError(f"invalid snapshot sequence: {sidecar}")
        cameras = value.get("cameras")
        if not isinstance(cameras, list) or [x.get("identity") for x in cameras] != [
            "front_left",
            "front",
            "front_right",
            "rear",
        ]:
            raise ReplayError(f"invalid four-camera order: {sidecar}")
        value["_sidecar"] = sidecar
        snapshots[(reset, sequence)] = value
    return snapshots


def _advice_events(
    evaluator: Path,
    episode_id: str,
    snapshots: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    path = evaluator / "step3_timeout_advice.jsonl"
    if not path.is_file():
        return [], 0
    all_episode = [row for row in _json_lines(path) if row.get("episode_id") == episode_id]
    events: list[dict[str, Any]] = []
    for row in all_episode:
        if row.get("status") != "ADVISE":
            continue
        action = row.get("advised_action")
        reset = row.get("reset_generation")
        sequence = row.get("trigger_sequence_id")
        if action not in ACTION_LABELS or not isinstance(reset, int) or not isinstance(sequence, int):
            raise ReplayError("ADVISE row has an invalid public primitive identity")
        snapshot = snapshots.get((reset, sequence))
        if snapshot is None:
            raise ReplayError(f"ADVISE lacks matching four-camera snapshot: reset={reset} sequence={sequence}")
        event = dict(row)
        event["_snapshot"] = snapshot
        event["_sim_stamp_ns"] = int(snapshot["sim_stamp_after_ns"])
        events.append(event)
    events.sort(key=lambda value: value["_sim_stamp_ns"])
    return events, len(all_episode)


def _snapshot_canvas(evaluator: Path, event: dict[str, Any], label_font: ImageFont.ImageFont) -> Image.Image:
    snapshot = event["_snapshot"]
    canvas = Image.new("RGB", VIEW_SIZE, "black")
    for index, camera in enumerate(snapshot["cameras"]):
        source = _inside(evaluator, evaluator / camera["path"], "Rev-C image")
        with Image.open(source) as opened:
            tile = _fit(opened, (480, 360))
        x = (index % 2) * 480
        y = (index // 2) * 360
        canvas.paste(tile, (x, y))
        _draw_label(canvas, CAMERA_LABELS[camera["identity"]], (x + 12, y + 10), font=label_font)
    return canvas


def _draw_panel(
    canvas: Image.Image,
    *,
    episode_id: str,
    mode: str,
    frame: dict[str, Any],
    first_sim_ns: int,
    event: dict[str, Any] | None,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((PANEL_X, 0, OUTPUT_SIZE[0], OUTPUT_SIZE[1]), fill=(18, 21, 27))
    y = 18
    for line, color, font in (
        ("InternNav T5 Replay", (120, 210, 255), title_font),
        (f"Episode: {episode_id}", (255, 255, 255), body_font),
        (f"Mode: {mode}", (255, 255, 255), body_font),
        (f"Sim: {(int(frame['sim_stamp_ns']) - first_sim_ns) / 1e9:7.1f} s", (210, 210, 210), body_font),
        (f"Reset: {frame.get('reset_generation')}", (210, 210, 210), body_font),
        (f"Sequence: {frame.get('sequence_id')}", (210, 210, 210), body_font),
    ):
        draw.text((PANEL_X + 16, y), line, fill=color, font=font)
        y += 37 if font is title_font else 29
    y += 18
    if event is None:
        text = "D435 RGB 5 Hz\n\nNo active Step3 advice"
        color = (170, 180, 190)
    else:
        action = int(event["advised_action"])
        text = (
            "STEP3 PUBLIC ADVICE\n\n"
            f"{ACTION_LABELS[action]}\n"
            f"action={action}\n"
            f"confidence={float(event.get('confidence', 0.0)):.2f}\n\n"
            f"reason={event.get('reason', '')}\n\n"
            f"request={event.get('trigger_request_id', '')}"
        )
        color = (255, 214, 92)
    for paragraph in text.splitlines():
        lines = textwrap.wrap(paragraph, width=27, break_long_words=True) or [""]
        for line in lines:
            draw.text((PANEL_X + 16, y), line, fill=color, font=body_font)
            y += 28


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_frame(stream: BinaryIO, image: Image.Image) -> None:
    stream.write(image.convert("RGB").tobytes())


def render(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = args.evaluator.resolve(strict=True)
    if evaluator.is_symlink() or not evaluator.is_dir():
        raise ReplayError("evaluator root must be a real directory")
    frames_path = evaluator / "d435_rgb_5hz/frames.jsonl"
    frames = [
        row for row in _json_lines(frames_path) if row.get("episode_id") == args.episode_id
    ]
    if not frames:
        raise ReplayError(f"episode has no D435 frames: {args.episode_id}")
    frames.sort(key=lambda value: (int(value["sim_stamp_ns"]), int(value["frame_index"])))
    if any(int(b["sim_stamp_ns"]) <= int(a["sim_stamp_ns"]) for a, b in zip(frames, frames[1:])):
        raise ReplayError("episode D435 sim stamps are not strictly increasing")
    snapshots = _load_snapshots(evaluator, args.episode_id)
    events, response_count = _advice_events(evaluator, args.episode_id, snapshots)
    if args.require_step3 and not events:
        raise ReplayError("Step3 rendering was requested but no ADVISE event is available")

    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ReplayError(f"refusing to overwrite output: {output}")
    receipt = output.with_suffix(output.suffix + ".json")
    if receipt.exists() or receipt.is_symlink():
        raise ReplayError(f"refusing to overwrite receipt: {receipt}")
    ffmpeg = _regular_file(args.ffmpeg.resolve(strict=True), "ffmpeg")
    if not os.access(ffmpeg, os.X_OK):
        raise ReplayError(f"ffmpeg is not executable: {ffmpeg}")

    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720",
        "-framerate", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-threads", str(args.threads), str(output),
    ]
    title_font = _font(23)
    body_font = _font(20)
    label_font = _font(20)
    event_canvases: dict[tuple[int, int], Image.Image] = {}
    hold_ns = round(args.step3_hold_sec * 1e9)
    first_sim_ns = int(frames[0]["sim_stamp_ns"])
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise ReplayError("ffmpeg stdin was not created")
        event_index = 0
        active: dict[str, Any] | None = None
        for frame in frames:
            sim_ns = int(frame["sim_stamp_ns"])
            while event_index < len(events) and events[event_index]["_sim_stamp_ns"] <= sim_ns:
                active = events[event_index]
                event_index += 1
            if active is not None and sim_ns - active["_sim_stamp_ns"] > hold_ns:
                active = None
            canvas = Image.new("RGB", OUTPUT_SIZE, "black")
            if active is None:
                source = _inside(
                    evaluator,
                    evaluator / "d435_rgb_5hz" / frame["path"],
                    "D435 frame",
                )
                with Image.open(source) as opened:
                    view = _fit(opened, VIEW_SIZE)
                canvas.paste(view, (0, 0))
                _draw_label(canvas, "D435 RGB", (12, 10), font=label_font)
            else:
                key = (int(active["reset_generation"]), int(active["trigger_sequence_id"]))
                if key not in event_canvases:
                    event_canvases[key] = _snapshot_canvas(evaluator, active, label_font)
                canvas.paste(event_canvases[key], (0, 0))
            _draw_panel(
                canvas,
                episode_id=args.episode_id,
                mode=args.mode,
                frame=frame,
                first_sim_ns=first_sim_ns,
                event=active,
                title_font=title_font,
                body_font=body_font,
            )
            _write_frame(process.stdin, canvas)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise ReplayError(f"ffmpeg exited with {return_code}")
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "episode_id": args.episode_id,
        "mode": args.mode,
        "source_evaluator": str(evaluator),
        "source_frames_jsonl": str(frames_path),
        "frame_count": len(frames),
        "fps": args.fps,
        "playback_duration_sec": len(frames) / args.fps,
        "first_sim_stamp_ns": int(frames[0]["sim_stamp_ns"]),
        "last_sim_stamp_ns": int(frames[-1]["sim_stamp_ns"]),
        "step3_response_count": response_count,
        "step3_advice_count": len(events),
        "step3_hold_sec": args.step3_hold_sec,
        "step3_public_fields_only": True,
        "hidden_reasoning_rendered": False,
        "four_camera_switching": bool(events),
        "output_mp4": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "ffmpeg": str(ffmpeg),
    }
    temporary = receipt.with_suffix(receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--step3-hold-sec", type=float, default=2.0)
    parser.add_argument("--require-step3", action="store_true")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=21)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.fps <= 0 or not 0.2 <= args.step3_hold_sec <= 10.0:
        parser.error("fps/Step3 hold interval is out of bounds")
    if not 0 <= args.crf <= 51 or not 1 <= args.threads <= 32:
        parser.error("CRF/thread count is out of bounds")
    return args


def main() -> int:
    try:
        payload = render(parse_args())
    except (OSError, ValueError, json.JSONDecodeError, ReplayError) as error:
        raise SystemExit(f"render_t5_episode_replay: {error}") from error
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
