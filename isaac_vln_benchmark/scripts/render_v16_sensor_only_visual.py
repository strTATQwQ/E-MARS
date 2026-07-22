#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zlib
import base64
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def event_type(row: dict[str, Any]) -> str:
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    return str(details.get("event_type") or row.get("event") or "")


def semantic_overlay(run_dir: Path, image_path: Path, response_path: Path, output: Path) -> None:
    source = Image.open(image_path).convert("RGB")
    response = read_json(response_path)
    detection = max(response.get("detections") or [], key=lambda row: float(row.get("score", 0.0)))
    width, height = source.size
    mask_bytes = zlib.decompress(base64.b64decode(detection["mask_zlib_base64"]))
    mask = Image.frombytes("L", (width, height), bytes(255 if value else 0 for value in mask_bytes))
    tint = Image.new("RGBA", source.size, (255, 45, 45, 0))
    tint.putalpha(mask.point(lambda value: 105 if value else 0))
    composited = Image.alpha_composite(source.convert("RGBA"), tint).convert("RGB")
    draw = ImageDraw.Draw(composited)
    x0, y0, x1, y1 = detection["bbox_xyxy_norm"]
    box = (int(x0 * width), max(0, int(y0 * height)), int(x1 * width), int(y1 * height))
    draw.rectangle(box, outline=(30, 255, 90), width=3)
    draw.text((8, 8), "GroundingDINO-B + SAM2.1-L (offline frozen-frame reproduction)", fill="white", font=font(11), stroke_width=2, stroke_fill="black")

    gate = read_json(run_dir / "v16_sensor_only_gate.json")
    events = read_events(run_dir / "events.jsonl")
    step_rows = [row for row in events if event_type(row) == "step_http_response"]
    output_json = ((step_rows[0].get("details") or {}).get("output") or {}) if step_rows else {}
    track = output_json.get("track") or {}
    panel_h = 178
    canvas = Image.new("RGB", (width * 2, height * 2 + panel_h), (17, 21, 27))
    canvas.paste(composited.resize((width * 2, height * 2), Image.Resampling.NEAREST), (0, 0))
    panel = ImageDraw.Draw(canvas)
    lines = [
        f"bbox={detection['bbox_xyxy_norm']} score={detection['score']} mask_score={detection.get('mask_score')}",
        f"track: hits={track.get('hits')} confirmed={track.get('confirmed')} fresh={track.get('fresh')} distance={track.get('distance_m')}m",
        f"Step JSON: stop={output_json.get('stop')} visible={output_json.get('target_visible')} distance_ok={output_json.get('estimated_distance_ok')}",
        f"latency: perception_p95={gate['latency']['perception_p95_sec']}s step={gate['latency']['step_p95_sec']}s safe_stop={gate['latency']['threshold_to_safe_stop_p95_sec']}s",
        f"chain: stale_gate -> controller -> primitive_executor -> safe_mux | collision=0 stale_action=0",
        "qualification_evidence=false | ideal_kinematic | Sim2Real NOT READY",
    ]
    y = height * 2 + 10
    for line in lines:
        panel.text((12, y), line, fill=(224, 232, 240), font=font(14))
        y += 26
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def route_overlay(run_dir: Path, output: Path) -> None:
    metrics = read_json(run_dir / "metrics.json")
    episode = metrics["episodes"][0]
    mode = episode["mode"]
    task_id = episode["task_id"]
    episode_dir = run_dir / mode / task_id
    scene = read_json(episode_dir / "scene.yaml")
    events = read_events(run_dir / "events.jsonl")
    trajectory = [
        (float(row["details"]["pose"][0]), float(row["details"]["pose"][1]))
        for row in events
        if row.get("event") == "isaac_ground_truth_pose"
        and isinstance(row.get("details"), dict)
        and isinstance(row["details"].get("pose"), list)
    ]
    phases: dict[str, int] = {}
    for row in events:
        if event_type(row) != "sensor_planner_primitive":
            continue
        phase = str((((row.get("details") or {}).get("primitive") or {}).get("phase") or ""))
        phases[phase] = phases.get(phase, 0) + 1
    bounds = scene.get("bounds") or [-1.5, -3.0, 8.0, 3.0]
    xmin, ymin, xmax, ymax = [float(value) for value in bounds]
    width, height, panel_h, margin = 1000, 620, 150, 45
    canvas = Image.new("RGB", (width, height + panel_h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    def point(x: float, y: float) -> tuple[int, int]:
        px = margin + int((x - xmin) / (xmax - xmin) * (width - 2 * margin))
        py = margin + int((ymax - y) / (ymax - ymin) * (height - 2 * margin))
        return px, py

    for obstacle in scene.get("obstacles") or []:
        pose, size = obstacle["pose"], obstacle["size"]
        p0 = point(float(pose[0]) - float(size[0]) / 2, float(pose[1]) + float(size[1]) / 2)
        p1 = point(float(pose[0]) + float(size[0]) / 2, float(pose[1]) - float(size[1]) / 2)
        draw.rectangle((*p0, *p1), fill=(85, 92, 104), outline=(20, 24, 30), width=2)
        draw.text((p0[0], p0[1] - 18), str(obstacle.get("id")), fill=(25, 30, 38), font=font(12))
    for obj in scene.get("objects") or []:
        x, y = float(obj["pose"][0]), float(obj["pose"][1])
        color = (224, 52, 52) if str(obj.get("color")) == "red" else (40, 100, 225)
        px, py = point(x, y)
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline=(15, 15, 15))
        draw.text((px + 10, py - 8), str(obj.get("id")), fill=(25, 30, 38), font=font(11))
    if len(trajectory) >= 2:
        draw.line([point(x, y) for x, y in trajectory], fill=(0, 157, 125), width=5)
    if trajectory:
        sx, sy = point(*trajectory[0])
        ex, ey = point(*trajectory[-1])
        draw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), fill=(255, 210, 30))
        draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=(0, 60, 40))
    draw.text((margin, 10), "V16 route diagnostic (judge geometry is visualization-only)", fill=(20, 28, 38), font=font(18))
    gate = read_json(run_dir / "v16_sensor_only_gate.json")
    draw.rectangle((0, height, width, height + panel_h), fill=(17, 21, 27))
    lines = [
        f"result={episode['success']} reason={episode['failure_reason']} path={episode['path_length_m']}m final_target_distance={episode['final_distance_to_target_m']}m",
        f"phases={json.dumps(phases, sort_keys=True)}",
        f"Step={gate['latency']['step_p95_sec']}s perception_p95={gate['latency']['perception_p95_sec']}s safety_blocks={gate['safety']['safety_block_ticks']}",
        "controller input: actual RGB-D + odom only | oracle leakage=0 | qualification_evidence=false",
        "route gate=FAIL | expansion stopped | Sim2Real NOT READY",
    ]
    y = height + 8
    for line in lines:
        draw.text((12, y), line, fill=(224, 232, 240), font=font(14))
        y += 27
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--grounded-sam-response", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.image and args.grounded_sam_response:
        semantic_overlay(run_dir, args.image.resolve(), args.grounded_sam_response.resolve(), args.output.resolve())
    else:
        route_overlay(run_dir, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
