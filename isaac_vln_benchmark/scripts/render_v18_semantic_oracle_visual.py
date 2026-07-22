#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--tasks", nargs="+", default=["ref_01", "comp_02", "rec_01"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run = Path(args.run)
    output = Path(args.output) if args.output else run / "visual" / "semantic_oracle_trajectory_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks_doc = load_document(run / "selected_tasks.yaml")
    scenes_doc = load_document(run / "selected_scenes.yaml")
    tasks = {str(task["task_id"]): task for task in tasks_doc.get("tasks", [])}
    scenes = {str(scene["scene_type"]): scene for scene in scenes_doc.get("scenes", [])}

    panel_width = 620
    panel_height = 690
    top = 72
    canvas = Image.new("RGB", (panel_width * len(args.tasks), panel_height + top), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24, bold=True)
    panel_title_font = _font(18, bold=True)
    body_font = _font(14)
    small_font = _font(12)
    draw.text(
        (canvas.width // 2, 18),
        "V18 Forced Semantic Oracle: actual motion with oracle-only judge overlay",
        fill="#202124",
        font=title_font,
        anchor="ma",
    )

    overlay_rows = []
    for panel_index, task_id in enumerate(args.tasks):
        x0 = panel_index * panel_width
        task = tasks[task_id]
        scene = scenes[str(task["scene_type"])]
        episode_dir = run / "forced_semantic_oracle" / task_id
        metrics = load_document(episode_dir / "metrics.json")
        with (episode_dir / "trajectory.csv").open("r", encoding="utf-8", newline="") as handle:
            trajectory = list(csv.DictReader(handle))
        points = [(float(row["x"]), float(row["y"])) for row in trajectory]
        marker_points = [tuple(float(value) for value in (obj.get("pose") or [0.0, 0.0])[:2]) for obj in scene.get("objects", [])]
        obstacle_points = [tuple(float(value) for value in (obj.get("pose") or [0.0, 0.0])[:2]) for obj in scene.get("obstacles", [])]
        all_points = points + marker_points + obstacle_points + [(0.0, 0.0)]
        min_x = min(point[0] for point in all_points) - 0.8
        max_x = max(point[0] for point in all_points) + 0.8
        min_y = min(point[1] for point in all_points) - 0.8
        max_y = max(point[1] for point in all_points) + 0.8
        plot = (x0 + 45, top + 62, x0 + panel_width - 25, top + 470)

        def pixel(point: tuple[float, float]) -> tuple[int, int]:
            px = plot[0] + (point[0] - min_x) / max(max_x - min_x, 1e-6) * (plot[2] - plot[0])
            py = plot[3] - (point[1] - min_y) / max(max_y - min_y, 1e-6) * (plot[3] - plot[1])
            return int(px), int(py)

        draw.rectangle((x0 + 8, top + 4, x0 + panel_width - 8, top + panel_height - 10), fill="#ffffff", outline="#c8ccd0", width=2)
        result = "SUCCESS" if metrics.get("success") else str(metrics.get("failure_reason") or "FAIL").upper()
        draw.text(
            (x0 + panel_width // 2, top + 18),
            f"{task_id} | {task['category']} | {result}",
            fill="#207245" if metrics.get("success") else "#b3261e",
            font=panel_title_font,
            anchor="ma",
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            gx = int(plot[0] + fraction * (plot[2] - plot[0]))
            gy = int(plot[1] + fraction * (plot[3] - plot[1]))
            draw.line((gx, plot[1], gx, plot[3]), fill="#e2e5e9", width=1)
            draw.line((plot[0], gy, plot[2], gy), fill="#e2e5e9", width=1)
        draw.rectangle(plot, outline="#8b929a", width=1)
        if len(points) >= 2:
            draw.line([pixel(point) for point in points], fill="#1f6f8b", width=4, joint="curve")
        if points:
            _circle(draw, pixel(points[0]), 6, "#202124")
            _cross(draw, pixel(points[-1]), 7, "#d1495b")
        for index, obj in enumerate(scene.get("objects", [])):
            pose = tuple(float(value) for value in (obj.get("pose") or [0.0, 0.0])[:2])
            px, py = pixel(pose)
            color = _plot_color(str(obj.get("color") or "gray"))
            draw.rectangle((px - 7, py - 7, px + 7, py + 7), fill=color, outline="#202124", width=2)
            label = f"{index}: {obj.get('semantic_marker_label', obj.get('class', 'marker'))}"
            if px > plot[2] - 170:
                draw.text((px - 10, py - 14), label, fill="#202124", font=small_font, anchor="ra")
            else:
                draw.text((px + 10, py - 14), label, fill="#202124", font=small_font)
        for obstacle in scene.get("obstacles", []):
            pose = tuple(float(value) for value in (obstacle.get("pose") or [0.0, 0.0])[:2])
            size = obstacle.get("size") or [0.4, 0.4, 1.0]
            cx, cy = pixel(pose)
            sx = max(8, int(float(size[0]) / max(max_x - min_x, 1e-6) * (plot[2] - plot[0])))
            sy = max(8, int(float(size[1]) / max(max_y - min_y, 1e-6) * (plot[3] - plot[1])))
            draw.rectangle((cx - sx // 2, cy - sy // 2, cx + sx // 2, cy + sy // 2), fill="#777777", outline="#333333")
        draw.text((plot[0], plot[3] + 8), "world x/y (m) | blue=trajectory | squares=semantic markers", fill="#555555", font=small_font)
        instruction_lines = textwrap.wrap(str(task.get("instruction") or ""), width=70)
        details = instruction_lines + [
            "",
            f"subgoals {metrics.get('semantic_subgoals_completed')}/{metrics.get('semantic_subgoals_expected')} | path {metrics.get('path_length_m')} m | time {metrics.get('mission_time_sec')} s",
            f"real-image calls {metrics.get('num_omninav_real_image_calls')} | fallback {metrics.get('num_omninav_fallback_calls')} | runtime stale {metrics.get('runtime_stale')}",
        ]
        draw.multiline_text((x0 + 25, top + 510), "\n".join(details), fill="#303134", font=body_font, spacing=4)
        overlay_rows.append(
            {
                "task_id": task_id,
                "category": task["category"],
                "success": bool(metrics.get("success")),
                "trajectory_points": len(trajectory),
                "objects": len(scene.get("objects", [])),
                "oracle_judge_overlay": True,
                "actual_viewport_evidence": False,
                "qualification_evidence": False,
            }
        )
    draw.text(
        (canvas.width // 2, canvas.height - 15),
        "Instrumented markers | ideal-kinematic Isaac | not open-vocabulary perception evidence | qualification=false",
        fill="#555555",
        font=small_font,
        anchor="ms",
    )
    canvas.save(output)
    (output.parent / "overlay_state.json").write_text(
        json.dumps({"schema_version": 1, "panels": overlay_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "panels": overlay_rows}, indent=2))
    return 0


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _circle(draw: ImageDraw.ImageDraw, point: tuple[int, int], radius: int, color: str) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _cross(draw: ImageDraw.ImageDraw, point: tuple[int, int], radius: int, color: str) -> None:
    x, y = point
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=3)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=3)


def _plot_color(value: str) -> str:
    return {
        "red": "#d1495b",
        "blue": "#2f6fdb",
        "green": "#3a8f5b",
        "yellow": "#e6b422",
        "white": "#f5f5f5",
        "gray": "#777777",
    }.get(value.lower(), "#888888")


if __name__ == "__main__":
    raise SystemExit(main())
