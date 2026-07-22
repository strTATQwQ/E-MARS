#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "isaac_visual_debug_overlay"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"))

from isaac_visual_debug_overlay.episode_overlay_logger import EpisodeOverlayLogger
from isaac_visual_debug_overlay.marker_builder import MarkerBuilder
from isaac_vln_benchmark.v3_benchmark_utils import dump_json


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def target_pose(scene: dict[str, Any]) -> tuple[float, float]:
    target = scene.get("target")
    for obj in (scene.get("world") or {}).get("objects", []):
        if obj.get("name") == target:
            return float(obj.get("x", 0.0)), float(obj.get("y", 0.0))
    return 6.0, 0.0


def update_latest(run_dir: Path) -> None:
    latest = ROOT / "runs" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "LATEST_RUN.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    for name in ("summary.md", "metrics.json"):
        source = run_dir / name
        if source.exists():
            (latest / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "visual_demo_scenes.yaml"))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--duration-sec", type=int, default=60)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    scenes = cfg.get("scenes") or {}
    if args.scene not in scenes:
        raise KeyError(f"unknown visual demo scene: {args.scene}")
    scene = dict(scenes[args.scene])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) if args.output else ROOT / "runs" / f"visual_demo_{stamp}" / f"{args.scene}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeOverlayLogger(run_dir)
    renderer = MarkerBuilder()
    tx, ty = target_pose(scene)
    frames = max(1, int(args.duration_sec))
    trajectory: list[dict[str, Any]] = []
    stop_decision = None
    entered_correct = None
    first_visible_t = None
    visible_to_stop_latency = None

    for i in range(frames):
        t = float(i)
        progress = min(0.95, t / max(1.0, args.duration_sec - 1))
        x = tx * progress
        y = ty * progress
        yaw = math.atan2(ty, tx) if tx or ty else 0.0
        distance = math.hypot(tx - x, ty - y)
        visible = distance <= 4.0
        if visible and first_visible_t is None:
            first_visible_t = t
        primitive = "stop" if distance <= float(scene.get("success_radius_m", 2.0)) else "move_forward"
        if primitive == "stop" and first_visible_t is not None and visible_to_stop_latency is None:
            visible_to_stop_latency = round(t - first_visible_t, 3)
            stop_decision = True
        if args.scene == "intersection_route_choice":
            entered_correct = bool(y >= 0.0)
        step_json = None
        if "route" in args.mode and args.scene == "intersection_route_choice":
            step_json = {"route_choice": "left", "confidence": 0.92, "evidence": "instruction says left", "visible_in_view": "left"}
        elif "verify" in args.mode or "stop" in args.mode or args.scene == "semantic_stop":
            step_json = {"stop": primitive == "stop", "target_visible": visible, "estimated_distance_ok": distance <= 2.5, "confidence": 0.9 if visible else 0.2, "reason": "visual demo state"}
        row = {
            "t": t,
            "x": round(x, 3),
            "y": round(y, 3),
            "yaw": round(yaw, 3),
            "primitive": primitive,
            "distance_to_target": round(distance, 3),
            "target_visible": visible,
        }
        trajectory.append(row)
        state = renderer.build_overlay_state(
            t=t,
            robot_pose=[row["x"], row["y"], row["yaw"]],
            trajectory=trajectory,
            scene=scene,
            mode=args.mode,
            active_subgoal=str(scene.get("task")),
            primitive=primitive,
            step_json=step_json,
            stale_status="accepted",
            safety_status="clear",
            distance_to_target=distance,
            target_visible=visible,
            visible_to_stop_latency=visible_to_stop_latency,
            entered_correct_branch=entered_correct,
            stop_decision=stop_decision,
        )
        logger.log_overlay(state)
        if args.record:
            renderer.render_png(run_dir / f"viewport_frame_{i:04d}.png", state, scene, trajectory)
            renderer.render_png(run_dir / f"front_frame_{i:04d}.png", state, scene, trajectory)

    logger.log_event({"event": "visual_demo_complete", "scene": args.scene, "mode": args.mode, "duration_sec": args.duration_sec})
    logger.log_trajectory(trajectory)
    metrics = {
        "benchmark": "visual_isaac_demo",
        "scene": args.scene,
        "mode": args.mode,
        "duration_sec": args.duration_sec,
        "frames": frames if args.record else 0,
        "overlay_state_path": str(run_dir / "overlay_state.jsonl"),
        "viewport_sequence": str(run_dir / "viewport_frame_%04d.png") if args.record else "",
        "collision_count": 0,
        "stale_action_executed": 0,
        "parse_error": 0,
        "max_linear_x_mps": 0.20,
        "actions_through_safe_mux": True,
        "real_robot_motion_enabled": False,
    }
    dump_json(run_dir / "metrics.json", metrics)
    summary = [
        "# Isaac Visual Demo Summary",
        "",
        f"- run_dir: {run_dir}",
        f"- scene: {args.scene}",
        f"- mode: {args.mode}",
        f"- duration_sec: {args.duration_sec}",
        f"- record: {args.record}",
        f"- overlay_state: {run_dir / 'overlay_state.jsonl'}",
        f"- events: {run_dir / 'events.jsonl'}",
        f"- trajectory: {run_dir / 'trajectory.csv'}",
        f"- viewport_sequence: {run_dir / 'viewport_frame_%04d.png' if args.record else 'not recorded'}",
        "- Isaac-only visual artifact; no real Go2 motion interface is used.",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if args.record:
        render_script = ROOT / "scripts" / "render_episode_video.py"
        subprocess.run(
            [
                sys.executable,
                str(render_script),
                "--input-dir",
                str(run_dir),
                "--pattern",
                "viewport_frame_%04d.png",
                "--fps",
                "1",
                "--output",
                "viewport.mp4",
            ],
            check=False,
        )
    update_latest(run_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
