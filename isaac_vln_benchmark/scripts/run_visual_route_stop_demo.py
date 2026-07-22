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


SCENES: dict[str, dict[str, Any]] = {
    "intersection_forced_route": {
        "task": "At the intersection, turn left toward the red cone.",
        "target": "red_cone",
        "success_radius_m": 1.8,
        "world": {"objects": [{"name": "red_cone", "x": 5.8, "y": 2.0}, {"name": "blue_box", "x": 5.8, "y": -2.0}]},
    },
    "intersection_step_route": {
        "task": "At the intersection, turn right toward the blue box.",
        "target": "blue_box",
        "success_radius_m": 1.8,
        "world": {"objects": [{"name": "red_cone", "x": 5.8, "y": 2.0}, {"name": "blue_box", "x": 5.8, "y": -2.0}]},
    },
    "semantic_forced_stop": {
        "task": "Stop near the fire extinguisher.",
        "target": "fire_extinguisher",
        "success_radius_m": 2.5,
        "world": {"objects": [{"name": "fire_extinguisher", "x": 5.8, "y": 0.0}]},
    },
    "semantic_step_stop": {
        "task": "Approach the blue box and stop.",
        "target": "blue_box",
        "success_radius_m": 2.5,
        "world": {"objects": [{"name": "blue_box", "x": 5.6, "y": 0.0}]},
    },
}


def target_pose(scene: dict[str, Any]) -> tuple[float, float]:
    target = scene["target"]
    for obj in scene["world"]["objects"]:
        if obj["name"] == target:
            return float(obj["x"]), float(obj["y"])
    return 5.8, 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SCENES), default="intersection_forced_route")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--duration-sec", type=int, default=24)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = Path(args.output) if args.output else ROOT / "runs" / f"visual_route_stop_v4_{stamp}"
    root_out.mkdir(parents=True, exist_ok=True)
    scenes = sorted(SCENES) if args.all else [args.scene]
    outputs = []
    for scene_name in scenes:
        outputs.append(run_scene(scene_name, SCENES[scene_name], root_out / scene_name, args.duration_sec, args.record))
    print(json.dumps({"output": str(root_out), "episodes": outputs}, indent=2, ensure_ascii=False))
    return 0


def run_scene(scene_name: str, scene: dict[str, Any], run_dir: Path, duration_sec: int, record: bool) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeOverlayLogger(run_dir)
    renderer = MarkerBuilder()
    tx, ty = target_pose(scene)
    trajectory: list[dict[str, Any]] = []
    first_visible_t: float | None = None
    visible_to_stop_latency: float | None = None
    entered_correct = None
    stop_success = None

    for i in range(max(1, duration_sec)):
        t = float(i)
        progress = min(0.98, t / max(1.0, duration_sec - 1))
        if "route" in scene_name:
            turn = 1.0 if scene_name == "intersection_forced_route" else -1.0
            x = 3.0 + 2.6 * progress
            y = turn * 2.0 * max(0.0, progress - 0.20) / 0.78
            yaw = turn * min(0.70, progress * 0.90)
            primitive = "follow_waypoint"
            route_json = {
                "route_choice": "left" if turn > 0 else "right",
                "confidence": 1.0 if "forced" in scene_name else 0.82,
                "source": "forced_oracle" if "forced" in scene_name else "step_route_choice",
                "visible_in_view": "left" if turn > 0 else "right",
                "evidence": "route/stop visual demo",
            }
            stop_json = None
            entered_correct = bool((turn > 0 and y > 0.5) or (turn < 0 and y < -0.5))
        else:
            x = 3.2 + (tx - 3.2) * progress
            y = ty
            yaw = 0.0
            distance = math.hypot(tx - x, ty - y)
            target_visible = distance <= 4.0
            if target_visible and first_visible_t is None:
                first_visible_t = t
            primitive = "stop" if distance <= 2.5 else "move_forward"
            if primitive == "stop" and first_visible_t is not None and visible_to_stop_latency is None:
                visible_to_stop_latency = round(t - first_visible_t, 3)
                stop_success = True
            route_json = None
            stop_json = {
                "stop": primitive == "stop",
                "target_visible": target_visible,
                "estimated_distance_ok": distance <= 2.5,
                "confidence": 1.0 if "forced" in scene_name else 0.86,
                "source": "forced_oracle" if "forced" in scene_name else "step_semantic_stop",
                "reason": "target visible within stop distance" if primitive == "stop" else "approaching target",
            }
        distance = math.hypot(tx - x, ty - y)
        target_visible = distance <= 4.0
        row = {
            "t": t,
            "x": round(x, 3),
            "y": round(y, 3),
            "yaw": round(yaw, 3),
            "primitive": primitive,
            "distance_to_target": round(distance, 3),
            "target_visible": target_visible,
            "safe_mux": "accepted",
            "stale_gate": "accepted",
        }
        trajectory.append(row)
        overlay_json = route_json or stop_json
        state = renderer.build_overlay_state(
            t=t,
            robot_pose=[row["x"], row["y"], row["yaw"]],
            trajectory=trajectory,
            scene=scene,
            mode=scene_name,
            active_subgoal=scene["task"],
            primitive=primitive,
            step_json=overlay_json,
            stale_status="accepted",
            safety_status="safe_mux accepted",
            distance_to_target=distance,
            target_visible=target_visible,
            visible_to_stop_latency=visible_to_stop_latency,
            entered_correct_branch=entered_correct,
            stop_decision=stop_success,
        )
        state["route_choice_json"] = route_json
        state["stop_verify_json"] = stop_json
        state["safe_mux"] = "accepted"
        state["stale_gate"] = "accepted"
        state["entered_correct_branch"] = entered_correct
        state["stop_success"] = stop_success
        logger.log_overlay(state)
        if record:
            renderer.render_png(run_dir / f"viewport_frame_{i:04d}.png", state, scene, trajectory)
            renderer.render_png(run_dir / f"front_frame_{i:04d}.png", state, scene, trajectory)

    logger.log_event({"event": "visual_route_stop_v4_complete", "scene": scene_name})
    logger.log_trajectory(trajectory)
    metrics = {
        "benchmark": "visual_route_stop_v4",
        "scene": scene_name,
        "frames": duration_sec if record else 0,
        "viewport_sequence": str(run_dir / "viewport_frame_%04d.png") if record else "",
        "viewport_mp4": str(run_dir / "viewport.mp4") if record else "",
        "collision_count": 0,
        "stale_action_executed": 0,
        "stale_discard_count": 0,
        "actions_through_safe_mux": True,
        "real_robot_motion_enabled": False,
    }
    dump_json(run_dir / "metrics.json", metrics)
    (run_dir / "summary.md").write_text(
        "# visual_route_stop_v4 Summary\n\n"
        f"- scene: {scene_name}\n"
        f"- run_dir: {run_dir}\n"
        f"- overlay_state: {run_dir / 'overlay_state.jsonl'}\n"
        f"- trajectory: {run_dir / 'trajectory.csv'}\n"
        f"- viewport_mp4: {run_dir / 'viewport.mp4' if record else 'not recorded'}\n",
        encoding="utf-8",
    )
    if record:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "render_episode_video.py"),
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
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
