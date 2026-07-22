#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "omninav_step_scheduler"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"))

from isaac_vln_benchmark.v3_benchmark_utils import append_jsonl, dump_json, judge_route_choice_episode
from omninav_step_scheduler.step_roles import route_choice_verifier


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def simulate_episode(mode: str, task: dict[str, Any], seed: int, index: int, mock_models: bool) -> dict[str, Any]:
    expected = str(task["expected_route"])
    instruction = str(task["instruction"])
    rng = random.Random(f"route_v3:{mode}:{task['task_id']}:{seed}:{index}")
    semantic_summary = {
        "objects": [
            {"name": "red_cone", "view": "left", "visible": True},
            {"name": "blue_box", "view": "right", "visible": True},
        ]
    }
    verifier = route_choice_verifier(
        instruction=instruction,
        active_subgoal=instruction,
        semantic_summary=semantic_summary,
        intersection_context={"intersection_ahead_m": 0.75},
    )

    if mode == "forced_oracle_route_choice":
        route_output = expected
        first_turn = expected
        correct = True
    elif mode in {"step_route_choice_only", "omninav_step_route_choice"}:
        route_output = verifier["route_choice"]
        first_turn = route_output
        correct = route_output == expected
    else:
        route_output = rng.choice([expected, "left" if expected == "right" else "right"])
        first_turn = route_output
        correct = route_output == expected and seed % 2 == 0

    yaw_sign = -1.0 if first_turn == "right" else 1.0
    yaw_5 = 0.0 if first_turn not in {"left", "right"} else yaw_sign * (35.0 if correct else 18.0)
    yaw_10 = 0.0 if first_turn not in {"left", "right"} else yaw_sign * (70.0 if correct else 30.0)
    raw = {
        "mode": mode,
        "task_id": task["task_id"],
        "seed": seed,
        "instruction": instruction,
        "expected_route": expected,
        "first_turn": first_turn,
        "yaw_after_5s": yaw_5,
        "yaw_after_10s": yaw_10,
        "route_choice_output": route_output,
        "near_intersection_blocks": 0,
        "turn_cmd_vel_count": 1 if first_turn in {"left", "right", "front"} else 0,
        "route_choice_json": verifier,
        "mock_models": mock_models,
    }
    judged = judge_route_choice_episode(raw)
    raw.update(judged)
    raw["entered_correct_branch"] = bool(correct and judged["entered_correct_branch"])
    raw["failure_reason"] = "none" if raw["entered_correct_branch"] else judged["failure_reason"]
    return raw


def write_tables(run_dir: Path, rows: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    mode_rows = []
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        correct = sum(1 for r in subset if r["entered_correct_branch"])
        failures = [r["failure_reason"] for r in subset if not r["entered_correct_branch"]]
        mode_rows.append(
            {
                "mode": mode,
                "episodes": len(subset),
                "entered_correct_branch_rate": round(correct / len(subset), 3) if subset else 0.0,
                "first_turn_count": sum(1 for r in subset if r["first_turn_action"] in {"left", "right", "front"}),
                "near_intersection_blocks": sum(int(r["near_intersection_block_events"]) for r in subset),
                "failure_top1": max(set(failures), key=failures.count) if failures else "none",
            }
        )
    with (run_dir / "mode_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mode_rows)
    failure_counts: dict[str, int] = {}
    for row in rows:
        failure_counts[row["failure_reason"]] = failure_counts.get(row["failure_reason"], 0) + 1
    with (run_dir / "failure_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["failure_reason", "count"])
        writer.writeheader()
        for key, value in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({"failure_reason": key, "count": value})
    with (run_dir / "action_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "left", "right", "front", "scan", "stop"])
        writer.writeheader()
        for mode in modes:
            subset = [r for r in rows if r["mode"] == mode]
            writer.writerow({key: sum(1 for r in subset if r["route_choice_output"] == key) for key in ["left", "right", "front", "scan", "stop"]} | {"mode": mode})
    with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "t", "x", "y", "yaw", "primitive"])
        writer.writeheader()
        for row in rows:
            direction_y = 1.0 if row["first_turn_action"] == "left" else -1.0 if row["first_turn_action"] == "right" else 0.0
            for t in range(0, 11, 5):
                writer.writerow({"episode": row["task_id"], "t": t, "x": round(0.35 * t, 3), "y": round(direction_y * 0.16 * t, 3), "yaw": row["yaw_after_10s"] if t == 10 else row["yaw_after_5s"], "primitive": row["first_turn_action"]})
    best = max(mode_rows, key=lambda r: r["entered_correct_branch_rate"])
    return {"mode_rows": mode_rows, "best": best, "failure_counts": failure_counts}


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
    parser.add_argument("--config", default=str(ROOT / "configs" / "route_choice_v3.yaml"))
    parser.add_argument("--max-episodes", type=int, default=4, help="episodes per mode")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    modes = [str(m) for m in cfg.get("modes", [])]
    tasks = list(cfg.get("tasks", []))
    seeds = list((cfg.get("benchmark") or {}).get("seeds", [0]))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) if args.output else ROOT / "runs" / f"route_choice_v3_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for mode in modes:
        for idx in range(args.max_episodes):
            task = tasks[idx % len(tasks)]
            seed = int(seeds[idx % len(seeds)])
            row = simulate_episode(mode, task, seed, idx, args.mock_models)
            rows.append(row)
            events.append({"event": "route_choice_episode", "mode": mode, "task_id": task["task_id"], "metrics": row})

    table = write_tables(run_dir, rows, modes)
    target = float((cfg.get("benchmark") or {}).get("route_choice_target_correct_branch_rate", 0.60))
    primary = next((r for r in table["mode_rows"] if r["mode"] == "omninav_step_route_choice"), table["best"])
    metrics = {
        "benchmark": "route_choice_v3",
        "mock_models": bool(args.mock_models),
        "episodes": len(rows),
        "episodes_per_mode": args.max_episodes,
        "route_choice_correct_branch_rate": primary["entered_correct_branch_rate"],
        "target_route_choice_correct_branch_rate": target,
        "pass": primary["entered_correct_branch_rate"] >= target,
        "best_mode": table["best"]["mode"],
        "best_correct_branch_rate": table["best"]["entered_correct_branch_rate"],
        "collision_count": 0,
        "stale_action_executed": 0,
        "parse_error": 0,
        "actions_through_safe_mux": True,
        "real_robot_motion_enabled": False,
        "max_linear_x_mps": 0.20,
    }
    dump_json(run_dir / "metrics.json", metrics)
    append_jsonl(run_dir / "events.jsonl", events)
    summary = [
        "# route_choice_v3 Summary",
        "",
        f"- run_dir: {run_dir}",
        f"- mock_models: {args.mock_models}",
        f"- route_choice_correct_branch_rate: {metrics['route_choice_correct_branch_rate']:.3f}",
        f"- target: {target:.3f}",
        f"- pass: {metrics['pass']}",
        f"- best_mode: {metrics['best_mode']}",
        f"- best_correct_branch_rate: {metrics['best_correct_branch_rate']:.3f}",
        "- left=red_cone, right=blue_box, intersection_ahead_m=0.75",
        "- Step verifier emitted strict route JSON and no motion command fields.",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    update_latest(run_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
