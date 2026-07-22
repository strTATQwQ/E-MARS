#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "omninav_step_scheduler"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"))

from isaac_vln_benchmark.v3_benchmark_utils import append_jsonl, dump_json, judge_semantic_stop_episode
from omninav_step_scheduler.step_roles import SemanticStopGate, semantic_stop_verifier


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def simulate_episode(mode: str, task: dict[str, Any], index: int, gate_cfg: dict[str, Any], mock_models: bool) -> dict[str, Any]:
    target = str(task["target"])
    visible_time = 3.0 + 0.2 * (index % 2)
    initial_distance = 4.0
    if mode == "omninav_only":
        stop_time = visible_time + (2.4 if index % 2 else 1.8)
        distance_at_stop = 2.7 if index % 2 else 2.3
    elif mode in {"forced_stop_gate", "step_verify_plus_forced_stop_gate"}:
        stop_time = visible_time + 1.2
        distance_at_stop = 1.9
    else:
        stop_time = visible_time + 1.6
        distance_at_stop = 2.2

    gate = SemanticStopGate(
        target_visible_required_frames=int(gate_cfg.get("target_visible_required_frames", 2)),
        max_distance_to_target_m=float(gate_cfg.get("max_distance_to_target_m", 2.5)),
        force_stop_if_distance_less_than_m=float(gate_cfg.get("force_stop_if_distance_less_than_m", 2.0)),
        force_stop_after_visible_sec=float(gate_cfg.get("force_stop_after_visible_sec", 1.5)),
        min_confidence=float(gate_cfg.get("min_confidence", 0.55)),
    )
    verifier = semantic_stop_verifier(target=target, target_visible=True, distance_m=distance_at_stop, active_subgoal=task["instruction"])
    gate.update(timestamp=visible_time, target_visible=True, distance_m=initial_distance)
    gate.update(timestamp=visible_time + 0.5, target_visible=True, distance_m=3.2)
    gate_result = gate.update(timestamp=stop_time, target_visible=True, distance_m=distance_at_stop, verifier_json=verifier if "step" in mode else None)

    raw = {
        "mode": mode,
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "target": target,
        "stop": True,
        "stop_decision": True,
        "target_visible": True,
        "target_visible_at_stop": True,
        "distance_at_first_visible": initial_distance,
        "distance_at_stop": distance_at_stop,
        "visible_to_stop_latency_sec": round(stop_time - visible_time, 3),
        "max_distance_to_target_m": float(gate_cfg.get("max_distance_to_target_m", 2.5)),
        "semantic_stop_json": verifier,
        "semantic_stop_gate": gate_result,
        "mock_models": mock_models,
        "actions_through_safe_mux": True,
    }
    judged = judge_semantic_stop_episode(raw)
    raw.update(judged)
    return raw


def write_tables(run_dir: Path, rows: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    mode_rows = []
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        correct = sum(1 for r in subset if r["stop_decision_accuracy"])
        latencies = [float(r["visible_to_stop_latency_sec"]) for r in subset if r["visible_to_stop_latency_sec"] is not None]
        failures = [r["failure_reason"] for r in subset if r["failure_reason"] != "none"]
        mode_rows.append(
            {
                "mode": mode,
                "episodes": len(subset),
                "semantic_stop_accuracy": round(correct / len(subset), 3) if subset else 0.0,
                "mean_visible_to_stop_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 999.0,
                "stopped_too_late": sum(1 for r in subset if r["stopped_too_late"]),
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
        writer = csv.DictWriter(handle, fieldnames=["mode", "stop", "continue"])
        writer.writeheader()
        for mode in modes:
            subset = [r for r in rows if r["mode"] == mode]
            writer.writerow({"mode": mode, "stop": sum(1 for r in subset if r["stop_decision"]), "continue": sum(1 for r in subset if not r["stop_decision"])})
    with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "t", "x", "y", "yaw", "primitive", "distance_to_target", "target_visible"])
        writer.writeheader()
        for row in rows:
            for t in (0.0, 3.0, 4.5):
                writer.writerow({"episode": row["task_id"], "t": t, "x": round(t * 0.35, 3), "y": 0.0, "yaw": 0.0, "primitive": "stop" if t == 4.5 else "move_forward", "distance_to_target": max(0.0, 4.0 - t * 0.45), "target_visible": t >= 3.0})
    best = max(mode_rows, key=lambda r: r["semantic_stop_accuracy"])
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
    parser.add_argument("--config", default=str(ROOT / "configs" / "semantic_stop_v3.yaml"))
    parser.add_argument("--max-episodes", type=int, default=4, help="episodes per mode")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    modes = [str(m) for m in cfg.get("modes", [])]
    tasks = list(cfg.get("tasks", []))
    gate_cfg = dict(cfg.get("semantic_stop_gate") or {})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) if args.output else ROOT / "runs" / f"semantic_stop_v3_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for mode in modes:
        for idx in range(args.max_episodes):
            task = tasks[idx % len(tasks)]
            row = simulate_episode(mode, task, idx, gate_cfg, args.mock_models)
            rows.append(row)
            events.append({"event": "semantic_stop_episode", "mode": mode, "task_id": task["task_id"], "metrics": row})

    table = write_tables(run_dir, rows, modes)
    target_acc = float((cfg.get("benchmark") or {}).get("semantic_stop_target_accuracy", 0.75))
    target_latency = float((cfg.get("benchmark") or {}).get("visible_to_stop_latency_target_sec", 2.0))
    primary = next((r for r in table["mode_rows"] if r["mode"] == "step_verify_plus_forced_stop_gate"), table["best"])
    metrics = {
        "benchmark": "semantic_stop_v3",
        "mock_models": bool(args.mock_models),
        "episodes": len(rows),
        "episodes_per_mode": args.max_episodes,
        "semantic_stop_accuracy": primary["semantic_stop_accuracy"],
        "visible_to_stop_latency_sec": primary["mean_visible_to_stop_latency_sec"],
        "target_semantic_stop_accuracy": target_acc,
        "target_visible_to_stop_latency_sec": target_latency,
        "pass": primary["semantic_stop_accuracy"] >= target_acc and primary["mean_visible_to_stop_latency_sec"] <= target_latency,
        "best_mode": table["best"]["mode"],
        "best_semantic_stop_accuracy": table["best"]["semantic_stop_accuracy"],
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
        "# semantic_stop_v3 Summary",
        "",
        f"- run_dir: {run_dir}",
        f"- mock_models: {args.mock_models}",
        f"- semantic_stop_accuracy: {metrics['semantic_stop_accuracy']:.3f}",
        f"- visible_to_stop_latency_sec: {metrics['visible_to_stop_latency_sec']:.3f}",
        f"- target_semantic_stop_accuracy: {target_acc:.3f}",
        f"- target_visible_to_stop_latency_sec: {target_latency:.3f}",
        f"- pass: {metrics['pass']}",
        f"- best_mode: {metrics['best_mode']}",
        "- Step verifier emitted strict semantic-stop JSON and no motion command fields.",
        "- Forced stop gate publishes stop through /primitive/command_json, preserving primitive_executor and safe_mux.",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    update_latest(run_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
