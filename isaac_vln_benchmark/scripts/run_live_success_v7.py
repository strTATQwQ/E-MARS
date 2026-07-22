#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.config_loader import load_data
from isaac_vln_benchmark.v7_value_utils import (
    evaluate_forced_oracle,
    evaluate_golden_smoke,
    evaluate_route_stop,
    evaluate_sim2real_v7,
    evaluate_step_signal,
    load_json,
    postprocess_v7_run,
    render_sim2real_gate,
    render_v7_summary,
    step_effect_counts,
)


MODE_SPECS: dict[str, dict[str, Any]] = {
    "forced_oracle": {
        "canonical": "omninav_forced_route_stop_oracle_v7",
        "config": "live_success_small_v7_forced_oracle.yaml",
        "modes": ["omninav_only_v6_golden", "omninav_forced_route_stop_oracle_v7"],
        "episodes_per_mode": 15,
        "run_kind": "forced_oracle",
    },
    "step_route_only": {
        "canonical": "omninav_step_route_only_v7",
        "config": "live_success_small_v7_step_route.yaml",
        "modes": ["omninav_step_route_only_v7"],
        "episodes_per_mode": 15,
        "run_kind": "step_route_only",
    },
    "step_stop_only": {
        "canonical": "omninav_step_stop_only_v7",
        "config": "live_success_small_v7_step_stop.yaml",
        "modes": ["omninav_step_stop_only_v7"],
        "episodes_per_mode": 15,
        "run_kind": "step_stop_only",
    },
    "step_route_stop": {
        "canonical": "omninav_step_route_stop_v7",
        "config": "live_success_small_v7_step_route_stop.yaml",
        "modes": ["omninav_step_route_stop_v7"],
        "episodes_per_mode": 15,
        "run_kind": "step_route_stop",
    },
}
ALIASES = {spec["canonical"]: key for key, spec in MODE_SPECS.items()}
ALIASES.update({key: key for key in MODE_SPECS})


def add_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", required=True, choices=sorted(ALIASES))
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--forced-oracle-result", default="")
    parser.add_argument("--route-result", default="")
    parser.add_argument("--stop-result", default="")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default="")
    parser.add_argument("--internnav-server", default="http://10.100.100.128:8087")
    parser.add_argument("--dgx-user", default="")
    parser.add_argument("--dgx-password", default="")


def common_remote_args(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = []
    if args.mock_models:
        cmd.append("--mock-models")
    if args.real_omninav:
        cmd.append("--real-omninav")
    if args.isaac_host and not args.mock_models:
        cmd += ["--isaac-host", args.isaac_host, "--isaac-user", args.isaac_user, "--isaac-password", args.isaac_password]
    if args.isaac_hostkey:
        cmd += ["--isaac-hostkey", args.isaac_hostkey]
    if args.internnav_server:
        cmd += ["--internnav-server", args.internnav_server]
    if args.dgx_user:
        cmd += ["--dgx-user", args.dgx_user]
    if args.dgx_password:
        cmd += ["--dgx-password", args.dgx_password]
    return cmd


def run_smoke(args: argparse.Namespace, parent_output: Path, short_mode: str) -> dict[str, Any]:
    smoke_dir = parent_output / f"golden_smoke_before_{short_mode}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_v7_golden_smoke.py"),
        "--output",
        str(smoke_dir),
    ] + common_remote_args(args)
    result = subprocess.run(cmd, check=False)
    metrics = load_json(smoke_dir / "metrics.json")
    evaluation = load_json(smoke_dir / "golden_smoke_result.json")
    if not evaluation:
        metrics = postprocess_v7_run(smoke_dir, run_kind="golden_smoke")
        evaluation = evaluate_golden_smoke(metrics)
    if result.returncode != 0 or not evaluation.get("pass", False):
        raise SystemExit(f"V7 golden smoke failed before {short_mode}: {evaluation}")
    return {"run_dir": str(smoke_dir), "evaluation": evaluation}


def evaluate_forced_prereq(path_text: str) -> dict[str, Any]:
    if not path_text:
        raise SystemExit("--forced-oracle-result is required before Step V7 modes")
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    metrics = load_json(path / "metrics.json")
    if "v7_summary" not in metrics:
        metrics = postprocess_v7_run(path, run_kind="forced_oracle")
    evaluation = evaluate_forced_oracle(metrics)
    if not evaluation.get("pass", False):
        raise SystemExit(f"ORACLE_FULL_NOT_EFFECTIVE: {evaluation}")
    return {"run_dir": str(path), "evaluation": evaluation}


def evaluate_route_stop_prereq(args: argparse.Namespace) -> dict[str, Any]:
    route_positive = False
    stop_positive = False
    details: dict[str, Any] = {}
    if args.route_result:
        path = _run_path(args.route_result)
        metrics = load_json(path / "metrics.json")
        if "v7_summary" not in metrics:
            metrics = postprocess_v7_run(path, run_kind="step_route_only")
        route_eval = evaluate_step_signal(metrics, "omninav_step_route_only_v7", kind="route")
        route_positive = bool(route_eval.get("positive_signal"))
        details["route"] = {"run_dir": str(path), "evaluation": route_eval}
    if args.stop_result:
        path = _run_path(args.stop_result)
        metrics = load_json(path / "metrics.json")
        if "v7_summary" not in metrics:
            metrics = postprocess_v7_run(path, run_kind="step_stop_only")
        stop_eval = evaluate_step_signal(metrics, "omninav_step_stop_only_v7", kind="stop")
        stop_positive = bool(stop_eval.get("positive_signal"))
        details["stop"] = {"run_dir": str(path), "evaluation": stop_eval}
    if not (route_positive or stop_positive):
        raise SystemExit("OMNINAV_STEP_VALUE_REMAINS_UNPROVEN: route-only and stop-only positive signals are missing")
    return details


def benchmark_cmd(args: argparse.Namespace, spec: dict[str, Any], out_dir: Path) -> list[str]:
    max_episodes_total = int(spec["episodes_per_mode"]) * len(spec["modes"])
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_live_success_benchmark.py"),
        "--config",
        str(ROOT / "configs" / spec["config"]),
        "--modes",
        *spec["modes"],
        "--max-episodes",
        str(max_episodes_total),
        "--output",
        str(out_dir),
    ] + common_remote_args(args)


def write_gate(out_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    gate = load_data(ROOT / "configs" / "sim2real_gate_v7.yaml")
    result = evaluate_sim2real_v7(metrics, gate)
    (out_dir / "sim2real_gate_v7.md").write_text(render_sim2real_gate(result, metrics), encoding="utf-8")
    (out_dir / "sim2real_gate_v7.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def evaluate_current(metrics: dict[str, Any], short_mode: str) -> dict[str, Any]:
    if short_mode == "forced_oracle":
        return evaluate_forced_oracle(metrics)
    if short_mode == "step_route_only":
        return evaluate_step_signal(metrics, "omninav_step_route_only_v7", kind="route")
    if short_mode == "step_stop_only":
        return evaluate_step_signal(metrics, "omninav_step_stop_only_v7", kind="stop")
    if short_mode == "step_route_stop":
        return evaluate_route_stop(metrics)
    raise KeyError(short_mode)


def _run_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one gated V7 live Isaac success experiment.")
    add_live_args(parser)
    args = parser.parse_args(argv)
    short_mode = ALIASES[args.mode]
    spec = MODE_SPECS[short_mode]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "runs" / f"v7_{short_mode}_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prereq: dict[str, Any] = {}
    if not args.skip_smoke:
        prereq["golden_smoke"] = run_smoke(args, out_dir, short_mode)
    if short_mode in {"step_route_only", "step_stop_only", "step_route_stop"}:
        prereq["forced_oracle"] = evaluate_forced_prereq(args.forced_oracle_result)
    if short_mode == "step_route_stop":
        prereq["component_signals"] = evaluate_route_stop_prereq(args)

    result = subprocess.run(benchmark_cmd(args, spec, out_dir), check=False)
    metrics = postprocess_v7_run(out_dir, run_kind=str(spec["run_kind"]))
    evaluation = evaluate_current(metrics, short_mode)
    evaluation["prerequisites"] = prereq
    evaluation["step_effect_counts"] = step_effect_counts(metrics)
    gate = write_gate(out_dir, metrics)
    evaluation["sim2real_gate"] = gate
    decision = "OMNINAV_STEP_VALUE_REMAINS_UNPROVEN"
    if short_mode == "forced_oracle" and not evaluation.get("pass", False):
        decision = "ORACLE_FULL_NOT_EFFECTIVE"
    elif short_mode == "step_route_stop" and evaluation.get("pass", False):
        decision = "OMNINAV_STEP_SHOWS_VALUE_IN_ISAAC_ROUTE_SEMANTIC_TASKS"
    evaluation["final_decision"] = decision
    (out_dir / "v7_evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_v7_summary(out_dir, metrics, evaluation, title=f"V7 {short_mode}"), encoding="utf-8")
    print(json.dumps({"output": str(out_dir), "evaluation": evaluation}, indent=2, ensure_ascii=False))
    if result.returncode != 0:
        return result.returncode
    return 0 if evaluation.get("pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
