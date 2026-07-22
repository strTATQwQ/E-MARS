#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.benchmark_runner import run_benchmark
from isaac_vln_benchmark.config_loader import dump_data, load_data
from isaac_vln_benchmark.diagnostics import write_diagnostic_artifacts


def remote_output_path(local_output: Path) -> str:
    try:
        rel = local_output.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        rel = Path("runs") / local_output.name
    return f"/home/song/dgx-unitree/isaac_vln_benchmark/{rel.as_posix()}"


def putty_args(binary: str, args: argparse.Namespace) -> list[str]:
    cmd = [binary, "-batch"]
    if args.isaac_hostkey:
        cmd += ["-hostkey", args.isaac_hostkey]
    cmd += ["-pw", args.isaac_password]
    return cmd


def select_tasks(project_root: Path, success_cfg: dict[str, Any], max_episodes: int | None) -> list[dict[str, Any]]:
    tasks_source = str(success_cfg.get("benchmark", {}).get("tasks_source") or "configs/tasks.yaml")
    tasks_path = Path(tasks_source)
    if not tasks_path.is_absolute():
        tasks_path = project_root / tasks_path
    tasks_doc = load_data(tasks_path)
    all_tasks = tasks_doc.get("tasks", [])
    selected: list[dict[str, Any]] = []
    task_cfg = success_cfg.get("tasks", {})
    for task_type, cfg in task_cfg.items():
        rows = [dict(task) for task in all_tasks if task.get("task_type") == task_type]
        wanted_ids = [str(value) for value in cfg.get("ids", [])]
        if wanted_ids:
            by_id = {str(task.get("task_id")): task for task in rows}
            missing = [task_id for task_id in wanted_ids if task_id not in by_id]
            if missing:
                raise KeyError(f"Unknown {task_type} task ids: {missing}")
            rows = [dict(by_id[task_id]) for task_id in wanted_ids]
        for row in rows[: int(cfg.get("count", len(rows)))]:
            row["timeout_sec"] = int(cfg.get("timeout_sec", row.get("timeout_sec", success_cfg.get("benchmark", {}).get("timeout_sec_default", 120))))
            selected.append(row)
    seeds = success_cfg.get("benchmark", {}).get("seeds", [0])
    preserve_task_ids = bool(success_cfg.get("benchmark", {}).get("preserve_task_ids", False))
    preserve_task_seeds = bool(success_cfg.get("benchmark", {}).get("preserve_task_seeds", False))
    expanded = []
    for seed in seeds:
        for task in selected:
            row = dict(task)
            row["seed"] = int(task.get("seed", seed)) if preserve_task_seeds else seed
            if not preserve_task_ids:
                row["task_id"] = f"{task['task_id']}_seed{seed}"
            expanded.append(row)
    if max_episodes is not None:
        expanded = expanded[:max_episodes]
    return expanded


def write_selected_tasks(output: Path, tasks: list[dict[str, Any]]) -> Path:
    path = output / "selected_tasks.yaml"
    dump_data({"tasks": tasks}, path)
    return path


def run_local(
    args: argparse.Namespace,
    success_cfg: dict[str, Any],
    selected_tasks: Path,
    scenes_path: Path | None,
) -> dict[str, Any]:
    benchmark = success_cfg.get("benchmark", {})
    modes = args.modes or benchmark.get("modes") or ["internnav_only", "step_internnav_event"]
    per_mode_episodes = 10_000
    if args.max_episodes is not None:
        per_mode_episodes = max(1, math.ceil(args.max_episodes / max(len(modes), 1)))
    ns = Namespace(
        modes=modes,
        tasks=str(selected_tasks),
        scenes=None if scenes_path is None else str(scenes_path),
        config=None,
        ablation_modes=None,
        delay_profiles=None,
        delay_profile="none",
        num_episodes=per_mode_episodes,
        output=str(args.output),
        use_isaac=False,
        mock_models=True,
        ros_domain_id=int(os.environ.get("ROS_DOMAIN_ID", "0")),
        seed=42,
        project_root=str(PROJECT_ROOT),
        live_startup_timeout_sec=20.0,
        live_settle_sec=1.0,
        live_episode_timeout_sec=0.0,
    )
    return run_benchmark(ns)


def fetch_remote_output(args: argparse.Namespace, remote_output: str) -> None:
    plink = shutil.which("plink")
    pscp = shutil.which("pscp")
    if not plink or not pscp:
        return
    remote_tar = f"{remote_output.rstrip('/')}/report_artifacts.tgz"
    pack_cmd = (
        f"cd {remote_output} && items='out'; "
        "[ -d step_snapshots ] && items=\"$items step_snapshots\"; "
        "[ -d qualification_bag ] && items=\"$items qualification_bag\"; "
        f"tar -czf {remote_tar} $items"
    )
    subprocess.check_call(putty_args(plink, args) + [f"{args.isaac_user}@{args.isaac_host}", pack_cmd])
    local_tar = args.output / "remote_report_artifacts.tgz"
    subprocess.check_call(
        putty_args(pscp, args) + [f"{args.isaac_user}@{args.isaac_host}:{remote_tar}", str(local_tar)]
    )
    extract_dir = args.output / "_remote"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as archive:
        archive.extractall(extract_dir)
    remote_out = extract_dir / "out"
    if remote_out.exists():
        for child in remote_out.iterdir():
            target = args.output / child.name
            if child.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
    remote_snapshots = extract_dir / "step_snapshots"
    if remote_snapshots.exists():
        target = args.output / "step_snapshots"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(remote_snapshots, target)


def run_remote(
    args: argparse.Namespace,
    success_cfg: dict[str, Any],
    selected_tasks: Path,
    scenes_path: Path | None,
) -> str:
    host = args.isaac_host
    if not host:
        raise SystemExit("--isaac-host is required for live mode without --mock-models")
    modes = args.modes or success_cfg.get("benchmark", {}).get("modes") or ["internnav_only", "step_internnav_event"]
    stop_remote_omninav_client_for_mock(args, modes)
    total_requested = args.max_episodes or len(select_tasks(PROJECT_ROOT, success_cfg, None)) * max(len(modes), 1)
    max_episodes = max(1, math.ceil(total_requested / max(len(modes), 1)))
    remote_output = remote_output_path(args.output)
    plink = shutil.which("plink")
    pscp = shutil.which("pscp")
    if not plink or not pscp:
        raise SystemExit("plink and pscp are required for remote live benchmark on Windows")
    subprocess.check_call(putty_args(plink, args) + [f"{args.isaac_user}@{host}", f"mkdir -p {remote_output}"])
    transfer = [str(selected_tasks), str(args.output / "config.yaml")]
    remote_scenes = ""
    if scenes_path is not None:
        copied_scenes = args.output / "selected_scenes.yaml"
        shutil.copy2(scenes_path, copied_scenes)
        transfer.append(str(copied_scenes))
        remote_scenes = f"{remote_output}/selected_scenes.yaml"
    scheduler_env = ""
    if args.scheduler_config:
        scheduler_config = Path(args.scheduler_config)
        if not scheduler_config.is_absolute():
            scheduler_config = PROJECT_ROOT.parent / scheduler_config
        if not scheduler_config.is_file():
            raise FileNotFoundError(f"scheduler config not found: {scheduler_config}")
        copied_scheduler = args.output / "scheduler_config.yaml"
        scheduler_doc = load_data(scheduler_config)
        if args.real_step:
            scheduler_doc.setdefault("model_clients", {}).setdefault("step_http", {})[
                "snapshot_output_dir"
            ] = f"{remote_output}/step_snapshots"
        dump_data(scheduler_doc, copied_scheduler)
        transfer.append(str(copied_scheduler))
        scheduler_env = f"SCHEDULER_CONFIG='{remote_output}/scheduler_config.yaml' "
    subprocess.check_call(
        putty_args(pscp, args) + transfer + [f"{args.isaac_user}@{host}:{remote_output}/"]
    )
    mock_omninav = "0" if args.real_omninav else ("1" if any("omninav" in mode and "internnav" not in mode for mode in modes) else "0")
    camera_contract = success_cfg.get("camera_contract") if isinstance(success_cfg.get("camera_contract"), dict) else {}
    prefer_synthetic_camera = bool(camera_contract.get("omninav_prefer_synthetic_primary", False))
    env = scheduler_env + (
        f"MODES='{ ' '.join(modes) }' "
        f"NUM_EPISODES={int(max_episodes)} "
        f"LIVE_EPISODE_TIMEOUT_SEC=0 "
        f"MOCK_MODELS=0 "
        f"MOCK_STEP={'0' if args.real_step else ('1' if any('step' in mode for mode in modes) else '0')} "
        f"MOCK_OMNINAV={mock_omninav} "
        f"INTERNNAV_SERVER='{args.internnav_server}' "
        f"TASKS_FILE='{remote_output}/selected_tasks.yaml' "
        f"SCENES_FILE='{remote_scenes}' "
        f"BENCHMARK_CONFIG='{remote_output}/config.yaml' "
        f"DECISION_STRESS_ENABLE={'1' if args.decision_stress else '0'} "
        f"DECISION_STRESS_DELAY_SEC={float(args.decision_stress_delay_sec)} "
        f"DECISION_STRESS_DROP_RATE={float(args.decision_stress_drop_rate)} "
        f"DECISION_STRESS_PROFILE='{args.decision_stress_profile}' "
        f"DECISION_STRESS_RESET_ON_RAW={'1' if args.decision_stress_reset_on_raw else '0'} "
        f"QUALIFICATION_BAG_ENABLE={'1' if args.qualification_bag else '0'} "
        f"PREFER_SYNTHETIC_CAMERA={'1' if prefer_synthetic_camera else '0'} "
    )
    cmd = env + f"/home/song/dgx-unitree/isaac_vln_benchmark/scripts/run_live_benchmark_epyc.sh {remote_output}"
    completed = subprocess.run(putty_args(plink, args) + [f"{args.isaac_user}@{host}", cmd], check=False)
    (args.output / "remote_launcher_return_code.txt").write_text(f"{completed.returncode}\n", encoding="utf-8")
    fetch_remote_output(args, remote_output)
    return remote_output


def stop_remote_omninav_client_for_mock(args: argparse.Namespace, modes: list[str]) -> None:
    """Prevent stale duplicate OmniNav responses when a mock client is selected.

    The Isaac-side launch intentionally does not start the real OmniNav model client.
    In two-machine runs, however, a previously-started DGX model client can still be
    alive in the same ROS domain. If we also start the mock OmniNav client, every
    request gets two responses; the second one is correctly discarded as
    old_response_after_reset. Stop only the targeted DGX client in mock mode.
    """
    if args.real_omninav:
        return
    if not any("omninav" in str(mode) and "internnav" not in str(mode) for mode in modes):
        return
    if not args.dgx_user or not args.dgx_password:
        return
    plink = shutil.which("plink")
    if not plink:
        return
    dgx_host = os.environ.get("DGX_HOST", "10.100.100.128")
    remote_cmd = "pkill -f '[o]mninav_model_client_node' 2>/dev/null || true"
    subprocess.call([plink, "-batch", "-ssh", f"{args.dgx_user}@{dgx_host}", "-pw", args.dgx_password, remote_cmd])


def ensure_audit(args: argparse.Namespace) -> None:
    out = Path(args.output)
    audit_path = out / "internnav_model_audit.json"
    if args.internnav_server:
        audit_script = PROJECT_ROOT / "scripts" / "audit_internnav_model.py"
        cmd = [sys.executable, str(audit_script), "--server", args.internnav_server, "--output", str(audit_path)]
        cmd += ["--summary-output", str(out / "internnav_model_audit_summary.md")]
        if args.dgx_user:
            cmd += ["--ssh-user", args.dgx_user]
        if args.dgx_password:
            cmd += ["--ssh-password", args.dgx_password]
        subprocess.call(cmd)
    elif not audit_path.exists():
        audit_path.write_text(json.dumps({"server_url": "unknown", "latency_source_assessment": "unknown; requires manual confirmation"}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--isaac-host", default=None)
    parser.add_argument("--isaac-user", default="song")
    parser.add_argument("--isaac-password", default="a")
    parser.add_argument("--isaac-hostkey", default=os.environ.get("ISAAC_HOSTKEY", ""))
    parser.add_argument("--internnav-server", default="")
    parser.add_argument("--dgx-user", default=None)
    parser.add_argument("--dgx-password", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--real-omninav", action="store_true")
    parser.add_argument("--real-step", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--scheduler-config", default="")
    parser.add_argument("--decision-stress", action="store_true")
    parser.add_argument("--decision-stress-delay-sec", type=float, default=0.0)
    parser.add_argument("--decision-stress-drop-rate", type=float, default=0.0)
    parser.add_argument("--decision-stress-profile", default="none")
    parser.add_argument("--decision-stress-reset-on-raw", action="store_true")
    parser.add_argument("--qualification-bag", action="store_true")
    args = parser.parse_args(argv)
    args.output = Path(args.output)
    if not args.output.is_absolute():
        if args.output.parts and args.output.parts[0] == PROJECT_ROOT.name:
            args.output = PROJECT_ROOT.parent / args.output
        else:
            args.output = PROJECT_ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    success_cfg = load_data(config_path)
    selected = select_tasks(PROJECT_ROOT, success_cfg, args.max_episodes)
    selected_path = write_selected_tasks(args.output, selected)
    scenes_source = str(success_cfg.get("runtime", {}).get("scenes_source") or "").strip()
    scenes_path = None
    if scenes_source:
        scenes_path = Path(scenes_source)
        if not scenes_path.is_absolute():
            scenes_path = PROJECT_ROOT / scenes_path
        if not scenes_path.is_file():
            raise FileNotFoundError(f"scenes source not found: {scenes_path}")
    dump_data(success_cfg, args.output / "config.yaml")

    if args.mock_models:
        metrics = run_local(args, success_cfg, selected_path, scenes_path)
        ensure_audit(args)
        write_diagnostic_artifacts(args.output, success_cfg)
        print(json.dumps({"output": metrics["output_dir"], "episodes": len(metrics["episodes"])}, indent=2))
    else:
        remote_output = run_remote(args, success_cfg, selected_path, scenes_path)
        ensure_audit(args)
        write_diagnostic_artifacts(args.output, success_cfg)
        print(json.dumps({"output": str(args.output), "remote": args.isaac_host, "remote_output": remote_output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
