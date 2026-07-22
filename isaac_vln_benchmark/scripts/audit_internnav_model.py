#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def open_json(url: str, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else None
    except Exception:
        return None


def run_command(args: list[str], *, timeout: float = 5.0) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception:
        return ""


def ssh_command(host: str, user: str | None, password: str | None, command: str) -> str:
    if not user:
        return ""
    if password and shutil.which("plink"):
        return run_command(["plink", "-batch", "-pw", password, f"{user}@{host}", command])
    if shutil.which("ssh"):
        return run_command(["ssh", "-o", "BatchMode=yes", f"{user}@{host}", command])
    return ""


def parse_pid_and_cmd(ps_output: str) -> tuple[int | None, str]:
    for line in ps_output.splitlines():
        if "start_server.py" in line or "internnav" in line.lower():
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            try:
                return int(parts[0]), parts[1] if len(parts) > 1 else line
            except ValueError:
                continue
    return None, ""


def infer_from_cmd(cmd: str) -> dict[str, Any]:
    lower = cmd.lower()
    model_name = "unknown"
    model_class = "unknown"
    internnav_mode = "unknown"
    checkpoint = "unknown"
    config_files: list[str] = []
    words = cmd.split()
    for i, word in enumerate(words):
        if word in {"--config", "-c"} and i + 1 < len(words):
            config_files.append(words[i + 1])
        if word in {"--ckpt", "--checkpoint", "--ckpt_path"} and i + 1 < len(words):
            checkpoint = words[i + 1]
    if "cma" in lower:
        model_name = "cma"
        model_class = "CmaAgent"
        internnav_mode = "system1"
    if "internvla" in lower:
        model_name = "InternVLA-N1"
        model_class = "InternVLAN1Agent"
        internnav_mode = "dual" if "dual" in lower else "system2"
    if "navdp" in lower or "rdp" in lower:
        model_class = "RdpAgent"
        internnav_mode = "system1"
    return {
        "checkpoint_path": checkpoint,
        "model_name": model_name,
        "model_class": model_class,
        "internnav_mode": internnav_mode,
        "config_files": config_files,
    }


def repo_commit(repo: Path) -> str:
    if not repo.exists():
        return "unknown"
    out = run_command(["git", "-C", str(repo), "rev-parse", "HEAD"])
    return out.strip() or "unknown"


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(args.server)
    host = parsed.hostname or ""
    openapi = open_json(args.server.rstrip("/") + "/openapi.json")
    ps_output = ssh_command(
        host,
        args.ssh_user,
        args.ssh_password,
        "pgrep -af 'scripts/eval/start_server.py|start_server.py|internnav' || true",
    )
    pid, cmd = parse_pid_and_cmd(ps_output)
    inferred = infer_from_cmd(cmd)
    device_output = ssh_command(
        host,
        args.ssh_user,
        args.ssh_password,
        "python3 - <<'PY'\ntry:\n import torch\n print('cuda' if torch.cuda.is_available() else 'cpu')\nexcept Exception:\n print('unknown')\nPY",
    ).strip()
    device = device_output.splitlines()[-1] if device_output else "unknown"
    gpu_used = device == "cuda"
    commit = repo_commit(Path(args.repo)) if args.repo else "unknown"
    routes = sorted((openapi or {}).get("paths", {}).keys()) if openapi else []
    return {
        "server_url": args.server,
        "server_pid": pid,
        "checkpoint_path": inferred["checkpoint_path"],
        "model_name": inferred["model_name"],
        "model_class": inferred["model_class"],
        "internnav_mode": inferred["internnav_mode"],
        "device": device,
        "parameter_count": None,
        "gpu_used": gpu_used,
        "rgb_input_shape": [256, 144, 3],
        "depth_input_shape": [256, 144],
        "depth_normalization": "CMA_0_1",
        "action_space": ["forward", "left", "right", "stop"],
        "action_mapping": {
            "forward": "move_forward",
            "left": "turn_left",
            "right": "turn_right",
            "stop": "stop",
            "unknown": "stop",
        },
        "history_length": None,
        "uses_rgb": True,
        "uses_depth": True,
        "uses_instruction": True,
        "repo_commit": commit,
        "config_files": inferred["config_files"],
        "server_routes": routes,
        "process_command": cmd or "unknown",
        "latency_source_assessment": (
            "unknown; requires manual confirmation"
            if inferred["model_class"] in {"unknown", "CmaAgent", "RdpAgent"}
            else "may be full InternVLA-N1; requires manual confirmation"
        ),
    }


def write_summary(path: Path, audit: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# InternNav Model Audit",
            "",
            f"- server_url: `{audit['server_url']}`",
            f"- server_pid: `{audit['server_pid']}`",
            f"- model_name: `{audit['model_name']}`",
            f"- model_class: `{audit['model_class']}`",
            f"- internnav_mode: `{audit['internnav_mode']}`",
            f"- device: `{audit['device']}`",
            f"- checkpoint_path: `{audit['checkpoint_path']}`",
            f"- latency_source_assessment: `{audit['latency_source_assessment']}`",
            "",
            "The current 40 ms latency must not be reported as full InternVLA-N1 unless the model class and checkpoint are independently confirmed.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2] / "InternNav"))
    parser.add_argument("--ssh-user", default=None)
    parser.add_argument("--ssh-password", default=None)
    args = parser.parse_args(argv)
    audit = build_audit(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = Path(args.summary_output) if args.summary_output else out.parent / "summary.md"
    write_summary(summary_path, audit)
    print(json.dumps({"output": str(out), "model_class": audit["model_class"], "internnav_mode": audit["internnav_mode"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
