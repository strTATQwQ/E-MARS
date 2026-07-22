#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def command(value: list[str]) -> dict:
    try:
        result = subprocess.run(value, text=True, capture_output=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"returncode": -1, "error": f"{type(exc).__name__}:{exc}"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture DGX model-server hardware and runtime evidence.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-variant", required=True)
    parser.add_argument("--precision-mode", required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--pid", action="append", type=int, default=[])
    args = parser.parse_args()
    packages = {}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "pyzmq",
        "qwen-vl-utils",
        "imageio",
        "imageio-ffmpeg",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    configs = []
    for value in args.config:
        path = Path(value).resolve()
        configs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    processes = []
    for pid in args.pid:
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists():
            processes.append({"pid": pid, "cmdline": cmdline.read_bytes().replace(b"\0", b" ").decode().strip()})
    payload = {
        "schema_version": 1,
        "captured_at": time.time(),
        "model_variant": args.model_variant,
        "precision_mode": args.precision_mode,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "configs": configs,
        "processes": processes,
        "uname": command(["uname", "-a"]),
        "memory": command(["free", "-b"]),
        "nvidia_smi_gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_processes": command(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
        ),
        "environment_allowlist": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM")
            if os.environ.get(key) is not None
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "model_variant": args.model_variant}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
