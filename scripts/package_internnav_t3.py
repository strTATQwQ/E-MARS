#!/usr/bin/env python3
"""Create a deterministic, privacy-scanned T3 source and evidence ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT_NAME = "internnav_t3_deliverables"
SOURCES = (
    "WORKLOG_INTERNNAV_T3.md",
    "docs/internnav_io_contract.md",
    "reports/internnav_t3.md",
    "configs/internnav_t3",
    "internvla_go2_controller",
    "internvla_nav2_adapter",
    "internvla_ros2",
    "internvla_ros2_msgs",
    "results/internnav_t3",
    "tests/test_internvla_nav2_generation_gap.py",
    "tests/test_nav2_oracle_agent_client.py",
    "tests/test_t3_obstacle_manifest.py",
)
SCRIPT_NAMES = (
    "audit_internnav_t3_delivery.py",
    "audit_internnav_go2_clearance.py",
    "build_internnav_t3_comparison.py",
    "build_internnav_t3_clearance_candidates.py",
    "build_t3_continuous_diagnostics_dataset.py",
    "build_t3_obstacle_dataset.py",
    "build_t3_static_maps.py",
    "extract_internnav_per_episode.py",
    "freeze_internnav_t3_continuous_episodes.py",
    "internnav_go2_runtime.py",
    "internvla_go2_continuous_agent_client.py",
    "internvla_ipc_agent_client.py",
    "internvla_nav2_oracle_agent_client.py",
    "package_internnav_t3.py",
    "postprocess_t3_flash.py",
    "preflight_go2_continuous.py",
    "recover_t3_flash_postprocess.py",
    "run_go2_continuous_phase.sh",
    "run_go2_continuous_diagnostics.sh",
    "run_go2_continuous_no_obstacle_canary.sh",
    "run_go2_continuous_no_obstacle_pilot.sh",
    "run_go2_continuous_oracle.sh",
    "run_go2_continuous_canary.sh",
    "run_go2_continuous_pilot.sh",
    "run_go2_obstacle_oracle.sh",
    "run_go2_obstacle_stress.sh",
    "run_go2_controller_fault_tests.py",
    "run_go2_controller_fault_tests.sh",
    "run_go2_flash_baseline_t3.sh",
    "run_internnav_go2_entrypoint.py",
    "sanitize_internnav_log.py",
    "summarize_go2_controller_records.py",
    "summarize_internnav_progress.py",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "build",
    "install",
    "log",
    "logs",
    ".codex-tmp",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip", ".tgz", ".gz", ".pt", ".pth", ".bin"}
PRIVATE_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(
        rb"(?<!\d)(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        rb"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d)"
    ),
    re.compile(rb"/home/(?:song|railgun)(?:/|\b)"),
)


def files_under(workspace: Path) -> list[Path]:
    roots = [workspace / item for item in SOURCES]
    roots.extend(workspace / "scripts" / name for name in SCRIPT_NAMES)
    output: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(workspace)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            output.append(path)
    return sorted(set(output), key=lambda item: item.relative_to(workspace).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = (workspace / "reports/internnav_t3.md").read_text(encoding="utf-8")
    if "Status: `DONE`" in report:
        package_status = "DONE"
    elif "Status: `BLOCKED`" in report:
        package_status = "BLOCKED"
    else:
        raise RuntimeError("refusing final package before the T3 report is DONE or BLOCKED")
    files = files_under(workspace)
    entries = []
    payloads: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(workspace).as_posix()
        data = path.read_bytes()
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(data):
                raise RuntimeError(f"privacy scan rejected {relative}")
        entries.append(
            {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
        payloads.append((f"{ROOT_NAME}/{relative}", data))
    manifest = {
        "schema_version": 1,
        "status": package_status,
        "file_count": len(entries),
        "privacy_scan": "PASS",
        "files": entries,
    }
    payloads.append(
        (
            f"{ROOT_NAME}/MANIFEST.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            archive.writestr(info, data)
    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_path.write_text(f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n")
    print(json.dumps({"status": "PASS", "zip": output.name, "file_count": len(entries)}, sort_keys=True))


if __name__ == "__main__":
    main()
