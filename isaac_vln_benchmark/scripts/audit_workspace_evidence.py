#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
RUNS = PROJECT / "runs"
EVIDENCE = WORKSPACE / "evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_ids() -> set[str]:
    path = EVIDENCE / "canonical_runs.yaml"
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("run_id:"):
            result.add(stripped.split(":", 1)[1].strip().strip("'\""))
    return result


def directory_stats(path: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            size += item.stat().st_size
    return files, size


def audit_remote_copy(run_dir: Path, apply: bool) -> dict:
    remote = run_dir / "_remote"
    remote_out = remote / "out"
    record = {
        "run_id": run_dir.name,
        "remote_path": str(remote),
        "files_compared": 0,
        "matching_files": 0,
        "missing_local": [],
        "different": [],
        "verified_duplicate": False,
        "deleted": False,
        "bytes_reclaimed": 0,
    }
    if not remote_out.is_dir():
        record["reason"] = "missing _remote/out"
        return record
    for source in sorted(path for path in remote_out.rglob("*") if path.is_file()):
        relative = source.relative_to(remote_out)
        target = run_dir / relative
        record["files_compared"] += 1
        if not target.is_file():
            record["missing_local"].append(relative.as_posix())
            continue
        if source.stat().st_size != target.stat().st_size or sha256(source) != sha256(target):
            record["different"].append(relative.as_posix())
            continue
        record["matching_files"] += 1
    record["verified_duplicate"] = bool(
        record["files_compared"]
        and record["matching_files"] == record["files_compared"]
        and not record["missing_local"]
        and not record["different"]
    )
    if apply and record["verified_duplicate"]:
        remote_logs = run_dir / "remote_logs"
        metadata = []
        for source in sorted(path for path in remote.rglob("*") if path.is_file() and remote_out not in path.parents):
            if source.suffix in {".tgz", ".gz"}:
                continue
            relative = source.relative_to(remote)
            target = remote_logs / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            metadata.append({"path": relative.as_posix(), "sha256": sha256(target), "bytes": target.stat().st_size})
        remote_files, remote_bytes = directory_stats(remote)
        record["remote_files_before_delete"] = remote_files
        record["remote_metadata_preserved"] = metadata
        shutil.rmtree(remote)
        for archive in (run_dir / "remote_report_artifacts.tgz", run_dir / "report_artifacts.tgz"):
            if archive.is_file():
                remote_bytes += archive.stat().st_size
                archive.unlink()
        record["deleted"] = True
        record["bytes_reclaimed"] = remote_bytes
    return record


def build_run_index(canonical: set[str]) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in RUNS.iterdir() if path.is_dir() and path.name != "latest"):
        files, size = directory_stats(run_dir)
        metrics_path = run_dir / "metrics.json"
        summary = {}
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                summary = {
                    "episodes": len(metrics.get("episodes", [])),
                    "aggregate_by_mode": metrics.get("aggregate_by_mode", {}),
                }
            except Exception as exc:
                summary = {"metrics_error": repr(exc)}
        rows.append(
            {
                "run_id": run_dir.name,
                "canonical": run_dir.name in canonical,
                "files": files,
                "bytes": size,
                "has_metrics": metrics_path.is_file(),
                **summary,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash-audit benchmark evidence and optionally remove verified remote duplicates.")
    parser.add_argument("--apply-verified-remote-cleanup", action="store_true")
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    canonical = canonical_ids()
    remote_records = [
        audit_remote_copy(run_dir, args.apply_verified_remote_cleanup)
        for run_dir in sorted(path for path in RUNS.iterdir() if (path / "_remote").is_dir())
    ]
    run_index = build_run_index(canonical)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply_verified_remote_cleanup else "dry_run",
        "canonical_run_ids": sorted(canonical),
        "remote_copy_audit": remote_records,
        "summary": {
            "runs": len(run_index),
            "canonical_present": sum(1 for row in run_index if row["canonical"]),
            "remote_copies": len(remote_records),
            "verified_duplicates": sum(1 for row in remote_records if row["verified_duplicate"]),
            "deleted": sum(1 for row in remote_records if row["deleted"]),
            "bytes_reclaimed": sum(int(row["bytes_reclaimed"]) for row in remote_records),
        },
    }
    (EVIDENCE / "duplicate_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (EVIDENCE / "run_index.json").write_text(json.dumps({"schema_version": 1, "runs": run_index}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
