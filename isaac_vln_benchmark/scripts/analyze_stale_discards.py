#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "ros2_ws" / "src" / "isaac_vln_benchmark"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from isaac_vln_benchmark.v4_benchmark_utils import analyze_stale_events, write_stale_analysis
from isaac_vln_benchmark.v5_timebase_utils import V5_ATTRIBUTIONS, analyze_stale_events_v5


def load_jsonl_limited(path: Path, max_records: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if max_records > 0 and len(rows) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def resolve_events_path(source: Path) -> Path:
    if source.is_file():
        return source
    for candidate in (source / "events.jsonl", source / "out" / "events.jsonl"):
        if candidate.exists():
            return candidate
    return source / "events.jsonl"


def resolve_metrics_path(source: Path, explicit: str) -> Path | None:
    if explicit:
        return Path(explicit)
    if not source.is_dir():
        return None
    for candidate in (source / "metrics.json", source / "out" / "metrics.json"):
        if candidate.exists():
            return candidate
    return source / "metrics.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="run directory or events.jsonl")
    parser.add_argument("--metrics", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-records", type=int, default=200000)
    parser.add_argument("--v5", action="store_true", help="write v5 attribution names and stale_attribution.csv")
    args = parser.parse_args(argv)

    source = Path(args.input)
    if source.is_dir():
        events_path = resolve_events_path(source)
        output = Path(args.output) if args.output else source / "stale_discard_analysis.md"
    else:
        events_path = source
        output = Path(args.output) if args.output else source.with_name("stale_discard_analysis.md")
    events = load_jsonl_limited(events_path, args.max_records)
    analysis = analyze_stale_events_v5(events) if args.v5 else analyze_stale_events(events)
    metrics = {}
    metrics_path = resolve_metrics_path(source, args.metrics)
    if metrics_path and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    analysis["metrics_stale_discard_count"] = metrics.get("stale_discard_count")
    if args.v5:
        write_v5_stale_analysis(output, analysis)
        write_v5_stale_csv(output.with_name("stale_attribution.csv"), analysis)
        total = analysis.get("total_discards", 0)
    else:
        write_stale_analysis(output, analysis)
        total = analysis.get("total_stale_discards", 0)
    print(json.dumps({"output": str(output), "total_stale_discards": total, "counts": analysis["counts"]}, indent=2))
    return 0


def write_v5_stale_analysis(path: Path, analysis: dict) -> None:
    lines = [
        "# Stale Discard Analysis V5",
        "",
        f"- total_stale_discards: {analysis.get('total_discards', 0)}",
        f"- metrics_stale_discard_count: {analysis.get('metrics_stale_discard_count')}",
        "",
        "| attribution | count |",
        "| --- | ---: |",
    ]
    counts = analysis.get("counts") if isinstance(analysis.get("counts"), dict) else {}
    for category in V5_ATTRIBUTIONS:
        lines.append(f"| {category} | {int(counts.get(category, 0))} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v5_stale_csv(path: Path, analysis: dict) -> None:
    counts = analysis.get("counts") if isinstance(analysis.get("counts"), dict) else {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attribution", "count"])
        writer.writeheader()
        for category in V5_ATTRIBUTIONS:
            writer.writerow({"attribution": category, "count": int(counts.get(category, 0))})


if __name__ == "__main__":
    raise SystemExit(main())
