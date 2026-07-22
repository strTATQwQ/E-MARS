from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class EpisodeOverlayLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_path = self.output_dir / "overlay_state.jsonl"
        self.events_path = self.output_dir / "events.jsonl"
        self.trajectory_path = self.output_dir / "trajectory.csv"
        self._trajectory_header_written = False

    def log_overlay(self, state: dict[str, Any]) -> None:
        _append_jsonl(self.overlay_path, state)

    def log_event(self, event: dict[str, Any]) -> None:
        _append_jsonl(self.events_path, event)

    def log_trajectory(self, rows: list[dict[str, Any]]) -> None:
        fieldnames = ["t", "x", "y", "yaw", "primitive", "distance_to_target", "target_visible"]
        with self.trajectory_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not self._trajectory_header_written and self.trajectory_path.stat().st_size == 0:
                writer.writeheader()
                self._trajectory_header_written = True
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_summary(self, text: str) -> None:
        (self.output_dir / "summary.md").write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
