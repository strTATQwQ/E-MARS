"""Exclusive, bounded-schema evidence writer for coordinator-run Oracle jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .contracts import SelectionDecision


class LocalizationEvidenceWriter:
    def __init__(
        self,
        result_dir: Path,
        *,
        record_filename: str,
        summary_filename: str,
    ) -> None:
        self.result_dir = result_dir.resolve()
        if self.result_dir.exists():
            raise FileExistsError("localization_result_dir_must_be_fresh")
        self.result_dir.mkdir(parents=True, exist_ok=False)
        self.record_path = self.result_dir / record_filename
        self.summary_path = self.result_dir / summary_filename
        self.deviation_path = self.result_dir / "localization_deviations.json"
        if len({self.record_path, self.summary_path, self.deviation_path}) != 3:
            raise ValueError("localization_evidence_filenames_must_be_distinct")
        for path in (self.record_path, self.summary_path, self.deviation_path):
            if path.parent != self.result_dir:
                raise ValueError("evidence_path_escape")
        # Reserve the append-only record atomically.  Reuse/append is forbidden.
        self.record_path.open("x", encoding="utf-8").close()
        self._decision_count = 0
        self._output_count = 0
        self._switch_count = 0
        self._deviations: list[dict[str, Any]] = []
        self._deviation_count = 0
        self._deviation_keys: set[str] = set()
        self._selection_counts: dict[str, int] = {}
        self._odometry_deferred_decision_count = 0
        self._maximum_switch_translation_jump_m = 0.0
        self._maximum_switch_rotation_jump_rad = 0.0
        self._closed = False

    def append(self, decision: SelectionDecision) -> None:
        if self._closed:
            raise RuntimeError("localization_evidence_writer_closed")
        payload = decision.to_dict()
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.record_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
        self._decision_count += 1
        self._output_count += int(decision.output is not None)
        self._switch_count += int(decision.switch_event)
        if decision.selected_source is not None:
            self._selection_counts[decision.selected_source] = (
                self._selection_counts.get(decision.selected_source, 0) + 1
            )
        self._odometry_deferred_decision_count += int(decision.odometry_deferred)
        if decision.switch_translation_jump_m is not None:
            self._maximum_switch_translation_jump_m = max(
                self._maximum_switch_translation_jump_m,
                decision.switch_translation_jump_m,
            )
        if decision.switch_rotation_jump_rad is not None:
            self._maximum_switch_rotation_jump_rad = max(
                self._maximum_switch_rotation_jump_rad,
                decision.switch_rotation_jump_rad,
            )
        if decision.deviation is not None:
            self._deviation_count += 1
            # Keep one representative per structured deviation shape.  The
            # aggregate count remains exact without retaining an unbounded
            # in-memory copy during a ten-episode Oracle.
            key = json.dumps(decision.deviation, sort_keys=True, allow_nan=False)
            if key not in self._deviation_keys:
                self._deviation_keys.add(key)
                self._deviations.append(decision.deviation)

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        # Reserve the final name exclusively before preparing the atomic body;
        # another process can no longer create a target between check/replace.
        path.open("x", encoding="utf-8").close()
        temporary_created = False
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                temporary_created = True
                stream.write(
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                    + "\n"
                )
            os.replace(temporary, path)
        except BaseException:
            try:
                if temporary_created:
                    temporary.unlink(missing_ok=True)
            finally:
                path.unlink(missing_ok=True)
            raise

    def close(self, selector_summary: dict[str, Any], *, status: str) -> None:
        if self._closed:
            raise RuntimeError("localization_evidence_writer_already_closed")
        if status not in {"PASS", "PASS_WITH_DEVIATION", "FAIL"}:
            raise ValueError("invalid_localization_status")
        self._closed = True
        self._atomic_json(
            self.deviation_path,
            {
                "schema_version": 1,
                "status": "WARN" if self._deviations else "NONE",
                "odometry_deferred": bool(self._deviations),
                "deviation_count": self._deviation_count,
                "deviations": self._deviations,
            },
        )
        self._atomic_json(
            self.summary_path,
            {
                "schema_version": 1,
                "status": status,
                "decision_count": self._decision_count,
                "output_count": self._output_count,
                "switch_count": self._switch_count,
                "selection_counts": dict(sorted(self._selection_counts.items())),
                "odometry_deferred_decision_count": (
                    self._odometry_deferred_decision_count
                ),
                "maximum_switch_translation_jump_m": (
                    self._maximum_switch_translation_jump_m
                ),
                "maximum_switch_rotation_jump_rad": (
                    self._maximum_switch_rotation_jump_rad
                ),
                "selector": selector_summary,
                "deviation_artifact": self.deviation_path.name,
            },
        )
