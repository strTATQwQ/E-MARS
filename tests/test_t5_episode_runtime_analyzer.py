from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_t5_episode_runtime.py"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _gate_row(
    episode_id: str,
    generation: int,
    request_id: str,
    action: int,
    kind: str,
    elapsed: float,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "reset_generation": generation,
        "decision": {
            "kind": kind,
            "elapsed_sim_sec": elapsed,
            "progress": 0.21,
            "required_progress": 0.20,
            "commanded_progress": 0.25,
        },
        "gate": {
            "pending": None,
            "stop_barrier": {"request_id": request_id, "action": action},
        },
    }


def _fixture(root: Path) -> None:
    evaluator = root / "remote/x86/evaluator/attempt"
    _write(
        evaluator / "per_episode.json",
        {
            "completed_episode_count": 2,
            "episodes": [
                {
                    "ordinal": 1,
                    "trajectory_id": "scene_10",
                    "duration_sec": 10.0,
                    "step_count": 100,
                },
                {
                    "ordinal": 2,
                    "trajectory_id": "scene_20",
                    "duration_sec": 20.0,
                    "step_count": 200,
                },
            ],
        },
    )
    _write_rows(
        evaluator / "go2_runtime_audit.jsonl",
        [{"physics_hz": 20.0}, {"physics_hz": 20.0}],
    )
    controller = []
    for episode_id, generation, base in (("a::10", 0, 1.0), ("a::20", 1, 3.0)):
        controller.extend(
            [
                {
                    "episode_id": episode_id,
                    "reset_generation": generation,
                    "identity_valid": True,
                    "state_only": False,
                    "motion_enabled": True,
                    "desired_angular_z": base,
                    "actual_angular_velocity": [0.0, 0.0, base / 2.0],
                    "command_age_sec": 0.1,
                    "wall_time_unix": 100.0 + base,
                },
                {
                    "episode_id": episode_id,
                    "reset_generation": generation,
                    "identity_valid": True,
                    "state_only": False,
                    "motion_enabled": False,
                    "desired_angular_z": 0.0,
                    "actual_angular_velocity": [0.0, 0.0, 0.0],
                    "command_age_sec": 0.3,
                    "wall_time_unix": 101.0 + base,
                },
            ]
        )
    controller.append(
        {
            "episode_id": "bootstrap",
            "reset_generation": 0,
            "identity_valid": False,
            "state_only": True,
            "desired_angular_z": 99.0,
            "actual_angular_velocity": [0.0, 0.0, 99.0],
            "command_age_sec": 99.0,
        }
    )
    _write_rows(root / "remote/dgx/onboard/controller_records.jsonl", controller)
    _write_rows(
        root / "remote/dgx/client/motion_observation_gate_records.jsonl",
        [
            _gate_row("a::10", 0, "a::10:0:1", 1, "safe_stop_complete", 1.5),
            _gate_row("a::20", 1, "a::20:1:1", 2, "safe_stop_timeout", 2.0),
        ],
    )


def _run(root: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    output = root / "analysis.json"
    completed = subprocess.run(
        [sys.executable, str(ANALYZER), "--result-dir", str(root), "--output", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_reports_estimated_rtf_controller_distributions_and_gate_outcomes(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    completed, payload = _run(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "PASS"
    first = payload["episodes"][0]
    assert first["wall_duration_sec"] == 10.0
    assert first["sim_duration_sec"] == 5.0
    assert first["sim_duration_estimated"] is True
    assert first["rtf"] == 0.5
    assert first["controller"]["all_updates"]["commanded_yaw_rate_rad_s"]["mean"] == 0.5
    assert first["controller"]["all_updates"]["measured_yaw_rate_rad_s"]["maximum"] == 0.5
    assert first["controller"]["all_updates"]["command_age_sec"]["p95"] == pytest.approx(0.29)
    assert payload["aggregate"]["rtf"] == 0.5
    gate = payload["aggregate"]["motion_gate"]
    assert gate["attempted"] == 2
    assert gate["completed"] == 1
    assert gate["timeout"] == 1
    assert gate["by_action"]["forward"]["completed"] == 1
    assert gate["by_action"]["left"]["timeout"] == 1


def test_rejects_inconsistent_physics_rate(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _write_rows(
        tmp_path / "remote/x86/evaluator/attempt/go2_runtime_audit.jsonl",
        [{"physics_hz": 20.0}, {"physics_hz": 30.0}],
    )
    completed, payload = _run(tmp_path)
    assert completed.returncode == 2
    assert payload["status"] == "FAIL"
    assert "inconsistent physics_hz" in payload["error"]


def test_rejects_ambiguous_evaluator_evidence(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _write(
        tmp_path / "remote/x86/evaluator/other/per_episode.json",
        {"completed_episode_count": 0, "episodes": []},
    )
    completed, payload = _run(tmp_path)
    assert completed.returncode == 2
    assert payload["status"] == "FAIL"
    assert "exactly one per_episode.json" in payload["error"]
