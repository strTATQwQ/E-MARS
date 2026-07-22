from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from t4_completion.recovery.offline_smoke import main, run_profile_smoke
from t4_completion.recovery import load_recovery_config


ROOT = Path(__file__).resolve().parents[1]
PROFILES = [
    ROOT / "configs/completion_sim/recovery/profile_a.json",
    ROOT / "configs/completion_sim/recovery/profile_b.json",
]


def test_both_reference_smokes_exercise_all_actions() -> None:
    for path in PROFILES:
        payload = run_profile_smoke(load_recovery_config(path))
        assert payload["status"] == "PASS"
        assert payload["old_trajectory_rejection_count"] == 2
        assert payload["old_trajectory_execution_count"] == 0
        assert payload["terminal_safe_stop_count"] == 0
        assert payload["state"] == "monitoring"
        assert len(payload["event_stream_sha256"]) == 64


def test_cli_prints_machine_readable_pass_without_online_resources() -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/t4_recovery_offline_smoke.py",
            *(str(path.relative_to(ROOT)) for path in PROFILES),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["online_resources_used"] == []
    assert [item["profile_id"] for item in payload["profiles"]] == ["A", "B"]


def test_cli_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    output = tmp_path / "already.json"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main([str(PROFILES[0]), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "existing\n"
