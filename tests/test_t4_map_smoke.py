from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from scripts.t4_container_process_cleanup import target_tokens
from t4_completion.map.contract import load_and_validate
from t4_completion.map.smoke import run_dry_smoke
from t4_completion.map.static_map import build_grid, grid_summary, require_simulation_boundary
from t4_completion.map.warn_relay import (
    bounded_command,
    interlock_state,
    validate_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_dry_smoke_marks_clears_inflates_and_never_starts_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry smoke attempted to start a subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    payload = run_dry_smoke()
    assert payload["status"] == "PASS"
    assert payload["processes_started"] == 0
    assert payload["network_accessed"] is False
    assert payload["shared_resources_used"] == []
    assert all(payload["checks"].values())


def test_static_map_matches_fixed_diagnostic_geometry() -> None:
    config = load_and_validate().smoke_map
    grid = build_grid(config)
    summary = grid_summary(config)
    assert len(grid) == 80 * 80
    assert set(grid) == {0, 100}
    assert summary["occupied_cells"] == 128
    assert summary["free_cells"] == 6272
    assert summary["map_to_odom"] == "identity"
    assert summary["qos"] == {
        "reliability": "reliable",
        "durability": "transient_local",
        "depth": 1,
    }


def test_warn_only_relay_is_bounded_stale_and_estop_safe() -> None:
    bounded = bounded_command(4.0, -8.0, 0.01, True, False)
    assert (bounded.linear_x, bounded.angular_z, bounded.stopped) == (0.25, -1.0, False)
    assert bounded_command(0.1, 0.2, 0.31, True, False).reason == "stale_command"
    assert bounded_command(0.1, 0.2, 0.01, False, False).reason == "estop_unobserved"
    assert bounded_command(0.1, 0.2, 0.01, True, True).reason == "simulation_estop"
    assert bounded_command(math.nan, 0.2, 0.01, True, False).reason == "nonfinite_input"


def test_oracle_motion_enable_is_a_fail_closed_sim_interlock() -> None:
    assert interlock_state(False, False, False, False) == (False, False)
    assert interlock_state(False, False, True, False) == (True, True)
    assert interlock_state(False, False, True, True) == (True, False)
    assert interlock_state(True, True, True, True) == (True, True)


def test_container_cleanup_targets_only_the_bound_deployment_relay() -> None:
    control_root = Path("/home/song/internnav/.t4-deployments/grant-ref-isaac")
    tokens = target_tokens(control_root)

    assert (
        "/home/song/internnav/.t4-deployments/grant-ref-isaac/"
        "t4_completion/map/warn_relay.py"
    ) in tokens
    assert "t4_completion/map/warn_relay.py" not in tokens


def test_warn_relay_evidence_requires_bounded_nonzero_motion(tmp_path: Path) -> None:
    evidence = tmp_path / "relay.jsonl"
    records = [
        {
            "schema_version": 1,
            "event": "startup",
            "sequence": 1,
            "reason": "startup",
            "stopped": True,
            "linear_x": 0.0,
            "angular_z": 0.0,
        },
        {
            "schema_version": 1,
            "event": "state_change",
            "sequence": 2,
            "reason": "bounded_forward",
            "stopped": False,
            "linear_x": 0.25,
            "angular_z": -1.0,
        },
        {
            "schema_version": 1,
            "event": "state_change",
            "sequence": 3,
            "reason": "stale_command",
            "stopped": True,
            "linear_x": 0.0,
            "angular_z": 0.0,
        },
    ]
    evidence.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = validate_evidence(evidence)

    assert summary["status"] == "PASS"
    assert summary["bounded_forward_count"] == 1
    assert summary["nonzero_output_count"] == 1


def test_warn_relay_evidence_rejects_all_zero_output(tmp_path: Path) -> None:
    evidence = tmp_path / "relay.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event": "startup",
                "sequence": 1,
                "reason": "startup",
                "stopped": True,
                "linear_x": 0.0,
                "angular_z": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        validate_evidence(evidence)


def test_warn_relay_evidence_warns_for_bounded_zero_only(tmp_path: Path) -> None:
    evidence = tmp_path / "relay.jsonl"
    records = [
        {
            "schema_version": 1,
            "event": "startup",
            "sequence": 1,
            "reason": "startup",
            "stopped": True,
            "linear_x": 0.0,
            "angular_z": 0.0,
        },
        {
            "schema_version": 1,
            "event": "state_change",
            "sequence": 2,
            "reason": "bounded_forward",
            "stopped": False,
            "linear_x": 0.0,
            "angular_z": 0.0,
        },
    ]
    evidence.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = validate_evidence(evidence)

    assert summary["status"] == "PASS"
    assert summary["bounded_forward_count"] == 1
    assert summary["nonzero_output_count"] == 0
    assert summary["warnings"] == ["no_nonzero_bounded_command_observed"]


def test_ros_publisher_rejects_missing_managed_sim_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERNNAV_RUNTIME_POLICY", raising=False)
    monkeypatch.delenv("INTERNNAV_SIMULATION_TARGET", raising=False)
    monkeypatch.delenv("INTERNNAV_T4_MAP_COMPANION_ACK", raising=False)
    with pytest.raises(RuntimeError):
        require_simulation_boundary()


def test_dry_run_cli_emits_machine_readable_pass() -> None:
    completed = subprocess.run(
        ["python", str(ROOT / "scripts/t4_map_smoke_dry_run.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["offline_only"] is True
