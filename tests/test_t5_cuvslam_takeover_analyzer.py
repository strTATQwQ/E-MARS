from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_t5_cuvslam_takeover.py"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _rows(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _fixture(root: Path, *, fallback_count: int = 0, success_count: int = 2) -> None:
    _write(
        root / "fast_lane_final_summary.json",
        {"status": "PASS", "runtime_summary": {"profile": "screen3", "lane": "a"}},
    )
    _write(
        root / "remote/dgx/onboard/controller_summary.json",
        {
            "pose_source": "external_odometry",
            "ground_truth_pose_used_for_nav": False,
            "external_odometry_nav_publish_count": 20,
        },
    )
    _write(
        root / "remote/dgx/cuvslam/takeover_runtime_contract.json",
        {
            "status": "PASS",
            "mode": "cuvslam_takeover",
            "primary_source": "cuvslam",
            "navigation_odometry_topic": "/odom",
            "single_odom_authority": True,
            "gt_fallback_count": fallback_count,
            "reset_pollution_count": 0,
        },
    )
    _rows(
        root / "remote/dgx/cuvslam/odometry_supervisor_records.jsonl",
        [
            {
                "event": "tracking_lost",
                "fail_safe_stop_published": True,
                "safe_stop_latency_wall_sec": 0.24,
            }
        ],
    )
    episodes = [
        {"ordinal": index + 1, "success": index < success_count}
        for index in range(3)
    ]
    _write(
        root / "remote/x86/evaluator/attempt/per_episode.json",
        {
            "expected_episode_count": 3,
            "completed_episode_count": 3,
            "success_count": success_count,
            "episodes": episodes,
        },
    )


def _run(root: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    output = root / "takeover_metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--result-root",
            str(root),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_analyzer_accepts_fixed3_without_gt_fallback(tmp_path: Path) -> None:
    _fixture(tmp_path)
    completed, payload = _run(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "NAV_CANDIDATE_PASS"
    assert payload["nav_candidate_pass"] is True


def test_analyzer_classifies_any_gt_fallback_separately(tmp_path: Path) -> None:
    _fixture(tmp_path, fallback_count=1)
    completed, payload = _run(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "SAFE_FALLBACK_PASS"
    assert payload["nav_candidate_pass"] is False


def test_analyzer_fails_closed_on_late_safe_stop(tmp_path: Path) -> None:
    _fixture(tmp_path)
    records = tmp_path / "remote/dgx/cuvslam/odometry_supervisor_records.jsonl"
    _rows(
        records,
        [
            {
                "event": "tracking_lost",
                "fail_safe_stop_published": True,
                "safe_stop_latency_wall_sec": 0.31,
            }
        ],
    )
    completed, payload = _run(tmp_path)
    assert completed.returncode == 2
    assert payload["status"] == "FAIL"
    assert payload["checks"]["safe_stop_within_wall_deadline"] is False


def test_analyzer_does_not_promote_one_of_three(tmp_path: Path) -> None:
    _fixture(tmp_path, success_count=1)
    completed, payload = _run(tmp_path)
    assert completed.returncode == 2
    assert payload["status"] == "FAIL"
    assert payload["checks"]["oracle_success_threshold"] is False
