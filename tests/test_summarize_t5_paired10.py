from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.summarize_t5_paired10 import PairedAnalysisError, analyze, main


SHA = "1" * 40


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


def _fixture(tmp_path: Path, *, partial: bool = False) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    paired = repo / "results/internnav_t5/paired"
    keys = {
        "paired10_a": [f"a{i}_1{i}" for i in range(10)],
        "paired10_b": [f"b{i}_2{i}" for i in range(10)],
    }
    manifest = {
        "status": "FROZEN_FOR_EXECUTION",
        "evidence_classification": {"held_out": False},
        "episode_sets": {
            name: {"lane": name[-1], "episode_keys": values}
            for name, values in keys.items()
        },
        "rounds": [
            {
                "round_id": "round1",
                "lane_a": {"pair_set": "paired10_a", "evaluation_arm": "internvla_only"},
                "lane_b": {"pair_set": "paired10_b", "evaluation_arm": "internvla_step3"},
            },
            {
                "round_id": "round2",
                "lane_a": {"pair_set": "paired10_a", "evaluation_arm": "internvla_step3"},
                "lane_b": {"pair_set": "paired10_b", "evaluation_arm": "internvla_only"},
            },
        ],
    }
    manifest_path = repo / "configs/internnav_t5/paired.json"
    _write(manifest_path, manifest)
    lane_results: dict[str, dict] = {}
    slots = [
        ("round1", "lane_a", "a", "paired10_a", "internvla_only"),
        ("round1", "lane_b", "b", "paired10_b", "internvla_step3"),
        ("round2", "lane_a", "a", "paired10_a", "internvla_step3"),
        ("round2", "lane_b", "b", "paired10_b", "internvla_only"),
    ]
    for ordinal, (round_id, lane_name, lane, pair_set, arm) in enumerate(slots):
        relative = f"results/internnav_t5/child-{ordinal}"
        lane_results.setdefault(round_id, {"status": "PASS"})[lane_name] = relative
        root = repo / relative
        binding = {
            "evaluation_arm": arm,
            "pair_set": pair_set,
            "execution_episode_keys": keys[pair_set],
        }
        _write(root / "fast_lane_summary.json", {
            "status": "PASS", "lane": lane, "pair_set": pair_set,
            "evaluation_arm": arm, "code_ref_sha": SHA, "input_binding": binding,
        })
        _write(root / "fast_lane_final_summary.json", {"status": "PASS"})
        episodes = []
        for index, key in enumerate(keys[pair_set]):
            success = (index + (arm == "internvla_step3")) % 2 == 0
            episodes.append({
                "trajectory_id": key, "success": success, "duration_sec": 10.0,
                "step_count": 20, "official_metrics": {
                    "sr": int(success), "spl": 0.5 + 0.1 * success, "ne_m": 2.0,
                }, "ndtw": 0.7, "termination_reason": "success" if success else "stuck",
            })
        _write(root / "remote/x86/evaluator/run/per_episode.json", {"episodes": episodes})
        _jsonl(root / "remote/x86/evaluator/run/go2_runtime_audit.jsonl", [{"physics_hz": 20.0}])
        timeline_rows = []
        for index, key in enumerate(keys[pair_set]):
            latency = {"internvla.inference": 10.0 + index}
            if arm == "internvla_step3":
                latency["step3.private_trace_response"] = 100.0 + index
            timeline_rows.append({
                "episode_id": f"{lane}::{key}",
                "payload": {"command_age_sec": 0.1 + index / 100.0, "wall_latencies_ms": latency},
            })
        timeline = root / "replay/timeline.jsonl"
        _jsonl(timeline, timeline_rows)
        _write(root / "replay/timeline_index.json", {
            "event_count": len(timeline_rows), "timeline_sha256": hashlib.sha256(timeline.read_bytes()).hexdigest(),
            # This deliberately wrong aggregate must never be reused.
            "wall_latency_summary": {"components": {"internvla.inference": {"p95_ms": 999999}}},
        })
    if partial:
        lane_results["round2"] = {"status": "NOT_RUN", "lane_a": None, "lane_b": None}
    _write(paired / "paired10_execution_summary.json", {
        "status": "FAIL" if partial else "PASS", "run_id": "paired-test",
        "code_ref_sha": SHA, "manifest": "configs/internnav_t5/paired.json",
        "lane_results": lane_results,
    })
    return repo, paired


def test_complete_report_pairs_within_lane_and_uses_raw_latency(tmp_path: Path) -> None:
    repo, paired = _fixture(tmp_path)
    value = analyze(paired, repo)
    assert value["status"] == "COMPLETE"
    assert value["overall"]["observed_execution_count"] == 40
    assert value["overall"]["outcomes"] == {
        "both_success": 0, "only_internvla": 10, "only_step3": 10,
        "neither": 0, "incomplete": 0, "pair_count": 20,
    }
    assert value["wall_response_latency"]["internvla"]["primary_response"]["count"] == 40
    assert value["wall_response_latency"]["internvla"]["primary_response"]["p95"] < 20
    assert value["wall_response_latency"]["step3"]["primary_response"]["count"] == 20
    assert value["evidence_classification"]["held_out"] is False


def test_partial_online_failure_is_incomplete_but_pair_drift_fails_closed(tmp_path: Path) -> None:
    repo, paired = _fixture(tmp_path, partial=True)
    assert analyze(paired, repo)["status"] == "INCOMPLETE"
    fast = repo / "results/internnav_t5/child-0/fast_lane_summary.json"
    value = json.loads(fast.read_text(encoding="utf-8"))
    value["pair_set"] = "paired10_b"
    _write(fast, value)
    with pytest.raises(PairedAnalysisError, match="wrong pair_set"):
        analyze(paired, repo)


def test_cli_writes_fail_report_for_contract_violation(tmp_path: Path) -> None:
    repo, paired = _fixture(tmp_path)
    summary = paired / "paired10_execution_summary.json"
    value = json.loads(summary.read_text(encoding="utf-8"))
    value["lane_results"]["round1"]["lane_b"] = value["lane_results"]["round1"]["lane_a"]
    _write(summary, value)
    output = paired / "analysis.json"
    assert main([str(paired), "--repo-root", str(repo), "--output", str(output)]) == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "FAIL"
