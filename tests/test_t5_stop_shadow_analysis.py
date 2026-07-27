from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_t5_stop_shadows import analyze


CODE_SHA = "a" * 40


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _lane(root: Path, lane: str, offset: int) -> Path:
    keys = [f"trajectory-{lane}-{index}_{offset + index}" for index in range(10)]
    _write(root / "fast_lane_final_summary.json", {"status": "PASS"})
    _write(
        root / "input_binding.json",
        {
            "lane": lane,
            "code_ref_sha": CODE_SHA,
            "candidate_profile": "recovery_a",
            "isaac_sensor_profile": "dual_lane_wp03_stop_shadow",
        },
    )
    _write(
        root / "remote/x86/ordered_episode_manifest.json",
        {"ordered_episode_keys": keys},
    )
    _write(
        root / "remote/x86/evaluator/run/per_episode.json",
        {
            "episodes": [
                {"trajectory_id": key, "success": True, "termination_reason": "success"}
                for key in keys
            ]
        },
    )
    active = []
    advice = []
    task = []
    gate = []
    for index, key in enumerate(keys):
        episode = key.rsplit("_", 1)[-1]
        identity = f"{lane}::{episode}"
        active.extend(
            [
                {
                    "event": "ablation_transform",
                    "episode_id": identity,
                    "reset_generation": 0,
                    "sequence_id": 2,
                    "oracle_distance_m": 4.0,
                    "original_model_stop": index == 0,
                },
                {
                    "event": "ablation_transform",
                    "episode_id": identity,
                    "reset_generation": 0,
                    "sequence_id": 5,
                    "oracle_distance_m": 2.0,
                    "original_model_stop": index < 1,
                },
            ]
        )
        if index < 2:
            advice.append(
                {
                    "status": "ARRIVED",
                    "episode_id": identity,
                    "reset_generation": 0,
                    "trigger_sequence_id": 5,
                }
            )
        if index < 3:
            gate.append(
                {
                    "event": "stop_shadow_step3_arrival_confirmed",
                    "episode_id": identity,
                    "reset_generation": 0,
                    "payload": {"trigger_sequence_id": 5},
                }
            )
        if index < 4:
            task.append(
                {
                    "event": "task_state_updated",
                    "episode_id": identity,
                    "reset_generation": 0,
                    "trigger_sequence_id": 5,
                    "final_arrival_candidate": True,
                }
            )
        if index < 5:
            token = f"{identity}:0:oracle-terminal-shadow:5"
            gate.extend(
                [
                    {
                        "event": "stop_shadow_oracle_reference",
                        "episode_id": identity,
                        "reset_generation": 0,
                        "token": f"{identity}:0:oracle-reference:5",
                        "payload": {"sequence_id": 5},
                    },
                    {
                        "event": "step3_oracle_terminal_shadow_started",
                        "episode_id": identity,
                        "reset_generation": 0,
                        "token": token,
                        "payload": {"sequence_id": 5},
                    },
                    {
                        "event": "step3_oracle_terminal_shadow_completed",
                        "episode_id": identity,
                        "reset_generation": 0,
                        "token": token,
                        "payload": {"confirmed": True},
                    },
                ]
            )
    _jsonl(root / "remote/dgx/onboard/active_records.jsonl", active)
    _jsonl(root / "remote/x86/evaluator/step3_timeout_advice.jsonl", advice)
    _jsonl(root / "remote/x86/evaluator/task_state/events.jsonl", task)
    _jsonl(root / "remote/dgx/client/motion_observation_gate_records.jsonl", gate)
    return root


def test_analyzer_merges_disjoint_lanes_and_ranks_operational_methods(tmp_path: Path) -> None:
    lane_a = _lane(tmp_path / "lane-a", "a", 0)
    lane_b = _lane(tmp_path / "lane-b", "b", 10)

    result = analyze(lane_a, lane_b)

    assert result["status"] == "PASS"
    assert result["episode_count"] == 20
    assert result["oracle_navigation_success_count"] == 20
    assert result["best_operational_method"] == "step3_task_state_target_found"
    assert result["promotion_recommendation"] == "PROMISING_SHADOW_METHOD"
    assert result["oracle_triggered_probe"]["ranking_eligible"] is False
    assert result["oracle_triggered_probe"]["attempted_count"] == 10
    assert result["oracle_triggered_probe"]["completed_count"] == 10
    assert result["oracle_triggered_probe"]["confirmed_count"] == 10
    assert result["oracle_triggered_probe"]["failed_count"] == 0
    model = next(
        row
        for row in result["operational_method_scores"]
        if row["method"] == "internvla_model_stop"
    )
    assert model["false_positive_candidate_count"] == 2
    assert model["candidate_count"] == model["candidate_frame_count"] == 4
    assert model["candidate_count_unit"] == (
        "unique_sequence_frames_not_independent_judgments"
    )
    assert model["candidate_episode_count"] == 2
    assert model["candidate_burst_count"] == model["candidate_onset_count"] == 4
    assert model["true_positive_episode_count"] == 2


def _append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def test_internvla_stop_frames_are_compacted_into_contiguous_bursts(
    tmp_path: Path,
) -> None:
    lane_a = _lane(tmp_path / "lane-a", "a", 0)
    lane_b = _lane(tmp_path / "lane-b", "b", 10)
    active = lane_a / "remote/dgx/onboard/active_records.jsonl"
    _append_jsonl(
        active,
        [
            {
                "event": "ablation_transform",
                "episode_id": "a::0",
                "reset_generation": 0,
                "sequence_id": sequence_id,
                "oracle_distance_m": 3.5,
                "original_model_stop": True,
            }
            for sequence_id in (3, 4)
        ],
    )

    result = analyze(lane_a, lane_b)
    model = next(
        row
        for row in result["operational_method_scores"]
        if row["method"] == "internvla_model_stop"
    )

    assert model["candidate_frame_count"] == 6
    assert model["candidate_episode_count"] == 2
    assert model["candidate_burst_count"] == 3
    assert model["candidate_onset_count"] == 3
    assert model["max_burst_length_frames"] == 4
    lane_a_episode = next(
        row
        for row in model["episode_generation_statistics"]
        if row["episode_id"] == "a::0"
    )
    assert lane_a_episode == {
        "episode_id": "a::0",
        "reset_generation": 0,
        "candidate_frame_count": 4,
        "burst_count": 1,
        "onset_sequence_ids": [2],
    }


def test_oracle_terminal_probe_reports_attempts_failures_and_reasons(
    tmp_path: Path,
) -> None:
    lane_a = _lane(tmp_path / "lane-a", "a", 0)
    lane_b = _lane(tmp_path / "lane-b", "b", 10)
    gate = lane_a / "remote/dgx/client/motion_observation_gate_records.jsonl"
    _append_jsonl(
        gate,
        [
            {
                "event": "stop_shadow_oracle_reference",
                "episode_id": "a::5",
                "reset_generation": 0,
                "token": "a::5:0:oracle-reference:5",
                "payload": {"sequence_id": 5},
            },
            {
                "event": "step3_oracle_terminal_shadow_started",
                "episode_id": "a::5",
                "reset_generation": 0,
                "token": "a::5:0:oracle-terminal-shadow:5",
                "payload": {"sequence_id": 5},
            },
            {
                "event": "step3_arrival_not_confirmed",
                "episode_id": "a::5",
                "reset_generation": 0,
                "token": "a::5:0:oracle-terminal-shadow:5",
                "payload": {"reason": "step3_arrival_not_confirmed"},
            },
            {
                "event": "step3_oracle_terminal_shadow_completed",
                "episode_id": "a::5",
                "reset_generation": 0,
                "token": "a::5:0:oracle-terminal-shadow:5",
                "payload": {"confirmed": False},
            },
            {
                "event": "stop_shadow_oracle_reference",
                "episode_id": "a::6",
                "reset_generation": 0,
                "token": "a::6:0:oracle-reference:5",
                "payload": {"sequence_id": 5},
            },
            {
                "event": "step3_oracle_terminal_shadow_unavailable",
                "episode_id": "a::6",
                "reset_generation": 0,
                "token": "a::6:0:oracle-terminal-shadow:5",
                "payload": {"sequence_id": 5},
            },
            {
                "event": "stop_shadow_oracle_reference",
                "episode_id": "a::7",
                "reset_generation": 0,
                "token": "a::7:0:oracle-reference:5",
                "payload": {"sequence_id": 5},
            },
            {
                "event": "step3_oracle_terminal_shadow_started",
                "episode_id": "a::7",
                "reset_generation": 0,
                "token": "a::7:0:oracle-terminal-shadow:5",
                "payload": {"sequence_id": 5},
            },
        ],
    )

    probe = analyze(lane_a, lane_b)["oracle_triggered_probe"]

    assert probe["candidate_count"] == probe["confirmed_count"] == 10
    assert probe["attempted_count"] == 13
    assert probe["started_count"] == 12
    assert probe["completed_count"] == 11
    assert probe["failed_count"] == 3
    assert probe["failure_reason_counts"] == {
        "oracle_terminal_shadow_unavailable": 1,
        "probe_incomplete_after_start": 1,
        "step3_arrival_not_confirmed": 1,
    }
