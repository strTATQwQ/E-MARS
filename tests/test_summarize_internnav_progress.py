from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_internnav_progress import metric_records, summarize
from scripts.extract_internnav_per_episode import extract_complete


def test_summarize_attaches_authoritative_metrics_to_matching_episode(
    tmp_path: Path,
) -> None:
    metric = {
        "episode": {
            "trajectory_id": "route-a",
            "episode_id": "007",
            "NE": 1.25,
            "success": 1.0,
            "osr": 1.0,
            "spl": 0.75,
            "TL": 4.5,
            "ndtw": 0.8,
        }
    }
    log = tmp_path / "eval.log"
    log.write_text(
        "evaluator metrics: "
        + json.dumps(metric, indent=2)
        + "\n[1/1][step_index:9] finish: [trajectory_id:route-a_007]"
        "[duration:2.50 s][step_count:10][fps:4.00][result:success]\n",
        encoding="utf-8",
    )

    summary = summarize(log)

    assert summary["completed_episode_count"] == 1
    assert summary["episodes"][0]["official_metrics"] == {
        "sr": 1,
        "os": 1,
        "spl": 0.75,
        "ne_m": 1.25,
    }
    assert summary["episodes"][0]["TL"] == 4.5
    assert summary["episodes"][0]["ndtw"] == 0.8


def test_summarize_does_not_attach_metrics_from_another_episode(
    tmp_path: Path,
) -> None:
    metric = {
        "trajectory_id": "route-a",
        "episode_id": "006",
        "NE": 3.0,
        "success": 0,
        "osr": 0,
        "spl": 0.0,
    }
    log = tmp_path / "eval.log"
    log.write_text(
        json.dumps(metric)
        + "\n[1/1][step_index:9] finish: [trajectory_id:route-a_007]"
        "[duration:2.50 s][step_count:10][fps:4.00][result:not_reach_goal]\n",
        encoding="utf-8",
    )

    summary = summarize(log)

    assert "official_metrics" not in summary["episodes"][0]


def test_summarize_decodes_metrics_before_trailing_kit_log_lines(
    tmp_path: Path,
) -> None:
    metric = {
        "default_eval_name": [
            {
                "trajectory_id": 433,
                "episode_id": 121,
                "NE": 4.1472,
                "success": 0.0,
                "osr": 0.0,
                "spl": 0.0,
                "ndtw": 0.9998,
            }
        ]
    }
    log = tmp_path / "eval.log"
    log.write_text(
        "2026Z [py stderr]: 433_121: "
        + json.dumps(metric, indent=2)
        + "\n2026Z [Warning] unrelated Kit diagnostic\n"
        + "[1/1][step_index:2600] finish: [trajectory_id:433_121]"
        "[duration:453.67 s][step_count:2600][fps:5.73][result:stuck]\n",
        encoding="utf-8",
    )

    summary = summarize(log)

    assert summary["episodes"][0]["official_metrics"] == {
        "sr": 0,
        "os": 0,
        "spl": 0.0,
        "ne_m": 4.1472,
    }
    assert summary["episodes"][0]["ndtw"] == 0.9998


def test_metric_records_extracts_evaluator_json_without_finish_line(
    tmp_path: Path,
) -> None:
    log = tmp_path / "common.log"
    log.write_text(
        "2026Z [py stderr]: 433_121: "
        + json.dumps(
            {
                "default_eval_name": [
                    {
                        "trajectory_id": 433,
                        "episode_id": 121,
                        "NE": 4.1,
                        "success": 0.0,
                        "osr": 0.0,
                        "spl": 0.0,
                        "ndtw": 0.9,
                    }
                ]
            },
            indent=2,
        )
        + "\n2026Z later diagnostic\n",
        encoding="utf-8",
    )

    records = metric_records(log)

    assert records["433_121"]["ndtw"] == 0.9


def test_extract_complete_joins_progress_and_common_metric_logs(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress" / "run.log"
    progress.parent.mkdir()
    progress.write_text(
        "[1/1][step_index:9] finish: [trajectory_id:433_121]"
        "[duration:2.50 s][step_count:10][fps:4.00][result:not_reach_goal]\n",
        encoding="utf-8",
    )
    common = tmp_path / "common" / "run.log"
    common.parent.mkdir()
    common.write_text(
        "metrics: "
        + json.dumps(
            {
                "default_eval_name": [
                    {
                        "trajectory_id": 433,
                        "episode_id": 121,
                        "NE": 1.2,
                        "success": 0.0,
                        "osr": 1.0,
                        "spl": 0.0,
                        "ndtw": 0.8,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = extract_complete(tmp_path, 1)

    assert payload["episodes"][0]["official_metrics"] == {
        "sr": 0,
        "os": 1,
        "spl": 0.0,
        "ne_m": 1.2,
    }
    assert payload["episodes"][0]["ndtw"] == 0.8
