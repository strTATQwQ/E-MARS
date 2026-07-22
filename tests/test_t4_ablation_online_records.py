from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.build_t4_ablation_episode_records import (
    _termination,
    build_records,
    latency_summary,
)
from t4_completion.ablation.contract import load_matrix, resolve_variant_configs


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/completion_sim/ablation/frozen_matrix_v1.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _archive_inputs(path: Path, episodes: list[dict], config: dict) -> None:
    raw = json.dumps({"episodes": episodes}, separators=(",", ":"), sort_keys=True).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(raw)
    members = {
        "smoke_dataset/val_unseen/val_unseen.json.gz": buffer.getvalue(),
        "ablation_configs/variants/full_system1_system2.json": (
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_latency_summary_is_monotonic_and_empty_is_explicit() -> None:
    assert latency_summary([]) == {
        "count": 0,
        "mean": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
    summary = latency_summary([10.0, 30.0, 20.0])
    assert summary["count"] == 3
    assert summary["mean"] == 20.0
    assert summary["p50"] <= summary["p95"] <= summary["max"]


def test_evaluator_step_threshold_is_recorded_as_timeout() -> None:
    termination = _termination(
        factors={"termination_mode": "model_stop"},
        result_reason="exceed_total_max_step",
        official={"ne_m": 4.0},
        client_rows=[{"model_stop": False}],
    )
    assert termination["reason"] == "timeout"
    assert termination["effective_source"] == "timeout"


@pytest.mark.parametrize(
    ("metrics_in_per_episode", "client_gap_kind"),
    [(True, "none"), (False, "none"), (False, "correlated"), (False, "source_timeout")],
)
def test_online_converter_uses_frozen_dataset_order_and_official_metrics(
    tmp_path: Path,
    metrics_in_per_episode: bool,
    client_gap_kind: str,
) -> None:
    result = tmp_path / "results/arm"
    result.mkdir(parents=True)
    matrix = load_matrix(MATRIX_PATH)
    config = resolve_variant_configs(matrix)["full_system1_system2"]
    dataset_episodes = [
        {"trajectory_id": f"route-{index}", "episode_id": f"episode-{index}"}
        for index in range(20)
    ]
    _archive_inputs(result / "migration_inputs.tgz", dataset_episodes, config)

    completed = []
    clients = []
    transforms = []
    audits = []
    controls = []
    for index, episode in enumerate(dataset_episodes):
        episode_id = episode["episode_id"]
        source_timeout = client_gap_kind == "source_timeout" and index == 15
        completed_row = {
            "trajectory_id": f"route-{index}_{episode_id}",
            "step_count": 1,
            "termination_reason": "exceed_total_max_step" if source_timeout else "success",
        }
        if metrics_in_per_episode:
            completed_row["official_metrics"] = {
                "sr": 0 if source_timeout else 1,
                "os": 0 if source_timeout else 1,
                "spl": 0.0 if source_timeout else 0.5,
                "ne_m": 5.0 if source_timeout else 1.0,
            }
        completed.append(completed_row)
        client_rows = [
                {
                    "episode_id": episode_id,
                    "action_source": 1,
                    "inference_latency_sec": 0.02,
                    "nav2_resolution_latency_sec": 0.001,
                    "model_stop": False,
                },
                {
                    "episode_id": episode_id,
                    "action_source": 2,
                    "inference_latency_sec": 0.03,
                    "nav2_resolution_latency_sec": 0.002,
                    "model_stop": True,
                },
            ]
        if not (client_gap_kind != "none" and index == 15):
            clients.extend(client_rows)
        if not source_timeout:
            transforms.append(
                {
                    "episode_id": episode_id,
                    "event": "ablation_transform",
                    "original_action_source": 1,
                    "original_point_count": 2,
                    "sequence_id": index,
                }
            )
            transforms.append(
                {
                    "episode_id": episode_id,
                    "action_source": 1,
                    "sequence_id": index,
                }
            )
        audits.append(
            {
                "episode_id": episode_id,
                "event": "ablation_history_generation_reset",
            }
        )
        audits.append(
            {
                "episode_id": episode_id,
                "event": "ablation_history_frame_consumed",
                "sequence_id": index,
            }
        )
        controls.append(
            {
                "episode_id": episode_id,
                "identity_valid": True,
                "motion_enabled": True,
                "command_fresh": True,
                "desired_linear_x": 0.1,
                "desired_angular_z": 0.2,
                "physical_collision": False,
                "fallen": False,
            }
        )
    # Evaluator completion order is intentionally different from dataset order.
    _write_json(result / "isaac-run/per_episode.json", {"episodes": completed[::-1]})
    if not metrics_in_per_episode:
        progress = []
        for index, episode in enumerate(dataset_episodes, 1):
            episode_id = episode["episode_id"]
            key = f"route-{index - 1}_{episode_id}"
            source_timeout = client_gap_kind == "source_timeout" and index == 16
            metric = {
                "NE": 5.0 if source_timeout else 1.0,
                "success": 0.0 if source_timeout else 1.0,
                "osr": 0.0 if source_timeout else 1.0,
                "spl": 0.0 if source_timeout else 0.5,
                "episode_id": episode_id,
                "trajectory_id": f"route-{index - 1}",
            }
            progress.append(f"{key}: " + json.dumps({"evaluation": [metric]}, indent=2))
            progress.append(
                f"[{index}/20][step_index:1] finish: [trajectory_id:{key}]"
                "[duration:1.0 s][step_count:1][fps:1.0]"
                f"[result:{'exceed_total_max_step' if source_timeout else 'success'}]"
            )
        log = result / "isaac-run/logs_sanitized/eval.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(progress) + "\n", encoding="utf-8")
    _write_jsonl(result / "isaac-run/client_records.jsonl", clients)
    _write_json(
        result / "isaac-run/client_summary.json",
        {"status": "FINISHED", "step_count": len(clients)},
    )
    _write_jsonl(result / "dgx-run/active_records.jsonl", transforms)
    _write_jsonl(result / "model-run/model_recovery_audit.jsonl", audits)
    _write_jsonl(result / "dgx-run/controller_records.jsonl", controls)
    _write_json(result / "dgx-run/controller_summary.json", {"stale_motion_execution_count": 0})
    _write_json(
        result / "dgx-run/warn_only_relay_summary.json",
        {
            "status": "PASS",
            "maximum_abs_linear_mps": 0.25,
            "maximum_abs_angular_rps": 1.0,
            "reason_counts": {"simulation_estop": 0},
        },
    )

    records = build_records(
        result,
        "full_system1_system2",
        "a" * 40,
        repository_root=tmp_path,
    )

    assert len(records) == 20
    assert records[0]["episode_id"] == "route-0_episode-0"
    assert records[-1]["episode_id"] == "route-19_episode-19"
    assert records[0]["official_metrics"] == {"sr": 1, "os": 1, "spl": 0.5, "ne_m": 1.0}
    assert records[0]["activation"]["system1_count"] == 1
    assert records[0]["activation"]["system2_count"] == 1
    if client_gap_kind == "correlated":
        assert records[15]["warnings"] == ["recorder_partial_frame"]
        assert records[15]["diagnostics"]["latency_ms"]["end_to_end"]["count"] == 0
        assert records[15]["activation"]["visual_frame_count"] == 1
    elif client_gap_kind == "source_timeout":
        assert records[15]["warnings"] == [
            "recorder_partial_frame",
            "source_timeout_extended",
        ]
        assert records[15]["termination"]["reason"] == "timeout"
        assert records[15]["diagnostics"]["latency_ms"]["end_to_end"]["count"] == 0
    assert (result / "episode-sources/019.json").is_file()
