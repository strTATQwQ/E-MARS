import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


splitter = _load(
    "materialize_t5_final_pilot_split",
    "scripts/materialize_t5_final_pilot_split.py",
)
finalizer = _load(
    "finalize_t5_final_pilot", "scripts/finalize_t5_final_pilot.py"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(
                (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )


def _fixture_contract(tmp_path: Path):
    episodes = []
    for key in splitter.FROZEN_KEYS:
        trajectory_id, episode_id = key.split("_", 1)
        episodes.append(
            {
                "trajectory_id": int(trajectory_id),
                "episode_id": int(episode_id),
                "instruction": {"instruction_text": f"episode {key}"},
            }
        )
    source = tmp_path / "source" / "val_unseen.json.gz"
    _write_gzip(source, {"episode_count": 20, "episodes": episodes, "split": "val_unseen"})
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    execution = tmp_path / "execution.json"
    pilot = tmp_path / "pilot.json"
    _write_json(
        execution,
        {
            "final_evaluation": {
                "pilot": {
                    "lane_a_episode_keys": list(splitter.FROZEN_LANE_A_KEYS),
                    "lane_b_episode_keys": list(splitter.FROZEN_LANE_B_KEYS),
                    "aggregate_episode_count": 20,
                    "held_out_tuning_forbidden": True,
                }
            }
        },
    )
    _write_json(
        pilot,
        {
            "episode_count": 20,
            "episode_keys": list(splitter.FROZEN_KEYS),
            "overlay_sha256": source_sha,
        },
    )
    return source, source_sha, execution, pilot


def _materialize(tmp_path: Path, prefix: str = "one"):
    source, source_sha, execution, pilot = _fixture_contract(tmp_path)
    audit_path = tmp_path / f"{prefix}-split-audit.json"
    audit = splitter.materialize_split(
        source,
        tmp_path / f"{prefix}-a",
        tmp_path / f"{prefix}-b",
        execution_manifest_path=execution,
        pilot_manifest_path=pilot,
        audit_output=audit_path,
        expected_source_sha256=source_sha,
    )
    return source_sha, execution, pilot, audit_path, audit


def test_final_pilot_split_is_deterministic_and_disjoint(tmp_path):
    source_sha, execution, pilot, audit_path, audit = _materialize(tmp_path, "one")
    second_audit = splitter.materialize_split(
        tmp_path / "source" / "val_unseen.json.gz",
        tmp_path / "two-a",
        tmp_path / "two-b",
        execution_manifest_path=execution,
        pilot_manifest_path=pilot,
        audit_output=tmp_path / "two-split-audit.json",
        expected_source_sha256=source_sha,
    )
    assert audit["status"] == "PASS"
    assert audit_path.is_file()
    assert set(audit["lanes"]["a"]["episode_keys"]).isdisjoint(
        audit["lanes"]["b"]["episode_keys"]
    )
    assert audit["lanes"]["a"]["episode_keys"] + audit["lanes"]["b"][
        "episode_keys"
    ] == list(splitter.FROZEN_KEYS)
    for lane in ("a", "b"):
        first = Path(audit["lanes"][lane]["output_dataset"])
        second = Path(second_audit["lanes"][lane]["output_dataset"])
        assert first.read_bytes() == second.read_bytes()
        with gzip.open(first, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        assert payload["episode_count"] == 10
        observed = [f'{row["trajectory_id"]}_{row["episode_id"]}' for row in payload["episodes"]]
        assert observed == audit["lanes"][lane]["episode_keys"]


def test_final_pilot_split_rejects_hash_and_key_drift(tmp_path):
    source, source_sha, execution, pilot = _fixture_contract(tmp_path)
    hash_contract = json.loads(pilot.read_text(encoding="utf-8"))
    hash_contract["overlay_sha256"] = "0" * 64
    _write_json(pilot, hash_contract)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        splitter.materialize_split(
            source,
            tmp_path / "bad-hash-a",
            tmp_path / "bad-hash-b",
            execution_manifest_path=execution,
            pilot_manifest_path=pilot,
            expected_source_sha256="0" * 64,
        )
    drifted = hash_contract
    drifted["overlay_sha256"] = source_sha
    drifted["episode_keys"][0] = "drifted"
    _write_json(pilot, drifted)
    with pytest.raises(ValueError, match="frozen manifest contract failed"):
        splitter.materialize_split(
            source,
            tmp_path / "bad-key-a",
            tmp_path / "bad-key-b",
            execution_manifest_path=execution,
            pilot_manifest_path=pilot,
            expected_source_sha256=source_sha,
        )


def _pass():
    return {"status": "PASS", "checks": {"complete": True}}


def _make_lane_result(
    root: Path,
    lane: str,
    keys: list[str],
    dataset_sha: str,
    *,
    success_key: str | None = None,
    residual_count: int = 0,
) -> None:
    code_sha = "1" * 40
    resolution_sha = "2" * 64
    map_sha = "3" * 64
    binding = {
        **_pass(),
        "lane": lane,
        "code_ref_sha": code_sha,
        "candidate_profile": "a1+b1+c0",
        "candidate_resolution_sha256": resolution_sha,
        "candidate_binding": {"bundle": "frozen-final"},
        "execution_episode_count": 10,
        "execution_episode_keys": keys,
        "dataset_sha256": dataset_sha,
        "static_map_manifest_sha256": map_sha,
        "source_receipts": {"split_audit_sha256": "4" * 64},
    }
    dgx_status = {**_pass(), "residual_count": 0}
    x86_status = {
        **_pass(),
        "residual_count": residual_count,
        "socket_residual_count": 0,
        "clock_publishers_after_stop": 0,
        "lane_lock_released": True,
        "shared_asset_lock_fd_released": True,
    }
    runtime = {
        **_pass(),
        "lane": lane,
        "profile": "final10",
        "code_ref_sha": code_sha,
        "candidate_resolution_sha256": resolution_sha,
        "input_binding": binding,
        "dgx_status": dgx_status,
        "x86_status": x86_status,
    }
    lease = _pass()
    final = {
        **_pass(),
        "runtime_summary": runtime,
        "lease_release": lease,
    }
    _write_json(root / "input_binding.json", binding)
    _write_json(root / "fast_lane_summary.json", runtime)
    _write_json(root / "fast_lane_final_summary.json", final)
    _write_json(root / "lease_release_summary.json", lease)
    _write_json(root / "remote" / "dgx" / "lane_status.json", dgx_status)
    _write_json(root / "remote" / "x86" / "isaac_status.json", x86_status)
    ordered_keys = list(reversed(keys))
    ordered_ids = [key.rsplit("_", 1)[-1] for key in ordered_keys]
    _write_json(
        root / "remote" / "x86" / "ordered_episode_manifest.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "dataset_sha256": dataset_sha,
            "dataset_episode_count": 10,
            "raw_episode_keys": keys,
            "materialized_pre_reverse_episode_keys": keys,
            "ordered_episode_keys": ordered_keys,
            "ordered_episode_ids": ordered_ids,
        },
    )
    rows = []
    runtime_rows = []
    for index, (key, episode_id) in enumerate(zip(ordered_keys, ordered_ids)):
        success = key == success_key
        mean_age = 0.01 * (index + 1)
        rows.append(
            {
                "ordinal": index + 1,
                "trajectory_id": episode_id,
                "termination_reason": "stuck" if index == 9 else "done",
                "official_metrics": {
                    "sr": int(success),
                    "os": int(index % 2 == 0),
                    "spl": 0.1 * (index + 1),
                    "ndtw": 0.5,
                    "ne_m": float(index + 1),
                },
            }
        )
        runtime_rows.append(
            {
                "ordinal": index + 1,
                "trajectory_id": episode_id,
                "controller": {
                    "all_updates": {
                        "command_age_sec": {
                            "count": 4,
                            "minimum": mean_age - 0.005,
                            "mean": mean_age,
                            "p95": mean_age + 0.005,
                            "maximum": mean_age + 0.01,
                            "percentile_method": "linear_interpolation",
                        }
                    }
                },
            }
        )
    per_episode_path = (
        root / "remote" / "x86" / "evaluator" / "attempt_001" / "per_episode.json"
    )
    _write_json(
        per_episode_path,
        {
            "expected_episode_count": 10,
            "completed_episode_count": 10,
            "episodes": rows,
        },
    )
    relative_per_episode = per_episode_path.relative_to(root).as_posix()
    _write_json(
        root / "analysis" / "episode_runtime_metrics.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "checks": {
                "episode_count_matches_summary": True,
                "every_episode_has_controller_samples": True,
                "every_episode_has_command_age_samples": True,
            },
            "episodes": runtime_rows,
            "sources": {
                relative_per_episode: {
                    "sha256": hashlib.sha256(per_episode_path.read_bytes()).hexdigest()
                }
            },
        },
    )


def _finalize_fixture(tmp_path: Path, *, one_success: bool, residual_b: int = 0):
    source_sha, execution, pilot, audit_path, audit = _materialize(tmp_path, "split")
    lane_a = tmp_path / "result-a"
    lane_b = tmp_path / "result-b"
    _make_lane_result(
        lane_a,
        "a",
        list(finalizer.FROZEN_LANE_KEYS["a"]),
        audit["lanes"]["a"]["output_sha256"],
        success_key=finalizer.FROZEN_LANE_KEYS["a"][0] if one_success else None,
    )
    _make_lane_result(
        lane_b,
        "b",
        list(finalizer.FROZEN_LANE_KEYS["b"]),
        audit["lanes"]["b"]["output_sha256"],
        residual_count=residual_b,
    )
    report = finalizer.finalize(
        lane_a,
        lane_b,
        audit_path,
        execution_manifest_path=execution,
        pilot_manifest_path=pilot,
        expected_source_sha256=source_sha,
    )
    return report, lane_a, lane_b


def _refinalize(base: Path, lane_a: Path, lane_b: Path):
    pilot = json.loads((base / "pilot.json").read_text(encoding="utf-8"))
    return finalizer.finalize(
        lane_a,
        lane_b,
        base / "split-split-audit.json",
        execution_manifest_path=base / "execution.json",
        pilot_manifest_path=base / "pilot.json",
        expected_source_sha256=pilot["overlay_sha256"],
    )


def test_finalizer_separates_integrity_from_promotion_and_aggregates(tmp_path):
    report, _lane_a, _lane_b = _finalize_fixture(tmp_path, one_success=False)
    assert report["integrity"]["status"] == "PASS"
    assert report["promotion"]["status"] == "NOT_ELIGIBLE"
    assert report["aggregate"]["episode_count"] == 20
    assert report["aggregate"]["success_count"] == 0
    assert report["aggregate"]["SR"] == 0.0
    assert report["aggregate"]["OS"] == 0.5
    assert report["aggregate"]["SPL"] == pytest.approx(0.55)
    assert report["aggregate"]["nDTW"] == 0.5
    assert report["aggregate"]["NE"] == pytest.approx(5.5)
    assert report["aggregate"]["stuck_count"] == 2
    assert report["aggregate"]["command_age_sec_mean"] == pytest.approx(0.055)
    assert report["aggregate"]["command_age_sec_episode_p95_mean"] == pytest.approx(
        0.06
    )
    assert report["aggregate"]["command_age_sec_episode_p95_max"] == pytest.approx(
        0.105
    )
    assert report["aggregate"]["command_age_sec_max"] == pytest.approx(0.11)
    assert report["lanes"]["a"]["execution_episode_keys"] == list(
        reversed(finalizer.FROZEN_LANE_KEYS["a"])
    )
    assert report["lanes"]["a"]["metrics"][0]["command_age_source"].startswith(
        "analysis/episode_runtime_metrics.json"
    )


def test_finalizer_promotes_one_success_but_rejects_residual(tmp_path):
    eligible, _lane_a, _lane_b = _finalize_fixture(
        tmp_path / "eligible", one_success=True
    )
    assert eligible["integrity"]["status"] == "PASS"
    assert eligible["promotion"]["status"] == "ELIGIBLE"
    assert eligible["aggregate"]["success_count"] == 1
    assert eligible["aggregate"]["SR"] == 0.05

    residual, _lane_a, _lane_b = _finalize_fixture(
        tmp_path / "residual", one_success=True, residual_b=1
    )
    assert residual["integrity"]["status"] == "FAIL"
    assert residual["integrity"]["checks"]["both_lanes_zero_residual"] is False
    assert residual["promotion"]["status"] == "NOT_EVALUABLE"
    assert residual["aggregate"] is None


def test_finalizer_rejects_episode_key_drift(tmp_path):
    report, lane_a, _lane_b = _finalize_fixture(tmp_path, one_success=True)
    assert report["integrity"]["status"] == "PASS"
    metrics_path = (
        lane_a
        / "remote"
        / "x86"
        / "evaluator"
        / "attempt_001"
        / "per_episode.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["episodes"][0]["trajectory_id"] = "cross-lane-drift"
    _write_json(metrics_path, metrics)
    drifted = _refinalize(tmp_path, lane_a, _lane_b)
    assert drifted["integrity"]["status"] == "FAIL"
    assert drifted["lanes"]["a"]["checks"]["metrics_complete_and_ordered"] is False
    assert drifted["promotion"]["status"] == "NOT_EVALUABLE"


def test_finalizer_rejects_runtime_command_age_missing_or_identity_drift(tmp_path):
    missing_base = tmp_path / "missing"
    report, lane_a, lane_b = _finalize_fixture(missing_base, one_success=True)
    assert report["integrity"]["status"] == "PASS"
    runtime_path = lane_a / "analysis" / "episode_runtime_metrics.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    del runtime["episodes"][0]["controller"]["all_updates"]["command_age_sec"]
    _write_json(runtime_path, runtime)
    missing = _refinalize(missing_base, lane_a, lane_b)
    assert missing["integrity"]["status"] == "FAIL"
    assert "command-age" in missing["lanes"]["a"]["source_errors"]["episode_metrics"]

    drift_base = tmp_path / "runtime-drift"
    report, lane_a, lane_b = _finalize_fixture(drift_base, one_success=True)
    assert report["integrity"]["status"] == "PASS"
    runtime_path = lane_a / "analysis" / "episode_runtime_metrics.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["episodes"][0]["trajectory_id"] = "wrong-runtime-identity"
    _write_json(runtime_path, runtime)
    drifted = _refinalize(drift_base, lane_a, lane_b)
    assert drifted["integrity"]["status"] == "FAIL"
    assert "trajectory_id/order" in drifted["lanes"]["a"]["source_errors"][
        "episode_metrics"
    ]


def test_finalizer_rejects_order_manifest_drift(tmp_path):
    report, lane_a, lane_b = _finalize_fixture(tmp_path, one_success=True)
    assert report["integrity"]["status"] == "PASS"
    order_path = lane_a / "remote" / "x86" / "ordered_episode_manifest.json"
    order = json.loads(order_path.read_text(encoding="utf-8"))
    order["materialized_pre_reverse_episode_keys"] = list(
        reversed(order["materialized_pre_reverse_episode_keys"])
    )
    _write_json(order_path, order)
    drifted = _refinalize(tmp_path, lane_a, lane_b)
    assert drifted["integrity"]["status"] == "FAIL"
    assert "raw/pre-reverse" in drifted["lanes"]["a"]["source_errors"][
        "episode_metrics"
    ]
