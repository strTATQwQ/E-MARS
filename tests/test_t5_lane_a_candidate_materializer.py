from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_t5_lane_a_candidates import (  # noqa: E402
    aggregate_prefix,
    canonical_json_sha256,
    code_bundle_sha256,
)
from materialize_t5_lane_a_candidate_result import (  # noqa: E402
    materialize_blocked_family,
    materialize_candidate_result,
)
from resolve_t5_lane_a_candidate import resolve  # noqa: E402


MANIFEST_PATH = (
    ROOT / "configs" / "internnav_t5" / "lane_a_candidates" / "manifest.json"
)
CODE_REF = "bfc08bae39710500a3d8693b58a8b72b904175e5"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _episode(key: str, index: int) -> dict[str, object]:
    success = index == 0
    return {
        "trajectory_id": key,
        "success": success,
        "stuck": not success,
        "termination_reason": "success" if success else "stuck",
        "official_metrics": {
            "sr": int(success),
            "os": int(success),
            "spl": 0.4 if success else 0.0,
            "ne_m": 1.0 + index,
            "ndtw": 0.7 - index * 0.05,
        },
        "command_age_sec": 0.05 + index * 0.01,
    }


def _fast_result(tmp_path: Path, selector: str = "a1", profile: str = "screen3") -> Path:
    run_id = "candidate-run-001"
    root = tmp_path / f"fast-lane-a-{profile}-{run_id}"
    root.mkdir()
    resolution = resolve(ROOT, selector)
    _write(root / "candidate_resolution.json", resolution)
    _write(root / "audits" / "dgx_candidate_resolution.json", resolution)
    _write(root / "remote" / "dgx" / "candidate_resolution.json", resolution)
    counts = {"screen1": 1, "screen3": 3, "fixed5": 5}
    count = counts[profile]
    fixed_keys = resolution["provenance"]["fixed_episode_keys"]
    binding = {
        "schema_version": 1,
        "status": "PASS",
        "checks": {"sealed": True},
        "lane": "a",
        "candidate_profile": selector,
        "candidate_resolution_sha256": resolution["resolution_sha256"],
        "candidate_resolution_file_sha256": __import__("hashlib").sha256(
            (root / "candidate_resolution.json").read_bytes()
        ).hexdigest(),
        "candidate_binding": resolution["canonical_binding"],
        "execution_profile": profile,
        "execution_episode_count": count,
        "execution_episode_keys": fixed_keys[:count],
        "episode_keys": fixed_keys,
        "code_ref_sha": CODE_REF,
        "source_receipts": {
            "prepare_summary_sha256": "1" * 64,
            "prepare_final_sha256": "2" * 64,
            "fast_prepare_input_sha256": "3" * 64,
            "dataset_audit_sha256": "4" * 64,
        },
    }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "checks": {"online": True},
        "lane": "a",
        "profile": profile,
        "run_id": run_id,
        "code_ref_sha": CODE_REF,
        "candidate_profile": selector,
        "candidate_resolution_sha256": resolution["resolution_sha256"],
        "candidate_binding": resolution["canonical_binding"],
        "input_binding": binding,
    }
    final = {
        "schema_version": 1,
        "status": "PASS",
        "checks": {"runtime_pass": True},
        "runtime_summary": summary,
    }
    _write(root / "input_binding.json", binding)
    _write(root / "fast_lane_summary.json", summary)
    _write(root / "fast_lane_final_summary.json", final)
    # Evaluator execution order may be reversed; materialization must restore
    # the frozen prefix order by exact episode identity.
    _write(
        root
        / "remote"
        / "x86"
        / "evaluator"
        / "attempt_001"
        / "per_episode.json",
        {
            "schema_version": 1,
            "expected_episode_count": count,
            "completed_episode_count": count,
            "episodes": list(
                reversed([_episode(key, index) for index, key in enumerate(fixed_keys[:count])])
            ),
        },
    )
    return root


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_materializer_preserves_exact_episode_metrics_and_provenance(
    tmp_path: Path,
) -> None:
    root = _fast_result(tmp_path)
    payload = materialize_candidate_result(root)
    manifest = _manifest()
    expected_keys = manifest["fixed_evaluation"]["episode_keys"][:3]

    assert payload["candidate_id"] == "a1_action_gate_recovery_a"
    assert [item["episode_key"] for item in payload["episodes"]] == expected_keys
    assert payload["episodes"][0] == {
        "episode_key": expected_keys[0],
        "success": True,
        "stuck": False,
        "SPL": 0.4,
        "nDTW": 0.7,
        "NE": 1.0,
        "command_age_sec": 0.05,
        "termination_reason": "success",
        "oracle_success": True,
        "source_record_sha256": canonical_json_sha256(_episode(expected_keys[0], 0)),
    }
    assert payload["provenance"] == {
        "predecessor_candidate_ids": [],
        "code_bundle_sha256": manifest["provenance_contract"]["code_bundle_sha256"],
        "candidate_config_sha256": manifest["candidate_families"][0]["config_sha256"],
        "candidate_manifest_sha256": canonical_json_sha256(manifest),
    }
    assert payload["source_provenance"]["code_ref_sha"] == CODE_REF
    assert payload["source_provenance"]["episode_metrics_relative_path"].endswith(
        "per_episode.json"
    )
    aggregate = aggregate_prefix(payload, expected_keys, 3)
    assert aggregate["complete"] is True
    assert aggregate["success_count"] == 1
    assert aggregate["stuck_count"] == 2


@pytest.mark.parametrize(
    ("delete_path", "message"),
    [
        (("official_metrics", "ndtw"), "nDTW"),
        (("command_age_sec",), "command_age_sec"),
        (("official_metrics", "spl"), "SPL"),
        (("official_metrics", "ne_m"), "NE"),
    ],
)
def test_materializer_fails_closed_on_missing_per_episode_metric(
    tmp_path: Path, delete_path: tuple[str, ...], message: str
) -> None:
    root = _fast_result(tmp_path)
    metrics_path = next((root / "remote" / "x86" / "evaluator").rglob("per_episode.json"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = metrics["episodes"][0]
    target = record
    for name in delete_path[:-1]:
        target = target[name]
    del target[delete_path[-1]]
    _write(metrics_path, metrics)
    with pytest.raises(ValueError, match=message):
        materialize_candidate_result(root)


def test_materializer_requires_an_explicit_episode_success(tmp_path: Path) -> None:
    root = _fast_result(tmp_path)
    metrics_path = next((root / "remote" / "x86" / "evaluator").rglob("per_episode.json"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["episodes"][0].pop("success")
    metrics["episodes"][0]["official_metrics"].pop("sr")
    _write(metrics_path, metrics)
    with pytest.raises(ValueError, match="success"):
        materialize_candidate_result(root)


def test_materializer_aggregates_only_explicit_sim_time_command_age(
    tmp_path: Path,
) -> None:
    root = _fast_result(tmp_path, profile="screen1")
    metrics_path = next(
        (root / "remote" / "x86" / "evaluator").rglob("per_episode.json")
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["episodes"][0].pop("command_age_sec")
    _write(metrics_path, metrics)
    key = resolve(ROOT, "a1")["provenance"]["fixed_episode_keys"][0]
    episode_id = key.rsplit("_", 1)[1]
    _write(
        root / "remote" / "x86" / "ordered_episode_manifest.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "ordered_episode_keys": [key],
            "ordered_episode_ids": [episode_id],
        },
    )
    records = root / "remote" / "dgx" / "onboard" / "controller_records.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {
                    "episode_id": f"a::{episode_id}",
                    "reset_generation": 0,
                    "state_only": False,
                    "identity_valid": True,
                    "command_age_sec": 0.1,
                },
                {
                    "episode_id": f"a::{episode_id}",
                    "reset_generation": 0,
                    "state_only": False,
                    "identity_valid": True,
                    "command_age_sec": 0.3,
                },
                {
                    "episode_id": f"a::{episode_id}",
                    "reset_generation": 0,
                    "state_only": True,
                    "identity_valid": True,
                    "command_age_sec": 9.0,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    payload = materialize_candidate_result(root)

    assert payload["episodes"][0]["command_age_sec"] == pytest.approx(0.2)
    provenance = payload["source_provenance"]["command_age_provenance"]
    assert provenance["sample_count_by_episode"] == {key: 2}
    assert set(provenance["source_sha256"]) == {
        "controller_records",
        "ordered_episode_manifest",
    }


def test_materializer_uses_actual_loader_order_for_command_age_identity(
    tmp_path: Path,
) -> None:
    root = _fast_result(tmp_path, profile="screen3")
    metrics_path = next(
        (root / "remote" / "x86" / "evaluator").rglob("per_episode.json")
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in metrics["episodes"]:
        row.pop("command_age_sec")
    _write(metrics_path, metrics)

    frozen_keys = resolve(ROOT, "a1")["provenance"]["fixed_episode_keys"][:3]
    ordered_keys = list(reversed(frozen_keys))
    ordered_ids = [key.rsplit("_", 1)[1] for key in ordered_keys]
    _write(
        root / "remote" / "x86" / "ordered_episode_manifest.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "ordered_episode_keys": ordered_keys,
            "ordered_episode_ids": ordered_ids,
        },
    )
    records = root / "remote" / "dgx" / "onboard" / "controller_records.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_id": f"a::{episode_id}",
                    "reset_generation": generation,
                    "state_only": False,
                    "identity_valid": True,
                    "command_age_sec": 0.1 + generation * 0.1,
                },
                sort_keys=True,
            )
            for generation, episode_id in enumerate(ordered_ids)
        )
        + "\n",
        encoding="utf-8",
    )

    payload = materialize_candidate_result(root)

    by_key = {row["episode_key"]: row for row in payload["episodes"]}
    assert by_key[ordered_keys[0]]["command_age_sec"] == pytest.approx(0.1)
    assert by_key[ordered_keys[1]]["command_age_sec"] == pytest.approx(0.2)
    assert by_key[ordered_keys[2]]["command_age_sec"] == pytest.approx(0.3)
    assert [row["episode_key"] for row in payload["episodes"]] == frozen_keys


def test_materializer_rejects_aggregate_substitution(tmp_path: Path) -> None:
    root = _fast_result(tmp_path)
    metrics_path = next((root / "remote" / "x86" / "evaluator").rglob("per_episode.json"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in metrics["episodes"]:
        row.pop("command_age_sec")
        row["official_metrics"].pop("ndtw")
    metrics["mean_command_age_sec"] = 0.1
    metrics["nDTW"] = 0.5
    _write(metrics_path, metrics)
    with pytest.raises(ValueError, match="nDTW"):
        materialize_candidate_result(root)


def test_materializer_rejects_cross_host_resolution_drift(tmp_path: Path) -> None:
    root = _fast_result(tmp_path)
    path = root / "remote" / "dgx" / "candidate_resolution.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["candidate_selector"] = "a0"
    _write(path, value)
    with pytest.raises(ValueError, match="differs"):
        materialize_candidate_result(root)


def _analysis(tmp_path: Path, winner: str) -> Path:
    manifest = _manifest()
    code_sha = code_bundle_sha256(
        ROOT, manifest["provenance_contract"]["code_paths"]
    )
    value = {
        "schema_version": 1,
        "status": "INCOMPLETE",
        "validated_provenance": {
            "candidate_manifest_sha256": canonical_json_sha256(manifest),
            "code_bundle_sha256": code_sha,
            "config_sha256_by_family": {
                family["family_id"]: family["config_sha256"]
                for family in manifest["candidate_families"]
            },
        },
        "family_reports": [
            {
                "family_id": "action_observation_recovery",
                "status": "COMPLETE",
                "winner": winner,
            }
        ],
    }
    path = tmp_path / "analysis.json"
    _write(path, value)
    return path


def test_outer_can_mark_c_family_blocked_for_exact_a0_winner(tmp_path: Path) -> None:
    analysis = _analysis(tmp_path, "a0_action_gate_only")
    marker = materialize_blocked_family(analysis)
    assert marker["status"] == "BLOCKED"
    assert marker["family_id"] == "trajectory_horizon_refresh"
    assert marker["reason_code"] == "RECOVERY_A_NOT_SELECTED"
    assert marker["blocked_by_candidate_id"] == "a0_action_gate_only"


def test_outer_c_block_rejects_a1_winner(tmp_path: Path) -> None:
    analysis = _analysis(tmp_path, "a1_action_gate_recovery_a")
    with pytest.raises(ValueError, match="requires an exact a0"):
        materialize_blocked_family(analysis)
