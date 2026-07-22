from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_t5_lane_a_candidates import (  # noqa: E402
    aggregate_prefix,
    analyze,
    canonical_json_sha256,
    code_bundle_sha256,
)


MANIFEST_PATH = (
    ROOT / "configs" / "internnav_t5" / "lane_a_candidates" / "manifest.json"
)


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _context(manifest):
    config_hashes = {}
    for family in manifest["candidate_families"]:
        value = json.loads((ROOT / family["config"]).read_text(encoding="utf-8"))
        config_hashes[family["family_id"]] = canonical_json_sha256(value)
    return {
        "manifest_sha256": canonical_json_sha256(manifest),
        "config_sha256_by_family": config_hashes,
        "actual_code_bundle_sha256": code_bundle_sha256(
            ROOT, manifest["provenance_contract"]["code_paths"]
        ),
    }


def _family(manifest, candidate_id):
    return next(
        family
        for family in manifest["candidate_families"]
        if candidate_id in family["candidate_ids"]
    )


def test_code_bundle_hash_is_checkout_newline_independent(tmp_path: Path) -> None:
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    lf_root.mkdir()
    crlf_root.mkdir()
    relative_paths = ["alpha.py", "nested/beta.sh"]
    (lf_root / "nested").mkdir()
    (crlf_root / "nested").mkdir()
    for relative in relative_paths:
        (lf_root / relative).write_bytes(b"one\ntwo\n")
        (crlf_root / relative).write_bytes(b"one\r\ntwo\r\n")
    assert code_bundle_sha256(lf_root, relative_paths) == code_bundle_sha256(
        crlf_root, relative_paths
    )


def _result(
    candidate_id: str,
    *,
    predecessors: list[str],
    success_count: int,
    ne: float,
    stuck_count: int,
    spl: float = 0.2,
    ndtw: float = 0.3,
    command_age_sec: float = 0.1,
    oracle_success_count: int | None = None,
):
    manifest = _manifest()
    keys = manifest["fixed_evaluation"]["episode_keys"]
    family = _family(manifest, candidate_id)
    oracle_count = success_count if oracle_success_count is None else oracle_success_count
    return {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "provenance": {
            "predecessor_candidate_ids": predecessors,
            "code_bundle_sha256": manifest["provenance_contract"][
                "code_bundle_sha256"
            ],
            "candidate_config_sha256": family["config_sha256"],
            "candidate_manifest_sha256": canonical_json_sha256(manifest),
        },
        "episodes": [
            {
                "episode_key": key,
                "success": index < success_count,
                # OS remains diagnostic input and must not affect ranking.
                "oracle_success": index < oracle_count,
                "SPL": spl if index < success_count else 0.0,
                "nDTW": ndtw,
                "NE": ne,
                "command_age_sec": command_age_sec,
                "stuck": index < stuck_count,
            }
            for index, key in enumerate(keys)
        ],
    }


def _analyze(results):
    manifest = _manifest()
    return analyze(manifest, results, **_context(manifest))


def test_manifest_preregisters_three_disjoint_candidate_families() -> None:
    manifest = _manifest()
    families = manifest["candidate_families"]
    assert [item["family_id"] for item in families] == [
        "action_observation_recovery",
        "camera_history_alignment",
        "trajectory_horizon_refresh",
    ]
    all_ids = [value for family in families for value in family["candidate_ids"]]
    assert len(all_ids) == len(set(all_ids)) == 7
    assert [
        item["cumulative_episode_count"]
        for item in manifest["successive_halving"]["rounds"]
    ] == [1, 3, 5]
    assert manifest["successive_halving"]["ranking_priority"] == [
        "success_count_desc",
        "stuck_count_asc",
        "mean_SPL_desc",
        "mean_nDTW_desc",
        "mean_NE_asc",
        "mean_command_age_sec_asc",
        "candidate_id_asc",
    ]


def test_analyzer_reports_first_family_then_blocks_predecessors() -> None:
    report = _analyze({})
    assert report["status"] == "INCOMPLETE"
    assert report["plateau"]["status"] == "NOT_EVALUATED"
    assert len(report["pending_runs"]) == 2
    assert report["family_reports"][1]["status"] == "BLOCKED_BY_PREDECESSOR"


def _complete_results():
    return {
        "a0_action_gate_only": _result(
            "a0_action_gate_only",
            predecessors=[],
            success_count=0,
            ne=6.0,
            stuck_count=3,
        ),
        "a1_action_gate_recovery_a": _result(
            "a1_action_gate_recovery_a",
            predecessors=[],
            success_count=1,
            ne=5.0,
            stuck_count=2,
        ),
        "b0_go2_history_off": _result(
            "b0_go2_history_off",
            predecessors=["a1_action_gate_recovery_a"],
            success_count=1,
            ne=5.5,
            stuck_count=3,
        ),
        "b1_go2_history_on": _result(
            "b1_go2_history_on",
            predecessors=["a1_action_gate_recovery_a"],
            success_count=1,
            ne=4.9,
            stuck_count=2,
        ),
        "c0_upstream_mean": _result(
            "c0_upstream_mean",
            predecessors=["a1_action_gate_recovery_a", "b1_go2_history_on"],
            success_count=1,
            ne=5.2,
            stuck_count=3,
        ),
        "c1_depth_geometry_rerank": _result(
            "c1_depth_geometry_rerank",
            predecessors=["a1_action_gate_recovery_a", "b1_go2_history_on"],
            success_count=1,
            ne=4.8,
            stuck_count=2,
        ),
        "c2_rerank_short_fresh": _result(
            "c2_rerank_short_fresh",
            predecessors=["a1_action_gate_recovery_a", "b1_go2_history_on"],
            success_count=0,
            ne=5.0,
            stuck_count=1,
        ),
    }


def test_successive_halving_selects_each_family_winner_and_emits_plateau() -> None:
    report = _analyze(_complete_results())
    assert report["status"] == "COMPLETE"
    assert report["family_winners"] == [
        "a1_action_gate_recovery_a",
        "b1_go2_history_on",
        "c1_depth_geometry_rerank",
    ]
    assert report["plateau"]["status"] == "PLATEAU"


def test_frozen_ranking_prefers_less_stuck_before_spl_ndtw_ne_age_or_os() -> None:
    results = {
        "a0_action_gate_only": _result(
            "a0_action_gate_only",
            predecessors=[],
            success_count=1,
            ne=9.0,
            stuck_count=0,
            spl=0.01,
            ndtw=0.01,
            command_age_sec=1.0,
            oracle_success_count=0,
        ),
        "a1_action_gate_recovery_a": _result(
            "a1_action_gate_recovery_a",
            predecessors=[],
            success_count=1,
            ne=1.0,
            stuck_count=1,
            spl=1.0,
            ndtw=1.0,
            command_age_sec=0.01,
            oracle_success_count=5,
        ),
    }
    report = _analyze(results)
    assert report["family_reports"][0]["winner"] == "a0_action_gate_only"


def test_command_age_is_the_last_metric_tie_break() -> None:
    results = {
        "a0_action_gate_only": _result(
            "a0_action_gate_only",
            predecessors=[],
            success_count=1,
            ne=4.0,
            stuck_count=0,
            command_age_sec=0.2,
        ),
        "a1_action_gate_recovery_a": _result(
            "a1_action_gate_recovery_a",
            predecessors=[],
            success_count=1,
            ne=4.0,
            stuck_count=0,
            command_age_sec=0.1,
        ),
    }
    report = _analyze(results)
    assert report["family_reports"][0]["winner"] == "a1_action_gate_recovery_a"


def test_analyzer_rejects_predecessor_or_sha_drift() -> None:
    results = _complete_results()
    results["b0_go2_history_off"]["provenance"]["predecessor_candidate_ids"] = []
    try:
        _analyze(results)
    except ValueError as exc:
        assert "predecessor tuple mismatch" in str(exc)
    else:
        raise AssertionError("predecessor drift was accepted")


def test_analyzer_rejects_unregistered_result_identity() -> None:
    try:
        _analyze({"not_preregistered": {"episodes": []}})
    except ValueError as exc:
        assert "unregistered" in str(exc)
    else:
        raise AssertionError("unregistered candidate was accepted")


@pytest.mark.parametrize("value", ["false", "true", 2, -1])
def test_stuck_metric_rejects_non_boolean_non_binary_values(value) -> None:
    payload = _result(
        "a0_action_gate_only",
        predecessors=[],
        success_count=0,
        ne=1.0,
        stuck_count=0,
    )
    key = payload["episodes"][0]["episode_key"]
    payload["episodes"][0]["stuck"] = value
    with pytest.raises(ValueError, match="stuck"):
        aggregate_prefix(payload, [key], 1)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("SPL", -0.01, "SPL"),
        ("SPL", 1.01, "SPL"),
        ("nDTW", -0.01, "nDTW"),
        ("nDTW", 1.01, "nDTW"),
        ("NE", -0.01, "NE"),
        ("command_age_sec", -0.01, "command age"),
    ],
)
def test_candidate_metric_ranges_fail_closed(metric, value, message) -> None:
    payload = _result(
        "a0_action_gate_only",
        predecessors=[],
        success_count=0,
        ne=1.0,
        stuck_count=0,
    )
    key = payload["episodes"][0]["episode_key"]
    payload["episodes"][0][metric] = value
    with pytest.raises(ValueError, match=message):
        aggregate_prefix(payload, [key], 1)
