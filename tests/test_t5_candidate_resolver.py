from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.resolve_t5_lane_a_candidate import _code_bundle_sha256, resolve


ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "scripts/resolve_t5_lane_a_candidate.py"
FAST = ROOT / "coordination/run_t5_fast_lane_online.sh"
DGX = ROOT / "scripts/run_t5_dgx_lane.sh"


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def candidate_root(tmp_path: Path) -> Path:
    code_paths = ["code/runtime.py", "code/adapter.py"]
    for relative, content in zip(code_paths, ("runtime = 1\n", "adapter = 1\n")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    configs = {
        "action_observation_recovery.json": {
            "family_id": "action_observation_recovery",
            "variants": [
                {
                    "candidate_id": "a0_action_gate_only",
                    "runtime_overrides": {
                        "INTERNNAV_T5_CANDIDATE_PROFILE": "baseline"
                    },
                },
                {
                    "candidate_id": "a1_action_gate_recovery_a",
                    "runtime_overrides": {
                        "INTERNNAV_T5_CANDIDATE_PROFILE": "recovery_a"
                    },
                },
            ],
        },
        "camera_history_alignment.json": {
            "family_id": "camera_history_alignment",
            "variants": [
                {
                    "candidate_id": "b0_go2_history_off",
                    "runtime_overrides": {
                        "INTERNVLA_T4_VIEW_MODE": "go2_view",
                        "INTERNVLA_T4_HISTORY_MODE": "off",
                    },
                },
                {
                    "candidate_id": "b1_go2_history_on",
                    "runtime_overrides": {
                        "INTERNVLA_T4_VIEW_MODE": "go2_view",
                        "INTERNVLA_T4_HISTORY_MODE": "on",
                    },
                },
            ],
        },
        "trajectory_horizon_refresh.json": {
            "family_id": "trajectory_horizon_refresh",
            "variants": [
                {
                    "candidate_id": "c0_upstream_mean",
                    "runtime_overrides": {
                        "INTERNVLA_T5_TRAJECTORY_RERANK": "0",
                        "INTERNVLA_T4_PROGRESS_HORIZON_SEC": "5.0",
                        "INTERNVLA_T4_REFRESH_DISTANCE_M": "0.30",
                        "INTERNVLA_T4_REFRESH_TIME_SEC": "2.0",
                        "INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC": "5.0",
                    },
                },
                {
                    "candidate_id": "c1_depth_geometry_rerank",
                    "runtime_overrides": {
                        "INTERNVLA_T5_TRAJECTORY_RERANK": "1",
                        "INTERNVLA_T4_PROGRESS_HORIZON_SEC": "5.0",
                        "INTERNVLA_T4_REFRESH_DISTANCE_M": "0.30",
                        "INTERNVLA_T4_REFRESH_TIME_SEC": "2.0",
                        "INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC": "5.0",
                    },
                },
                {
                    "candidate_id": "c2_rerank_short_fresh",
                    "runtime_overrides": {
                        "INTERNVLA_T5_TRAJECTORY_RERANK": "1",
                        "INTERNVLA_T4_PROGRESS_HORIZON_SEC": "4.0",
                        "INTERNVLA_T4_REFRESH_DISTANCE_M": "0.20",
                        "INTERNVLA_T4_REFRESH_TIME_SEC": "1.5",
                        "INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC": "3.0",
                    },
                },
            ],
        },
    }
    base = tmp_path / "configs/internnav_t5/lane_a_candidates"
    for name, value in configs.items():
        write_json(base / name, value)
    families = []
    ids = (
        ("action_observation_recovery", "action_observation_recovery.json"),
        ("camera_history_alignment", "camera_history_alignment.json"),
        ("trajectory_horizon_refresh", "trajectory_horizon_refresh.json"),
    )
    for family_id, name in ids:
        value = configs[name]
        families.append(
            {
                "family_id": family_id,
                "config": f"configs/internnav_t5/lane_a_candidates/{name}",
                "config_sha256": canonical_sha256(value),
                "candidate_ids": [item["candidate_id"] for item in value["variants"]],
            }
        )
    write_json(
        base / "manifest.json",
        {
            "schema_version": 1,
            "lane": "a",
            "successive_halving": {
                "rounds": [
                    {"cumulative_episode_count": 1},
                    {"cumulative_episode_count": 3},
                    {"cumulative_episode_count": 5},
                ]
            },
            "fixed_evaluation": {
                "episode_count": 5,
                "episode_keys": ["e1", "e2", "e3", "e4", "e5"],
            },
            "provenance_contract": {
                "code_bundle_sha256": _code_bundle_sha256(tmp_path, code_paths),
                "code_paths": code_paths,
                "result_required_fields": [
                    "predecessor_candidate_ids",
                    "code_bundle_sha256",
                    "candidate_config_sha256",
                    "candidate_manifest_sha256",
                ],
            },
            "candidate_families": families,
        },
    )
    return tmp_path


def update_family_sha(root: Path, family_index: int) -> None:
    manifest_path = root / "configs/internnav_t5/lane_a_candidates/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family = manifest["candidate_families"][family_index]
    config = json.loads((root / family["config"]).read_text(encoding="utf-8"))
    family["config_sha256"] = canonical_sha256(config)
    write_json(manifest_path, manifest)


def test_full_composite_resolves_only_preregistered_environment(tmp_path: Path) -> None:
    root = candidate_root(tmp_path)
    result = resolve(root, "a1+b0+c2")
    assert result["status"] == "PASS"
    assert result["selected_candidate_ids"] == [
        "a1_action_gate_recovery_a",
        "b0_go2_history_off",
        "c2_rerank_short_fresh",
    ]
    assert result["effective_candidate_profile"] == "recovery_a"
    assert result["provenance"]["predecessor_candidate_ids"] == [
        "a1_action_gate_recovery_a",
        "b0_go2_history_off",
    ]
    unsigned = dict(result)
    assert unsigned.pop("resolution_sha256") == canonical_sha256(unsigned)
    assert result["runtime_overrides"] == {
        "INTERNNAV_T5_CANDIDATE_PROFILE": "recovery_a",
        "INTERNVLA_T4_HISTORY_MODE": "off",
        "INTERNVLA_T4_PROGRESS_HORIZON_SEC": "4.0",
        "INTERNVLA_T4_REFRESH_DISTANCE_M": "0.20",
        "INTERNVLA_T4_REFRESH_TIME_SEC": "1.5",
        "INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC": "3.0",
        "INTERNVLA_T4_VIEW_MODE": "go2_view",
        "INTERNVLA_T5_TRAJECTORY_RERANK": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(RESOLVER),
            "--root",
            str(root),
            "--selector",
            "a1+b0+c2",
            "--format",
            "env",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert set(completed.stdout.splitlines()) == {
        f"{key}={value}" for key, value in result["runtime_overrides"].items()
    }


@pytest.mark.parametrize("selector", ("a0", "a1", "a0+b0", "a1+b1"))
def test_dependency_ordered_partial_composites_are_supported(
    tmp_path: Path, selector: str
) -> None:
    result = resolve(candidate_root(tmp_path), selector)
    assert result["candidate_selector"] == selector
    assert len(result["selected_candidate_ids"]) == len(selector.split("+"))
    assert result["successive_halving_episode_counts"] == [1, 3, 5]


def test_legacy_profiles_remain_compatible_without_candidate_manifest(
    tmp_path: Path,
) -> None:
    assert resolve(tmp_path, "baseline")["runtime_overrides"] == {
        "INTERNNAV_T5_CANDIDATE_PROFILE": "baseline"
    }
    assert resolve(tmp_path, "recovery_a")["effective_candidate_profile"] == (
        "recovery_a"
    )


@pytest.mark.parametrize(
    "selector",
    ("", "b1", "c2", "a2", "a1+c2", "a1+b1+c3", "a1+b1+c2;id"),
)
def test_selector_grammar_cannot_escape_preregistered_dependency_order(
    tmp_path: Path, selector: str
) -> None:
    with pytest.raises(ValueError):
        resolve(candidate_root(tmp_path), selector)


@pytest.mark.parametrize(
    "selector",
    (
        "a0+b0+c0",
        "a0+b0+c1",
        "a0+b0+c2",
        "a0+b1+c0",
        "a0+b1+c1",
        "a0+b1+c2",
    ),
)
def test_trajectory_family_requires_recovery_enabled_predecessor(
    tmp_path: Path, selector: str
) -> None:
    with pytest.raises(ValueError, match="requires recovery-enabled a1"):
        resolve(candidate_root(tmp_path), selector)


@pytest.mark.parametrize(
    ("key", "value"),
    (("LD_PRELOAD", "/tmp/inject.so"), ("INTERNVLA_T4_HISTORY_MODE", "maybe")),
)
def test_tampered_manifest_cannot_inject_key_or_value(
    tmp_path: Path, key: str, value: str
) -> None:
    root = candidate_root(tmp_path)
    path = (
        root
        / "configs/internnav_t5/lane_a_candidates/camera_history_alignment.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config["variants"][0]["runtime_overrides"][key] = value
    write_json(path, config)
    update_family_sha(root, 1)
    with pytest.raises(ValueError, match="not allowed|override scope"):
        resolve(root, "a0+b0")


@pytest.mark.parametrize(
    ("family_index", "key"),
    (
        (0, "INTERNNAV_T5_CANDIDATE_PROFILE"),
        (1, "INTERNVLA_T4_VIEW_MODE"),
        (2, "INTERNVLA_T4_REFRESH_TIME_SEC"),
    ),
)
def test_each_family_requires_exact_override_key_set(
    tmp_path: Path, family_index: int, key: str
) -> None:
    root = candidate_root(tmp_path)
    manifest_path = root / "configs/internnav_t5/lane_a_candidates/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = root / manifest["candidate_families"][family_index]["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["variants"][0]["runtime_overrides"][key]
    write_json(config_path, config)
    update_family_sha(root, family_index)
    with pytest.raises(ValueError, match="overrides missing|override scope"):
        resolve(root, "a1+b0+c1")


def test_code_bundle_mismatch_fails_closed(tmp_path: Path) -> None:
    root = candidate_root(tmp_path)
    (root / "code/runtime.py").write_text("runtime = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code bundle changed"):
        resolve(root, "a1")


def test_code_bundle_treats_windows_crlf_as_deployed_lf(tmp_path: Path) -> None:
    root = candidate_root(tmp_path)
    path = root / "code/runtime.py"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    result = resolve(root, "a1")
    manifest = json.loads((
        root / "configs/internnav_t5/lane_a_candidates/manifest.json"
    ).read_text(encoding="utf-8"))
    assert result["provenance"]["code_bundle_sha256"] == (
        manifest["provenance_contract"]["code_bundle_sha256"]
    )


def test_predecessor_provenance_contract_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    root = candidate_root(tmp_path)
    path = root / "configs/internnav_t5/lane_a_candidates/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provenance_contract"]["result_required_fields"].remove(
        "predecessor_candidate_ids"
    )
    write_json(path, manifest)
    with pytest.raises(ValueError, match="predecessor provenance"):
        resolve(root, "a1+b0")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("episode_count", 4),
        ("episode_keys", ["e1", "e2", "e3", "e4", "e4"]),
    ),
)
def test_fixed_episode_binding_mismatch_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    root = candidate_root(tmp_path)
    path = root / "configs/internnav_t5/lane_a_candidates/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["fixed_evaluation"][field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError, match="fixed episode"):
        resolve(root, "a1")


def test_runners_use_array_forwarding_and_never_evaluate_candidate_output() -> None:
    fast = FAST.read_text(encoding="utf-8")
    dgx = DGX.read_text(encoding="utf-8")
    assert "resolve_t5_lane_a_candidate.py" in fast
    assert "resolve_t5_lane_a_candidate.py" in dgx
    assert '"${candidate_env[@]}"' in dgx
    assert dgx.count('env "${common_env[@]}" "${recovery_env[@]}" "${candidate_env[@]}"') == 2
    assert 'setsid env "${common_env[@]}" "${candidate_env[@]}"' in dgx
    candidate_block = dgx[
        dgx.index("mapfile -t candidate_env") : dgx.index(
            'if test "$effective_candidate_profile" = recovery_a; then'
        )
    ]
    assert "eval" not in candidate_block
    assert "source" not in candidate_block
    assert "screen1|screen3|fixed5" in fast
    assert "a0+b0+c0" not in fast
    assert "a0+b0+c0" not in dgx
    for marker in (
        "resolution_sha256",
        "code_bundle_sha256",
        "fixed_episode_keys",
        "predecessor_candidate_ids",
        "registered_config_sha256_by_family",
    ):
        assert marker in fast
        assert marker in dgx
