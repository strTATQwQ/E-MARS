from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


receipt_builder = _load(
    "build_t5_final_pilot_prepare_receipt",
    "scripts/build_t5_final_pilot_prepare_receipt.py",
)
binding_resolver = _load(
    "resolve_t5_final_pilot_lane_binding",
    "scripts/resolve_t5_final_pilot_lane_binding.py",
)
bundle_finalizer = _load(
    "finalize_t5_final_pilot_bundle",
    "scripts/finalize_t5_final_pilot_bundle.py",
)


CODE_SHA = "1" * 40
LANE_KEYS = {
    "a": [f"ta{index}_ea{index}" for index in range(10)],
    "b": [f"tb{index}_eb{index}" for index in range(10)],
}
ALL_KEYS = LANE_KEYS["a"] + LANE_KEYS["b"]
SCENES = ["scene_a", "scene_b", "scene_c", "scene_d", "scene_e"]


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> Path:
    value = {
        "schema_version": 2,
        "status": "PASS",
        "candidate_selector": "a1+b1+c1",
        "canonical_binding": {
            "candidate_selector": "a1+b1+c1",
            "fixed_episode_count": 5,
        },
        "provenance": {
            "fixed_episode_count": 5,
            "fixed_episode_keys": [f"dev_{index}" for index in range(5)],
        },
    }
    value["resolution_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return _write(path, value)


def _recovery_candidate(path: Path) -> Path:
    value = {
        "schema_version": 2,
        "status": "PASS",
        "candidate_selector": "recovery_a",
        "selector_kind": "legacy",
        "canonical_binding": {
            "candidate_selector": "recovery_a",
            "effective_candidate_profile": "recovery_a",
            "fixed_episode_count": None,
        },
        "provenance": {
            "fixed_episode_count": None,
            "fixed_episode_keys": None,
        },
    }
    value["resolution_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return _write(path, value)


def _fixture(tmp_path: Path, monkeypatch, *, lane_a_only: bool = False):
    common = tmp_path / "portable-results"
    base = common / "d0-prepare"
    pilot = common / "pilot-prepare"
    base.mkdir(parents=True)
    pilot.mkdir(parents=True)
    roots = {
        "dgx_a": "/home/railgun/internnav-t1-t2/.t5-deployments/exact-a",
        "dgx_b": "/home/rail/internnav-t1-t2/.t5-deployments/exact-b",
        "x86_a": "/home/song/internnav-t1-t2/.t5-deployments/exact-x86-a",
        "x86_b": "/home/song/internnav-t1-t2/.t5-deployments/exact-x86-b",
        "x86_prepare": "/home/song/internnav-t1-t2/.t5-deployments/exact-x86-prepare",
    }
    if lane_a_only:
        roots = {key: roots[key] for key in ("dgx_a", "x86_a", "x86_prepare")}
    scope = (
        {"prepare_scope": "lane-a", "prepared_lanes": ["a"]}
        if lane_a_only
        else {}
    )
    base_input = _write(
        base / "fast_prepare_input.json", {"code_ref_sha": CODE_SHA, **scope}
    )
    base_summary = _write(
        base / "d0_prepare_summary.json",
        {"status": "PASS", "code_ref_sha": CODE_SHA, "deployment_roots": roots, "checks": {"ok": True}, **scope},
    )
    _write(
        base / "d0_prepare_final_summary.json",
        {
            "status": "PASS",
            "code_ref_sha": CODE_SHA,
            "deployment_roots": roots,
            "preparation_summary_sha256": _sha(base_summary),
            "checks": {"ok": True},
            **scope,
        },
    )

    source = pilot / "assets" / "source" / "val_unseen" / "val_unseen.json.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-fixture")
    source_sha = _sha(source)
    lane_paths = {}
    lane_records = {}
    for lane in ("a", "b"):
        path = pilot / "assets" / f"lane_{lane}" / "val_unseen" / "val_unseen.json.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"lane-{lane}-fixture".encode())
        lane_paths[lane] = path
        lane_records[lane] = {
            "output_dataset": str(path),
            "output_sha256": _sha(path),
            "episode_count": 10,
            "episode_keys": LANE_KEYS[lane],
        }

    execution_manifest = _write(
        common / "execution.json",
        {
            "final_evaluation": {
                "pilot": {
                    "lane_a_episode_keys": LANE_KEYS["a"],
                    "lane_b_episode_keys": LANE_KEYS["b"],
                    "aggregate_episode_count": 20,
                    "held_out_tuning_forbidden": True,
                }
            }
        },
    )
    pilot_manifest = _write(
        common / "pilot.json",
        {
            "episode_count": 20,
            "episode_keys": ALL_KEYS,
            "overlay_sha256": source_sha,
            "scene_ids": [f"mp3d/{scene}/{scene}.glb" for scene in SCENES],
        },
    )
    monkeypatch.setattr(receipt_builder, "EXECUTION_MANIFEST", execution_manifest)
    monkeypatch.setattr(receipt_builder, "PILOT_MANIFEST", pilot_manifest)
    monkeypatch.setattr(receipt_builder, "EXPECTED_SOURCE_SHA256", source_sha)
    split_path = _write(
        pilot / "assets" / "final_pilot_split_audit.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "source": {
                "dataset": str(source),
                "sha256": source_sha,
                "episode_count": 20,
                "episode_keys": ALL_KEYS,
            },
            "lanes": lane_records,
            "manifests": {
                "execution": {"sha256": _sha(execution_manifest)},
                "pilot": {"sha256": _sha(pilot_manifest)},
            },
            "checks": {"ok": True},
        },
    )

    maps = pilot / "maps" / "final_pilot_static_maps"
    maps.mkdir(parents=True)
    map_rows = []
    for scene in SCENES:
        filename = f"{scene}.bin"
        (maps / filename).write_bytes(scene.encode())
        map_rows.append({"scan": scene, "file": filename})
    generations = [
        {"trajectory_id": key.rsplit("_", 1)[0], "episode_id": key.rsplit("_", 1)[1]}
        for key in ALL_KEYS
    ]
    _write(
        maps / "manifest.json",
        {
            "schema_version": 1,
            "dataset_sha256": source_sha,
            "episode_count": 20,
            "map_count": 5,
            "maps": {row["scan"]: row for row in map_rows},
            "generations": generations,
            "t4_truth_isolation": {
                "status": "PASS",
                "dataset_sha256": source_sha,
                "runtime_ground_truth_pose_used_for_map_selection": False,
            },
        },
    )
    archive = pilot / "maps" / "final_pilot_static_maps.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(maps.iterdir()):
            stream.add(path, arcname=path.name)

    active_lanes = ("a",) if lane_a_only else ("a", "b")
    remote_dataset_roots = {
        lane: roots[f"x86_{lane}"] + f"/inputs/final_{lane}10"
        for lane in active_lanes
    }
    remote_map_manifests = {
        lane: roots[f"dgx_{lane}"] + "/inputs/final_maps/manifest.json"
        for lane in active_lanes
    }
    remote_x86 = _write(
        pilot / "remote" / "x86_prepare.json",
        {
            "status": "PASS",
            "code_ref_sha": CODE_SHA,
            "source_sha256": source_sha,
            "prepare_scope": "lane-a" if lane_a_only else "dual",
            "prepared_lanes": list(active_lanes),
            "lane_dataset_sha256": {lane: lane_records[lane]["output_sha256"] for lane in active_lanes},
            "lane_dataset_roots": remote_dataset_roots,
            "map_manifest_sha256": _sha(maps / "manifest.json"),
        },
    )
    remote_map_receipts = {}
    for lane in active_lanes:
        remote_map_receipts[lane] = _write(
            pilot / "remote" / f"dgx_{lane}_map.json",
            {
                "status": "PASS",
                "lane": lane,
                "code_ref_sha": CODE_SHA,
                "manifest_path": remote_map_manifests[lane],
                "manifest_sha256": _sha(maps / "manifest.json"),
            },
        )

    receipt = receipt_builder.build_receipt(
        code_sha=CODE_SHA,
        base_prepare_root=base,
        split_audit_path=split_path,
        map_dir=maps,
        map_archive=archive,
        remote_x86_receipt_path=remote_x86,
        remote_lane_map_receipt_paths=remote_map_receipts,
        remote_lane_dataset_roots=remote_dataset_roots,
        remote_lane_map_manifests=remote_map_manifests,
        output=pilot / "final_pilot_prepare_receipt.json",
    )
    assert receipt["status"] == "PASS"
    assert set(receipt["deployment_roots"]) == (
        {"dgx_a", "x86_a"}
        if lane_a_only
        else {"dgx_a", "dgx_b", "x86_a", "x86_b"}
    )
    assert not Path(receipt["split"]["audit_relative_path"]).is_absolute()
    assert not Path(receipt["static_maps"]["manifest_relative_path"]).is_absolute()
    return common, pilot, receipt


def test_prepare_receipt_and_final10_binding_are_portable(tmp_path, monkeypatch):
    _common, pilot, receipt = _fixture(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path / "candidate.json")
    outputs = {}
    for lane in ("a", "b"):
        output = tmp_path / f"binding-{lane}.json"
        payload = binding_resolver.resolve_binding(
            prepare_root=pilot,
            lane=lane,
            code_sha=CODE_SHA,
            candidate_resolution_path=candidate,
            runtime_profile=dict(binding_resolver.EXPECTED_PROFILE),
            output=output,
        )
        outputs[lane] = payload
        assert payload["status"] == "PASS"
        assert payload["execution_profile"] == "final10"
        assert payload["execution_episode_count"] == 10
        assert payload["split_audit_sha256"] == receipt["split"]["audit_sha256"]
    assert outputs["a"]["candidate_binding"] == outputs["b"]["candidate_binding"]
    assert outputs["a"]["static_map_manifest_sha256"] == outputs["b"]["static_map_manifest_sha256"]


def test_wp03_stop_shadow_reuses_only_frozen_a10_b10_assets(tmp_path, monkeypatch):
    _common, pilot, receipt = _fixture(tmp_path, monkeypatch)
    candidate = _recovery_candidate(tmp_path / "recovery-candidate.json")
    outputs = {}
    for lane in ("a", "b"):
        output = tmp_path / f"shadow-binding-{lane}.json"
        outputs[lane] = binding_resolver.resolve_binding(
            prepare_root=pilot,
            lane=lane,
            code_sha=CODE_SHA,
            candidate_resolution_path=candidate,
            runtime_profile=dict(binding_resolver.WP03_STOP_SHADOW_PROFILE),
            output=output,
        )
        assert outputs[lane]["status"] == "PASS"
        assert outputs[lane]["wp03_stop_shadow_overlay"] is True
        assert outputs[lane]["source_final_profile"] == receipt["final_profile"]
        assert outputs[lane]["execution_episode_count"] == 10
    assert set(outputs["a"]["execution_episode_keys"]).isdisjoint(
        outputs["b"]["execution_episode_keys"]
    )


def test_pilot_screen1_selects_one_frozen_lane_episode_without_rebinding_source(
    tmp_path, monkeypatch
):
    _common, pilot, receipt = _fixture(tmp_path, monkeypatch)
    candidate = _recovery_candidate(tmp_path / "screen-candidate.json")
    selected = LANE_KEYS["a"][4]
    payload = binding_resolver.resolve_binding(
        prepare_root=pilot,
        lane="a",
        code_sha=CODE_SHA,
        candidate_resolution_path=candidate,
        runtime_profile=dict(binding_resolver.WP03_STOP_SHADOW_PROFILE),
        output=tmp_path / "pilot-screen1-binding.json",
        execution_profile="pilot-screen1",
        episode_key=selected,
    )
    lane_receipt = receipt["split"]["lanes"]["a"]
    assert payload["status"] == "PASS"
    assert payload["source_execution_profile"] == "final10"
    assert payload["execution_profile"] == "pilot-screen1"
    assert payload["execution_episode_count"] == 1
    assert payload["execution_episode_keys"] == [selected]
    assert payload["screen_episode_key"] == selected
    assert payload["episode_count"] == 10
    assert payload["episode_keys"] == LANE_KEYS["a"]
    assert payload["dataset_sha256"] == lane_receipt["dataset_sha256"]
    assert payload["split_audit_sha256"] == receipt["split"]["audit_sha256"]


@pytest.mark.parametrize(
    "episode_key",
    [None, LANE_KEYS["b"][0], "external_episode", "../ta0_ea0"],
)
def test_pilot_screen1_rejects_missing_cross_lane_or_external_episode(
    tmp_path, monkeypatch, episode_key
):
    _common, pilot, _receipt = _fixture(tmp_path, monkeypatch)
    candidate = _recovery_candidate(tmp_path / "reject-candidate.json")
    output = tmp_path / "rejected-pilot-screen1-binding.json"
    with pytest.raises(ValueError, match="pilot-screen1"):
        binding_resolver.resolve_binding(
            prepare_root=pilot,
            lane="a",
            code_sha=CODE_SHA,
            candidate_resolution_path=candidate,
            runtime_profile=dict(binding_resolver.WP03_STOP_SHADOW_PROFILE),
            output=output,
            execution_profile="pilot-screen1",
            episode_key=episode_key,
        )
    assert not output.exists()


def test_lane_a_prepare_receipt_omits_b_and_resolver_rejects_b(tmp_path, monkeypatch):
    _common, pilot, receipt = _fixture(tmp_path, monkeypatch, lane_a_only=True)
    candidate = _candidate(tmp_path / "candidate-a-only.json")
    assert receipt["prepare_scope"] == "lane-a"
    assert receipt["prepared_lanes"] == ["a"]
    assert set(receipt["deployment_roots"]) == {"dgx_a", "x86_a"}
    assert set(receipt["split"]["lanes"]) == {"a"}
    assert set(receipt["remote_receipts"]["dgx_maps"]) == {"a"}
    payload = binding_resolver.resolve_binding(
        prepare_root=pilot,
        lane="a",
        code_sha=CODE_SHA,
        candidate_resolution_path=candidate,
        runtime_profile=dict(binding_resolver.EXPECTED_PROFILE),
        output=tmp_path / "binding-a-only.json",
    )
    assert payload["status"] == "PASS"
    assert payload["prepare_scope"] == "lane-a"
    assert payload["deployment_roots"] == {
        "dgx": receipt["deployment_roots"]["dgx_a"],
        "x86": receipt["deployment_roots"]["x86_a"],
    }
    try:
        binding_resolver.resolve_binding(
            prepare_root=pilot,
            lane="b",
            code_sha=CODE_SHA,
            candidate_resolution_path=candidate,
            runtime_profile=dict(binding_resolver.EXPECTED_PROFILE),
            output=tmp_path / "binding-b-forbidden.json",
        )
    except ValueError as error:
        assert "not prepared" in str(error)
    else:
        raise AssertionError("Lane B must be rejected by a Lane-A-only receipt")


def test_bundle_wrapper_rebinds_stale_split_paths_after_relocation(tmp_path, monkeypatch):
    common, pilot, _receipt = _fixture(tmp_path, monkeypatch)
    candidate = _candidate(common / "candidate.json")
    lane_roots = {}
    for lane in ("a", "b"):
        lane_root = common / f"fast-lane-{lane}"
        lane_root.mkdir()
        binding_resolver.resolve_binding(
            prepare_root=pilot,
            lane=lane,
            code_sha=CODE_SHA,
            candidate_resolution_path=candidate,
            runtime_profile=dict(binding_resolver.EXPECTED_PROFILE),
            output=lane_root / "input_binding.json",
        )
        lane_roots[lane] = lane_root
    stub = common / "metric_finalizer.py"
    stub.write_text(
        "def finalize(a,b,split):\n"
        " return {'status':'PASS','integrity':{'status':'PASS','checks':{'metric':True}},"
        "'promotion':{'status':'NOT_ELIGIBLE','checks':{'integrity_pass':True}},"
        "'aggregate':{'episode_count':20},'lanes':{'a':{'result_root':str(a)},'b':{'result_root':str(b)}}}\n",
        encoding="utf-8",
    )
    first = bundle_finalizer.finalize_bundle(
        lane_a_root=lane_roots["a"],
        lane_b_root=lane_roots["b"],
        prepare_root=pilot,
        output=common / "reports" / "first.json",
        metric_finalizer_path=stub,
    )
    assert first["status"] == "PASS"
    assert first["path_contract"]["all_local_paths_relative"] is True
    assert not Path(first["lanes"]["a"]["result_root"]).is_absolute()

    relocated = tmp_path / "relocated-results"
    shutil.move(str(common), relocated)
    second = bundle_finalizer.finalize_bundle(
        lane_a_root=relocated / "fast-lane-a",
        lane_b_root=relocated / "fast-lane-b",
        prepare_root=relocated / "pilot-prepare",
        output=relocated / "reports" / "second.json",
        metric_finalizer_path=relocated / "metric_finalizer.py",
    )
    assert second["status"] == "PASS"
    assert second["integrity"]["runner_contract"]["status"] == "PASS"


def test_online_entrypoints_keep_disjoint_leases_and_explicit_final10():
    fast = (ROOT / "coordination" / "run_t5_fast_lane_online.sh").read_text(encoding="utf-8")
    dual = (ROOT / "coordination" / "run_t5_final_pilot_dual_online.sh").read_text(encoding="utf-8")
    prepare = (ROOT / "coordination" / "run_t5_final_pilot_prepare_online.sh").read_text(encoding="utf-8")
    distributed = (ROOT / "scripts" / "run_t5_distributed_isaac.sh").read_text(encoding="utf-8")
    assert "fixed5 | pilot-screen1 | final10" in fast
    assert 'pilot-screen1|final10' in fast
    assert '--execution-profile "$profile"' in fast
    assert '--episode-key "$screen_episode_key"' in fast
    assert "INTERNNAV_T5_FINAL_PILOT_LANE" in fast
    assert "run_t5_fast_lane_online.sh" in dual
    assert "with_resource_lease.sh\" all-lanes" not in dual
    assert "INTERNNAV_T5_CANDIDATE_PROFILE=a1+b1+c1" in dual
    assert "INTERNNAV_T5_FINAL_PILOT_LANE" in distributed
    assert 'case "$dataset_episode_count" in 1|10)' in distributed
    assert "flock -x -w 1800 /tmp/internnav_t5_isaac_shared_assets.lock" in prepare
    assert 'allowed_root_keys=required_root_keys|{"x86_prepare"}' in prepare
    assert "set(roots).issubset(allowed_root_keys)" in prepare
    assert 'lane_a_scope=all(scope=="lane-a" and lanes==["a"]' in prepare
    assert "taskset -c '$lane_a_cpuset'" in prepare
    assert 'tar -C "$result_dir/assets" -czf "$asset_archive" lane_a' in prepare
    assert 'if test "$prepare_scope" = dual; then' in prepare
    assert 'map_records=raw_maps.values() if isinstance(raw_maps,dict) else raw_maps' in prepare
    assert "Z6MFQCViBuw" in prepare
    assert "d0_fixed5_static_maps" not in prepare


def test_final10_model_phase_routes_to_pilot_instead_of_usage_exit():
    distributed = (ROOT / "scripts" / "run_t5_distributed_isaac.sh").read_text(encoding="utf-8")
    phase_case = distributed.rsplit('case "$dataset_episode_count" in', 1)[1].split("esac", 1)[0]
    final_pilot_branch = phase_case.split("10|20)", 1)[1].split(";;", 1)[0]
    reject_branch = phase_case.split("*)", 1)[1]

    assert "export INTERNVLA_T4_PHASE_OVERRIDE=pilot" in final_pilot_branch
    assert "exit 64" not in final_pilot_branch
    assert "10/20 pilot episodes" in reject_branch
    assert "exit 64" in reject_branch
