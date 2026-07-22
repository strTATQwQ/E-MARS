from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "coordination" / "run_t5_d0_dual_online.sh"
HELPER = ROOT / "scripts" / "finalize_t5_d0_dual.py"
SINGLE_LANE_SCRIPT = ROOT / "coordination" / "run_t5_d0_lane_online.sh"
FROZEN_EPISODE_KEYS = [
    "trajectory1_e1",
    "trajectory2_e2",
    "trajectory3_e3",
    "trajectory4_e4",
    "trajectory5_e5",
]
REVERSED_EPISODE_KEYS = list(reversed(FROZEN_EPISODE_KEYS))


def _minimal_d0_manifest() -> dict[str, object]:
    return {
        "fixed_input": {
            "episode_keys": list(FROZEN_EPISODE_KEYS),
            "dataset_file_sha256": "4" * 64,
        }
    }


def _load_helper():
    spec = importlib.util.spec_from_file_location("finalize_t5_d0_dual", HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dual = _load_helper()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_order_evidence(
    x86_root: Path,
    dataset_sha256: str,
    ordered_keys: list[str] | None = None,
) -> tuple[Path, Path]:
    ordered = list(ordered_keys or FROZEN_EPISODE_KEYS)
    order_path = x86_root / "ordered_episode_manifest.json"
    _write(
        order_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "dataset_sha256": dataset_sha256,
            "dataset_episode_count": 5,
            "raw_episode_keys": list(FROZEN_EPISODE_KEYS),
            "materialized_pre_reverse_episode_keys": list(reversed(ordered)),
            "ordered_episode_keys": ordered,
            "ordered_episode_ids": [key.rsplit("_", 1)[-1] for key in ordered],
            "loader_contract": {
                "materializer": "BasePathKeyEpisodeloader.path_key_data",
                "fresh_resumable_order": "reverse_materialized_path_keys",
                "rank": 0,
                "world_size": 1,
            },
        },
    )
    readiness_path = x86_root / "health" / "runtime_readiness_evidence.json"
    _write(
        readiness_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "mode": "model",
            "model_episode_order_manifest": {"sha256": _sha(order_path)},
        },
    )
    return order_path, readiness_path


def _pass(**extra: object) -> dict[str, object]:
    return {"schema_version": 1, "status": "PASS", "checks": {"complete": True}, **extra}


def _predecessor_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    prep = tmp_path / "results" / "internnav_t5" / "d0-0-prepare-t5d0020260719t100000"
    d01 = tmp_path / "results" / "internnav_t5" / "d0-1-lane-a-t5d0120260719t110000"
    d02 = tmp_path / "results" / "internnav_t5" / "d0-2-lane-b-t5d0220260719t120000"
    roots = {
        "dgx_a": "/home/railgun/t5/a",
        "dgx_b": "/home/rail/t5/b",
        "x86_a": "/home/song/t5/a",
        "x86_b": "/home/song/t5/b",
    }
    common = {
        "prep_grant_id": "t5d0020260719t100000",
        "code_ref_sha": "1" * 40,
        "golden_bundle_canonical_sha256": "2" * 64,
        "run_manifest_canonical_sha256": "3" * 64,
        "dataset_root": "/episodes/fixed5",
        "dataset_file_sha256": "4" * 64,
        "static_map_manifest_sha256": "5" * 64,
    }
    _write(prep / "d0_prepare_final_summary.json", _pass())
    _write(
        prep / "d0_prepare_summary.json",
        _pass(
            grant_id=common["prep_grant_id"],
            deployment_roots=roots,
            memory_admission={"minimum_available_bytes": 9_000_000_000},
        ),
    )
    _write(prep / "lease_release_summary.json", _pass())
    _write(prep / "remote" / "x86" / "dataset_audit.json", {"status": "PASS"})
    common["source_receipts"] = {
        "prepare_summary_sha256": _sha(prep / "d0_prepare_summary.json"),
        "prepare_release_sha256": _sha(prep / "lease_release_summary.json"),
        "dataset_audit_sha256": _sha(prep / "remote" / "x86" / "dataset_audit.json"),
        "prepare_final_sha256": _sha(prep / "d0_prepare_final_summary.json"),
    }
    for lane, root, stage in (
        ("a", d01, "d0_1_lane_a_fixed_5"),
        ("b", d02, "d0_2_lane_b_same_fixed_5"),
    ):
        runtime = {
            "status": "PASS",
            "stage": stage,
            "lane": lane,
            "code_ref_sha": common["code_ref_sha"],
            "golden_bundle_canonical_sha256": common[
                "golden_bundle_canonical_sha256"
            ],
            "run_manifest_canonical_sha256": common[
                "run_manifest_canonical_sha256"
            ],
            "episode_count": 5,
        }
        _write(root / "d0_lane_summary.json", _pass(runtime_summary=runtime))
        _write(root / "preparation_chain_binding.json", _pass())
        binding = dict(common)
        binding["deployment_roots"] = {
            "dgx": roots[f"dgx_{lane}"],
            "x86": roots[f"x86_{lane}"],
        }
        _write(root / "prep_binding.json", binding)
        _write(
            root / "remote" / "x86" / "isaac_contract.json",
            {"started_unix": 100.0},
        )
        _write(
            root / "remote" / "x86" / "isaac_status.json",
            {"finished_unix": 200.0},
        )
        _write_order_evidence(root / "remote" / "x86", "4" * 64)
        _write(
            root / "remote" / "dgx" / "client" / "client_summary.json",
            {
                "status": "FINISHED",
                "step_count": 10,
                "mean_inference_latency_sec": 1.0,
                "mean_action_round_trip_latency_sec": 2.0,
                "mean_nav2_resolution_latency_sec": 0.5,
            },
        )
        _write(
            root / "remote" / "dgx" / "onboard" / "controller_summary.json",
            {"status": "FINISHED", "measured_control_hz": 30.0},
        )
        prefix = f"{lane}::"
        records = [
            {"episode_id": f"{prefix}e{index}", "sequence_id": index}
            for index in range(1, 6)
        ]
        for relative in (
            Path("remote/dgx/client/client_records.jsonl"),
            Path("remote/dgx/onboard/controller_records.jsonl"),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
        final_path = root / "d0_lane_summary.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        final["runtime_summary"]["key_file_sha256"] = {
            "remote/dgx/client/client_summary.json": _sha(
                root / "remote" / "dgx" / "client" / "client_summary.json"
            ),
            "remote/dgx/onboard/controller_summary.json": _sha(
                root / "remote" / "dgx" / "onboard" / "controller_summary.json"
            ),
            "remote/dgx/client/client_records.jsonl": _sha(
                root / "remote" / "dgx" / "client" / "client_records.jsonl"
            ),
            "remote/dgx/onboard/controller_records.jsonl": _sha(
                root / "remote" / "dgx" / "onboard" / "controller_records.jsonl"
            ),
        }
        _write(final_path, final)
    _write(
        d01 / "predecessor_binding.json",
        _pass(
            predecessor_stage="d0_0_online_prepare",
            predecessor_result_root=str(prep).replace("\\", "/"),
            predecessor_receipt_sha256=_sha(prep / "d0_prepare_final_summary.json"),
        ),
    )

    def bind_internal_hashes(root: Path) -> None:
        final_path = root / "d0_lane_summary.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        runtime = final["runtime_summary"]
        runtime["prep_binding_sha256"] = _sha(root / "prep_binding.json")
        runtime["predecessor_binding_sha256"] = _sha(
            root / "predecessor_binding.json"
        )
        runtime["preparation_chain_binding_sha256"] = _sha(
            root / "preparation_chain_binding.json"
        )
        _write(final_path, final)

    bind_internal_hashes(d01)
    _write(
        d02 / "predecessor_binding.json",
        _pass(
            predecessor_stage="d0_1_lane_a_fixed_5",
            predecessor_result_root=str(d01).replace("\\", "/"),
            predecessor_receipt_sha256=_sha(d01 / "d0_lane_summary.json"),
        ),
    )
    bind_internal_hashes(d02)
    grant = {
        "grant_id": "t5d0320260719t130000",
        "authorization_ref_sha": "0" * 40,
        "code_ref_sha": common["code_ref_sha"],
        "golden_bundle_sha256": common["golden_bundle_canonical_sha256"],
        "run_manifest_sha256": common["run_manifest_canonical_sha256"],
        "predecessor_stage": "d0_2_lane_b_same_fixed_5",
        "predecessor_result_root": str(d02).replace("\\", "/"),
        "predecessor_receipt_sha256": _sha(d02 / "d0_lane_summary.json"),
    }
    return prep, d01, d02, grant


def test_predecessors_require_both_lanes_to_share_exact_d00_chain(tmp_path: Path) -> None:
    prep, d01, d02, grant = _predecessor_fixture(tmp_path)
    output = tmp_path / "predecessor_chain.json"
    value = dual.validate_predecessors(
        prep_root=prep,
        d01_root=d01,
        d02_root=d02,
        grant=grant,
        manifest=_minimal_d0_manifest(),
        output=output,
    )
    assert value["status"] == "PASS"
    assert value["single_lane_baseline"]["a"]["runtime_duration_sec"] == 100.0
    assert value["single_lane_baseline"]["b"]["runtime_duration_sec"] == 100.0
    assert value["deployment_roots"]["x86_a"] != value["deployment_roots"]["x86_b"]

    drift = json.loads((d02 / "prep_binding.json").read_text(encoding="utf-8"))
    drift["prep_grant_id"] = "different-preparation"
    _write(d02 / "prep_binding.json", drift)
    with pytest.raises(dual.ContractError, match="predecessor chain failed"):
        dual.validate_predecessors(
            prep_root=prep,
            d01_root=d01,
            d02_root=d02,
            grant=grant,
            manifest=_minimal_d0_manifest(),
            output=tmp_path / "failed.json",
        )


def test_predecessor_rejects_replaced_lane_binding_file(tmp_path: Path) -> None:
    prep, d01, d02, grant = _predecessor_fixture(tmp_path)
    _write(d01 / "preparation_chain_binding.json", _pass(tampered_after_run=True))
    with pytest.raises(dual.ContractError, match="predecessor chain failed"):
        dual.validate_predecessors(
            prep_root=prep,
            d01_root=d01,
            d02_root=d02,
            grant=grant,
            manifest=_minimal_d0_manifest(),
            output=tmp_path / "replaced_binding.json",
        )


def _ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event": "started", "component": "worker", "pid": 123, "scope": "host"},
        {
            "event": "verified_absent",
            "component": "worker",
            "pid": 123,
            "scope": "host",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _dual_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    result = tmp_path / "d03"
    manifest = tmp_path / "d0.json"
    predecessor = result / "predecessor_chain.json"
    _write(
        manifest,
        {
            "fixed_input": {
                "episode_keys": list(FROZEN_EPISODE_KEYS),
                "dataset_file_sha256": "e" * 64,
                "same_episode_on_two_lanes_counts_once": True,
            },
            "acceptance": {"maximum_simultaneous_slowdown_fraction": 0.2},
            "lanes": {
                "a": {
                    "ros_domain_id": 75,
                    "namespace": "/t5/lane_a",
                    "controller_port": 25137,
                    "model_client_port": 25139,
                    "oracle_port": 25140,
                    "clock_port": 25141,
                    "isaac_render_gpu_physical": 0,
                    "cpuset": "0-7,16-23",
                },
                "b": {
                    "ros_domain_id": 76,
                    "namespace": "/t5/lane_b",
                    "controller_port": 25138,
                    "model_client_port": 25239,
                    "oracle_port": 25240,
                    "clock_port": 25241,
                    "isaac_render_gpu_physical": 1,
                    "cpuset": "8-15,24-31",
                },
            },
        },
    )
    _write(
        predecessor,
        _pass(
            grant_id="g",
            authorization_ref_sha="a" * 40,
            code_ref_sha="b" * 40,
            golden_bundle_canonical_sha256="c" * 64,
            run_manifest_canonical_sha256="d" * 64,
            static_map_manifest_sha256="f" * 64,
            single_lane_baseline={
                lane: {
                    "runtime_duration_sec": 100.0,
                    "performance_signals": {
                        "lower_is_better": {
                            "mean_inference_latency_sec": 1.0,
                            "mean_action_round_trip_latency_sec": 2.0,
                            "mean_nav2_resolution_latency_sec": 0.5,
                        },
                        "higher_is_better": {"measured_control_hz": 30.0},
                        "sources": {},
                    },
                }
                for lane in ("a", "b")
            },
        ),
    )
    for name in (
        "coordinator_cleanup_receipt.json",
        "cross_lane_isolation.json",
        "capacity_prestart.json",
    ):
        _write(result / "audits" / name, _pass())
    _write(
        result / "audits" / "credential_exposure_audit.json",
        {"status": "PASS", "exact_secret_match_count": 0},
    )
    for lane in ("a", "b"):
        domain = 75 if lane == "a" else 76
        namespace = f"/t5/lane_{lane}"
        gpu = 0 if lane == "a" else 1
        cpuset = "0-7,16-23" if lane == "a" else "8-15,24-31"
        ports = (
            {"controller": 25137, "model_client": 25139, "oracle": 25140, "clock": 25141}
            if lane == "a"
            else {"controller": 25138, "model_client": 25239, "oracle": 25240, "clock": 25241}
        )
        base = result / "remote" / f"lane_{lane}"
        dgx, x86 = base / "dgx", base / "x86"
        _write(dgx / "lane_status.json", {"status": "PASS", "residual_count": 0})
        _write(
            dgx / "lane_ready.json",
            {
                "status": "READY",
                "lane": lane,
                "mode": "model",
                "golden_bundle_canonical_sha256": "c" * 64,
            },
        )
        _write(
            dgx / "model_identity_audit.json",
            _pass(golden_bundle_canonical_sha256="c" * 64),
        )
        _write(
            dgx / "hf_token_process_audit.json",
            _pass(
                exact_secret_match_count=1,
                allowed_model_match_count=1,
                disallowed_match_count=0,
                parent_match_count=0,
                onboard_match_count=0,
                evaluator_match_count=0,
                secret_value_recorded=False,
                secret_digest_recorded=False,
            ),
        )
        _write(
            dgx / "lane_contract.json",
            {
                "lane": lane,
                "identity_prefix": f"{lane}::",
                "ros_domain_id": domain,
                "namespace": namespace,
                "inputs": {"static_map_manifest": {"sha256": "f" * 64}},
            },
        )
        _ledger(dgx / "pid_ledger.jsonl")
        _write(
            dgx / "client" / "client_summary.json",
            {
                "status": "FINISHED",
                "step_count": 10,
                "mean_inference_latency_sec": 1.1,
                "mean_action_round_trip_latency_sec": 2.2,
                "mean_nav2_resolution_latency_sec": 0.55,
                "reset_id": f"{lane}::0",
            },
        )
        (dgx / "client").mkdir(parents=True, exist_ok=True)
        client_rows = [
            {
                "event": "step",
                "episode_id": f"{lane}::e{index}",
                "request_id": f"{lane}::request-{index}",
            }
            for index in range(1, 6)
        ]
        (dgx / "client" / "client_records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in client_rows),
            encoding="utf-8",
        )
        controller_rows = [
            {
                "update_index": index,
                "episode_id": f"{lane}::e{index}",
                "reset_generation": index - 1,
            }
            for index in range(1, 6)
        ]
        (dgx / "onboard").mkdir(parents=True, exist_ok=True)
        (dgx / "onboard" / "controller_records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in controller_rows),
            encoding="utf-8",
        )
        started = 100.0 if lane == "a" else 102.0
        finished = started + 110.0
        _write(
            x86 / "isaac_contract.json",
            {
                "lane": lane,
                "identity": {"episode_prefix": f"{lane}::"},
                "ros_domain_id": domain,
                "namespace": namespace,
                "ports": ports,
                "cpuset": cpuset,
                "started_unix": started,
                "dataset": {"sha256": "e" * 64},
                "cache_roots": {
                    "xdg": f"/profiles/{lane}/xdg",
                    "tmp": f"/profiles/{lane}/tmp",
                },
                "health_endpoint": f"unix:///tmp/{lane}/health.sock",
                "ipc_alias": f"/tmp/{lane}/ipc",
                "full_mp4_encoding_allowed": False,
                "video_policy": "jsonl_keyframes_only",
                "shared_asset_lock_mode": "shared_read",
            },
        )
        _write(
            x86 / "isaac_status.json",
            {
                "status": "PASS",
                "residual_count": 0,
                "socket_residual_count": 0,
                "clock_publishers_after_stop": 0,
                "lane_lock_released": True,
                "execution_profile": "fixed_dataset",
                "episode_acceptance_claimed": True,
                "evaluation_completed_naturally": True,
                "termination_reason": "evaluator_natural_exit",
                "finished_unix": finished,
            },
        )
        _write_order_evidence(x86, "e" * 64)
        _write(
            x86 / "gpu_mapping.json",
            {"status": "PASS", "host_physical_gpu_index": gpu},
        )
        _write(x86 / "kit_gpu_audit.json", {"status": "PASS"})
        _write(
            x86 / "controller_summary.json",
            {"status": "FINISHED", "measured_control_hz": 27.0},
        )
        (x86 / "health").mkdir(parents=True, exist_ok=True)
        (x86 / "health" / "sensor_frames.jsonl").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "episode_id": f"{lane}::e1",
                    "reset_id": f"{lane}::reset-1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _ledger(x86 / "pid_ledger.jsonl")
        attempt = x86 / "evaluator" / f"t5_lane_{lane}_model_model_attempt_001"
        _write(
            attempt / "result.json",
            {"val_unseen": {"SR": 0.2, "OS": 0.4, "SPL": 0.1, "NE": 2.0, "Count": 5}},
        )
        _write(
            attempt / "isaac_remote_validation.json",
            {
                "status": "PASS",
                "episode_count": 5,
                "expected_episode_count": 5,
                "minimum_success_rate": 0.0,
                "episode_identity": f"{lane}::fixed5",
            },
        )
    return result, predecessor, manifest


def test_finalize_counts_fixed_five_once_and_enforces_slowdown(tmp_path: Path) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    output = result / "d0_dual_runtime_summary.json"
    value = dual.finalize_dual(
        result_root=result,
        predecessor_binding=predecessor,
        manifest_path=manifest,
        output=output,
    )
    assert value["status"] == "PASS"
    assert value["paired_statistics"]["unique_statistical_sample_count"] == 5
    assert value["paired_statistics"]["hardware_reproduction_observation_count"] == 10
    assert value["paired_statistics"]["pooled_sr_os_spl_ne_forbidden"] is True
    assert value["performance"]["a"]["slowdown_fraction"] == pytest.approx(0.1)

    status_path = result / "remote" / "lane_b" / "x86" / "isaac_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["finished_unix"] = 232.0  # 130 s versus the 100 s single-Lane baseline.
    _write(status_path, status)
    with pytest.raises(dual.ContractError, match="finalization failed"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / "failed_summary.json",
        )
    failed = json.loads((result / "failed_summary.json").read_text(encoding="utf-8"))
    assert failed["status"] == "FAIL"
    assert failed["fallback"] == {
        "action": "INTERLEAVED_SINGLE_LANE_EXECUTION",
        "preserve_failed_parallel_evidence": True,
        "required": True,
    }


def test_formal_dual_runtime_forces_fixed_dataset_profile() -> None:
    coordinator = (ROOT / "coordination" / "run_t5_d0_dual_online.sh").read_text(
        encoding="utf-8"
    )
    lane_coordinator = (
        ROOT / "coordination" / "run_t5_d0_lane_online.sh"
    ).read_text(encoding="utf-8")
    finalizer = (ROOT / "scripts" / "finalize_t5_d0_dual.py").read_text(
        encoding="utf-8"
    )
    assert "INTERNNAV_T5_ENGINEERING_CANARY_SEC=0" in coordinator
    assert "t5_lane_{lane}_model*_model_attempt_*" in coordinator
    assert "t5_lane_{lane}_model*_model_attempt_*" in lane_coordinator
    assert "model_model_attempt_001" not in lane_coordinator
    assert '"fixed_dataset_execution"' in finalizer
    assert 'x86_status.get("execution_profile")' in finalizer


def test_formal_finalizer_accepts_run_token_bound_attempt_directory(
    tmp_path: Path,
) -> None:
    attempt = (
        tmp_path
        / "evaluator"
        / "t5_lane_a_model_t5_fast_fixed5_run123_model_attempt_001"
    )
    _write(attempt / "result.json", {"val_unseen": {"Count": 5}})
    _write(attempt / "isaac_remote_validation.json", {"status": "PASS"})
    observed, result, validation = dual._find_attempt(tmp_path, "a")
    assert observed == attempt
    assert result["val_unseen"]["Count"] == 5
    assert validation["status"] == "PASS"


def test_finalize_rejects_cross_lane_identity(tmp_path: Path) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    result_path = (
        result
        / "remote"
        / "lane_a"
        / "dgx"
        / "client"
        / "client_records.jsonl"
    )
    result_path.write_text(
        result_path.read_text(encoding="utf-8")
        + json.dumps({"event": "step", "request_id": "b::foreign-request"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(dual.ContractError, match="finalization failed"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / "failed_cross_lane.json",
        )


def test_finalize_rejects_structurally_escaped_cross_lane_prefix(tmp_path: Path) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    path = result / "remote" / "lane_a" / "dgx" / "client" / "escaped.json"
    path.write_text('{"request_id":"b\\u003a\\u003aforeign"}\n', encoding="utf-8")
    with pytest.raises(dual.ContractError, match="finalization failed"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / "failed_escaped_cross_lane.json",
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "out_of_order", "unprefixed"),
)
@pytest.mark.parametrize(
    "record_relative",
    (
        Path("client/client_records.jsonl"),
        Path("onboard/controller_records.jsonl"),
    ),
)
def test_finalize_requires_exact_frozen_five_episode_order_and_uniqueness(
    tmp_path: Path, mutation: str, record_relative: Path
) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    path = (
        result
        / "remote"
        / "lane_b"
        / "dgx"
        / record_relative
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows[-1]["episode_id"] = "b::e4"
    elif mutation == "out_of_order":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        rows[2]["episode_id"] = "e3"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(dual.ContractError, match="finalization failed"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / f"failed_episode_{mutation}.json",
        )


def test_controller_prebind_state_only_sentinels_do_not_count_as_episodes(
    tmp_path: Path,
) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    path = (
        result
        / "remote"
        / "lane_a"
        / "dgx"
        / "onboard"
        / "controller_records.jsonl"
    )
    startup = [
        {"episode_id": "uninitialized", "state_only": True},
        {"episode_id": "bootstrap-episode-0", "state_only": True},
    ]
    official_rows = path.read_text(encoding="utf-8").splitlines(keepends=True)
    official_rows.insert(
        1, json.dumps({"episode_id": "bootstrap-episode-1", "state_only": True}) + "\n"
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in startup)
        + "".join(official_rows),
        encoding="utf-8",
    )
    value = dual.finalize_dual(
        result_root=result,
        predecessor_binding=predecessor,
        manifest_path=manifest,
        output=result / "accepted_prebind_state_only.json",
    )
    assert (
        value["lanes"]["a"]["episode_coverage"]["controller"][
            "ignored_prebind_state_only_count"
        ]
        == 3
    )


@pytest.mark.parametrize(
    ("row", "after_first"),
    [
        ({"episode_id": "bootstrap-episode-0", "state_only": False}, False),
        ({"episode_id": "b::e0", "state_only": True}, False),
        ({"episode_id": "uninitialized", "state_only": True}, True),
    ],
)
def test_controller_never_ignores_unsafe_or_postbind_unprefixed_rows(
    tmp_path: Path, row: dict[str, object], after_first: bool
) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    path = (
        result
        / "remote"
        / "lane_a"
        / "dgx"
        / "onboard"
        / "controller_records.jsonl"
    )
    rows = path.read_text(encoding="utf-8").splitlines(keepends=True)
    index = 1 if after_first else 0
    rows.insert(index, json.dumps(row) + "\n")
    path.write_text("".join(rows), encoding="utf-8")
    with pytest.raises(dual.ContractError, match="finalization failed"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / "rejected_unsafe_state_only.json",
        )


def test_standalone_lane_episode_receipt_accepts_bound_top_level_keys(
    tmp_path: Path,
) -> None:
    dgx = tmp_path / "dgx"
    ordered_ids = [key.rsplit("_", 1)[-1] for key in REVERSED_EPISODE_KEYS]
    for component in ("client", "onboard"):
        path = dgx / component / (
            "client_records.jsonl" if component == "client" else "controller_records.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps({"episode_id": f"a::{episode_id}"}) + "\n"
                for episode_id in ordered_ids
            ),
            encoding="utf-8",
        )
    manifest = tmp_path / "grant_validation.json"
    _write(
        manifest,
        {
            "episode_keys": list(FROZEN_EPISODE_KEYS),
            "dataset_file_sha256": "e" * 64,
        },
    )
    order_path, readiness_path = _write_order_evidence(
        tmp_path / "x86", "e" * 64, REVERSED_EPISODE_KEYS
    )
    output = tmp_path / "coverage.json"
    value = dual.finalize_episode_coverage(
        dgx_root=dgx,
        lane="a",
        manifest_path=manifest,
        order_manifest_path=order_path,
        readiness_evidence_path=readiness_path,
        output=output,
    )
    assert value["status"] == "PASS"
    assert value["expected_prefixed_episode_ids"] == [
        f"a::{episode_id}" for episode_id in ordered_ids
    ]
    assert value["episode_order_contract"]["sha256"] == _sha(order_path)
    assert output.is_file()


@pytest.mark.parametrize(
    ("relative", "field", "bad_value"),
    [
        (("dgx", "client", "client_summary.json"), "mean_inference_latency_sec", 1.21),
        (
            ("dgx", "client", "client_summary.json"),
            "mean_action_round_trip_latency_sec",
            2.41,
        ),
        (
            ("dgx", "client", "client_summary.json"),
            "mean_nav2_resolution_latency_sec",
            0.61,
        ),
        (("x86", "controller_summary.json"), "measured_control_hz", 23.9),
    ],
)
def test_finalize_rejects_any_performance_dimension_over_twenty_percent(
    tmp_path: Path, relative: tuple[str, ...], field: str, bad_value: float
) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    path = result / "remote" / "lane_a"
    for component in relative:
        path /= component
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = bad_value
    _write(path, value)
    with pytest.raises(dual.ContractError, match="finalization failed"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / f"failed_{field}.json",
        )


def test_finalize_requires_fail_closed_model_only_token_scope(tmp_path: Path) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    path = result / "remote" / "lane_b" / "dgx" / "hf_token_process_audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["disallowed_match_count"] = 1
    _write(path, value)
    with pytest.raises(dual.ContractError, match="finalization failed"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / "failed_token_scope.json",
        )


def test_finalize_fails_closed_when_required_performance_evidence_is_missing(
    tmp_path: Path,
) -> None:
    result, predecessor, manifest = _dual_fixture(tmp_path)
    (result / "remote" / "lane_a" / "dgx" / "client" / "client_summary.json").unlink()
    with pytest.raises(dual.ContractError, match="cannot read JSON evidence"):
        dual.finalize_dual(
            result_root=result,
            predecessor_binding=predecessor,
            manifest_path=manifest,
            output=result / "missing_performance.json",
        )


def test_online_entry_has_atomic_four_lock_parallel_and_cleanup_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'with_resource_lease.sh" all-lanes' in text
    assert "INTERNNAV_T5_RESOURCE_LEASE_ACK=all-lanes" in text
    assert "DGX_A -> DGX_B -> ISAAC_GPU0 -> ISAAC_GPU1" in text
    assert "d0_3_dual_lane_same_fixed_5" in text
    assert 'diff --name-only "$code_ref" "$authorization_ref"' in text
    assert '[[ "$changed_path" = "$board_relative" ]]' in text
    assert "'lane_scope':'all_lanes'" in text
    assert "'resource_profile':'all-lanes'" in text
    assert "run_t5_dgx_lane.sh" in text
    assert "run_t5_distributed_isaac.sh" in text
    assert "internnav_t5_isaac_a internnav_t5_isaac_b" in text
    assert "live_isolation_snapshots.jsonl" in text
    assert "maximum_simultaneous_slowdown_fraction" in text
    assert "INTERLEAVED_SINGLE_LANE_EXECUTION" in text
    assert "supervisor_action \"$key\" TERM" in text
    assert "supervisor_action \"$key\" KILL" in text
    assert "coordinator_cleanup_receipt.json" in text
    assert "credential_exposure_audit.json" in text
    assert "run_t5_d0_lane_online.sh" not in text


def test_dual_persistent_quarantine_requires_owned_cleanup_on_all_hosts() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'source "$root/scripts/t5_quarantine_common.sh"' in text
    assert 'source "$root/scripts/t5_remote_compute_audit_common.sh"' in text
    assert 'quarantine_run_tag="${stage}:${grant_id}"' in text
    assert "isaac_gpu0_quarantine_file=/tmp/internnav_isaac_gpu0.quarantine" in text
    assert "isaac_gpu1_quarantine_file=/tmp/internnav_isaac_gpu1.quarantine" in text
    for role in ("dgx_a", "dgx_b"):
        assert f't5_quarantine_arm "${role}_target"' in text
        assert f't5_quarantine_owned_clear "${role}_target"' in text
    assert 't5_quarantine_arm "$x86_target"' in text
    assert 't5_quarantine_owned_clear "$x86_target"' in text
    assert "x86_gpu0_quarantine_armed=true" in text
    assert "x86_gpu1_quarantine_armed=true" in text
    assert "coordinator_cleanup_lane_a_receipt.json" in text
    assert "coordinator_cleanup_lane_b_receipt.json" in text
    assert "dgx_a_structured_compute_prestart.json" in text
    assert "dgx_b_structured_compute_prestart.json" in text
    assert "dgx_a_structured_compute_poststop.json" in text
    assert "dgx_b_structured_compute_poststop.json" in text
    assert "dgx_structured_compute_absent" in text
    assert "cleanup_receipt_a" in text and "cleanup_receipt_b" in text
    assert "x86_a_clean" in text and "x86_b_clean" in text
    assert text.index('t5_quarantine_arm "$x86_target"') < text.index(
        "capacity_program"
    )
    assert text.index("x86_gpu1_quarantine_armed=true") < text.index(
        "dgx_a_structured_compute_prestart.json"
    )
    assert text.index("dgx_b_structured_compute_prestart.json") < text.index(
        "capacity_program"
    )
    assert text.index('t5_quarantine_arm "$x86_target"') < text.index(
        "docker start internnav_t5_isaac_a"
    )
    assert "run_root_processes_absent" in text
    assert "runtime_ledger_error is None" in text
    assert "internnav.t5.deployment_root" in text
    assert "25137 25139 25140 25141" in text
    assert "25138 25239 25240 25241" in text
    assert text.index("request_linked_stop || incoming=1") < text.index(
        't5_quarantine_owned_clear "$dgx_a_target"'
    )


def test_single_lane_receipts_bind_dual_performance_baselines() -> None:
    text = SINGLE_LANE_SCRIPT.read_text(encoding="utf-8")
    assert "result/'remote/dgx/client/client_summary.json'" in text
    assert "result/'remote/dgx/onboard/controller_summary.json'" in text
    assert "result/'remote/dgx/client/client_records.jsonl'" in text
    assert "result/'remote/dgx/onboard/controller_records.jsonl'" in text
    assert 'finalize_t5_d0_dual.py" episodes' in text
    assert "fixed_five_episode_coverage.json" in text
    assert "'exact_frozen_episode_coverage'" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_dual_online_entry_is_valid_bash() -> None:
    if os.name == "nt" and shutil.which("wsl"):
        linux_path = subprocess.check_output(
            ["wsl", "wslpath", "-a", str(SCRIPT)], text=True
        ).strip()
        subprocess.run(["wsl", "bash", "-n", linux_path], check=True)
    else:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
