import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_t5_fast_dual_canary.py"
CANARY_CHECKS = {
    "bounded_duration_valid": True,
    "ready_interval_complete": True,
    "ready_probe_pass": True,
    "ready_baselines_valid": True,
    "cutoff_values_valid": True,
    "cutoff_counter_not_rolled_back": True,
    "sample_baselines_match_ready": True,
    "periodic_sample_count": True,
    "timeline_starts_near_zero": True,
    "timeline_strictly_increasing": True,
    "timeline_covers_configured_window": True,
    "timeline_sample_gap_bounded": True,
    "health_running_every_sample": True,
    "model_action_counts_valid": True,
    "model_action_count_positive_at_ready": True,
    "model_action_counts_monotonic": True,
    "model_actions_advance": True,
    "model_step_error_counts_valid": True,
    "model_step_error_counts_monotonic_from_ready": True,
    "new_model_step_error_counts_valid": True,
    "model_step_errors_absent": True,
    "new_model_step_errors_absent": True,
    "fresh_five_episode_evaluator_state": True,
    "model_action_count_not_rolled_back_at_tail": True,
    "model_step_error_count_not_rolled_back_at_tail": True,
    "evaluator_total_path_stable_at_tail": True,
    "clock_state_age_seconds_le_15": True,
    "clock_counters_monotonic": True,
    "physics_steps_advance": True,
    "physics_not_stagnant_over_30_seconds": True,
    "evaluator_stop_completed_before_analysis": True,
}
ISOLATION = {
    "a": {
        "gpu": 0,
        "domain": 75,
        "namespace": "/t5/lane_a",
        "cpuset": "0,2,4,6,8,10,12,14,16",
        "ports": {
            "controller": 25137,
            "model_client": 25139,
            "oracle": 25140,
            "clock": 25141,
        },
    },
    "b": {
        "gpu": 1,
        "domain": 76,
        "namespace": "/t5/lane_b",
        "cpuset": "1,3,5,7,9,11,13,15,17",
        "ports": {
            "controller": 25138,
            "model_client": 25239,
            "oracle": 25240,
            "clock": 25241,
        },
    },
}


def write_lane(
    root: Path,
    lane: str,
    started: float,
    finished: float,
    code_sha: str = "a" * 40,
    receipt_seed: str = "b",
) -> None:
    receipt = receipt_seed * 64
    binding = {
        "status": "PASS",
        "lane": lane,
        "code_ref_sha": code_sha,
        "deployment_roots": {
            "dgx": "/different/dgx/%s" % lane,
            "x86": "/different/x86/%s" % lane,
        },
        "dataset_sha256": "c" * 64,
        "static_map_manifest_sha256": "d" * 64,
        "episode_count": 5,
        "episode_keys": ["episode-%d" % index for index in range(5)],
        "source_receipts": {
            "prepare_summary_sha256": receipt,
            "prepare_final_sha256": "e" * 64,
            "fast_prepare_input_sha256": "7" * 64,
            "dataset_audit_sha256": "f" * 64,
        },
        "checks": {"prepared": True},
    }
    canary = {
        "status": "PASS",
        "profile": "engineering_canary",
        "lane": lane,
        "configured_seconds": 60,
        "observed_seconds": 60,
        "started_unix": started,
        "finished_unix": finished,
        "episode_acceptance_claimed": False,
        "evaluation_completed_naturally": False,
        "checks": CANARY_CHECKS,
    }
    (root / "remote" / "x86").mkdir(parents=True)
    canary_path = root / "remote" / "x86" / "engineering_canary.json"
    canary_path.write_text(json.dumps(canary), encoding="utf-8")
    canary_sha256 = hashlib.sha256(canary_path.read_bytes()).hexdigest()
    isolation = ISOLATION[lane]
    x86_status = {
        "status": "PASS",
        "lane": lane,
        "isaac_render_gpu_physical_index": isolation["gpu"],
        "cuda_visible_devices": str(isolation["gpu"]),
        "host_cuda_visible_devices": str(isolation["gpu"]),
        "container_logical_cuda_visible_devices": "0",
        "isaac_physics_gpu_visible_index": 0,
        "ros_domain_id": isolation["domain"],
        "namespace": isolation["namespace"],
        "cpuset": isolation["cpuset"],
        "ports": isolation["ports"],
    }
    runtime = {
        "status": "PASS",
        "lane": lane,
        "profile": "canary60",
        "resource_profile": "lane-%s" % lane,
        "code_ref_sha": code_sha,
        "input_binding": binding,
        "x86_status": x86_status,
        "sealed_evidence": {
            "engineering_canary": {
                "applicability": "required",
                "relative_path": "remote/x86/engineering_canary.json",
                "sha256": canary_sha256,
            }
        },
        "checks": {"runtime": True},
    }
    final = {
        "status": "PASS",
        "runtime_summary": copy.deepcopy(runtime),
        "checks": {"final": True},
    }
    (root / "fast_lane_summary.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )
    (root / "fast_lane_final_summary.json").write_text(
        json.dumps(final), encoding="utf-8"
    )


def run_finalizer(tmp_path: Path, a_root: Path, b_root: Path):
    output = tmp_path / "paired.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lane-a-result",
            str(a_root),
            "--lane-b-result",
            str(b_root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def rewrite_runtime_and_final(root: Path, mutate) -> None:
    runtime_path = root / "fast_lane_summary.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    mutate(runtime)
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    final_path = root / "fast_lane_final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["runtime_summary"] = copy.deepcopy(runtime)
    final_path.write_text(json.dumps(final), encoding="utf-8")


def test_two_canaries_with_at_least_45_seconds_overlap_pass(tmp_path: Path) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "a", 100.0, 160.0)
    write_lane(b_root, "b", 115.0, 175.0)

    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 0
    assert value["status"] == "PASS"
    assert value["simultaneous_running_window"]["overlap_seconds"] == 45.0
    assert value["episode_acceptance_claimed"] is False
    assert value["evaluation_completed_naturally"] is False
    assert value["statistical_sample_count"] == 0
    assert value["preparation_binding"]["source_receipts"]
    # Per-lane deployment roots are intentionally different and not treated as drift.
    assert value["checks"]["same_preparation_binding"] is True
    assert value["checks"]["gpus_disjoint"] is True
    assert value["checks"]["ros_domains_disjoint"] is True
    assert value["checks"]["namespaces_disjoint"] is True
    assert value["checks"]["cpusets_disjoint"] is True
    assert value["checks"]["ports_disjoint"] is True
    assert all(
        value["lanes"][lane]["checks"]["canary_sha256_matches_sealed_summary"]
        for lane in ("a", "b")
    )


def test_less_than_45_seconds_overlap_fails_closed(tmp_path: Path) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "a", 100.0, 160.0)
    write_lane(b_root, "b", 115.01, 175.01)

    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 75
    assert value["status"] == "FAIL"
    assert value["checks"]["minimum_45_second_overlap"] is False


def test_code_or_prepare_drift_fails_closed(tmp_path: Path) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "a", 100.0, 160.0)
    write_lane(b_root, "b", 100.0, 160.0, code_sha="9" * 40, receipt_seed="8")

    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 75
    assert value["checks"]["same_code_sha"] is False
    assert value["checks"]["same_preparation_binding"] is False
    assert value["code_ref_sha"] is None
    assert value["preparation_binding"] is None


def test_swapped_lane_or_episode_claim_fails_closed(tmp_path: Path) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "b", 100.0, 160.0)
    write_lane(b_root, "b", 100.0, 160.0)
    canary_path = b_root / "remote" / "x86" / "engineering_canary.json"
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    canary["episode_acceptance_claimed"] = True
    canary_path.write_text(json.dumps(canary), encoding="utf-8")

    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 75
    assert value["lanes"]["a"]["checks"]["lane_identity"] is False
    assert (
        value["lanes"]["b"]["checks"]["no_episode_acceptance_claim"] is False
    )


def test_tampered_final_embedding_or_non_sixty_canary_fails(tmp_path: Path) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "a", 100.0, 160.0)
    write_lane(b_root, "b", 100.0, 160.0)
    final_path = a_root / "fast_lane_final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["runtime_summary"]["run_id"] = "tampered"
    final_path.write_text(json.dumps(final), encoding="utf-8")
    canary_path = b_root / "remote" / "x86" / "engineering_canary.json"
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    canary["configured_seconds"] = 59
    canary_path.write_text(json.dumps(canary), encoding="utf-8")

    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 75
    assert (
        value["lanes"]["a"]["checks"]["final_embeds_exact_runtime_summary"]
        is False
    )
    assert value["lanes"]["b"]["checks"]["configured_sixty_seconds"] is False


def test_missing_evidence_still_writes_machine_readable_failure(tmp_path: Path) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "a", 100.0, 160.0)
    b_root.mkdir()

    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 75
    assert value["status"] == "FAIL"
    assert value["lanes"]["b"]["checks"]["all_source_files_regular"] is False
    assert set(value["lanes"]["b"]["source_errors"]) == {
        "final",
        "runtime",
        "canary",
    }


def test_post_final_canary_tamper_is_rejected_by_sealed_sha(tmp_path: Path) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "a", 100.0, 160.0)
    write_lane(b_root, "b", 100.0, 160.0)
    canary_path = a_root / "remote" / "x86" / "engineering_canary.json"
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    canary["post_final_tamper"] = True
    canary_path.write_text(json.dumps(canary), encoding="utf-8")

    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 75
    assert value["status"] == "FAIL"
    assert (
        value["lanes"]["a"]["checks"][
            "canary_sha256_matches_sealed_summary"
        ]
        is False
    )
    assert value["lanes"]["a"]["checks"]["canary_pass"] is True


def test_fixed_isolation_drift_and_cross_lane_overlap_are_rejected(
    tmp_path: Path,
) -> None:
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_lane(a_root, "a", 100.0, 160.0)
    write_lane(b_root, "b", 100.0, 160.0)

    def collide_with_lane_a(runtime):
        status = runtime["x86_status"]
        status.update(
            {
                "isaac_render_gpu_physical_index": 0,
                "cuda_visible_devices": "0",
                "host_cuda_visible_devices": "0",
                "ros_domain_id": 75,
                "namespace": "/t5/lane_a",
                "cpuset": "0,2,4,6,8,10,12,14,16",
                "ports": copy.deepcopy(ISOLATION["a"]["ports"]),
            }
        )

    rewrite_runtime_and_final(b_root, collide_with_lane_a)
    completed, value = run_finalizer(tmp_path, a_root, b_root)

    assert completed.returncode == 75
    lane_checks = value["lanes"]["b"]["checks"]
    assert lane_checks["fixed_gpu_contract"] is False
    assert lane_checks["fixed_ros_domain_contract"] is False
    assert lane_checks["fixed_namespace_contract"] is False
    assert lane_checks["fixed_cpuset_contract"] is False
    assert lane_checks["fixed_ports_contract"] is False
    assert value["checks"]["gpus_disjoint"] is False
    assert value["checks"]["ros_domains_disjoint"] is False
    assert value["checks"]["namespaces_disjoint"] is False
    assert value["checks"]["cpusets_disjoint"] is False
    assert value["checks"]["ports_disjoint"] is False
