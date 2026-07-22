from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "configs/internnav_t5/candidates/active_isaac_epyc_resources.json"


def test_active_resource_candidate_is_fixed_without_adoption_threshold() -> None:
    value = json.loads(ACTIVE.read_text(encoding="utf-8"))
    assert value["status"] == "ACTIVE_ENGINEERING_RESOURCE_MAP"
    assert value["isaac_host"] == {
        "host": "10.100.120.123",
        "user": "song",
        "role": "isaac_simulator_only",
    }
    assert value["lanes"]["a"] == {
        "gpu": 0,
        "cpuset": "0,2,4,6,8,10,12,14,16",
    }
    assert value["lanes"]["b"] == {
        "gpu": 1,
        "cpuset": "1,3,5,7,9,11,13,15,17",
    }
    assert not value["adoption"]["performance_comparison_required"]
    assert not value["adoption"]["performance_threshold_required"]
    assert not value["adoption"]["online_experiment_required_for_activation"]
    assert not any(value["frozen_evidence_policy"].values())


def test_active_fast_path_and_workers_match_resource_candidate() -> None:
    value = json.loads(ACTIVE.read_text(encoding="utf-8"))
    host = value["isaac_host"]["host"]
    lane_a = value["lanes"]["a"]["cpuset"]
    lane_b = value["lanes"]["b"]["cpuset"]
    paths = (
        ROOT / "coordination/run_t5_fast_prepare_online.sh",
        ROOT / "coordination/run_t5_fast_lane_online.sh",
        ROOT / "scripts/prepare_t5_isaac_workers.sh",
        ROOT / "scripts/run_t5_distributed_isaac.sh",
        ROOT / "scripts/run_t5_dgx_lane.sh",
    )
    texts = {path.name: path.read_text(encoding="utf-8") for path in paths}
    assert host in texts["run_t5_fast_prepare_online.sh"]
    assert host in texts["run_t5_fast_lane_online.sh"]
    assert host in texts["prepare_t5_isaac_workers.sh"]
    assert host in texts["run_t5_dgx_lane.sh"]
    for name in (
        "run_t5_fast_prepare_online.sh",
        "run_t5_fast_lane_online.sh",
        "prepare_t5_isaac_workers.sh",
        "run_t5_distributed_isaac.sh",
    ):
        assert lane_a in texts[name]
        assert lane_b in texts[name]


def test_v0_manifests_remain_historical_evidence() -> None:
    topology = json.loads(
        (ROOT / "configs/internnav_t5/topology.json").read_text(encoding="utf-8")
    )
    d0 = json.loads(
        (ROOT / "configs/internnav_t5/d0_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert topology["roles"]["isaac_x86"]["host"] == "10.100.120.111"
    assert d0["lanes"]["a"]["isaac"] == "song@10.100.120.111"
    assert d0["lanes"]["b"]["isaac"] == "song@10.100.120.111"


def test_gpu_leases_default_to_active_host_but_legacy_global_does_not() -> None:
    lease = (ROOT / "scripts/with_resource_lease.sh").read_text(encoding="utf-8")
    assert lease.count('host="${ISAAC_HOST:-10.100.120.123}"') == 2
    assert lease.count('host="${ISAAC_HOST:-10.100.120.111}"') == 1


def test_quarantine_recovery_targets_the_active_isaac_host() -> None:
    recovery = (ROOT / "scripts/recover_t5_resource_quarantine.sh").read_text(
        encoding="utf-8"
    )
    assert "t5_isaac_ip=\"${INTERNVLA_T5_ISAAC_IP:-10.100.120.123}\"" in recovery
    assert "legacy_isaac_ip" not in recovery
    assert 'song@$t5_isaac_ip|/tmp/internnav_isaac.quarantine' in recovery
    assert 'song@$t5_isaac_ip|/tmp/internnav_isaac_gpu0.quarantine' in recovery
    assert 'song@$t5_isaac_ip|/tmp/internnav_isaac_gpu1.quarantine' in recovery
