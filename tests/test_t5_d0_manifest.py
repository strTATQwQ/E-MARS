from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def canonical(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_d0_manifest_uses_frozen_five_and_current_golden() -> None:
    manifest = json.loads((ROOT / "configs/internnav_t5/d0_run_manifest.json").read_text())
    golden = ROOT / "configs/internnav_t5/golden_bundle_manifest.json"
    assert manifest["golden_bundle_canonical_sha256"] == canonical(golden)
    assert manifest["fixed_input"]["episode_count"] == 5
    assert len(set(manifest["fixed_input"]["episode_keys"])) == 5
    assert manifest["fixed_input"]["dataset_file_sha256"] == (
        "c568e33a55b6668f7669ddc5beef1af0967a674f4712eef29a3cf3151fba94d4"
    )
    assert manifest["fixed_input"]["same_episode_on_two_lanes_counts_once"] is True
    assert manifest["stage_order"][0] == "d0_0_online_prepare"


def test_d0_lanes_are_symmetric_except_isolation_fields() -> None:
    value = json.loads((ROOT / "configs/internnav_t5/d0_run_manifest.json").read_text())
    a, b = value["lanes"]["a"], value["lanes"]["b"]
    assert set(a) == set(b)
    assert a["resource_profile"] == "lane-a"
    assert b["resource_profile"] == "lane-b"
    assert a["cpuset"] == "0-7,16-23"
    assert b["cpuset"] == "8-15,24-31"
    assert a["identity_prefix"] == "a::"
    assert b["identity_prefix"] == "b::"
    assert a["ros_domain_id"] != b["ros_domain_id"]
    assert a["isaac_cuda_visible_devices"] != b["isaac_cuda_visible_devices"]
    assert a["isaac_render_gpu_physical"] == 0
    assert b["isaac_render_gpu_physical"] == 1
    assert a["isaac_physics_gpu_logical"] == b["isaac_physics_gpu_logical"] == 0
    assert a["kit_gpu_foundation_log_audit_required"] is True
    assert b["kit_gpu_foundation_log_audit_required"] is True


def test_d0_requires_zero_pollution_and_twenty_percent_capacity_gate() -> None:
    value = json.loads((ROOT / "configs/internnav_t5/d0_run_manifest.json").read_text())
    acceptance = value["acceptance"]
    assert acceptance["cross_lane_ros_or_reset_count"] == 0
    assert acceptance["residual_pid_pgid_socket_lock_count"] == 0
    assert acceptance["maximum_simultaneous_slowdown_fraction"] == 0.2
    assert value["shared_x86_policy"]["full_mp4_encoding_concurrent"] is False
    assert value["x86_host_inventory"]["d0_dual_runtime_recheck_required"] is True
