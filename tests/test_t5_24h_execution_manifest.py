import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/internnav_t5/t5_24h_execution_manifest.json"
GOLDEN = ROOT / "configs/internnav_t5/golden_bundle_manifest.json"
RESOURCES = ROOT / "configs/internnav_t5/candidates/active_isaac_epyc_resources.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_baseline_and_disjoint_source_control_ownership() -> None:
    manifest = load(MANIFEST)
    golden = load(GOLDEN)

    assert manifest["baseline"]["t5_commit_sha"] == (
        "2aa660cf1452a40a15248f06d5872115bb1a7032"
    )
    assert manifest["baseline"]["t4_functional_done_commit_sha"] == golden[
        "source_control"
    ]["t4_functional_done_commit_sha"]
    branches = manifest["source_control"]
    assert len(
        {
            branches["integration_branch"],
            branches["lane_a_branch"],
            branches["lane_b_branch"],
            branches["isaac_x86_branch"],
        }
    ) == 4
    owners = manifest["ownership"]
    assert set(owners) == {"lane_a", "lane_b", "isaac_x86", "codex_00"}


def test_fixed_resource_partition_matches_active_epyc_profile() -> None:
    manifest = load(MANIFEST)
    active = load(RESOURCES)
    for lane_id, active_id in (("lane_a", "a"), ("lane_b", "b")):
        expected = active["lanes"][active_id]
        actual = manifest["resources"][lane_id]
        assert actual["isaac"] == "song@10.100.120.123"
        assert actual["isaac_gpu"] == expected["gpu"]
        assert actual["isaac_cpuset"] == expected["cpuset"]
    a = set(manifest["resources"]["lane_a"]["isaac_cpuset"].split(","))
    b = set(manifest["resources"]["lane_b"]["isaac_cpuset"].split(","))
    assert a.isdisjoint(b)
    assert manifest["resources"]["normal_runs_use_all_lanes_lock"] is False


def test_internvla_search_space_and_action_gate_are_closed() -> None:
    search = load(MANIFEST)["internvla_search"]
    assert search["round_episode_counts"] == [1, 3, 5]
    assert [family["id"] for family in search["families"]] == [
        "action_observation_recovery_a",
        "camera_pose_history",
        "trajectory32_refresh",
    ]
    assert search["freeze_label"] == "LOCAL_SEARCH_PLATEAU"
    gate = search["action_gate"]
    assert gate["timebase"] == "sim_time"
    assert gate["turn"]["deadline_sim_sec"] == 2.0
    assert gate["turn"]["minimum_measured_yaw_rad"] >= 0.2094
    assert gate["forward"]["deadline_sim_sec"] == 3.0
    assert gate["forward"]["minimum_measured_distance_m"] == 0.2
    assert gate["before_next_observation"] == "safe_stop"


def test_step3_and_frontend_do_not_gain_low_level_control() -> None:
    manifest = load(MANIFEST)
    step3 = manifest["step3"]
    assert step3["protocol"] == "slow_planner_v1"
    assert step3["snapshot_id_format"] == "b::<episode>::<reset>::<sequence>"
    assert step3["ordered_views"] == ["front_left", "front", "front_right", "rear"]
    assert {"cmd_vel", "terminal_stop", "raw_chain_of_thought"} <= set(
        step3["forbidden_outputs"]
    )
    frontend = manifest["frontend"]
    assert frontend["read_only"] is True
    assert frontend["control_routes_allowed"] is False
    assert frontend["raw_text_allowed"] is False
    assert "GET /api/v1/cameras/{view_id}.jpg" in frontend["routes"]


def test_revc_geometry_and_cuvslam_rig_are_distinct() -> None:
    manifest = load(MANIFEST)
    cameras = manifest["revc_cameras"]
    assert cameras["resolution"] == [640, 480]
    assert cameras["temporary_base_transform"]["translation_m"] == [0.14, 0.0, 0.18]
    by_id = {view["id"]: view for view in cameras["views"]}
    assert by_id["front_left"]["position_mm"] == [30.0, 51.962, 20.0]
    assert by_id["front_right"]["yaw_deg"] == -60.0
    assert by_id["rear"]["yaw_deg"] == 180.0
    assert cameras["capture"] == "on_demand_same_render_tick"
    assert cameras["cuvslam_stereo_is_separate"] is True
    assert manifest["strict_extensions"]["cuvslam"]["stereo_resolution"] == [320, 240]


def test_final_pilot_partition_is_exactly_the_frozen_twenty() -> None:
    manifest = load(MANIFEST)
    golden = load(GOLDEN)
    pilot = manifest["final_evaluation"]["pilot"]
    a = pilot["lane_a_episode_keys"]
    b = pilot["lane_b_episode_keys"]
    frozen = golden["dataset_and_episodes"]["frozen_pilot_twenty"]["episode_keys"]
    assert len(a) == len(b) == 10
    assert set(a).isdisjoint(b)
    assert a + b == frozen
    assert pilot["aggregate_episode_count"] == 20


def test_manifest_contains_no_credentials_or_training_escape_hatches() -> None:
    manifest = load(MANIFEST)
    serialized = json.dumps(manifest).lower()
    for forbidden in ("password", "api_key"):
        assert forbidden not in serialized
    assert re.search(r"hf_[a-z0-9]{20,}", serialized) is None
    assert re.search(r"sk-[a-z0-9]{20,}", serialized) is None
    assert {"NVFP4", "LoRA", "model_finetuning"} <= set(manifest["prohibited"])
