from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_two_isaac_ros_workers_have_private_pid_ipc_gpu_and_cpu_sets() -> None:
    script = (ROOT / "scripts/prepare_t5_isaac_workers.sh").read_text()
    assert "internnav_t5_isaac_a" in script
    assert "internnav_t5_isaac_b" in script
    assert "--ipc private" in script
    assert "--pid host" not in script
    assert 'prepare_lane a 0 75 "$lane_a_cpuset"' in script
    assert 'prepare_lane b 1 76 "$lane_b_cpuset"' in script
    assert 'lane_a_cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"' in script
    assert 'lane_b_cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"' in script
    assert "internnav_t5_isaac_shared_assets.lock" in script
    assert "flock -x" in script
    assert "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" in script
    assert 'ROS_NAMESPACE=$namespace' in script
    assert 'ROS_STATIC_PEERS=$static_peer' in script
    assert 'INTERNNAV_T5_ID_PREFIX=$identity_prefix' in script
    assert 'isaac_ip="${INTERNVLA_T5_ISAAC_IP:-10.100.120.123}"' in script
    assert 'test "$isaac_ip" = 10.100.120.123' in script
    assert 'grep -Fq " $isaac_ip/"' in script


def test_worker_prepare_requires_real_dual_gpu_lease_and_separate_profiles() -> None:
    script = (ROOT / "scripts/prepare_t5_isaac_workers.sh").read_text()
    assert 'case "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" in' in script
    assert "all-lanes)" in script
    assert "isaac)" in script
    assert "requires all-lanes or isaac lease" in script
    for variable in (
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "OV_CACHE_ROOT",
        "NVIDIA_SHADER_CACHE_PATH",
        "TMPDIR",
    ):
        assert variable in script
    assert "full_mp4_encoding_allowed_concurrently" in script
    assert "INTERNVLA_T5_X86_LANE_A_ROOT" in script
    assert "INTERNVLA_T5_X86_LANE_B_ROOT" in script
    assert '--mount "type=bind,src=$root,dst=$root,readonly"' in script
    assert '--mount "type=bind,src=$lane_deployment_root,dst=$lane_deployment_root"' in script
    assert '--mount "type=bind,src=$lane_root,dst=$lane_root"' in script
    assert '-v "$root:$root"' not in script
