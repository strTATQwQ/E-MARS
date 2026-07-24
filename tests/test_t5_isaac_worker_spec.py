import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_t5_isaac_worker_spec.py"
SPEC = importlib.util.spec_from_file_location("t5_worker_spec", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def expected() -> dict:
    root = "/home/song/internnav-t1-t2"
    return {
        "schema_version": 1,
        "spec_version": 1,
        "container": "internnav_t5_isaac_a",
        "lane": "a",
        "gpu": 0,
        "ros_domain_id": 75,
        "cpuset": "0-7,16-23",
        "control_root": root,
        "lane_deployment_root": f"{root}/.t5-deployments/grant-isaac-a",
        "other_lane_deployment_root": f"{root}/.t5-deployments/grant-isaac-b",
        "lane_profile_root": f"{root}/runtime/t5_isaac_workers/a",
        "other_lane_profile_root": f"{root}/runtime/t5_isaac_workers/b",
        "ros_workspace": "/home/song/internnav-t4/isaac_ros_ws_45",
        "image_reference": "isaac:test",
        "image_id": "sha256:expected",
        "host_user_uid": 1000,
        "host_user_gid": 1000,
        "namespace": "/t5/lane_a",
        "static_peer": "10.100.100.128",
        "identity_prefix": "a::",
    }


def inspected(value: dict) -> dict:
    profile = value["lane_profile_root"]
    labels = {
        "internnav.t5.spec_version": "1",
        "internnav.t5.role": "isaac_ros_worker",
        "internnav.t5.lane": value["lane"],
        "internnav.t5.gpu": str(value["gpu"]),
        "internnav.t5.ros_domain_id": str(value["ros_domain_id"]),
        "internnav.t5.control_root": value["control_root"],
        "internnav.t5.deployment_root": value["lane_deployment_root"],
        "internnav.t5.worker_profile_root": profile,
        "vendor.image.label": "allowed",
    }
    environment = [
        f'ROS_DOMAIN_ID={value["ros_domain_id"]}',
        "ROS_LOCALHOST_ONLY=0",
        f'ROS_NAMESPACE={value["namespace"]}',
        "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
        f'ROS_STATIC_PEERS={value["static_peer"]}',
        f'INTERNNAV_T5_LANE={value["lane"]}',
        f'INTERNNAV_T5_ID_PREFIX={value["identity_prefix"]}',
        "CUDA_VISIBLE_DEVICES=0",
        f'NVIDIA_VISIBLE_DEVICES={value["gpu"]}',
        f"XDG_CACHE_HOME={profile}/cache/xdg",
        f"XDG_CONFIG_HOME={profile}/config",
        f"XDG_DATA_HOME={profile}/data",
        f"OV_CACHE_ROOT={profile}/cache/ov",
        f"NVIDIA_SHADER_CACHE_PATH={profile}/cache/nvidia",
        f"TMPDIR={profile}/tmp",
        "USERNAME=admin",
        f'HOST_USER_UID={value["host_user_uid"]}',
        f'HOST_USER_GID={value["host_user_gid"]}',
    ]
    mounts = [
        {"Type": "bind", "Source": value["control_root"], "Destination": value["control_root"], "RW": False},
        {"Type": "bind", "Source": value["lane_deployment_root"], "Destination": value["lane_deployment_root"], "RW": True},
        {"Type": "bind", "Source": profile, "Destination": profile, "RW": True},
        {"Type": "bind", "Source": value["ros_workspace"], "Destination": "/workspaces/isaac", "RW": False},
    ]
    return {
        "Id": "container-id",
        "Name": f'/{value["container"]}',
        "Image": value["image_id"],
        "Config": {
            "Image": value["image_reference"], "Labels": labels, "Env": environment,
            "Entrypoint": ["/usr/local/bin/scripts/workspace-entrypoint.sh"],
            "Cmd": ["sleep", "infinity"],
        },
        "HostConfig": {
            "CpusetCpus": value["cpuset"], "PidMode": "", "IpcMode": "private",
            "NetworkMode": "host", "Privileged": False, "RestartPolicy": {"Name": "no"},
            "DeviceRequests": [{"Driver": "", "Count": 0, "DeviceIDs": [str(value["gpu"])],
                                "Capabilities": [["gpu"]], "Options": {}}],
        },
        "Mounts": mounts,
    }


def replace_env(actual: dict, key: str, value: str) -> None:
    actual["Config"]["Env"] = [
        f"{key}={value}" if item.startswith(f"{key}=") else item
        for item in actual["Config"]["Env"]
    ]


def test_exact_fixture_passes() -> None:
    expected_value = expected()
    result = MODULE.validate(expected_value, inspected(expected_value))
    assert result["status"] == "PASS"
    assert result["failed_checks"] == []


def test_stale_domain_fixture_fails_closed() -> None:
    expected_value = expected()
    actual = inspected(expected_value)
    replace_env(actual, "ROS_DOMAIN_ID", "76")
    actual["Config"]["Labels"]["internnav.t5.ros_domain_id"] = "76"
    result = MODULE.validate(expected_value, actual)
    assert result["status"] == "FAIL"
    assert {"labels_exact", "environment_contract"} <= set(result["failed_checks"])


def test_stale_cache_fixture_fails_closed() -> None:
    expected_value = expected()
    actual = inspected(expected_value)
    replace_env(actual, "OV_CACHE_ROOT", "/shared/ov-cache")
    result = MODULE.validate(expected_value, actual)
    assert result["status"] == "FAIL"
    assert "environment_contract" in result["failed_checks"]


def test_stale_dds_peer_or_identity_fixture_fails_closed() -> None:
    expected_value = expected()
    actual = inspected(expected_value)
    replace_env(actual, "ROS_STATIC_PEERS", "10.100.120.122")
    replace_env(actual, "INTERNNAV_T5_ID_PREFIX", "b::")
    result = MODULE.validate(expected_value, actual)
    assert result["status"] == "FAIL"
    assert {
        "environment_contract",
        "lane_identity_environment_exact",
    } <= set(result["failed_checks"])


def test_stale_image_fixture_fails_closed() -> None:
    expected_value = expected()
    actual = inspected(expected_value)
    actual["Image"] = "sha256:stale"
    result = MODULE.validate(expected_value, actual)
    assert result["status"] == "FAIL"
    assert "image_id" in result["failed_checks"]


def test_cross_lane_writable_mount_fixture_fails_closed() -> None:
    expected_value = expected()
    actual = inspected(expected_value)
    actual["Mounts"].append({
        "Type": "bind",
        "Source": expected_value["other_lane_deployment_root"],
        "Destination": expected_value["other_lane_deployment_root"],
        "RW": True,
    })
    result = MODULE.validate(expected_value, actual)
    assert result["status"] == "FAIL"
    assert {"mounts_exact", "other_lane_deployment_not_writable"} <= set(result["failed_checks"])


def test_prepare_recreates_only_a_stopped_drifted_container_under_exclusive_lock() -> None:
    script = (ROOT / "scripts" / "prepare_t5_isaac_workers.sh").read_text()
    assert "flock -x -w 30 8" in script
    assert "test \"$(docker inspect -f '{{.State.Running}}' \"$container\")\" != true" in script
    assert 'if ! python3 "$validator" "$expected" "$inspected" "$audit"; then' in script
    assert 'docker rm "$container"' in script
    assert "create_lane_container" in script
