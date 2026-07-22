from __future__ import annotations

from scripts.t5_network_probe import Host, _ros_command, evaluate, parse_ping


def test_parse_linux_ping_summary() -> None:
    value = parse_ping(
        "20 packets transmitted, 20 received, 0% packet loss, time 971ms\n"
        "rtt min/avg/max/mdev = 0.210/0.350/0.900/0.120 ms\n"
    )
    assert value == {
        "transmitted": 20,
        "received": 20,
        "packet_loss_percent": 0.0,
        "rtt_min_ms": 0.21,
        "rtt_avg_ms": 0.35,
        "rtt_max_ms": 0.9,
        "jitter_ms": 0.12,
    }


def test_evaluate_passes_complete_low_latency_measurement() -> None:
    inventory = {
        "dgx_model": {
            "user": "railgun",
            "selected_interface": {"ifname": "eth0"},
            "ros_jazzy_present": True,
            "process_counts": {"model": 0, "nav2": 0, "isaac": 0},
        },
        "dgx_edge": {
            "user": "rail",
            "selected_interface": {"ifname": "eth0"},
            "ros_jazzy_present": True,
            "process_counts": {"model": 0, "nav2": 0, "isaac": 0},
        },
        "isaac_x86": {
            "user": "song",
            "selected_interface": {"ifname": "eth0"},
            "ros_jazzy_present": True,
            "process_counts": {"model": 0, "nav2": 0, "isaac": 0},
        },
    }
    measurements = {
        "inventory": inventory,
        "ping": [
            {"packet_loss_percent": 0.0, "rtt_max_ms": 1.0, "jitter_ms": 0.1}
        ],
        "bandwidth": [
            {"client_bits_per_second": 1e9, "server_bits_per_second": 9e8}
        ],
        "clock": [{"best_offset_ns": 1_000_000}],
        "dds": [{"observed": {"dgx_model": True, "dgx_edge": True}}],
    }
    summary = evaluate(measurements)
    assert summary["status"] == "PASS"
    assert all(summary["checks"].values())


def test_ros_setup_temporarily_disables_nounset_on_hosts_and_container() -> None:
    model = _ros_command(Host("dgx_model", "railgun", "10.0.0.1"), 75, "ros2 topic list")
    isaac = _ros_command(Host("isaac_x86", "song", "10.0.0.2"), 75, "ros2 topic list")
    assert "set +u; source /opt/ros/jazzy/setup.bash; set -u" in model
    assert "docker exec --user admin" in isaac
    assert "internnav_t4_isaac_ros" in isaac
