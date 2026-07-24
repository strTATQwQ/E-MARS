from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_strict_runtime_probe_is_read_only_and_lane_specific() -> None:
    text = (ROOT / "scripts/probe_t5_dgx_strict_runtime.sh").read_text(
        encoding="utf-8"
    )
    assert "BatchMode=yes" in text
    assert "10.100.100.128" in text
    assert "10.100.120.122" in text
    assert "apt-cache" in text
    assert "ros2 pkg prefix" in text
    assert "nvblox_ros" in text
    assert "isaac_ros_visual_slam" in text
    for mutating_command in ("apt-get install", "sudo ", "docker run", "rm -rf"):
        assert mutating_command not in text
