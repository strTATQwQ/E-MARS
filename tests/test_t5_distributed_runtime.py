from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_t5_interface_carries_dual_time_and_compressed_transport() -> None:
    metadata = (ROOT / "internvla_ros2_msgs/msg/ObservationMetadata.msg").read_text()
    action = (ROOT / "internvla_ros2_msgs/action/Step.action").read_text()
    client = (ROOT / "internvla_ros2/internvla_ros2/client_node.py").read_text()
    model = (ROOT / "internvla_ros2/internvla_ros2/model_node.py").read_text()
    for text in (metadata, action):
        assert "uint64 client_wall_monotonic_ns" in text
        assert "builtin_interfaces/Time sim_stamp" in text
    assert 'declare_parameter("observation_transport", "raw")' in client
    assert 'declare_parameter("observation_transport", "raw")' in model
    assert "/internvla/observation/rgb/compressed" in client
    assert "/internvla/observation/depth/compressed" in model
    assert "deadline_ns / 1e9 - time.time()" not in client


def test_t5_lanes_are_symmetric_complete_stacks_and_concurrent() -> None:
    topology = json.loads(
        (ROOT / "configs/internnav_t5/topology.json").read_text(encoding="utf-8")
    )
    execution = topology["execution_model"]
    assert execution["lane_a_does_not_lock_dgx_b"] is True
    assert execution["lane_b_does_not_lock_dgx_a"] is True
    assert execution["each_dgx_runs_its_model_and_navigation_concurrently"] is True
    model = (ROOT / "scripts/run_t5_distributed_model.sh").read_text()
    edge = (ROOT / "scripts/run_t5_distributed_edge.sh").read_text()
    lane = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text()
    isaac = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    assert "run_t5_dgx_lane.sh" in model
    assert "run_t5_dgx_lane.sh" in edge
    assert "run_t4_model_server.sh" in lane
    assert "run_t4_dgx_onboard.sh" in lane
    assert "model_and_nav2_colocated" in lane
    assert "run_t4_sensor_gate.sh" in isaac


def test_isaac_overlay_starts_no_local_model_or_navigation_client(tmp_path: Path) -> None:
    output = tmp_path / "phase.sh"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_t5_isaac_remote_phase_overlay.py"),
            "--source",
            str(ROOT / "scripts/run_go2_continuous_phase.sh"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = output.read_text(encoding="utf-8")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert "ros2 run internvla_t4_sensors internvla_t4_client" not in text
    assert "ros2 run internvla_ros2 internvla_nav2_oracle_bridge" not in text
    assert "INTERNVLA_CLIENT_ENDPOINT" in text
    assert "INTERNVLA_ORACLE_ENDPOINT" in text
    assert "/dev/tcp" not in text
    assert "bare TCP" in text
    assert "materialize_t5_episode_order.py" in text
    assert "INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST" in text
    assert "BasePathKeyEpisodeloader" not in text
    assert 'T0_CONTROL_ROOT="$CONTROL_ROOT"' in text
    assert 'export INTERNNAV_T0_CONTROL_ROOT="$CONTROL_ROOT"' in text
    assert value["local_ros_navigation_or_client_processes_started"] is False
    assert value["t0_control_root_binding"] == (
        "exact_t5_deployment_control_root"
    )


def test_x86_is_sole_clock_source_and_has_two_isolated_gpu_workers() -> None:
    publisher = (ROOT / "scripts/t5_clock_publisher.py").read_text()
    runtime = (ROOT / "scripts/internnav_go2_runtime.py").read_text()
    isaac = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    assert 'create_publisher(Clock, "/clock", 10)' in publisher
    assert 'bind(("127.0.0.1", port))' in publisher
    assert 'parser.add_argument("--state", type=Path, required=True)' in publisher
    assert 'self._write_state("RUNNING")' in publisher
    assert '"received_step_count": self.received_count' in publisher
    assert "if value <= 0:" in publisher
    assert "if self.received_any and value <= self.last_clock_ns:" in publisher
    assert "_advance_t5_sim_clock(self.dt)" in runtime
    assert "CUDA_VISIBLE_DEVICES" in isaac
    assert "internnav_t5_isaac_a" in isaac
    assert "internnav_t5_isaac_b" in isaac


def test_oracle_preflight_binds_frozen_parent_dataset_outside_deployment() -> None:
    isaac = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    assert "t5_host_repository_root=/home/song/internnav-t1-t2" in isaac
    assert (
        't3_base_oracle_dataset_root="$t5_host_repository_root/episodes/h1_nav2_oracle"'
        in isaac
    )
    assert (
        'export INTERNVLA_T3_BASE_ORACLE_DATASET_ROOT="$t3_base_oracle_dataset_root"'
        in isaac
    )
    assert 'if test "$mode" = oracle; then' in isaac
    assert ')" = 10' in isaac


def test_active_nvblox_fixed_three_uses_its_frozen_two_of_three_threshold() -> None:
    isaac = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    assert 'nvblox_mode="${INTERNNAV_T5_NVBLOX_MODE:-off}"' in isaac
    assert 'test "$nvblox_mode" = active_local_gt' in isaac
    assert 'test "$dataset_episode_count" = 3' in isaac
    assert "INTERNVLA_T4_MIN_SR_OVERRIDE=0.6666666666666666" in isaac
    assert 'elif test "$mode" = oracle; then' in isaac
    assert "INTERNVLA_T4_MIN_SR_OVERRIDE=0.8" in isaac
    assert '"nvblox_mode": os.environ["INTERNNAV_T5_NVBLOX_MODE"]' in isaac


def test_primary_profile_keeps_t4_identity_and_functional_thresholds() -> None:
    profile = json.loads(
        (ROOT / "configs/internnav_t5/primary_completion_sim.json").read_text()
    )
    assert profile["frozen_t4_identity"]["commit_sha"] == (
        "d6332b75a3830c0e2f9752446340c90398a2c489"
    )
    assert profile["acceptance"] == {
        "oracle_episode_count": 5,
        "oracle_minimum_success_count": 4,
        "pilot_episode_count": 20,
        "pilot_minimum_success_count": 1,
    }


def test_legacy_t5_1_runner_cannot_authorize_from_superseded_board() -> None:
    runner = (ROOT / "coordination/run_t5_1_online.sh").read_text()
    assert "permanently disabled" in runner
    assert "exit 69" in runner
    assert "with_resource_lease.sh" not in runner
    tombstone = (ROOT / "coordination/T5_TASK_BOARD.md").read_text()
    assert "INTERNAV_T5_ONLINE_GRANT_V1" not in tombstone
    assert "no longer an authorization source" in tombstone
