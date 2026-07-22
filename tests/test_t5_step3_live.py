from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from PIL import Image

_NUMPY_STUBBED = False
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")
    _NUMPY_STUBBED = True


ROOT = Path(__file__).resolve().parents[1]
for package_root in (
    ROOT / "internvla_nav2_adapter",
    ROOT / "internvla_go2_controller",
    ROOT / "scripts",
):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from internvla_nav2_adapter.lane_b_frontier_transaction import (  # noqa: E402
    FrontierTransactionError,
    FrontierTransactionStore,
    PreparedFrontier,
)
from internnav_t5_lane_b_step3_agent_client import (  # noqa: E402
    LaneBStep3ROS2IPCAgentClient as ROS2IPCAgentClient,
    MAX_ADVISOR_IMAGE_BYTES,
    MAX_ADVISOR_TOTAL_IMAGE_BYTES,
    STEP3_SNAPSHOT_WAIT_SEC,
    STEP3_VIEW_ORDER,
    _step3_contract_values,
)

if _NUMPY_STUBBED:
    sys.modules.pop("numpy", None)


IDENTITY = ("b::episode-1", 0, 4, "b::request-4", "b::episode-1::0::4")
PATH_SHA = hashlib.sha256(b"frontier-path").hexdigest()


def _frontier(frontier_id: int = 2) -> PreparedFrontier:
    return PreparedFrontier(
        frontier_id=frontier_id,
        discrete_action=1,
        relative_x=0.0,
        relative_z=0.75,
        distance_m=0.75,
        bearing_deg=0.0,
        path_sha256=PATH_SHA,
        path={"saved": True},
    )


def test_frontier_transaction_is_one_shot_and_identity_bound() -> None:
    now = [100.0]
    store = FrontierTransactionStore(
        clock=lambda: now[0], token_factory=lambda: "1" * 32
    )
    prepared = store.prepare(IDENTITY, [_frontier()], prepared_sim_ns=9_000)
    with pytest.raises(FrontierTransactionError, match="identity or token"):
        store.consume(
            identity=(IDENTITY[0], 1, *IDENTITY[2:]),
            token=prepared.token,
            select_frontier=True,
            frontier_id=2,
            expected_path_sha256=PATH_SHA,
        )
    assert store.current is None
    with pytest.raises(FrontierTransactionError, match="no prepared"):
        store.consume(
            identity=IDENTITY,
            token=prepared.token,
            select_frontier=True,
            frontier_id=2,
            expected_path_sha256=PATH_SHA,
        )


def test_frontier_transaction_rejects_expiry_id_and_path_changes() -> None:
    now = [10.0]
    tokens = iter(("2" * 32, "3" * 32, "4" * 32))
    store = FrontierTransactionStore(clock=lambda: now[0], token_factory=lambda: next(tokens))
    prepared = store.prepare(IDENTITY, [_frontier()], prepared_sim_ns=12_000)
    now[0] += 15.001
    with pytest.raises(FrontierTransactionError, match="expired"):
        store.consume(
            identity=IDENTITY,
            token=prepared.token,
            select_frontier=True,
            frontier_id=2,
            expected_path_sha256=PATH_SHA,
        )
    prepared = store.prepare(IDENTITY, [_frontier()], prepared_sim_ns=12_000)
    with pytest.raises(FrontierTransactionError, match="not current"):
        store.consume(
            identity=IDENTITY,
            token=prepared.token,
            select_frontier=True,
            frontier_id=99,
            expected_path_sha256=PATH_SHA,
        )
    prepared = store.prepare(IDENTITY, [_frontier()], prepared_sim_ns=12_000)
    with pytest.raises(FrontierTransactionError, match="digest mismatch"):
        store.consume(
            identity=IDENTITY,
            token=prepared.token,
            select_frontier=True,
            frontier_id=2,
            expected_path_sha256="f" * 64,
        )


def test_frontier_transaction_explicit_release_has_no_motion_payload() -> None:
    store = FrontierTransactionStore(token_factory=lambda: "5" * 32)
    prepared = store.prepare(IDENTITY, [_frontier()], prepared_sim_ns=15_000)
    selected, prepared_sim_ns = store.consume(
        identity=IDENTITY,
        token=prepared.token,
        select_frontier=False,
    )
    assert selected is None
    assert prepared_sim_ns == 15_000


def _live_client_fixture(tmp_path: Path) -> tuple[ROS2IPCAgentClient, Path]:
    contract_path = ROOT / "configs/internnav_t5/revc_four_camera_snapshot.json"
    contract_sha, extrinsics, cameras = _step3_contract_values(contract_path)
    client = ROS2IPCAgentClient.__new__(ROS2IPCAgentClient)
    client.connection = None
    client._step3_live_enabled = True
    client._step3_direct_high_level = False
    client._step3_result_root = tmp_path
    client._step3_request_path = tmp_path / "revc_snapshot.request.json"
    client._step3_ack_path = tmp_path / "revc_snapshot.ack.json"
    client._step3_contract_sha256 = contract_sha
    client._step3_extrinsics = extrinsics
    client._step3_contract_cameras = cameras
    client._step3_episode_id = "b::episode-1"
    client._step3_reset_generation = 2
    client._step3_next_sequence = 3
    client.episode_ordinal = 0
    return client, contract_path


def _write_snapshot(client: ROS2IPCAgentClient, tmp_path: Path) -> Path:
    snapshot_dir = tmp_path / "revc_snapshots" / "snapshot-fixture"
    snapshot_dir.mkdir(parents=True)
    rows = []
    for index, view_id in enumerate(STEP3_VIEW_ORDER):
        camera = client._step3_contract_cameras[view_id]
        path = snapshot_dir / f"{index:02d}_{view_id}.png"
        Image.new("RGB", (640, 480), (index * 40, 80, 120)).save(path, "PNG")
        payload = path.read_bytes()
        rows.append(
            {
                "identity": view_id,
                "order_index": index,
                "sensor_name": camera["sensor_name"],
                "frame_id": camera["frame_id"],
                "prim_path": camera["prim_path"],
                "position_F_M_mm": camera["position_F_M_mm"],
                "yaw_deg": camera["yaw_deg"],
                "encoding": "png_rgb8",
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    sidecar = {
        "schema_version": 1,
        "contract_sha256": client._step3_contract_sha256,
        "same_render_tick": True,
        "sim_stamp_before_ns": 8_000_000_000,
        "sim_stamp_after_ns": 8_000_000_000,
        "episode_id": client._step3_episode_id,
        "reset_generation": client._step3_reset_generation,
        "sequence_id": client._step3_next_sequence,
        "camera_order": list(STEP3_VIEW_ORDER),
        "cameras": rows,
    }
    sidecar_path = snapshot_dir / "snapshot.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    request_id = "b::episode-1:step3:2:3"
    assert client._step3_ack_path is not None
    client._step3_ack_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CAPTURED",
                "request_id": request_id,
                "episode_id": "b::episode-1",
                "reset_generation": 2,
                "sequence_id": 3,
                "sidecar": sidecar_path.relative_to(tmp_path).as_posix(),
            }
        ),
        encoding="utf-8",
    )
    return sidecar_path


def test_live_snapshot_validates_and_transports_exact_four_bounded_jpegs(
    tmp_path: Path,
) -> None:
    client, _ = _live_client_fixture(tmp_path)
    _write_snapshot(client, tmp_path)
    value = client._step3_load_snapshot()
    assert value is not None
    assert value["snapshot_id"] == "b::episode-1::2::3"
    assert [item["view_id"] for item in value["images"]] == list(STEP3_VIEW_ORDER)
    sizes = [len(__import__("base64").b64decode(item["jpeg_base64"])) for item in value["images"]]
    assert max(sizes) <= MAX_ADVISOR_IMAGE_BYTES
    assert sum(sizes) <= MAX_ADVISOR_TOTAL_IMAGE_BYTES


def test_live_snapshot_wait_closes_capture_ack_race(tmp_path: Path) -> None:
    client, _ = _live_client_fixture(tmp_path)

    def delayed_capture() -> None:
        time.sleep(0.05)
        _write_snapshot(client, tmp_path)

    thread = threading.Thread(target=delayed_capture)
    thread.start()
    try:
        value = client._step3_wait_snapshot(timeout_sec=0.5, poll_sec=0.005)
    finally:
        thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert value is not None
    assert value["snapshot_id"] == "b::episode-1::2::3"
    assert STEP3_SNAPSHOT_WAIT_SEC == 5.0


def test_live_snapshot_wait_accepts_ack_at_observed_3p2_seconds(tmp_path: Path) -> None:
    client, _ = _live_client_fixture(tmp_path)
    now = [100.0]
    expected = {"snapshot_id": "b::episode-1::2::3"}
    client._step3_load_snapshot = lambda: expected if now[0] >= 103.2 else None
    value = client._step3_wait_snapshot(
        timeout_sec=5.0,
        poll_sec=0.1,
        clock=lambda: now[0],
        sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
    )
    assert value is expected
    assert 3.2 <= now[0] - 100.0 <= 3.3


def test_live_snapshot_wait_falls_back_at_five_second_wall_timeout(
    tmp_path: Path,
) -> None:
    client, _ = _live_client_fixture(tmp_path)
    now = [10.0]
    assert client._step3_wait_snapshot(
        timeout_sec=5.0,
        poll_sec=0.1,
        clock=lambda: now[0],
        sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
    ) is None
    assert now[0] == pytest.approx(15.0)


def test_direct_snapshot_timeout_fails_closed_without_sending_request() -> None:
    client = ROS2IPCAgentClient.__new__(ROS2IPCAgentClient)
    client.connection = None
    client._step3_live_enabled = True
    client._step3_direct_high_level = True
    client._step3_wait_snapshot = lambda: None
    sent: list[dict[str, object]] = []

    def send(_self: object, request: dict[str, object]) -> dict[str, object]:
        sent.append(request)
        return {"status": "ok"}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "internvla_ipc_agent_client.ROS2IPCAgentClient._exchange", send
        )
        with pytest.raises(RuntimeError, match="was not captured"):
            client._exchange({"operation": "step"})
    assert sent == []


@pytest.mark.parametrize("mutation", ["hash", "order", "cross_reset", "missing_image"])
def test_live_snapshot_corruption_is_deterministic_fallback(
    tmp_path: Path, mutation: str
) -> None:
    client, _ = _live_client_fixture(tmp_path)
    sidecar_path = _write_snapshot(client, tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if mutation == "hash":
        sidecar["cameras"][0]["sha256"] = "0" * 64
    elif mutation == "order":
        sidecar["camera_order"] = list(reversed(STEP3_VIEW_ORDER))
    elif mutation == "cross_reset":
        sidecar["reset_generation"] = 9
    else:
        (tmp_path / sidecar["cameras"][0]["path"]).unlink()
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert client._step3_load_snapshot() is None


def test_live_capture_arm_tracks_initialize_next_and_reset(tmp_path: Path) -> None:
    client, _ = _live_client_fixture(tmp_path)
    client._step3_update_identity(
        {"episode_id": "b::episode-1", "reset_generation": 0}, next_sequence=0
    )
    client._step3_arm_next_snapshot()
    assert json.loads(client._step3_request_path.read_text())["sequence_id"] == 0
    client._step3_request_path.unlink()
    client._step3_update_identity(
        {"episode_id": "b::episode-1", "reset_generation": 0}, next_sequence=1
    )
    client._step3_arm_next_snapshot()
    assert json.loads(client._step3_request_path.read_text())["sequence_id"] == 1
    client._step3_request_path.unlink()
    client._step3_update_identity(
        {"episode_id": "b::episode-2", "reset_generation": 1}, next_sequence=0
    )
    client._step3_arm_next_snapshot()
    request = json.loads(client._step3_request_path.read_text())
    assert (request["episode_id"], request["reset_generation"], request["sequence_id"]) == (
        "b::episode-2",
        1,
        0,
    )


def test_private_commit_interface_has_no_coordinate_velocity_or_stop_authority() -> None:
    commit = (ROOT / "internnav_t5_lane_b_msgs/srv/CommitFrontier.srv").read_text()
    request = commit.split("---", 1)[0]
    assert "frontier_id" in request
    for forbidden in ("cmd_vel", "geometry_msgs", "float32 x", "float32 y", "stop"):
        assert forbidden not in request.lower()


def test_coexistence_probe_is_observation_only_and_frozen_five_plus_five() -> None:
    probe = (ROOT / "scripts/probe_t5_internvla_observation_latency.py").read_text()
    coordinator = (ROOT / "coordination/run_t5_dgx_b_coexistence_online.sh").read_text()
    publisher_lines = [
        line.strip() for line in probe.splitlines() if "create_publisher(" in line
    ]
    assert len(publisher_lines) == 3
    assert all("observation" in line or line.endswith("(") for line in publisher_lines)
    assert "from internvla_ros2_msgs.msg import NavigationCommand" not in probe
    assert '"/internvla/discrete_action"' in probe
    assert '"/internvla/stop"' in probe
    assert '"/cmd_vel"' in probe
    assert 'args.count != 5' in probe
    assert "int(response.reset_barrier_sequence_id) != 0" in probe
    assert "reset returned nonzero new-generation sequence barrier" in probe
    assert coordinator.count("--count 5") == 2
    assert "internvla_only_latency.json" in coordinator
    assert "co_loaded_latency.json" in coordinator
    assert '"$target:$remote_result/logs"' in coordinator
    assert "p95_degradation <= 20.0" in coordinator


def test_live_coordinator_requires_snapshot_request_and_advisor_sim_tick_identity() -> None:
    coordinator = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/lane_b_step3_live.py"
    ).read_text()
    assert "sim_stamp_ns != int(request_sim_ns)" in coordinator
    assert "sim_before_ns != int(request_sim_ns)" in coordinator
    assert "sim_after_ns != sim_before_ns" in coordinator


def test_failed_step3_commit_blocks_fallback_until_follow_path_cleanup() -> None:
    adapter = (
        ROOT
        / "internvla_t4_recovery/internvla_t4_recovery/lane_b_step3_adapter_node.py"
    ).read_text()
    assert "step3_commit_cleanup_blocked" in adapter
    guarded_resolve = adapter.split(
        'if getattr(self, "step3_commit_cleanup_blocked", False):', 1
    )[1]
    assert "self._publish_motion(False)" in guarded_resolve
    assert "self._cancel_active(wait=True)" in guarded_resolve


def test_step3_commit_does_not_wait_for_cmd_while_sim_clock_is_frozen() -> None:
    adapter = (
        ROOT
        / "internvla_t4_recovery/internvla_t4_recovery/lane_b_step3_adapter_node.py"
    ).read_text()
    commit = adapter.split("def _commit_step3_frontier_locked", 1)[1].split(
        "def main", 1
    )[0]
    assert "self._send_path(selected.path)" in commit
    assert "self._wait_for_fresh_cmd" not in commit
    assert "action = int(selected.discrete_action)" in commit
    assert '"initial_control_before_clock_resume": "safe_zero"' in commit


def test_lane_a_frozen_candidate_code_bundle_is_unchanged() -> None:
    from scripts.analyze_t5_lane_a_candidates import code_bundle_sha256

    manifest = json.loads(
        (ROOT / "configs/internnav_t5/lane_a_candidates/manifest.json").read_text()
    )
    provenance = manifest["provenance_contract"]
    assert code_bundle_sha256(ROOT, provenance["code_paths"]) == provenance[
        "code_bundle_sha256"
    ]


def test_step3_uses_private_dispatchers_but_preserves_public_process_names() -> None:
    recovery_setup = (ROOT / "internvla_t4_recovery/setup.py").read_text()
    sensor_setup = (ROOT / "internvla_t4_sensors/setup.py").read_text()
    entrypoint = (ROOT / "scripts/run_internnav_go2_entrypoint.py").read_text()
    assert "internvla_t4_adapter = internvla_t4_recovery.adapter_dispatch:main" in recovery_setup
    assert "internvla_t4_client = internvla_t4_sensors.client_dispatch:main" in sensor_setup
    assert "LaneBStep3ContinuousROS2IPCAgentClient as Client" in entrypoint


def test_private_client_fallback_calls_frozen_resolver_without_recursion() -> None:
    private_client = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/lane_b_step3_client_node.py"
    ).read_text()
    coordinator = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/lane_b_step3_live.py"
    ).read_text()
    assert "def _resolve_frozen_nav2" in private_client
    assert "return super()._resolve_nav2(command)" in private_client
    assert "self.node._resolve_nav2(command)" not in coordinator
    assert "self.node._resolve_frozen_nav2(command)" in coordinator


def test_lane_b_prepare_does_not_expand_unset_destination_in_local_assignment() -> None:
    prepare = (ROOT / "coordination/run_t5_lane_b_prepare_online.sh").read_text()
    assert 'local target="$1" destination="$2" stage\n' in prepare
    assert 'stage="/tmp/t5-step3-${tag}-$(basename "$destination").tar.gz"' in prepare
