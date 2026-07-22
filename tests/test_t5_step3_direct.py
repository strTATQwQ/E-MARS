from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_direct_launchers_are_lane_b_only_and_never_start_internvla() -> None:
    dgx = (ROOT / "scripts/run_t5_step3_direct_dgx_lane.sh").read_text()
    online = (
        ROOT / "coordination/run_t5_step3_direct_fixed5_online.sh"
    ).read_text()
    services = (ROOT / "scripts/run_t5_step3_live_services.sh").read_text()
    isaac = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text()
    for path in (
        ROOT / "scripts/run_t5_step3_direct_dgx_lane.sh",
        ROOT / "scripts/run_t5_step3_live_services.sh",
        ROOT / "scripts/run_t5_distributed_isaac.sh",
        ROOT / "coordination/run_t5_step3_direct_fixed5_online.sh",
    ):
        completed = subprocess.run(
            ["bash", "-n", path.relative_to(ROOT).as_posix()],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    assert 'bash "$root/scripts/run_t4_model_server.sh"' not in dgx
    assert "audit_no_internvla" in dgx
    assert "INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL=1" in dgx
    assert "audit_no_internvla" in dgx
    assert "internvla_fallback_allowed\": False" in dgx
    assert "step3_site" in dgx
    assert 'if (Path(value) / "zmq").is_dir()' in dgx
    assert "import zmq; assert zmq.__version__" in dgx
    assert "pip install" not in dgx
    assert "lane_b_step3_direct_fixed5" in isaac
    assert "build_t5_lane_b_step3_direct_runtime_overlay.py" in isaac
    assert "dgx_b" in online and "isaac_gpu1" in online
    assert "dgx_a" in online and "isaac_gpu0" in online  # release proof excludes both
    assert 'value["dataset_root"]' in online
    assert 'value["dataset_remote_path"]' not in online
    assert "planner_mode=direct_high_level" in services


def test_direct_private_dispatch_and_failure_contract_are_explicit() -> None:
    client_dispatch = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/client_dispatch.py"
    ).read_text()
    adapter_dispatch = (
        ROOT / "internvla_t4_recovery/internvla_t4_recovery/adapter_dispatch.py"
    ).read_text()
    client = (
        ROOT
        / "internvla_t4_sensors/internvla_t4_sensors/lane_b_step3_client_node.py"
    ).read_text()
    coordinator = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/lane_b_step3_live.py"
    ).read_text()
    for source in (client_dispatch, adapter_dispatch):
        assert "INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL" in source
    assert "_LaneBStep3ExecutionBridge" in client
    assert "InternVLAClientNode" in client
    assert "return self._execute_direct_step3" in client
    assert '"internvla_model_loaded": False' in client
    assert '"internvla_fallback_used": False' in client
    assert "DirectStep3Failure" in coordinator
    assert "return self.node._resolve_frozen_nav2(command)" in coordinator
    assert "LaneBIntent.FRONTIER_GOAL_CANDIDATE" in coordinator
    assert "and not outcome.requires_safe_stop" in coordinator
    direct_branch = coordinator.split(
        "if self.mode is LaneBPlannerMode.DIRECT_HIGH_LEVEL:", 1
    )[1]
    assert "_direct_safe_stop" in direct_branch


def test_frontier_pure_helpers_are_static_for_instance_call_binding() -> None:
    source = (
        ROOT
        / "internvla_t4_recovery/internvla_t4_recovery/lane_b_step3_adapter_node.py"
    ).read_text()
    tree = ast.parse(source)
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LaneBStep3RecoveryAdapter"
    )
    expected = {
        "_step3_path_sha256",
        "_step3_identity",
        "_expected_step3_snapshot_id",
    }
    helpers = {
        node.name: node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef) and node.name in expected
    }
    assert set(helpers) == expected
    for helper in helpers.values():
        assert [
            decorator.id
            for decorator in helper.decorator_list
            if isinstance(decorator, ast.Name)
        ] == ["staticmethod"]


def _write_navigation(root: Path, *, sr: float, stuck: int) -> None:
    run = root / "remote/x86/evaluator/run"
    run.mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps(
            {
                "val_unseen": {
                    "SR": sr, "OS": sr, "SPL": sr, "NE": 4.0,
                    "Count": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "per_episode.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"termination_reason": "stuck" if index < stuck else "success"}
                    for index in range(5)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_direct_comparison_reports_calls_latency_safe_stop_and_no_fake_ndtw(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    direct = tmp_path / "direct"
    _write_navigation(baseline, sr=0.2, stuck=5)
    _write_navigation(direct, sr=0.4, stuck=3)
    dgx = direct / "remote/dgx_b"
    (dgx / "client").mkdir(parents=True)
    (dgx / "onboard").mkdir()
    (dgx / "step3").mkdir()
    (dgx / "client/client_records.jsonl").write_text(
        "".join(
            json.dumps(
                {"planning_latency_sec": value, "command_age_sec": value}
            )
            + "\n"
            for value in (8.0, 10.0)
        ),
        encoding="utf-8",
    )
    (dgx / "onboard/step3_advisor_records.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"phase": "prepare", "episode_id": "b::e1", "recorded_wall_time": 1.0},
                {"phase": "commit", "episode_id": "b::e1", "status": "COMMITTED", "latency_sec": 0.4},
                {"phase": "prepare", "episode_id": "b::e1", "recorded_wall_time": 12.0},
            )
        ),
        encoding="utf-8",
    )
    (dgx / "step3/frontend_decisions.jsonl").write_text(
        json.dumps(
            {
                "decision": {
                    "source_decision": "abstain", "requires_safe_stop": True,
                    "safe_stop_reason": "step3_abstain", "fallback_used": False,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_t5_step3_direct_comparison.py"),
            "--baseline", str(baseline), "--direct", str(direct),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["baseline"]["sr"] == 0.2
    assert value["step3_direct"]["sr"] == 0.4
    assert value["step3_direct"]["ndtw"] is None
    assert value["step3_direct"]["step3_call_count"] == 2
    assert value["step3_direct"]["step3_calls_per_episode"] == {"b::e1": 2}
    assert value["step3_direct"]["abstain_count"] == 1
    assert value["step3_direct"]["safe_stop_count"] == 1
    assert value["step3_direct"]["fallback_count"] == 0
    assert value["step3_direct"]["internvla_model_loaded"] is False
