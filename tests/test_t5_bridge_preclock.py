from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py"


def _policy_helper():
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef)
        and item.name == "_is_t5_preclock_zero_tf_warn_drop"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BRIDGE), "exec"), namespace)
    return namespace[node.name]


def test_only_opted_in_t5_completion_lane_drops_initial_zero_tf() -> None:
    policy = _policy_helper()
    common = {
        "runtime_policy": "completion_sim",
        "opt_in": True,
        "lane_identity_prefix": "a::",
        "part": "tf",
        "stamp": 0,
        "positive_stamp_seen": False,
    }
    assert policy(**common) is True
    for override in (
        {"runtime_policy": "strict_evidence"},
        {"opt_in": False},
        {"lane_identity_prefix": ""},
        {"lane_identity_prefix": "lane-a"},
        {"part": "clock"},
        {"part": "identity"},
        {"part": "lidar"},
        {"part": "depth"},
        {"stamp": -1},
        {"stamp": 1},
        {"positive_stamp_seen": True},
    ):
        values = {**common, **override}
        assert policy(**values) is False


def test_preclock_relaxation_is_default_off_and_t5_only_wired() -> None:
    bridge = BRIDGE.read_text(encoding="utf-8")
    t4 = (ROOT / "scripts/run_t4_dgx_onboard.sh").read_text(encoding="utf-8")
    t5 = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text(encoding="utf-8")
    assert 'self.declare_parameter("allow_preclock_zero_tf_warn_drop", False)' in bridge
    assert 'self.runtime_policy.name != "completion_sim"' in bridge
    assert 'or self.lane_identity_prefix not in {"a::", "b::"}' in bridge
    assert 'if part == "tf" and stamp > 0:' in bridge
    assert '"preclock_zero_tf_dropped": 0' in bridge
    assert 'self.counts["preclock_zero_tf_dropped"] == 1' in bridge
    assert '"action": "record_and_drop_preclock_tf_then_continue"' in bridge
    assert 'INTERNVLA_T4_ALLOW_PRECLOCK_ZERO_TF_WARN_DROP:-0' in t4
    assert "-p allow_preclock_zero_tf_warn_drop:=" in t4
    assert "INTERNVLA_T4_ALLOW_PRECLOCK_ZERO_TF_WARN_DROP=1" in t5


@pytest.mark.parametrize("prefix", ["", "a::", "b::"])
def test_preclock_policy_accepts_only_frozen_prefix_values(prefix: str) -> None:
    policy = _policy_helper()
    assert policy(
        runtime_policy="completion_sim",
        opt_in=True,
        lane_identity_prefix=prefix,
        part="tf",
        stamp=0,
        positive_stamp_seen=False,
    ) is bool(prefix)
