from isaac_vln_benchmark.sim2real_vnext_utils import ALLOWED_WHILE_NOT_READY, evaluate_sim2real_vnext


def _session():
    return {
        "pass": True,
        "max_linear_x_mps": 0.20,
        "runtime_stale_discards": 0,
        "timebase_error": 0,
        "parse_error": 0,
        "collision_count": 0,
        "fall_count": 0,
        "stale_action_executed": 0,
    }


def test_sim2real_gate_requires_all_post_value_qualification_evidence():
    result = evaluate_sim2real_vnext(
        value_confirmation={"pass": True},
        route_micro={"correct_branch_rate": 1.0},
        stop_micro={"stop_decision_accuracy": 1.0, "visible_to_stop_latency_p95_sec": 1.83},
        independent_sessions=[_session(), _session(), _session()],
        stress={"pass": True},
        replay={"pass": True},
    )
    assert result["ready"] is True
    assert result["real_go2_autonomy_enabled"] is False


def test_sim2real_gate_stays_not_ready_when_confirmation_and_stress_are_missing():
    result = evaluate_sim2real_vnext(
        value_confirmation={},
        route_micro={"correct_branch_rate": 1.0},
        stop_micro={"stop_decision_accuracy": 1.0, "visible_to_stop_latency_p95_sec": 1.83},
        independent_sessions=[],
        stress={},
        replay={},
    )
    assert result["ready"] is False
    assert result["status"] == "NOT READY FOR REAL ROBOT AUTONOMY"
    assert result["allowed_only"] == ALLOWED_WHILE_NOT_READY
