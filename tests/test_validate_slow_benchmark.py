from scripts.validate_slow_benchmark_run import is_evidenced_slow_only_target_terminal


def slow_row(decision: str = "target_found") -> dict:
    return {"kind": "slow", "decision": {"decision": decision}}


def test_accepts_evidenced_false_target_found_before_fast_policy() -> None:
    episode = {
        "success": False,
        "failure_reason": "false_target_found",
        "fast_steps": 0,
        "slow_decisions": 1,
        "executed_path_m": 0.0,
        "target_found_tp": 0,
        "target_found_fp": 1,
    }
    assert is_evidenced_slow_only_target_terminal(episode, [slow_row()])


def test_accepts_evidenced_true_target_found_before_fast_policy() -> None:
    episode = {
        "success": True,
        "failure_reason": "",
        "fast_steps": 0,
        "slow_decisions": 1,
        "executed_path_m": 0.0,
        "target_found_tp": 1,
        "target_found_fp": 0,
    }
    assert is_evidenced_slow_only_target_terminal(episode, [slow_row()])


def test_rejects_unexplained_episode_without_fast_policy() -> None:
    episode = {
        "success": False,
        "failure_reason": "max_fast_steps",
        "fast_steps": 0,
        "slow_decisions": 1,
        "executed_path_m": 0.0,
        "target_found_tp": 0,
        "target_found_fp": 0,
    }
    assert not is_evidenced_slow_only_target_terminal(episode, [slow_row("select_frontier")])


def test_rejects_target_terminal_with_unaccounted_movement_or_slow_rows() -> None:
    episode = {
        "success": False,
        "failure_reason": "false_target_found",
        "fast_steps": 0,
        "slow_decisions": 2,
        "executed_path_m": 0.1,
        "target_found_tp": 0,
        "target_found_fp": 1,
    }
    assert not is_evidenced_slow_only_target_terminal(episode, [slow_row()])
