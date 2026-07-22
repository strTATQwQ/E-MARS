from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_internnav_go2_entrypoint import (  # noqa: E402
    _install_t5_pre_warmup_reset_order,
    _registered_vln_distributed_evaluator,
)


class _Agent:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def reset(self, env_ids: list[int]) -> None:
        self.events.append(("agent_reset", list(env_ids)))


class _Env:
    def __init__(
        self, events: list[tuple[str, object]], new_infos: list[object]
    ) -> None:
        self.events = events
        self.new_infos = new_infos

    def reset(self, env_ids: list[int]) -> tuple[list[str], list[object]]:
        self.events.append(("env_reset", list(env_ids)))
        return ["observation"], list(self.new_infos)


def _evaluator_class():
    class Evaluator:
        def terminate_ops(self, *_args):
            if self.phase == "new_episode":
                return self.env.reset([0])
            if self.phase == "warmup_finished":
                self.agent.reset([0])
                return "normal"
            raise AssertionError(self.phase)

    return Evaluator


def test_t5_reset_runs_after_env_reset_and_before_warmup_duplicate() -> None:
    events: list[tuple[str, object]] = []
    evaluator_class = _evaluator_class()
    _install_t5_pre_warmup_reset_order(evaluator_class)
    evaluator = evaluator_class()
    evaluator.agent = _Agent(events)
    evaluator.env = _Env(events, [SimpleNamespace()])

    evaluator.phase = "new_episode"
    evaluator.terminate_ops(None, None, None)
    assert events == [("env_reset", [0]), ("agent_reset", [0])]
    assert evaluator._internnav_t5_pre_warmup_reset_ids == {0}

    evaluator.phase = "warmup_finished"
    evaluator.terminate_ops(None, None, None)
    assert events == [("env_reset", [0]), ("agent_reset", [0])]
    assert evaluator._internnav_t5_pre_warmup_reset_ids == set()


def test_t5_final_env_reset_does_not_advance_agent_identity() -> None:
    events: list[tuple[str, object]] = []
    evaluator_class = _evaluator_class()
    _install_t5_pre_warmup_reset_order(evaluator_class)
    evaluator = evaluator_class()
    evaluator.agent = _Agent(events)
    evaluator.env = _Env(events, [])
    evaluator.phase = "new_episode"

    evaluator.terminate_ops(None, None, None)

    assert events == [("env_reset", [0])]
    assert evaluator._internnav_t5_pre_warmup_reset_ids == set()


def test_t5_failed_agent_reset_aborts_before_warmup() -> None:
    events: list[tuple[str, object]] = []
    evaluator_class = _evaluator_class()
    _install_t5_pre_warmup_reset_order(evaluator_class)
    evaluator = evaluator_class()
    evaluator.agent = _Agent(events)
    evaluator.env = _Env(events, [SimpleNamespace()])
    evaluator.phase = "new_episode"

    def fail_reset(env_ids: list[int]) -> None:
        events.append(("agent_reset_failed", list(env_ids)))
        raise RuntimeError("reset failed")

    evaluator.agent.reset = fail_reset
    with pytest.raises(RuntimeError, match="reset failed"):
        evaluator.terminate_ops(None, None, None)

    assert events == [
        ("env_reset", [0]),
        ("agent_reset_failed", [0]),
    ]


def test_t5_reset_order_patch_is_idempotent() -> None:
    evaluator_class = _evaluator_class()
    original = evaluator_class.terminate_ops

    _install_t5_pre_warmup_reset_order(evaluator_class)
    wrapped = evaluator_class.terminate_ops
    _install_t5_pre_warmup_reset_order(evaluator_class)

    assert wrapped is evaluator_class.terminate_ops
    assert wrapped is not original


def test_registered_evaluator_uses_registry_not_decorated_module_symbol() -> None:
    evaluator_class = _evaluator_class()
    evaluator_base = SimpleNamespace(
        evaluators={"vln_distributed": evaluator_class}
    )

    assert _registered_vln_distributed_evaluator(evaluator_base) is evaluator_class
