from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import types
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_ros2"))

from internvla_ros2.trajectory_capture import (  # noqa: E402
    CapturedTrajectoryCandidates,
    MetricTrajectoryCapture,
)
from internvla_ros2.trajectory_rerank import (  # noqa: E402
    TrajectoryRerankConfig,
    rerank_trajectories,
)


MODEL_NODE = ROOT / "internvla_ros2" / "internvla_ros2" / "model_node.py"


def _top_level_function(name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(MODEL_NODE.read_text(encoding="utf-8"))
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(MODEL_NODE), "exec"), namespace)
    return namespace[name]


def _candidates(count: int = 32) -> np.ndarray:
    values = np.zeros((count, 33, 2), dtype=np.float32)
    for index in range(count):
        values[index, :, 0] = np.linspace(0.0, 0.20 + index * 0.04, 33)
    return values


def test_exact_32_candidates_are_ranked_deterministically_without_oracle_inputs() -> None:
    depth = np.ones((480, 640, 1), dtype=np.float32)
    first = rerank_trajectories(_candidates(), depth)
    second = rerank_trajectories(_candidates(), depth)
    assert first.fallback_reason is None
    assert first.selected_index == second.selected_index == 31
    assert np.array_equal(first.selected_trajectory, _candidates()[31])
    assert first.audit_mapping()["oracle_or_ground_truth_inputs"] is False
    parameters = set(inspect.signature(rerank_trajectories).parameters)
    assert parameters == {"candidate_trajectories", "normalized_depth", "config"}


def test_non_32_upstream_tensor_falls_back_and_records_observed_shape() -> None:
    result = rerank_trajectories(
        _candidates(31), np.ones((480, 640, 1), dtype=np.float32)
    )
    assert result.selected_trajectory is None
    assert result.selected_index is None
    assert result.upstream_shape == (31, 33, 2)
    assert result.fallback_reason == "upstream_candidate_tensor_is_not_32_by_33_by_2"


@pytest.mark.parametrize(
    "raw_shape",
    [(32, 2, 2), (32, 32, 4), (32, 31, 3), (32, 33, 2)],
)
def test_malformed_raw_diffusion_shapes_fall_back_before_ranking(raw_shape) -> None:
    malformed = np.zeros(raw_shape, dtype=np.float32)
    result = rerank_trajectories(
        CapturedTrajectoryCandidates(raw_shape=raw_shape, trajectories=malformed),
        np.ones((480, 640, 1), dtype=np.float32),
    )
    assert result.selected_trajectory is None
    assert (
        result.fallback_reason
        == "upstream_raw_diffusion_tensor_is_not_32_by_32_by_3"
    )


def test_exact_raw_diffusion_shape_is_the_only_shape_derived_to_32_by_33(
    monkeypatch,
) -> None:
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        @property
        def shape(self):
            return self.value.shape

        def detach(self):
            return self

        def clone(self):
            return FakeTensor(self.value.copy())

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    def original(_tensor, use_discrate_action=True):
        if use_discrate_action:
            return [{"action": [1], "ideal_flag": True}]
        return np.zeros((33, 2), dtype=np.float32)

    modules = {
        "internnav": types.ModuleType("internnav"),
        "internnav.model": types.ModuleType("internnav.model"),
        "internnav.model.utils": types.ModuleType("internnav.model.utils"),
        "internnav.model.utils.vln_utils": types.ModuleType(
            "internnav.model.utils.vln_utils"
        ),
        "internnav.model.basemodel": types.ModuleType(
            "internnav.model.basemodel"
        ),
        "internnav.model.basemodel.internvla_n1": types.ModuleType(
            "internnav.model.basemodel.internvla_n1"
        ),
        "internnav.model.basemodel.internvla_n1.internvla_n1_policy": types.ModuleType(
            "internnav.model.basemodel.internvla_n1.internvla_n1_policy"
        ),
    }
    modules["internnav.model.utils"].vln_utils = modules[
        "internnav.model.utils.vln_utils"
    ]
    modules["internnav.model.basemodel.internvla_n1"].internvla_n1_policy = modules[
        "internnav.model.basemodel.internvla_n1.internvla_n1_policy"
    ]
    modules["internnav.model.utils.vln_utils"].traj_to_actions = original
    modules[
        "internnav.model.basemodel.internvla_n1.internvla_n1_policy"
    ].traj_to_actions = original
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    capture = MetricTrajectoryCapture(capture_candidates=True)
    capture.install()
    capture.begin_step()
    modules["internnav.model.utils.vln_utils"].traj_to_actions(
        FakeTensor(np.zeros((32, 32, 3), dtype=np.float32)), True
    )
    batch = capture.take_candidates()
    assert isinstance(batch, CapturedTrajectoryCandidates)
    assert batch.raw_shape == (32, 32, 3)
    assert batch.trajectories.shape == (32, 33, 2)


def test_invalid_depth_falls_back_without_selecting_a_candidate() -> None:
    depth = np.zeros((480, 640, 1), dtype=np.float32)
    result = rerank_trajectories(_candidates(), depth)
    assert result.selected_trajectory is None
    assert result.fallback_reason == "insufficient_valid_depth"


def test_default_capture_and_runtime_are_off_and_t5_scoped() -> None:
    capture = MetricTrajectoryCapture()
    capture.begin_step()
    assert capture.take_candidates() is None
    model_source = (
        ROOT / "internvla_ros2" / "internvla_ros2" / "model_node.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ.get("INTERNVLA_T5_TRAJECTORY_RERANK", "0")' in model_source
    assert "trajectory rerank is restricted to exact T5 Isaac completion_sim" in model_source


def test_rerank_seed_is_stable_per_identity_and_excludes_observation_digest() -> None:
    derive_seed = _top_level_function(
        "_trajectory_rerank_seed",
        {
            "RequestIdentity": object,
            "MODEL_REVISION": "model-revision",
            "CHECKPOINT_REVISION": "checkpoint-revision",
            "hashlib": hashlib,
            "json": json,
        },
    )
    identity = types.SimpleNamespace(
        episode_id="a::628",
        reset_generation=2,
        sequence_id=17,
        request_id="a::628:2:17",
    )
    same_identity = types.SimpleNamespace(
        episode_id="a::628",
        reset_generation=2,
        sequence_id=17,
        request_id="request-id-is-not-seed-material",
    )
    different_sequence = types.SimpleNamespace(
        episode_id="a::628", reset_generation=2, sequence_id=18
    )

    seed = derive_seed(identity)
    assert seed == derive_seed(same_identity)
    assert seed != derive_seed(different_sequence)
    assert 0 <= seed <= np.iinfo(np.uint32).max
    assert "observation" not in inspect.signature(derive_seed).parameters
    seed_source = inspect.getsource(derive_seed)
    assert "observation_digest" not in seed_source
    assert '"rgb"' not in seed_source
    assert '"depth"' not in seed_source


def test_rerank_seed_changes_with_every_seed_contract_identity_field() -> None:
    derive_seed = _top_level_function(
        "_trajectory_rerank_seed",
        {
            "RequestIdentity": object,
            "MODEL_REVISION": "model-revision",
            "CHECKPOINT_REVISION": "checkpoint-revision",
            "hashlib": hashlib,
            "json": json,
        },
    )
    identity = types.SimpleNamespace(
        episode_id="a::628", reset_generation=2, sequence_id=17
    )
    baseline = derive_seed(identity)
    assert baseline != derive_seed(
        types.SimpleNamespace(
            episode_id="a::259", reset_generation=2, sequence_id=17
        )
    )
    assert baseline != derive_seed(
        types.SimpleNamespace(
            episode_id="a::628", reset_generation=3, sequence_id=17
        )
    )
    assert baseline != derive_seed(
        identity, model_revision="different-model-revision"
    )
    assert baseline != derive_seed(
        identity, checkpoint_revision="different-checkpoint-revision"
    )


def test_rerank_seeds_python_numpy_and_torch_cpu_cuda(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda seed: calls.append(("torch_cpu", seed))
    fake_torch.cuda = types.SimpleNamespace(
        manual_seed_all=lambda seed: calls.append(("torch_cuda", seed)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(random, "seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: calls.append(("numpy", seed)))
    seed_rngs = _top_level_function(
        "_seed_trajectory_rerank_rngs", {"random": random, "np": np}
    )

    seed_rngs(1234)

    assert calls == [
        ("python", 1234),
        ("numpy", 1234),
        ("torch_cpu", 1234),
        ("torch_cuda", 1234),
    ]


def test_rerank_rng_seed_failure_is_fail_closed(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda _seed: None
    fake_torch.cuda = types.SimpleNamespace(
        manual_seed_all=lambda _seed: (_ for _ in ()).throw(
            RuntimeError("cuda seed failed")
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    seed_rngs = _top_level_function(
        "_seed_trajectory_rerank_rngs", {"random": random, "np": np}
    )

    with pytest.raises(RuntimeError, match="cuda seed failed"):
        seed_rngs(1234)


def test_rerank_seed_is_fail_closed_and_applied_before_real_agent_step() -> None:
    source = MODEL_NODE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InternVLAModelNode"
    )
    execute_step = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_step"
    )
    calls = [
        (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id, node.lineno)
        for node in ast.walk(execute_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "step")
            or (
                isinstance(node.func, ast.Name)
                and node.func.id == "_seed_trajectory_rerank_rngs"
            )
        )
    ]
    seed_line = next(line for name, line in calls if name == "_seed_trajectory_rerank_rngs")
    step_line = next(line for name, line in calls if name == "step")
    assert seed_line < step_line
    scope = "\n".join(source.splitlines()[seed_line - 5 : seed_line])
    assert 'if self.backend == "real":' in scope
    assert "if self._trajectory_rerank_enabled:" in scope
    assert "try:" not in "\n".join(
        source.splitlines()[seed_line - 1 : step_line - 1]
    )
    assert "torch.use_deterministic_algorithms" not in source
    assert '"seed": int(seed)' in source
    assert '"seed_contract": _TRAJECTORY_RERANK_SEED_CONTRACT' in source


def test_candidate_family_registers_mean_rerank_and_horizon_refresh_expiry() -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "internnav_t5"
            / "lane_a_candidates"
            / "trajectory_horizon_refresh.json"
        ).read_text(encoding="utf-8")
    )
    assert config["rerank_contract"]["candidate_count"] == 32
    variants = config["variants"]
    assert [item["runtime_overrides"]["INTERNVLA_T5_TRAJECTORY_RERANK"] for item in variants] == [
        "0",
        "1",
        "1",
    ]
    for item in variants:
        overrides = item["runtime_overrides"]
        assert "INTERNVLA_T4_PROGRESS_HORIZON_SEC" in overrides
        assert "INTERNVLA_T4_REFRESH_DISTANCE_M" in overrides
        assert "INTERNVLA_T4_REFRESH_TIME_SEC" in overrides
        assert "INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC" in overrides
