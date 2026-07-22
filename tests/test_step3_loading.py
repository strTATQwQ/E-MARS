from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from slow_planner.step3_vl_10b import (
    CHECKPOINT_KEY_MAPPING,
    CHECKPOINT_KEY_MAPPING_NAME,
    Step3VLSlowPlanner,
)


class _Parameter:
    dtype = "torch.bfloat16"

    def numel(self) -> int:
        return 7

    def element_size(self) -> int:
        return 2


class _Model:
    config = SimpleNamespace(model_type="step_robotics", image_token_id=1)

    def parameters(self):
        return [_Parameter()]

    def eval(self) -> None:
        self.was_evaled = True


class _Processor:
    tokenizer = object()
    image_processor = None


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, loading_info: dict) -> dict:
    calls: dict = {}

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["processor"] = (args, kwargs)
            return _Processor()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["model"] = (args, kwargs)
            return _Model(), loading_info

    fake_transformers = SimpleNamespace(
        __version__="4.57.6",
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoProcessor=AutoProcessor,
    )
    fake_torch = SimpleNamespace(bfloat16="torch.bfloat16")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls


def test_step3_loader_explicitly_maps_checkpoint_and_records_clean_load(monkeypatch, tmp_path):
    calls = _install_fake_runtime(
        monkeypatch,
        {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
    )
    planner = Step3VLSlowPlanner.from_pretrained(tmp_path)
    kwargs = calls["model"][1]
    assert kwargs["key_mapping"] == CHECKPOINT_KEY_MAPPING
    assert kwargs["output_loading_info"] is True
    health = planner.health()
    assert health["checkpoint_key_mapping"] == CHECKPOINT_KEY_MAPPING_NAME
    assert health["checkpoint_load_clean"] is True
    assert health["checkpoint_loading_counts"] == {
        "missing_keys": 0,
        "unexpected_keys": 0,
        "mismatched_keys": 0,
        "error_msgs": 0,
    }
    assert health["parameter_dtype_counts"] == {"torch.bfloat16": 7}


def test_step3_loader_rejects_missing_checkpoint_weights(monkeypatch, tmp_path):
    _install_fake_runtime(
        monkeypatch,
        {"missing_keys": ["model.language_model.layers.0.weight"], "unexpected_keys": []},
    )
    with pytest.raises(RuntimeError, match="checkpoint load was not clean"):
        Step3VLSlowPlanner.from_pretrained(tmp_path)
