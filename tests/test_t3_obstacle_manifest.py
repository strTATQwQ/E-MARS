import hashlib
import importlib.util
from pathlib import Path


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_t3_obstacle_dataset.py"
    spec = importlib.util.spec_from_file_location("build_t3_obstacle_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_instruction_digest_matches_vln_task_observation():
    builder = _load_builder()
    instruction = {
        "instruction_text": "turn left",
        "instruction_tokens": [10, 20, 30],
    }

    runtime_text = builder.evaluator_instruction_string(instruction)

    assert runtime_text == "turn left"
    assert hashlib.sha256(runtime_text.encode("utf-8")).hexdigest() == hashlib.sha256(
        instruction["instruction_text"].encode("utf-8")
    ).hexdigest()
    assert "10" not in runtime_text


def test_runtime_instruction_digest_accepts_already_flattened_text():
    builder = _load_builder()
    assert builder.evaluator_instruction_string("go forward") == "go forward"
