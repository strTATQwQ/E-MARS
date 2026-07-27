from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from slow_planner.base import (
    CandidateFrontier,
    OrderedImage,
    PlannerDecision,
    PlannerMetrics,
    PlannerRunner,
    SlowPlannerRequest,
    StructuredPlannerDecision,
)
from slow_planner.hf_base import (
    GenerationThreadTimeout,
    _contains_complete_decision_json,
    instrumented_generate,
)
from slow_planner.serve import SlowPlannerServer
from slow_planner.step3_vl_10b import (
    STEP3_CHAT_GENERATION_SUFFIX,
    STEP3_NO_THINKING_PREFILL,
    Step3VLSlowPlanner,
    _interleaved_labeled_images,
)


ROOT = Path(__file__).resolve().parents[1]


class _Processor:
    image_processor = None
    tokenizer = object()

    def __init__(self, suffix: str = STEP3_CHAT_GENERATION_SUFFIX) -> None:
        self.suffix = suffix
        self.inputs: dict[str, object] | None = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert messages[-1]["role"] == "user"
        return "templated-prefix" + self.suffix

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return {"input_ids": [[1, 2, 3]]}


def _planner(processor: _Processor) -> Step3VLSlowPlanner:
    planner = Step3VLSlowPlanner(
        model=SimpleNamespace(config=SimpleNamespace(model_type="step_robotics")),
        processor=processor,
        device="cpu",
        max_new_tokens=96,
        processor_use_fast=False,
        stop_on_complete_json=True,
        generation_wall_budget_s=10.5,
        generation_join_grace_s=0.5,
    )
    planner.skip_private_reasoning = True
    return planner


def test_step3_closes_forced_thinking_block_before_json_generation() -> None:
    processor = _Processor()
    planner = _planner(processor)
    planner.prepare_inputs(images=[object()] * 4, prompt="JSON only")
    assert processor.inputs is not None
    text = processor.inputs["text"][0]
    assert text.endswith(STEP3_CHAT_GENERATION_SUFFIX + STEP3_NO_THINKING_PREFILL)
    assert text.endswith("<think>\n</think>\n")
    health = planner.health()
    assert health["skip_private_reasoning"] is True
    assert health["stop_on_complete_json"] is True
    assert health["max_new_tokens"] == 96
    assert health["generation_wall_budget_s"] == 10.5
    assert health["generation_join_grace_s"] == 0.5


def test_step3_interleaves_each_image_with_its_exact_view_label() -> None:
    images = [object()] * 4
    prompt = (
        'INPUT={"views":[{"i":0,"id":"front_left"},'
        '{"i":1,"id":"front"},{"i":2,"id":"front_right"},'
        '{"i":3,"id":"rear"}],"history":'
        '["termination_candidate=periodic_arrival_probe"]}\nRules: JSON only'
    )

    content = _interleaved_labeled_images(images, prompt)

    assert [item["type"] for item in content] == [
        "text",
        "image",
        "text",
        "image",
        "text",
        "image",
        "text",
        "image",
    ]
    assert [content[index]["text"] for index in range(0, 8, 2)] == [
        "IMAGE_INDEX=0; VIEW_ID=front_left\n",
        "IMAGE_INDEX=1; VIEW_ID=front\n",
        "IMAGE_INDEX=2; VIEW_ID=front_right\n",
        "IMAGE_INDEX=3; VIEW_ID=rear\n",
    ]
    assert [content[index]["image"] for index in range(1, 8, 2)] == images


def test_step3_keeps_frozen_image_only_layout_for_navigation_requests() -> None:
    images = [object()] * 4
    prompt = (
        'INPUT={"views":[{"i":0,"id":"front_left"},'
        '{"i":1,"id":"front"},{"i":2,"id":"front_right"},'
        '{"i":3,"id":"rear"}],"history":["task_state_v1={}"]}'
    )

    content = _interleaved_labeled_images(images, prompt)

    assert content == [{"type": "image", "image": image} for image in images]


def test_step3_refuses_unknown_chat_template_instead_of_leaking_reasoning() -> None:
    planner = _planner(_Processor(suffix="<assistant>"))
    with pytest.raises(RuntimeError, match="unexpected Step3 chat-template"):
        planner.prepare_inputs(images=[object()] * 4, prompt="JSON only")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"decision":"select_frontier","frontier_id":1,'
            '"target_relative_xz":null,"confidence":0.8}',
            True,
        ),
        (
            'prefix {"decision":"abstain","frontier_id":null,'
            '"target_relative_xz":null,"confidence":0.0} trailing',
            False,
        ),
        (
            '{"decision":"abstain","frontier_id":null,'
            '"target_relative_xz":null,"confidence":0.0',
            False,
        ),
        (
            '{"decision":"abstain","frontier_id":null,'
            '"target_relative_xz":null,"confidence":0.0} trailing',
            False,
        ),
        (
            '{"decision":"abstain","frontier_id":null,"frontier_id":null,'
            '"target_relative_xz":null,"confidence":0.0}',
            False,
        ),
        (
            '{"decision":"abstain","frontier_id":null,'
            '"target_relative_xz":null,"confidence":0.0,"reasoning":"private"}',
            False,
        ),
    ],
)
def test_complete_json_stop_requires_exact_frozen_decision_shape(
    text: str, expected: bool
) -> None:
    assert _contains_complete_decision_json(text) is expected


def test_step3_deadline_config_uses_one_bounded_deterministic_attempt() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/slow_models/step3_vl_10b_bf16.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["max_new_tokens"] == 96
    assert config["max_retries"] == 0
    assert config["deterministic_decoding"] is True
    assert config["skip_private_reasoning"] is True
    assert config["stop_on_complete_json"] is True
    assert config["generation_wall_budget_s"] == 10.5
    assert config["generation_join_grace_s"] == 0.5
    assert config["redact_raw_text"] is True


def _request() -> SlowPlannerRequest:
    return SlowPlannerRequest(
        episode_id="episode-1",
        snapshot_id="b::episode-1::0::1",
        instruction="Reach the doorway.",
        ordered_images=(OrderedImage("front", (0.0, 0.0, 0.0), b"jpeg", 640, 480),),
        candidate_frontiers=(CandidateFrontier(7, (0.2, 1.0), 1.02, 11.3),),
        agent_pose=(0.0, 0.0, 0.0),
        timestamp=time.time(),
    )


def test_step3_structured_parser_accepts_inert_wrapper_and_rejects_prose() -> None:
    planner = _planner(_Processor())
    structured = (
        '{"scene_summary":"open doorway ahead","target_evidence":[],'
        '"blocked_directions":["rear"],"recommended_frontier":7,'
        '"confidence":0.75,"target_found":false,"abstain":false}'
    )
    parsed = planner.parse_decision(
        _request(), "</think>\n```json\n" + structured + "\n```", attempts=1
    )
    assert isinstance(parsed, StructuredPlannerDecision)
    assert parsed.decision == "select_frontier"
    assert parsed.frontier_id == 7
    assert parsed.scene_summary == "open doorway ahead"
    assert parsed.blocked_directions == ("rear",)

    planner.generate_raw = lambda request, correction="": (  # type: ignore[method-assign]
        "analysis " + structured,
        PlannerMetrics(model_variant=planner.model_variant),
    )
    fallback, _ = PlannerRunner(planner, max_retries=0).decide(_request())
    assert fallback.fallback_used is True
    assert fallback.fallback_reason.startswith("schema_failure:")


def test_step3_target_found_is_a_semantic_abstain_without_stop_or_coordinates() -> None:
    raw = (
        '{"scene_summary":"destination room reached",'
        '"target_evidence":["matching white cabinet visible"],'
        '"blocked_directions":[],"recommended_frontier":null,'
        '"confidence":0.9,"target_found":true,"abstain":true}'
    )
    parsed = _planner(_Processor()).parse_decision(_request(), raw, attempts=1)
    assert parsed.decision == "abstain"
    assert parsed.frontier_id is None
    assert parsed.target_relative_xz is None
    assert parsed.target_found is True
    assert parsed.abstain is True


class _FakeRow:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __getitem__(self, value):
        if isinstance(value, slice):
            return _FakeRow(self.values[value])
        return self.values[value]

    def numel(self) -> int:
        return len(self.values)


class _FakeMatrix:
    device = "cpu"

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __iter__(self):
        return iter(_FakeRow(row) for row in self.rows)

    def __getitem__(self, value):
        row, column = value
        return _FakeRow(self.rows[row][column])


class _FakeInputs(dict):
    def to(self, device: str):
        assert device == "cpu"
        return self


class _FakeTokenizer:
    pieces = {
        10: '{"decision":"target_found",',
        11: '"frontier_id":null,',
        12: '"target_relative_xz":[0.2,1.0],',
        13: '"confidence":0.75}',
        14: "must-not-generate",
    }

    def decode(self, values: Any, **kwargs: Any) -> str:
        del kwargs
        raw = values.values if isinstance(values, _FakeRow) else list(values)
        return "".join(self.pieces.get(value, "") for value in raw)


class _FakeStreamer:
    def __init__(self, tokenizer: Any, **kwargs: Any) -> None:
        del tokenizer
        self.skip_prompt = bool(kwargs["skip_prompt"])
        self.next_tokens_are_prompt = True

    def put(self, value: Any) -> None:
        del value
        self.next_tokens_are_prompt = False

    def end(self) -> None:
        return None


class _FakeStoppingCriteria:
    pass


class _FakeStoppingCriteriaList(list):
    pass


class _FakeTorch:
    bool = bool
    cuda = SimpleNamespace(is_available=lambda: False)

    @staticmethod
    def tensor(values, **kwargs):
        del kwargs
        return list(values)

    @staticmethod
    def full(shape, value, **kwargs):
        del kwargs
        return [value] * shape[0]


def _install_fake_generation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            StoppingCriteria=_FakeStoppingCriteria,
            StoppingCriteriaList=_FakeStoppingCriteriaList,
            TextIteratorStreamer=_FakeStreamer,
        ),
    )


def test_48_token_direct_generation_stops_on_first_complete_json(monkeypatch) -> None:
    _install_fake_generation_runtime(monkeypatch)

    class Model:
        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 48
            generated: list[int] = []
            kwargs["streamer"].put(kwargs["input_ids"])
            for token in (10, 11, 12, 13, 14):
                generated.append(token)
                accumulated = _FakeMatrix([[1, 2, *generated]])
                kwargs["streamer"].put(_FakeMatrix([[token]]))
                if any(criteria(accumulated, None)[0] for criteria in kwargs["stopping_criteria"]):
                    break
            kwargs["streamer"].end()
            return SimpleNamespace(sequences=_FakeMatrix([[1, 2, *generated]]))

    raw, metrics = instrumented_generate(
        model=Model(),
        tokenizer=_FakeTokenizer(),
        inputs=_FakeInputs(input_ids=_FakeMatrix([[1, 2]])),
        device="cpu",
        max_new_tokens=48,
        stop_on_complete_json=True,
        generation_wall_budget_s=10.5,
        generation_join_grace_s=0.5,
    )
    assert raw.endswith('"confidence":0.75}')
    assert "must-not-generate" not in raw
    assert metrics["output_token_count"] == 4


def test_generation_thread_timeout_is_fatal_and_bounded(monkeypatch) -> None:
    _install_fake_generation_runtime(monkeypatch)

    class HungModel:
        def generate(self, **kwargs):
            del kwargs
            time.sleep(0.08)

    started = time.perf_counter()
    with pytest.raises(GenerationThreadTimeout, match="service restart required"):
        instrumented_generate(
            model=HungModel(),
            tokenizer=_FakeTokenizer(),
            inputs=_FakeInputs(input_ids=_FakeMatrix([[1, 2]])),
            device="cpu",
            max_new_tokens=48,
            stop_on_complete_json=True,
            generation_wall_budget_s=0.01,
            generation_join_grace_s=0.01,
        )
    assert time.perf_counter() - started < 0.06


def test_generation_wall_criterion_stops_cooperative_model(monkeypatch) -> None:
    _install_fake_generation_runtime(monkeypatch)

    class SlowModel:
        def generate(self, **kwargs):
            generated: list[int] = []
            kwargs["streamer"].put(kwargs["input_ids"])
            for _ in range(kwargs["max_new_tokens"]):
                time.sleep(0.004)
                generated.append(14)
                accumulated = _FakeMatrix([[1, 2, *generated]])
                kwargs["streamer"].put(_FakeMatrix([[14]]))
                if any(criteria(accumulated, None)[0] for criteria in kwargs["stopping_criteria"]):
                    break
            kwargs["streamer"].end()
            return SimpleNamespace(sequences=_FakeMatrix([[1, 2, *generated]]))

    _, metrics = instrumented_generate(
        model=SlowModel(),
        tokenizer=_FakeTokenizer(),
        inputs=_FakeInputs(input_ids=_FakeMatrix([[1, 2]])),
        device="cpu",
        max_new_tokens=48,
        stop_on_complete_json=True,
        generation_wall_budget_s=0.01,
        generation_join_grace_s=0.02,
    )
    assert 1 <= metrics["output_token_count"] < 48
    assert metrics["model_generate_ms"] < 50.0


def _decision(raw_text: str = "<think>private chain</think>") -> PlannerDecision:
    return PlannerDecision(
        episode_id="episode-1",
        snapshot_id="b::episode-1::0::1",
        decision="select_frontier",
        frontier_id=7,
        target_relative_xz=None,
        confidence=0.7,
        raw_text=raw_text,
    )


def test_step3_service_response_and_jsonl_redact_raw_generation(tmp_path) -> None:
    request = _request()

    class Socket:
        def __init__(self) -> None:
            self.received = False
            self.response: bytes | None = None

        def recv_multipart(self):
            if self.received:
                raise StopIteration
            self.received = True
            envelope = {"type": "decide", "request": request.metadata()}
            return [json.dumps(envelope).encode(), request.ordered_images[0].jpeg]

        def send(self, value: bytes) -> None:
            self.response = value

    server = object.__new__(SlowPlannerServer)
    server.socket = Socket()
    server.runner = SimpleNamespace(
        decide=lambda value: (_decision(), PlannerMetrics(model_variant="step3_vl_10b_bf16"))
    )
    server.planner = SimpleNamespace(model_variant="step3_vl_10b_bf16", precision_mode="bf16")
    server.max_request_age_s = 5.0
    server.redact_raw_text = True
    server.log_path = tmp_path / "service.jsonl"
    with pytest.raises(StopIteration):
        server.run()
    assert server.socket.response is not None
    response = server.socket.response.decode()
    log = server.log_path.read_text(encoding="utf-8")
    for serialized in (response, log):
        assert "raw_text" not in serialized
        assert "private chain" not in serialized
        assert "<think>" not in serialized


def test_fatal_generation_timeout_does_not_accept_queued_request() -> None:
    request = _request()

    class Socket:
        recv_count = 0
        send_count = 0

        def recv_multipart(self):
            self.recv_count += 1
            envelope = {"type": "decide", "request": request.metadata()}
            return [json.dumps(envelope).encode(), request.ordered_images[0].jpeg]

        def send(self, value: bytes) -> None:
            del value
            self.send_count += 1

        def send_json(self, value: dict[str, Any]) -> None:
            del value
            self.send_count += 1

    server = object.__new__(SlowPlannerServer)
    server.socket = Socket()
    server.runner = SimpleNamespace(
        decide=lambda value: (_ for _ in ()).throw(GenerationThreadTimeout("fatal"))
    )
    server.planner = SimpleNamespace(model_variant="step3_vl_10b_bf16", precision_mode="bf16")
    server.max_request_age_s = 5.0
    server.redact_raw_text = True
    with pytest.raises(GenerationThreadTimeout, match="fatal"):
        server.run()
    assert server.socket.recv_count == 1
    assert server.socket.send_count == 0
