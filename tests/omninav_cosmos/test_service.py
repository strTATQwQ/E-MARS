from dataclasses import replace

from omninav_cosmos.backbones.base import BackboneAdapter
from omninav_cosmos.contracts import NavigationOutput, NavigationRequest
from omninav_cosmos.service import LatestResponseGate, NavigationInferenceService


class FakeAdapter(BackboneAdapter):
    model_variant = "fake"
    precision_mode = "bf16"

    def __init__(self):
        self.resets = []

    def reset_episode(self, episode_id):
        self.resets.append(episode_id)

    def infer(self, request):
        return NavigationOutput(
            episode_id=request.episode_id,
            frame_id=request.frame_id,
            waypoints=((0.5, 0.0),),
            heading_sin_cos=((0.0, 1.0),),
            arrive_or_stop=False,
            confidence=0.8,
            model_latency_ms=10.0,
        )


def request(episode="ep-a", frame=1, timestamp=100.0, reset=False):
    return NavigationRequest(
        episode_id=episode,
        frame_id=frame,
        timestamp=timestamp,
        instruction="go to the chair",
        rgb_front=b"image",
        reset_episode=reset,
    )


def test_service_rejects_stale_and_duplicate_frames_with_safe_stop():
    adapter = FakeAdapter()
    service = NavigationInferenceService(adapter, clock=lambda: 100.0)
    assert service.process(request(frame=1)).safe_stop_reason == ""
    duplicate = service.process(request(frame=1))
    assert duplicate.safe_stop_reason == "stale_or_duplicate_frame"
    assert duplicate.arrive_or_stop is True
    stale = service.process(request(frame=2, timestamp=90.0))
    assert stale.safe_stop_reason == "stale_request"


def test_episode_change_requires_reset_and_retired_episode_cannot_return():
    adapter = FakeAdapter()
    service = NavigationInferenceService(adapter, clock=lambda: 100.0)
    service.process(request("ep-a", 1))
    rejected = service.process(request("ep-b", 0))
    assert rejected.safe_stop_reason == "episode_reset_required"
    accepted = service.process(request("ep-b", 0, reset=True))
    assert accepted.safe_stop_reason == ""
    retired = service.process(request("ep-a", 2, reset=True))
    assert retired.safe_stop_reason == "retired_episode"
    assert adapter.resets == ["ep-a", "ep-b"]


def test_latest_response_gate_rejects_old_or_wrong_episode_results():
    gate = LatestResponseGate()
    gate.reset("ep-a")
    base = FakeAdapter().infer(request("ep-a", 4))
    assert gate.accept(base) is True
    assert gate.accept(replace(base, frame_id=3)) is False
    assert gate.accept(replace(base, frame_id=5, episode_id="ep-b")) is False

