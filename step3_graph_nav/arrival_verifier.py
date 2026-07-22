from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .model import Step3SplitModel
from .protocol import GraphNavProtocolError


ARRIVAL_PROTOCOL_VERSION = 2

ARRIVAL_SYSTEM_PROMPT = (
    "You are ArrivalVerifier, a binary classifier independent from NavigationPolicy. Decide whether the "
    "agent's CURRENT viewpoint already satisfies the instruction's final stopping destination. Current "
    "panorama images and recent real trajectory keyframes are supplied separately. Use the trajectory "
    "history to distinguish seeing a destination ahead from already reaching it. You never choose a "
    "candidate, direction, waypoint, or motion. Think briefly, then classify arrived true or false."
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GraphNavProtocolError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GraphNavProtocolError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise GraphNavProtocolError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ArrivalCurrentView:
    view_index: int
    relative_heading_deg: float
    jpeg: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if isinstance(self.view_index, bool) or not isinstance(self.view_index, int) or self.view_index < 0:
            raise GraphNavProtocolError("view_index must be a non-negative integer")
        heading = _finite(self.relative_heading_deg, "relative_heading_deg")
        if heading < -180.000001 or heading > 180.000001:
            raise GraphNavProtocolError("relative_heading_deg must be normalized")
        if not isinstance(self.jpeg, bytes) or not self.jpeg:
            raise GraphNavProtocolError("current-view JPEG must be non-empty bytes")
        if self.width <= 0 or self.height <= 0:
            raise GraphNavProtocolError("current-view dimensions must be positive")

    def metadata(self) -> dict[str, Any]:
        return {
            "view_index": self.view_index,
            "relative_heading_deg": self.relative_heading_deg,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class ArrivalHistoryFrame:
    history_index: int
    jpeg: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.history_index, bool)
            or not isinstance(self.history_index, int)
            or self.history_index < 0
        ):
            raise GraphNavProtocolError("history_index must be a non-negative integer")
        if not isinstance(self.jpeg, bytes) or not self.jpeg:
            raise GraphNavProtocolError("history JPEG must be non-empty bytes")
        if self.width <= 0 or self.height <= 0:
            raise GraphNavProtocolError("history dimensions must be positive")

    def metadata(self) -> dict[str, Any]:
        return {"history_index": self.history_index, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class ArrivalRequest:
    episode_id: str
    snapshot_id: str
    instruction: str
    step_index: int
    current_views: tuple[ArrivalCurrentView, ...]
    history_frames: tuple[ArrivalHistoryFrame, ...]
    action_history: tuple[str, ...]
    timestamp: float = field(default_factory=time.time)
    protocol_version: int = ARRIVAL_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != ARRIVAL_PROTOCOL_VERSION:
            raise GraphNavProtocolError("unsupported arrival protocol version")
        if not self.episode_id.strip() or not self.snapshot_id.strip() or not self.instruction.strip():
            raise GraphNavProtocolError("arrival episode, snapshot, and instruction are required")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise GraphNavProtocolError("arrival step_index must be a non-negative integer")
        if not self.current_views:
            raise GraphNavProtocolError("ArrivalVerifier requires a current panorama")
        if [item.view_index for item in self.current_views] != list(range(len(self.current_views))):
            raise GraphNavProtocolError("current view_index values must be zero-based ordinals")
        if not 2 <= len(self.history_frames) <= 4:
            raise GraphNavProtocolError("ArrivalVerifier requires 2 to 4 real history frames")
        if [item.history_index for item in self.history_frames] != list(range(len(self.history_frames))):
            raise GraphNavProtocolError("history_index values must be zero-based ordinals")
        if len(self.action_history) != len(self.history_frames):
            raise GraphNavProtocolError("action history must align with history frames")
        if any(not isinstance(item, str) or not item.strip() for item in self.action_history):
            raise GraphNavProtocolError("action history entries must be non-empty strings")
        _finite(self.timestamp, "timestamp")

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "instruction": self.instruction,
            "step_index": self.step_index,
            "current_views": [item.metadata() for item in self.current_views],
            "history_frames": [item.metadata() for item in self.history_frames],
            "action_history": list(self.action_history),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_wire(cls, metadata: Mapping[str, Any], image_parts: Sequence[bytes]) -> "ArrivalRequest":
        current_rows = metadata.get("current_views") or []
        history_rows = metadata.get("history_frames") or []
        if not isinstance(current_rows, list) or not isinstance(history_rows, list):
            raise GraphNavProtocolError("arrival frame metadata must be arrays")
        if len(image_parts) != len(current_rows) + len(history_rows):
            raise GraphNavProtocolError("arrival metadata/payload count mismatch")
        current = []
        for row, payload in zip(current_rows, image_parts[: len(current_rows)]):
            if not isinstance(row, Mapping):
                raise GraphNavProtocolError("current-view metadata must be objects")
            current.append(
                ArrivalCurrentView(
                    view_index=int(row.get("view_index", -1)),
                    relative_heading_deg=_finite(row.get("relative_heading_deg"), "relative_heading_deg"),
                    jpeg=bytes(payload),
                    width=int(row.get("width") or 0),
                    height=int(row.get("height") or 0),
                )
            )
        history = []
        for row, payload in zip(history_rows, image_parts[len(current_rows) :]):
            if not isinstance(row, Mapping):
                raise GraphNavProtocolError("history metadata must be objects")
            history.append(
                ArrivalHistoryFrame(
                    history_index=int(row.get("history_index", -1)),
                    jpeg=bytes(payload),
                    width=int(row.get("width") or 0),
                    height=int(row.get("height") or 0),
                )
            )
        return cls(
            episode_id=str(metadata.get("episode_id") or ""),
            snapshot_id=str(metadata.get("snapshot_id") or ""),
            instruction=str(metadata.get("instruction") or ""),
            step_index=int(metadata.get("step_index", -1)),
            current_views=tuple(current),
            history_frames=tuple(history),
            action_history=tuple(str(item) for item in (metadata.get("action_history") or [])),
            timestamp=_finite(metadata.get("timestamp"), "timestamp"),
            protocol_version=int(metadata.get("protocol_version", ARRIVAL_PROTOCOL_VERSION)),
        )


def parse_arrival(raw_text: str) -> bool:
    try:
        value = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise GraphNavProtocolError(f"arrival output is not one JSON object: {exc.msg}") from exc
    if not isinstance(value, dict) or set(value) != {"arrived"}:
        raise GraphNavProtocolError("arrival JSON must contain exactly arrived")
    if not isinstance(value["arrived"], bool):
        raise GraphNavProtocolError("arrived must be a JSON boolean")
    return bool(value["arrived"])


def build_arrival_prompt(request: ArrivalRequest) -> str:
    current_rows = "\n".join(
        f"- current_image_index={item.view_index}, relative_heading_deg={item.relative_heading_deg:.1f}"
        for item in request.current_views
    )
    history_offset = len(request.current_views)
    history_rows = "\n".join(
        f"- history_image_index={history_offset + item.history_index}, order=oldest_to_newest_{item.history_index}"
        for item in request.history_frames
    )
    actions = "\n".join(f"- {item}" for item in request.action_history)
    return f"""INSTRUCTION:
{request.instruction}

CURRENT STEP COUNT: {request.step_index}

CURRENT 360-DEGREE PANORAMA SLICES:
{current_rows}

RECENT REAL VIEWPOINT KEYFRAMES (oldest to newest):
{history_rows}

EXECUTED MOVE / TURN SUMMARY (oldest to newest):
{actions}

Decide whether the current position, not a future direction, already fulfills the final destination.
Briefly reason using visual change across history. The final response is strictly one of:
{{"arrived":true}}
{{"arrived":false}}

Do not choose or mention any candidate ID."""


class ArrivalVerifier:
    def __init__(self, model: Step3SplitModel) -> None:
        self.model = model

    def verify(self, request: ArrivalRequest, *, threshold: float) -> tuple[str, dict[str, Any]]:
        threshold = _finite(threshold, "arrival_threshold")
        started = time.perf_counter()
        payloads = [item.jpeg for item in request.current_views]
        payloads.extend(item.jpeg for item in request.history_frames)
        images, image_decode_ms = self.model.prepare_images(payloads)
        scores, metrics = self.model.score_branches(
            system_prompt=ARRIVAL_SYSTEM_PROMPT,
            user_prompt=build_arrival_prompt(request),
            images=images,
            assistant_prefix='{"arrived":',
            branch_texts={"true": "true", "false": "false"},
        )
        margin = scores["true"] - scores["false"]
        arrived = bool(margin >= threshold)
        raw_output = json.dumps({"arrived": arrived}, separators=(",", ":"))
        return raw_output, {
            **metrics,
            "image_decode_ms": image_decode_ms,
            "current_view_count": len(request.current_views),
            "history_frame_count": len(request.history_frames),
            "arrived_true_logit": scores["true"],
            "arrived_false_logit": scores["false"],
            "arrived_margin": margin,
            "arrival_threshold": threshold,
            "verifier_total_ms": (time.perf_counter() - started) * 1000.0,
            "output_schema": "arrived_boolean_only",
        }

