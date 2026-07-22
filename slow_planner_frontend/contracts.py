"""Frozen, dependency-free contracts consumed by the operator panel.

This module intentionally contains only the small public projection shared
with the InternNav runtime.  The model, planner, Nav2 adapter and motion stack
remain outside this repository.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


LANE_ID = "b"
REV_C_VIEW_ORDER = ("front_left", "front", "front_right", "rear")
REV_C_IMAGE_SIZE = (640, 480)
SNAPSHOT_ID_TEMPLATE = "b::<episode>::<reset>::<sequence>"

PLANNER_METRIC_FIELDS = frozenset(
    {
        "image_count",
        "raw_image_resolutions",
        "visual_token_count",
        "input_token_count",
        "output_token_count",
        "image_decode_ms",
        "prompt_template_ms",
        "processor_ms",
        "preprocessing_ms",
        "prefill_ttft_ms",
        "decode_ms",
        "decode_tokens_per_s",
        "model_generate_ms",
        "end_to_end_ms",
        "peak_memory_mib",
        "network_ms",
        "retry_count",
        "model_variant",
        "precision_mode",
    }
)

_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_SNAPSHOT_RE = re.compile(
    rf"^b::(?P<episode>{_IDENTIFIER})::(?P<reset>0|[1-9][0-9]*)::"
    rf"(?P<sequence>0|[1-9][0-9]*)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FrontendContractError(ValueError):
    """Raised when a runtime artifact violates the frozen panel contract."""


@dataclass(frozen=True)
class LaneBSnapshotIdentity:
    episode_id: str
    reset_id: int
    sequence_id: int

    @property
    def snapshot_id(self) -> str:
        return f"{LANE_ID}::{self.episode_id}::{self.reset_id}::{self.sequence_id}"


def parse_lane_b_snapshot_id(snapshot_id: str) -> LaneBSnapshotIdentity:
    match = _SNAPSHOT_RE.fullmatch(str(snapshot_id))
    if match is None:
        raise FrontendContractError(
            f"snapshot_id must match {SNAPSHOT_ID_TEMPLATE!r}; received {snapshot_id!r}"
        )
    return LaneBSnapshotIdentity(
        episode_id=match.group("episode"),
        reset_id=int(match.group("reset")),
        sequence_id=int(match.group("sequence")),
    )


def _sha256(value: Any, name: str) -> str:
    raw = str(value or "")
    if not _SHA256_RE.fullmatch(raw):
        raise FrontendContractError(f"{name} must be a lowercase SHA-256 digest")
    return raw


def snapshot_content_sha256(record: Mapping[str, Any]) -> str:
    """Hash the immutable image/config/extrinsic identity of one snapshot."""

    snapshot_id = str(record.get("snapshot_id") or "")
    parse_lane_b_snapshot_id(snapshot_id)
    config_sha256 = _sha256(record.get("config_sha256"), "config_sha256")
    rows = record.get("images")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise FrontendContractError("snapshot images must be an array")
    images = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise FrontendContractError("snapshot image entries must be objects")
        images.append(
            {
                "view_id": str(row.get("view_id") or ""),
                "jpeg_sha256": _sha256(
                    row.get("jpeg_sha256"), "image.jpeg_sha256"
                ),
                "extrinsic_sha256": _sha256(
                    row.get("extrinsic_sha256"), "image.extrinsic_sha256"
                ),
            }
        )
    if tuple(row["view_id"] for row in images) != REV_C_VIEW_ORDER:
        raise FrontendContractError(
            "snapshot content hash requires the fixed Rev-C view order"
        )
    canonical = json.dumps(
        {
            "snapshot_id": snapshot_id,
            "config_sha256": config_sha256,
            "images": images,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class ValidatedRevCSnapshot(Protocol):
    """Structural type accepted from an integration-owned snapshot producer."""

    identity: LaneBSnapshotIdentity
    frames: Sequence[Any]
    frame_ages_s: Sequence[float]

    def sidecar_record(self) -> dict[str, Any]: ...
