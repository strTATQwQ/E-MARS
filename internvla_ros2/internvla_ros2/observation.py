"""Deterministic observation hashing shared by both ROS 2 roles."""

from __future__ import annotations

import hashlib

import numpy as np


def _framed_hash(digest: object, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "little"))
    digest.update(value)


def observation_digest(
    rgb: np.ndarray,
    depth: np.ndarray,
    instruction: str,
    tokens: list[int],
    gps: list[float],
    rotation: list[float],
) -> str:
    digest = hashlib.sha256()
    _framed_hash(digest, np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    _framed_hash(digest, np.ascontiguousarray(depth, dtype="<f4").tobytes())
    _framed_hash(digest, instruction.encode("utf-8"))
    _framed_hash(digest, np.asarray(tokens, dtype="<i8").tobytes())
    _framed_hash(digest, np.asarray(gps, dtype="<f8").tobytes())
    _framed_hash(digest, np.asarray(rotation, dtype="<f8").tobytes())
    return digest.hexdigest()


def compressed_observation_digest(
    rgb_jpeg: bytes,
    depth_png: bytes,
    instruction: str,
    tokens: list[int],
    gps: list[float],
    rotation: list[float],
) -> str:
    """Hash the exact cross-host payload and semantic observation metadata."""

    digest = hashlib.sha256()
    _framed_hash(digest, b"internnav-t5-compressed-observation-v1")
    _framed_hash(digest, bytes(rgb_jpeg))
    _framed_hash(digest, bytes(depth_png))
    _framed_hash(digest, instruction.encode("utf-8"))
    _framed_hash(digest, np.asarray(tokens, dtype="<i8").tobytes())
    _framed_hash(digest, np.asarray(gps, dtype="<f8").tobytes())
    _framed_hash(digest, np.asarray(rotation, dtype="<f8").tobytes())
    return digest.hexdigest()
