"""Bounded T5 observation codecs shared by DGX_EDGE and DGX_MODEL.

RGB uses JPEG because it dominates the edge-to-model link.  Normalized depth
uses lossless 16-bit PNG after a deterministic [0, 1] -> [0, 65535]
quantization.  The encoded bytes, rather than a second local decode, are
covered by the observation digest so decoder/library differences cannot create
false identity collisions across hosts.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


OBSERVATION_WIDTH = 640
OBSERVATION_HEIGHT = 480
DEFAULT_JPEG_QUALITY = 85
DEFAULT_PNG_COMPRESSION = 3
DEFAULT_MAX_RGB_BYTES = 512 * 1024
DEFAULT_MAX_DEPTH_BYTES = 768 * 1024


class ObservationCodecError(ValueError):
    """Raised before an invalid or over-bound observation reaches the model."""


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(rgb, dtype=np.uint8)
    if value.shape != (OBSERVATION_HEIGHT, OBSERVATION_WIDTH, 3):
        raise ObservationCodecError(
            f"RGB must be {OBSERVATION_WIDTH}x{OBSERVATION_HEIGHT}x3"
        )
    return value


def _validate_depth(depth: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(depth, dtype=np.float32)
    if value.shape != (OBSERVATION_HEIGHT, OBSERVATION_WIDTH, 1):
        raise ObservationCodecError(
            f"depth must be {OBSERVATION_WIDTH}x{OBSERVATION_HEIGHT}x1"
        )
    if not np.isfinite(value).all():
        raise ObservationCodecError("depth contains NaN/Inf")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ObservationCodecError("depth is outside normalized range [0,1]")
    return value


def _bounded(payload: bytes, maximum: int, label: str) -> bytes:
    value = bytes(payload)
    if not value:
        raise ObservationCodecError(f"{label} payload is empty")
    if maximum < 1 or len(value) > maximum:
        raise ObservationCodecError(
            f"{label} payload exceeds bound: {len(value)} > {maximum} bytes"
        )
    return value


def encode_rgb_jpeg(
    rgb: np.ndarray,
    *,
    quality: int = DEFAULT_JPEG_QUALITY,
    maximum_bytes: int = DEFAULT_MAX_RGB_BYTES,
) -> bytes:
    if quality < 1 or quality > 95:
        raise ObservationCodecError("JPEG quality must be in [1,95]")
    stream = BytesIO()
    Image.fromarray(_validate_rgb(rgb), mode="RGB").save(
        stream,
        format="JPEG",
        quality=int(quality),
        optimize=False,
        progressive=False,
        subsampling=2,
    )
    return _bounded(stream.getvalue(), maximum_bytes, "RGB JPEG")


def decode_rgb_jpeg(
    payload: bytes, *, maximum_bytes: int = DEFAULT_MAX_RGB_BYTES
) -> np.ndarray:
    value = _bounded(payload, maximum_bytes, "RGB JPEG")
    try:
        with Image.open(BytesIO(value)) as image:
            if image.format != "JPEG":
                raise ObservationCodecError("RGB payload is not JPEG")
            if image.size != (OBSERVATION_WIDTH, OBSERVATION_HEIGHT):
                raise ObservationCodecError(
                    f"RGB JPEG must be {OBSERVATION_WIDTH}x{OBSERVATION_HEIGHT}"
                )
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ObservationCodecError(f"invalid RGB JPEG: {exc}") from exc
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def encode_depth_png(
    depth: np.ndarray,
    *,
    compression: int = DEFAULT_PNG_COMPRESSION,
    maximum_bytes: int = DEFAULT_MAX_DEPTH_BYTES,
) -> bytes:
    if compression < 0 or compression > 9:
        raise ObservationCodecError("PNG compression must be in [0,9]")
    normalized = _validate_depth(depth)[:, :, 0]
    quantized = np.rint(normalized * 65535.0).astype(np.uint16)
    stream = BytesIO()
    Image.fromarray(quantized).save(
        stream,
        format="PNG",
        compress_level=int(compression),
        optimize=False,
    )
    return _bounded(stream.getvalue(), maximum_bytes, "depth PNG")


def decode_depth_png(
    payload: bytes, *, maximum_bytes: int = DEFAULT_MAX_DEPTH_BYTES
) -> np.ndarray:
    value = _bounded(payload, maximum_bytes, "depth PNG")
    try:
        with Image.open(BytesIO(value)) as image:
            if image.format != "PNG":
                raise ObservationCodecError("depth payload is not PNG")
            if image.size != (OBSERVATION_WIDTH, OBSERVATION_HEIGHT):
                raise ObservationCodecError(
                    f"depth PNG must be {OBSERVATION_WIDTH}x{OBSERVATION_HEIGHT}"
                )
            if image.mode not in {"I;16", "I;16L", "I;16B", "I"}:
                raise ObservationCodecError(
                    f"depth PNG must contain one 16-bit channel, observed {image.mode!r}"
                )
            quantized = np.asarray(image, dtype=np.uint16)
    except (OSError, UnidentifiedImageError) as exc:
        raise ObservationCodecError(f"invalid depth PNG: {exc}") from exc
    depth = quantized.astype(np.float32) / np.float32(65535.0)
    return np.ascontiguousarray(depth[:, :, None], dtype=np.float32)
