from __future__ import annotations

import hashlib
import io
import os
import re
import time
from pathlib import Path

from PIL import Image


def safe_episode_filename(episode_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", episode_id).strip("._-")
    return value or hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_episode_video(
    video_dir: Path,
    episode_id: str,
    jpeg_frames: list[bytes],
    *,
    fps: float,
    codec: str,
    quality: int,
) -> dict:
    if not jpeg_frames:
        raise ValueError("at least one JPEG frame is required for video evidence")
    if fps <= 0:
        raise ValueError("video fps must be positive")
    video_dir.mkdir(parents=True, exist_ok=True)
    output = video_dir / f"{safe_episode_filename(episode_id)}.mp4"
    temporary = output.with_suffix(".tmp.mp4")
    started = time.perf_counter()
    try:
        import imageio.v2 as imageio
        import numpy as np

        with imageio.get_writer(
            temporary,
            format="FFMPEG",
            mode="I",
            fps=float(fps),
            codec=codec,
            quality=int(quality),
            macro_block_size=None,
            ffmpeg_log_level="error",
        ) as writer:
            expected_size = None
            for encoded in jpeg_frames:
                with Image.open(io.BytesIO(encoded)) as image:
                    rgb = np.asarray(image.convert("RGB"))
                size = (int(rgb.shape[1]), int(rgb.shape[0]))
                if expected_size is None:
                    expected_size = size
                elif size != expected_size:
                    raise ValueError(f"video frame size changed from {expected_size} to {size}")
                writer.append_data(rgb)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    encoding_seconds = time.perf_counter() - started
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "frame_count": len(jpeg_frames),
        "fps": float(fps),
        "codec": codec,
        "quality": int(quality),
        "encoding_seconds_excluded_from_episode_wall": encoding_seconds,
        "source": "isaac_front_render_product",
    }
