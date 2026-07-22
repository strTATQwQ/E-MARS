#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="viewport_frame_%04d.png")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default="viewport.mp4")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    if not output.is_absolute():
        output = input_dir / output
    first = input_dir / args.pattern.replace("%04d", "0000")
    if not first.exists():
        print(f"PNG sequence not found: {first}")
        return 2
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"ffmpeg not found; keeping PNG sequence in {input_dir}")
        return 0
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(input_dir / args.pattern),
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
