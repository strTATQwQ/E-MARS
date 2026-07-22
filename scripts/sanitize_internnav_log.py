#!/usr/bin/env python3
"""Create a reproducible credential/host/path-redacted text log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPLACEMENTS = [
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "<REDACTED_HF_TOKEN>"),
    (
        re.compile(
            r"(?<!\d)(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d)"
        ),
        "<REDACTED_PRIVATE_HOST>",
    ),
    (re.compile(r"/home/[^/\s]+"), "$HOME"),
    (
        re.compile(r"(?i)(password|passwd|token)(\s*[:=]\s*)[^\s,;]+"),
        r"\1\2<REDACTED>",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8", errors="replace")
    for pattern, replacement in REPLACEMENTS:
        text = pattern.sub(replacement, text)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
