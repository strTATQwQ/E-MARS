#!/usr/bin/env python3
"""Download one pinned Hugging Face Xet file through the official CAS API.

This is a transport fallback for environments where the normal hf_xet client
loops after transient reconstruction-service failures. It never stores or
prints the token and only publishes the destination after byte count and
SHA-256 match the caller's pinned values.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import lz4.frame
import requests


def retry_get(session, url, *, headers=None, attempts=12, timeout=90):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            if response.status_code in (200, 206):
                return response
            last = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(min(10, attempt))
    raise RuntimeError(f"GET failed after {attempts} attempts: {type(last).__name__}: {last}")


def bg4_regroup(grouped: bytes) -> bytes:
    size = len(grouped)
    split, remainder = divmod(size, 4)
    lengths = [split + int(remainder > index) for index in range(4)]
    groups = []
    offset = 0
    for length in lengths:
        groups.append(grouped[offset : offset + length])
        offset += length
    output = bytearray(size)
    for index in range(split):
        output[4 * index : 4 * index + 4] = bytes(group[index] for group in groups)
    for index in range(remainder):
        output[4 * split + index] = groups[index][split]
    return bytes(output)


def decode_xorb_range(payload: bytes) -> tuple[bytes, int]:
    position = 0
    output = bytearray()
    chunks = 0
    while position < len(payload):
        if len(payload) - position < 8:
            raise RuntimeError("truncated Xorb chunk header")
        version = payload[position]
        compressed_length = int.from_bytes(payload[position + 1 : position + 4], "little")
        scheme = payload[position + 4]
        uncompressed_length = int.from_bytes(payload[position + 5 : position + 8], "little")
        position += 8
        compressed = payload[position : position + compressed_length]
        if len(compressed) != compressed_length:
            raise RuntimeError("truncated Xorb compressed chunk")
        position += compressed_length
        if version != 0:
            raise RuntimeError(f"unsupported Xorb chunk version {version}")
        if scheme == 0:
            decoded = compressed
        elif scheme == 1:
            decoded = lz4.frame.decompress(compressed)
        elif scheme == 2:
            decoded = bg4_regroup(lz4.frame.decompress(compressed))
        else:
            raise RuntimeError(f"unsupported Xorb compression scheme {scheme}")
        if len(decoded) != uncompressed_length:
            raise RuntimeError(
                f"Xorb chunk length mismatch: {len(decoded)} != {uncompressed_length}"
            )
        output.extend(decoded)
        chunks += 1
    return bytes(output), chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    resolve_url = (
        f"{args.endpoint.rstrip('/')}/{prefix}{args.repo_id}/resolve/"
        f"{args.revision}/{quote(args.filename, safe='/')}"
    )
    auth_headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()
    metadata = session.head(resolve_url, headers=auth_headers, allow_redirects=False, timeout=30)
    if metadata.status_code not in (200, 302, 303, 307):
        raise SystemExit(f"metadata request failed: HTTP {metadata.status_code}")
    file_hash = metadata.headers.get("X-Xet-Hash")
    link = metadata.headers.get("Link", "")
    auth_match = re.search(r'<([^>]+)>;\s*rel="xet-auth"', link)
    if not file_hash or not auth_match:
        raise SystemExit("file does not expose official Xet reconstruction metadata")

    token_response = retry_get(session, auth_match.group(1), headers=auth_headers, timeout=30)
    xet_auth = token_response.json()
    reconstruction_url = f"{xet_auth['casUrl'].rstrip('/')}/v1/reconstructions/{file_hash}"
    reconstruction_headers = {
        "Authorization": f"Bearer {xet_auth['accessToken']}",
        "Accept": "application/vnd.xet.reconstruction.v1+json",
    }
    reconstruction = retry_get(
        session,
        reconstruction_url,
        headers=reconstruction_headers,
        attempts=30,
        timeout=30,
    ).json()

    assembled = bytearray()
    total_chunks = 0
    for term_index, term in enumerate(reconstruction["terms"]):
        term_range = term["range"]
        candidates = sorted(
            reconstruction["fetch_info"][term["hash"]],
            key=lambda item: item["range"]["start"],
        )
        decoded_term = bytearray()
        covered_start = term_range["start"]
        for item in candidates:
            item_range = item["range"]
            if item_range["end"] <= term_range["start"] or item_range["start"] >= term_range["end"]:
                continue
            if item_range["start"] != covered_start:
                raise RuntimeError("non-contiguous Xet fetch ranges")
            byte_range = item["url_range"]
            response = retry_get(
                session,
                item["url"],
                # CAS v1 reports url_range.end as an inclusive byte offset.
                headers={"Range": f"bytes={byte_range['start']}-{byte_range['end']}"},
                attempts=8,
                timeout=120,
            )
            decoded, chunks = decode_xorb_range(response.content)
            decoded_term.extend(decoded)
            total_chunks += chunks
            covered_start = item_range["end"]
        if covered_start != term_range["end"]:
            raise RuntimeError("incomplete Xet fetch range coverage")
        if len(decoded_term) != term["unpacked_length"]:
            raise RuntimeError(
                f"term {term_index} length mismatch: {len(decoded_term)} != {term['unpacked_length']}"
            )
        assembled.extend(decoded_term)

    offset = int(reconstruction.get("offset_into_first_range", 0))
    if offset:
        del assembled[:offset]
    if len(assembled) != args.expected_size:
        raise RuntimeError(f"file size mismatch: {len(assembled)} != {args.expected_size}")
    digest = hashlib.sha256(assembled).hexdigest()
    if digest.lower() != args.expected_sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch: {digest} != {args.expected_sha256}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".part")
    temporary.write_bytes(assembled)
    temporary.replace(args.output)
    print(
        f"PASS {args.output} bytes={len(assembled)} sha256={digest} "
        f"terms={len(reconstruction['terms'])} chunks={total_chunks}"
    )


if __name__ == "__main__":
    main()
