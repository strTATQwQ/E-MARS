from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import PurePosixPath
from typing import Any


def model_files_from_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    values: list[str] = []
    for flag in ("-m", "--model", "--mmproj"):
        if flag in tokens and tokens.index(flag) + 1 < len(tokens):
            values.append(tokens[tokens.index(flag) + 1])
    expanded = []
    for value in values:
        match = re.search(r"-00001-of-(\d{5})\.gguf$", value)
        if match:
            count = int(match.group(1))
            expanded.extend(
                re.sub(r"-00001-of-\d{5}\.gguf$", f"-{index:05d}-of-{count:05d}.gguf", value)
                for index in range(1, count + 1)
            )
        else:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def capture_dgx_model_fingerprint(
    *, plink: str, host: str, user: str, password: str
) -> dict[str, Any]:
    target = f"{user}@{host}"
    base = [plink, "-batch", "-ssh", "-pw", password, target]
    process = subprocess.run(
        base + ["ps -C llama-server -o args="],
        check=True,
        text=True,
        capture_output=True,
    )
    commands = [line.strip() for line in process.stdout.splitlines() if "llama-server" in line]
    if len(commands) != 1:
        raise RuntimeError(f"expected one llama-server process, found {len(commands)}")
    files = model_files_from_command(commands[0])
    quoted = " ".join(shlex.quote(value) for value in files)
    checksum = subprocess.run(
        base + [f"sha256sum {quoted}"],
        check=True,
        text=True,
        capture_output=True,
    )
    hashes = {}
    for line in checksum.stdout.splitlines():
        digest, path = line.split(maxsplit=1)
        hashes[path.lstrip("* ")] = digest
    return {
        "schema_version": 1,
        "host": host,
        "server_command": commands[0],
        "model_files": [
            {"path": value, "name": PurePosixPath(value).name, "sha256": hashes.get(value)}
            for value in files
        ],
    }


def write_fingerprint(path, fingerprint: dict[str, Any]) -> None:
    path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
