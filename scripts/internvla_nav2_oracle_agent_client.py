#!/usr/bin/env python3
"""Official-evaluator AgentClient facade for the fixed Nav2 oracle gate."""

from __future__ import annotations

import gzip
import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


MAX_MESSAGE_BYTES = 1024 * 1024
MAX_EPISODE_ID_BYTES = 256


def _lane_identity_prefix() -> str:
    prefix = os.environ.get("INTERNNAV_T5_ID_PREFIX", "")
    if prefix not in {"", "a::", "b::"}:
        raise RuntimeError("INTERNNAV_T5_ID_PREFIX must be empty, a::, or b::")
    return prefix


def _validated_episode_id(value: Any) -> str:
    episode_id = str(value)
    if not episode_id:
        raise ValueError("oracle dataset episode_id is empty")
    if len(episode_id.encode("utf-8")) > MAX_EPISODE_ID_BYTES:
        raise ValueError("oracle dataset episode_id is too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in episode_id):
        raise ValueError("oracle dataset episode_id contains control characters")
    return episode_id


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    output = bytearray()
    while len(output) < count:
        block = connection.recv(count - len(output))
        if not block:
            raise ConnectionError("oracle IPC disconnected")
        output.extend(block)
    return bytes(output)


def _exchange(connection: socket.socket, value: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    connection.sendall(struct.pack("!I", len(payload)) + payload)
    size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if size < 2 or size > MAX_MESSAGE_BYTES:
        raise RuntimeError("invalid oracle IPC response size")
    response = json.loads(_recv_exact(connection, size).decode("utf-8"))
    if response.get("status") != "ok":
        raise RuntimeError(str(response.get("message", "oracle IPC failed")))
    return response


class Nav2OracleAgentClient:
    def __init__(self, config: Any):
        del config
        dataset_file = Path(os.environ["INTERNVLA_ORACLE_DATASET_FILE"]).resolve()
        with gzip.open(dataset_file, "rt", encoding="utf-8") as stream:
            episodes = json.load(stream)["episodes"]
        self.by_instruction = {
            item["instruction"]["instruction_text"]: item for item in episodes
        }
        if len(self.by_instruction) != len(episodes):
            raise RuntimeError("oracle dataset contains duplicate instructions")
        self.identity_prefix = _lane_identity_prefix()
        endpoint = os.environ.get("INTERNVLA_ORACLE_ENDPOINT", "")
        self.connection = socket.socket(
            socket.AF_INET if endpoint else socket.AF_UNIX, socket.SOCK_STREAM
        )
        self.connection.settimeout(
            float(os.environ.get("INTERNVLA_ORACLE_IPC_TIMEOUT_SEC", "360"))
        )
        if endpoint:
            if not endpoint.startswith("tcp://"):
                raise RuntimeError("INTERNVLA_ORACLE_ENDPOINT must use tcp://")
            host, separator, port_text = endpoint[6:].rpartition(":")
            if not separator or not host or not port_text.isdigit():
                raise RuntimeError("invalid Oracle TCP endpoint")
            port = int(port_text)
            if not 1024 <= port <= 65535:
                raise RuntimeError("Oracle TCP port is outside bounds")
            self.connection.connect((host, port))
        else:
            self.connection.connect(
                os.environ.get("INTERNVLA_ORACLE_SOCKET", "/tmp/internvla_oracle.sock")
            )
        _exchange(self.connection, {"schema_version": 1, "operation": "initialize"})
        self.last_result: dict[str, Any] | None = None
        self.last_error = ""

    def step(self, obs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            if len(obs) != 1:
                raise ValueError("Nav2 oracle expects one environment")
            item = obs[0]
            episode = self.by_instruction[str(item["instruction"])]
            dataset_episode_id = _validated_episode_id(episode["episode_id"])
            episode_id = (
                f"{getattr(self, 'identity_prefix', '')}{dataset_episode_id}"
            )
            response = _exchange(
                self.connection,
                {
                    "schema_version": 1,
                    "operation": "step",
                    "episode_id": episode_id,
                    "global_gps": np.asarray(item["globalgps"], dtype=np.float64).tolist(),
                    "global_rotation": np.asarray(
                        item["globalrotation"], dtype=np.float64
                    ).tolist(),
                    "reference_path": episode["reference_path"],
                },
            )
            if str(response.get("episode_id", "")) != episode_id:
                raise RuntimeError("oracle IPC returned a different dataset episode_id")
            discrete_action = int(response["discrete_action"])
            result = [{"action": [discrete_action], "ideal_flag": True}]
            self.last_result = response
            self.last_error = ""
            print(
                "INTERNVLA_ORACLE_ACTION_OK "
                + json.dumps(
                    {
                        "schema_version": 1,
                        "episode_id": episode_id,
                        "discrete_action": discrete_action,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            return result
        except BaseException as exc:
            # Never let the continuous facade reuse an earlier successful
            # command after a typed Nav2/IPC failure.  Replaying that stale
            # identity would keep the physics window running past the safety
            # boundary and poison the next reset generation.
            self.last_result = None
            self.last_error = repr(exc)[:1024]
            print(
                "INTERNVLA_ORACLE_STEP_ERROR "
                + json.dumps(
                    {
                        "schema_version": 1,
                        "error": self.last_error,
                        "safe_stop": True,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            return [{"action": [0], "ideal_flag": True}]

    def reset(self, reset_index: Any = None) -> dict[str, Any]:
        del reset_index
        return _exchange(self.connection, {"schema_version": 1, "operation": "reset"})

    def close(self) -> None:
        if self.connection is None:
            return
        try:
            _exchange(self.connection, {"schema_version": 1, "operation": "shutdown"})
        except BaseException:
            pass
        self.connection.close()
        self.connection = None

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
