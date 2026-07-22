from __future__ import annotations

import json
import sys
import types
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    class _Array:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return list(self.value)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.float64 = float
    numpy_stub.bool_ = bool
    numpy_stub.ndarray = _Array
    numpy_stub.isscalar = lambda value: isinstance(
        value, (str, bytes, int, float, bool, complex)
    )
    numpy_stub.asarray = lambda value, dtype=None: _Array(value)
    sys.modules["numpy"] = numpy_stub

import internvla_nav2_oracle_agent_client as oracle_client


def test_continuous_oracle_forwards_typed_dataset_identity() -> None:
    source = (SCRIPTS / "internvla_go2_continuous_agent_client.py").read_text(
        encoding="utf-8"
    )
    oracle_facade = source.split("class ContinuousNav2OracleAgentClient", 1)[1]
    assert 'str(response["episode_id"])' in oracle_facade
    assert 'f"oracle-episode-{generation}"' not in oracle_facade


def test_step_failure_clears_last_result(monkeypatch, capsys):
    client = oracle_client.Nav2OracleAgentClient.__new__(
        oracle_client.Nav2OracleAgentClient
    )
    client.by_instruction = {
        "go": {
            "episode_id": "1474",
            "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }
    }
    client.connection = object()
    client.last_result = {"sequence_id": 7, "stop": False}
    client.last_error = ""

    def fail_exchange(*_args, **_kwargs):
        raise RuntimeError("typed Nav2 failure")

    monkeypatch.setattr(oracle_client, "_exchange", fail_exchange)
    result = client.step(
        [
            {
                "instruction": "go",
                "globalgps": [0.0, 0.0, 0.0],
                "globalrotation": [0.0, 0.0, 0.0, 1.0],
            }
        ]
    )

    assert result == [{"action": [0], "ideal_flag": True}]
    assert client.last_result is None
    assert "typed Nav2 failure" in client.last_error
    stderr = capsys.readouterr().err
    assert "INTERNVLA_ORACLE_STEP_ERROR" in stderr
    assert "INTERNVLA_ORACLE_ACTION_OK" not in stderr


def test_step_sends_and_requires_dataset_episode_id(monkeypatch, capsys):
    client = oracle_client.Nav2OracleAgentClient.__new__(
        oracle_client.Nav2OracleAgentClient
    )
    client.by_instruction = {
        "go": {
            "episode_id": 1474,
            "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }
    }
    client.connection = object()
    client.last_result = None
    client.last_error = ""
    requests = []

    def exchange(_connection, request):
        requests.append(request)
        return {
            "episode_id": "1474",
            "discrete_action": 1,
            "reset_generation": 0,
            "sequence_id": 0,
            "stop": False,
        }

    monkeypatch.setattr(oracle_client, "_exchange", exchange)
    result = client.step(
        [
            {
                "instruction": "go",
                "globalgps": [0.0, 0.0, 0.0],
                "globalrotation": [0.0, 0.0, 0.0, 1.0],
            }
        ]
    )

    assert result == [{"action": [1], "ideal_flag": True}]
    assert requests[0]["episode_id"] == "1474"
    assert client.last_result["episode_id"] == "1474"
    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("INTERNVLA_ORACLE_ACTION_OK ")
    ]
    assert len(lines) == 1
    assert json.loads(lines[0].split(" ", 1)[1]) == {
        "schema_version": 1,
        "episode_id": "1474",
        "discrete_action": 1,
    }


def test_step_rejects_episode_identity_drift(monkeypatch, capsys):
    client = oracle_client.Nav2OracleAgentClient.__new__(
        oracle_client.Nav2OracleAgentClient
    )
    client.by_instruction = {
        "go": {
            "episode_id": "1474",
            "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }
    }
    client.connection = object()
    client.last_result = None
    client.last_error = ""
    monkeypatch.setattr(
        oracle_client,
        "_exchange",
        lambda *_args, **_kwargs: {
            "episode_id": "676",
            "discrete_action": 1,
        },
    )

    result = client.step(
        [
            {
                "instruction": "go",
                "globalgps": [0.0, 0.0, 0.0],
                "globalrotation": [0.0, 0.0, 0.0, 1.0],
            }
        ]
    )

    assert result == [{"action": [0], "ideal_flag": True}]
    assert client.last_result is None
    assert "different dataset episode_id" in client.last_error
    stderr = capsys.readouterr().err
    assert "INTERNVLA_ORACLE_STEP_ERROR" in stderr
    assert "INTERNVLA_ORACLE_ACTION_OK" not in stderr
