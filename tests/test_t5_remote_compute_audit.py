from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "t5_remote_compute_audit_common.sh"


def _linux_path(path: Path) -> str:
    if os.name == "nt":
        return subprocess.check_output(
            ["wsl", "wslpath", "-a", str(path)], text=True
        ).strip()
    return str(path)


def _run_fixture(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    root = _linux_path(ROOT)
    output = _linux_path(tmp_path / f"{mode}.json")
    fixture = f"""
set -euo pipefail
root={root!r}
remote() {{
  shift
  case {mode!r} in
    pass)
      printf '%s\\n' '{{"schema_version":1,"status":"PASS","mode":"forbidden-compute","match_count":0,"matches":[],"errors":[]}}'
      return 0 ;;
    stale)
      printf '%s\\n' '{{"schema_version":1,"status":"FAIL","mode":"forbidden-compute","match_count":1,"matches":[{{"pid":42}}],"errors":[]}}'
      return 73 ;;
    error)
      printf '%s\\n' '{{"schema_version":1,"status":"ERROR","mode":"forbidden-compute","match_count":0,"matches":[],"errors":[{{"pid":7,"error":"PermissionError"}}]}}'
      return 74 ;;
    malformed)
      printf '%s\\n' 'not-json'
      return 0 ;;
  esac
}}
source "$root/scripts/t5_remote_compute_audit_common.sh"
t5_remote_compute_absent fixture-host {output!r}
"""
    command = ["bash", "-c", fixture]
    if os.name == "nt":
        command = ["wsl", "-e", *command]
    return subprocess.run(command, text=True, capture_output=True, check=False)


@pytest.mark.skipif(
    os.name == "nt" and shutil.which("wsl") is None,
    reason="WSL is required for the shell fixture",
)
def test_remote_compute_wrapper_accepts_only_exact_pass(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path, "pass")
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "pass.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert all(payload["checks"].values())
    assert len(payload["auditor_sha256"]) == 64


@pytest.mark.skipif(
    os.name == "nt" and shutil.which("wsl") is None,
    reason="WSL is required for the shell fixture",
)
@pytest.mark.parametrize("mode", ["stale", "error", "malformed"])
def test_remote_compute_wrapper_fails_closed(tmp_path: Path, mode: str) -> None:
    result = _run_fixture(tmp_path, mode)
    assert result.returncode == 75
    payload = json.loads((tmp_path / f"{mode}.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert not all(payload["checks"].values())


def test_helper_is_source_bound_and_atomic() -> None:
    text = HELPER.read_text(encoding="utf-8")
    assert '"$root/scripts/t5_process_identity_audit.py"' in text
    assert "_t5_compute_auditor_sha256" in text
    assert "mktemp /tmp/internnav-t5-compute-audit" in text
    assert "trap 'rm -f --" in text
    assert "\\$audit" in text
    assert 'os.replace(temporary, output)' in text
    assert 'raise SystemExit(0 if payload["status"] == "PASS" else 75)' in text
