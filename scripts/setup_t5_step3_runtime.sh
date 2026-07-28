#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s MODEL_PATH VENV_PATH RESULT_DIR\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 3 ]] || usage
model_path="$1"
venv_path="$2"
result_dir="$3"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
requirements="$root/configs/internnav_t5/step3_runtime_requirements.txt"

case "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" in
  dgx-a|lane-a)
    expected_user=railgun
    expected_home=/home/railgun
    ;;
  dgx-b|lane-b)
    expected_user=rail
    expected_home=/home/rail
    ;;
  *) exit 73 ;;
esac
test "$(id -un)" = "$expected_user"
test -d "$model_path"
test -s "$model_path/config.json"
test -f "$requirements"
[[ "$venv_path" = "$expected_home"/ai-stack/venvs/* ]]
[[ "$result_dir" = "$expected_home"/* ]]
test ! -L "$model_path"
test ! -L "$requirements"

mkdir -p -- "$(dirname -- "$venv_path")" "$result_dir"
venv_path="$(readlink -m -- "$venv_path")"
result_dir="$(readlink -m -- "$result_dir")"
requirements="$(readlink -f -- "$requirements")"
test ! -e "$venv_path" || {
  test -x "$venv_path/bin/python"
  test -f "$venv_path/T5_STEP3_RUNTIME_READY.json"
}

if test ! -e "$venv_path"; then
  build_path="${venv_path}.build.$(date -u +%Y%m%dT%H%M%SZ).$$"
  test ! -e "$build_path"
  python3 -m venv "$build_path"
  "$build_path/bin/python" -m pip install --disable-pip-version-check \
    --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    --timeout 120 --retries 10 --requirement "$requirements"
  "$build_path/bin/python" - "$requirements" "$model_path" \
    "$build_path/T5_STEP3_RUNTIME_READY.json" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import accelerate
import fastapi
import numpy
import PIL
import psutil
import safetensors
import sentencepiece
import tokenizers
import torch
import torchvision
import transformers
import uvicorn
import yaml
import zmq

requirements, model_path, output = map(Path, sys.argv[1:])
expected = {
    # The ARM64 CUDA wheels expose their PEP 440 local build tag at runtime.
    # Keep the install requirement portable while pinning the actually loaded
    # CUDA build exactly in the immutable runtime receipt.
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "transformers": "4.57.6",
    "accelerate": "1.14.0",
    "numpy": "2.3.5",
    "PIL": "12.3.0",
    "yaml": "6.0.3",
    "zmq": "27.1.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.1",
    "tokenizers": "0.22.2",
    "psutil": "7.2.2",
    "fastapi": "0.139.2",
    "uvicorn": "0.51.0",
}
modules = {
    "torch": torch,
    "torchvision": torchvision,
    "transformers": transformers,
    "accelerate": accelerate,
    "numpy": numpy,
    "PIL": PIL,
    "yaml": yaml,
    "zmq": zmq,
    "safetensors": safetensors,
    "sentencepiece": sentencepiece,
    "tokenizers": tokenizers,
    "psutil": psutil,
    "fastapi": fastapi,
    "uvicorn": uvicorn,
}
observed = {name: str(module.__version__) for name, module in modules.items()}
if observed != expected:
    raise SystemExit(f"pinned Step3 runtime mismatch: {observed!r}")
if not torch.cuda.is_available():
    raise SystemExit("Step3 runtime cannot see CUDA")
freeze = subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
).stdout
payload = {
    "schema_version": 1,
    "status": "RUNTIME_READY",
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
    "model_path": str(model_path.resolve()),
    "versions": observed,
    "cuda_available": True,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "created_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv -- "$build_path" "$venv_path"
fi

"$venv_path/bin/python" - "$venv_path/T5_STEP3_RUNTIME_READY.json" \
  "$result_dir/runtime_setup.json" "$requirements" "$model_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, output, requirements, model_path = map(Path, sys.argv[1:])
value = json.loads(source.read_text(encoding="utf-8"))
expected_versions = {
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "transformers": "4.57.6",
    "accelerate": "1.14.0",
    "numpy": "2.3.5",
    "PIL": "12.3.0",
    "yaml": "6.0.3",
    "zmq": "27.1.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.1",
    "tokenizers": "0.22.2",
    "psutil": "7.2.2",
    "fastapi": "0.139.2",
    "uvicorn": "0.51.0",
}
checks = {
    "ready": value.get("status") == "RUNTIME_READY",
    "cuda": value.get("cuda_available") is True,
    "architecture": value.get("architecture") == "aarch64",
    "versions": value.get("versions") == expected_versions,
    "requirements": value.get("requirements_sha256")
    == hashlib.sha256(requirements.read_bytes()).hexdigest(),
    "model_path": value.get("model_path") == str(model_path.resolve()),
}
if not all(checks.values()):
    raise SystemExit("existing Step3 runtime receipt is not ready")
value["receipt_revalidation"] = checks
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(value, sort_keys=True))
PY
