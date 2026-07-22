#!/usr/bin/env bash
set -euo pipefail

# Run on the DGX Spark model host. Credentials and host addresses are never
# accepted as command-line defaults and are not written to logs.
CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
MODEL_PYTHON="${INTERNNAV_MODEL_PYTHON:-$HOME/internnav-t0/venv-model/bin/python}"
RESULT_DIR="${INTERNNAV_T0_RESULT_DIR:-$CONTROL_ROOT/results/internnav_t0/server}"

"$MODEL_PYTHON" "$CONTROL_ROOT/scripts/internnav_t0_preflight.py" \
  --role server \
  --internnav-root "$INTERNNAV_ROOT" \
  --control-root "$CONTROL_ROOT"

mkdir -p "$RESULT_DIR"
cd "$INTERNNAV_ROOT"
export PYTHONPATH="$INTERNNAV_ROOT:$INTERNNAV_ROOT/third_party/diffusion-policy${PYTHONPATH:+:$PYTHONPATH}"

# Official AgentServer has no dedicated health route. Once Uvicorn is ready,
# GET http://<model-host>:8023/openapi.json is the non-mutating health probe.
set +e
"$MODEL_PYTHON" scripts/eval/start_server.py \
  --host 0.0.0.0 \
  --config scripts/eval/configs/h1_internvla_n1_async_cfg.py \
  2>&1 | tee "$RESULT_DIR/server.log"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"

