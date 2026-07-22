#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPLAY_ROOT="${INTERNNAV_T1_REPLAY_ROOT:-$HOME/internnav-t1-t2/replay/legacy_canary_120}"
RESULT_ROOT="${INTERNNAV_T1_CAPTURE_RESULT_ROOT:-$HOME/internnav-t1-t2/results/t1_0_legacy_canary}"
: "${INTERNNAV_SERVER_HOST:?set INTERNNAV_SERVER_HOST to the DGX Spark LAN address}"

[[ ! -e "$REPLAY_ROOT" ]] || {
  echo "refusing to overwrite replay root: $REPLAY_ROOT" >&2
  exit 1
}
mkdir -p "$RESULT_ROOT"

export INTERNNAV_T0_CONTROL_ROOT="$CONTROL_ROOT"
export INTERNNAV_T0_ISAAC_MODE=isaac6_compat
export INTERNNAV_T0_ISAAC_ENTRYPOINT="$CONTROL_ROOT/scripts/run_internnav_t1_replay_capture_entrypoint.py"
export INTERNNAV_T0_CONFIG_PATH="$CONTROL_ROOT/configs/internnav_t1_t2/t1_0_replay_capture_cfg.py"
export INTERNNAV_T0_RESULT_DIR="$RESULT_ROOT"
export INTERNNAV_T1_REPLAY_ROOT="$REPLAY_ROOT"
export INTERNNAV_T1_REPLAY_STEPS="${INTERNNAV_T1_REPLAY_STEPS:-120}"
export INTERNNAV_T1_CHECKPOINT_REVISION="a698a9e898b4001621a319e1bc89f02ec715cc86"
export INTERNNAV_T1_COMMIT="1d8d078aa9031a4a02a1ae05844d49a1768a10e4"

bash "$CONTROL_ROOT/scripts/run_internnav_canary.sh"

ISAAC_PYTHON="${INTERNNAV_T0_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
"$ISAAC_PYTHON" "$CONTROL_ROOT/scripts/capture_internnav_t1_replay.py" \
  --root "$REPLAY_ROOT" \
  --minimum 100 \
  --expected "$INTERNNAV_T1_REPLAY_STEPS" \
  | tee "$RESULT_ROOT/replay_validation.log"

cp "$HOME/internnav-t0/InternNav/logs/internnav_t1_replay_capture/result.json" \
  "$RESULT_ROOT/canary_result.json"
cp "$REPLAY_ROOT/manifest.json" "$RESULT_ROOT/replay_manifest.json"
