#!/usr/bin/env bash
set -euo pipefail

# Run on Isaac after both model servers and the local ROS client node are ready.
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
RESULT_ROOT="${INTERNVLA_SHADOW_RESULT_ROOT:-$CONTROL_ROOT/results/t1_2_shadow}"
: "${INTERNNAV_SERVER_HOST:?set INTERNNAV_SERVER_HOST for the legacy shadow-primary server}"

mkdir -p "$RESULT_ROOT"
export INTERNNAV_T0_CONTROL_ROOT="$CONTROL_ROOT"
export INTERNNAV_T0_ISAAC_MODE=isaac6_compat
export INTERNNAV_T0_ISAAC_ENTRYPOINT="$CONTROL_ROOT/scripts/run_internnav_t1_shadow_entrypoint.py"
export INTERNNAV_T0_CONFIG_PATH="$CONTROL_ROOT/configs/internnav_t1_t2/t1_2_shadow_cfg.py"
export INTERNNAV_T0_RESULT_DIR="$RESULT_ROOT"
export INTERNVLA_SHADOW_RESULT_DIR="$RESULT_ROOT/shadow"
export INTERNVLA_LOCAL_IPC_TIMEOUT_SEC="${INTERNVLA_LOCAL_IPC_TIMEOUT_SEC:-360}"

bash "$CONTROL_ROOT/scripts/run_internnav_canary.sh"

UPSTREAM_RESULT="$HOME/internnav-t0/InternNav/logs/internnav_t1_shadow/result.json"
test -f "$UPSTREAM_RESULT"
cp "$UPSTREAM_RESULT" "$RESULT_ROOT/evaluator_result.json"
"${INTERNNAV_T0_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}" \
  "$CONTROL_ROOT/scripts/internvla_shadow_agent_client.py" \
  --finalize "$RESULT_ROOT/shadow" \
  --evaluator-result "$RESULT_ROOT/evaluator_result.json" \
  --expected-episodes 5 \
  | tee "$RESULT_ROOT/shadow_validation.log"
