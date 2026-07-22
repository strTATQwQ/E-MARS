#!/usr/bin/env bash
set -euo pipefail

# The pilot launcher shares the exact official evaluator path with canary, but
# preflight requires a PASS canary attestation and a disjoint 20-episode overlay.
CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export INTERNNAV_T0_CONTROL_ROOT="$CONTROL_ROOT"
export INTERNNAV_T0_DATASET_ROOT="${INTERNNAV_T0_DATASET_ROOT:-$HOME/internnav-t0/episodes/pilot}"
export INTERNNAV_T0_RESULT_DIR="${INTERNNAV_T0_RESULT_DIR:-$CONTROL_ROOT/results/internnav_t0/pilot}"
export INTERNNAV_T0_ALLOW_PILOT=1

exec bash "$CONTROL_ROOT/scripts/run_internnav_canary.sh" pilot
