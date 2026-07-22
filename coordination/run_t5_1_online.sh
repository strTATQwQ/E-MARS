#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
run_t5_1_online.sh is permanently disabled.

The split DGX_MODEL/DGX_EDGE topology was terminated before model or Isaac
execution.  T5 online work is authorized only by T5_DUAL_LANE_BOARD.md and the
run_t5_d0_* symmetric-lane coordinators.
EOF
exit 69
