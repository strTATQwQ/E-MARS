#!/usr/bin/env bash
set -euo pipefail

# Compatibility name only. T5 no longer permits a model-only DGX role.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$root/scripts/run_t5_dgx_lane.sh" "$@"
