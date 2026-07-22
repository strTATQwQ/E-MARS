#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 BENCHMARK_CONFIG SLOW_MODEL_CONFIG EPISODE_MANIFEST OUTPUT_DIR RUN_ID" >&2
  exit 2
fi

: "${ISAAC_SLOW_BENCHMARK_ROOT:?set ISAAC_SLOW_BENCHMARK_ROOT}"
: "${MP3D_CONNECTIVITY_ROOT:?set MP3D_CONNECTIVITY_ROOT}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_config="$1"
slow_config="$2"
manifest="$3"
output="$4"
run_id="$5"

export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES

python3 "$root/scripts/run_isaac_mp3d_batch.py" \
  --config "$benchmark_config" \
  --slow-config "$slow_config" \
  --manifest "$manifest" \
  --collision-root "$ISAAC_SLOW_BENCHMARK_ROOT/mp3d_collision_v4" \
  --connectivity-root "$MP3D_CONNECTIVITY_ROOT" \
  --output "$output" \
  --run-id "$run_id" \
  --device cuda:1
