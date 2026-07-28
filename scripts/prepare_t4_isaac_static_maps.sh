#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: prepare_t4_isaac_static_maps.sh --dataset-root DIR --output-dir DIR" >&2
  exit 64
}

DATASET_ROOT=""
OUTPUT_DIR=""
while (($#)); do
  case "$1" in
    --dataset-root) (($# >= 2)) || usage; DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) (($# >= 2)) || usage; OUTPUT_DIR="$2"; shift 2 ;;
    *) usage ;;
  esac
done

test "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = dgx+isaac
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
ip -4 -o addr show | grep -Fq " 10.100.120.111/"

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
DATASET_FILE="$DATASET_ROOT/val_unseen/val_unseen.json.gz"
CLEARANCE="${INTERNVLA_T3_STATIC_CLEARANCE_GATE_M:-0.30}"
test -f "$DATASET_FILE"
test -d "$INTERNNAV_ROOT/data/scene_data/mp3d_pe"
test ! -e "$OUTPUT_DIR"
[[ "$CLEARANCE" =~ ^0\.[0-9]+$ ]]

DATASET_SHA256="$(sha256sum "$DATASET_FILE" | cut -d' ' -f1)"
CACHE_TAG="${CLEARANCE//./p}"
CACHE_DIR="$CONTROL_ROOT/runtime/static_maps/dgx_onboard_${DATASET_SHA256:0:16}_${CACHE_TAG}"
mkdir -p "$(dirname "$OUTPUT_DIR")" "$CONTROL_ROOT/runtime/static_maps"
if test ! -f "$CACHE_DIR/manifest.json"; then
  test ! -e "$CACHE_DIR"
  python3 "$CONTROL_ROOT/scripts/build_t3_static_maps.py" \
    --dataset "$DATASET_FILE" \
    --scene-root "$INTERNNAV_ROOT/data/scene_data/mp3d_pe" \
    --output-root "$CACHE_DIR" \
    --minimum-required-prefix-clearance-m "$CLEARANCE"
fi

mkdir "$OUTPUT_DIR"
cp "$CACHE_DIR"/*.bin "$OUTPUT_DIR/"
python3 "$CONTROL_ROOT/scripts/build_t4_truth_isolated_static_manifest.py" \
  --manifest "$CACHE_DIR/manifest.json" \
  --dataset "$DATASET_FILE" \
  --output "$OUTPUT_DIR/manifest.json"
python3 - "$OUTPUT_DIR/map_prepare_status.json" "$DATASET_FILE" \
  "$DATASET_SHA256" "$CLEARANCE" "$OUTPUT_DIR/manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[5])
payload = {
    "schema_version": 1,
    "status": "PASS",
    "host_role": "isaac_offline_scene_map_builder",
    "dataset_file": sys.argv[2],
    "dataset_sha256": sys.argv[3],
    "minimum_clearance_m": float(sys.argv[4]),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "destination_role": "dgx_onboard_compute",
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
