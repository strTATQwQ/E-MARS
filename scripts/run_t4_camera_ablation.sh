#!/usr/bin/env bash
set -eo pipefail

SPLIT="${1:-}"
VARIANT="${2:-}"
ATTEMPT="${3:-001}"
if test "$SPLIT" != dev && test "$SPLIT" != heldout; then
  echo "usage: $0 {dev|heldout} CAMERA_VARIANT [attempt]" >&2
  exit 2
fi

HEIGHT=0.62
PITCH=30
HFOV=90
VFOV=
CAMERA_MODEL=generic_rgbd
case "$VARIANT" in
  current) ;;
  height_085) HEIGHT=0.85 ;;
  height_100) HEIGHT=1.00 ;;
  h1_equivalent) HEIGHT=1.25 ;;
  pitch_20) PITCH=20 ;;
  pitch_40) PITCH=40 ;;
  hfov_80) HFOV=80 ;;
  hfov_100) HFOV=100 ;;
  d435i_low_062_p20) HEIGHT=0.62; PITCH=20; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  d435i_low_062_p30) HEIGHT=0.62; PITCH=30; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  d435i_low_062_p40) HEIGHT=0.62; PITCH=40; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  d435i_mid_085_p20) HEIGHT=0.85; PITCH=20; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  d435i_mid_085_p30) HEIGHT=0.85; PITCH=30; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  d435i_mid_085_p40) HEIGHT=0.85; PITCH=40; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  d435i_high_100_p30) HEIGHT=1.00; PITCH=30; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  d435i_h1_125_p30) HEIGHT=1.25; PITCH=30; HFOV=69.4; VFOV=42.5; CAMERA_MODEL=realsense_d435i_rgb ;;
  custom)
    HEIGHT="${INTERNVLA_T4_CAMERA_HEIGHT_M:?set custom camera height}"
    PITCH="${INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG:?set custom camera pitch}"
    HFOV="${INTERNVLA_T4_CAMERA_HFOV_DEG:?set custom camera HFOV}"
    ;;
  *) echo "unknown camera variant: $VARIANT" >&2; exit 2 ;;
esac

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
T0_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$HOME/internnav-t0/control}"
ISAAC_PYTHON="${INTERNNAV_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
DEV_ROOT="$CONTROL_ROOT/episodes/t4_camera_dev"
HELDOUT_ROOT="$CONTROL_ROOT/episodes/t4_camera_heldout"
D435I_HELDOUT_ROOT="$CONTROL_ROOT/episodes/t4_camera_d435i_heldout"

python3 "$CONTROL_ROOT/scripts/prepare_t4_camera_splits.py" \
  --canary "$CONTROL_ROOT/episodes/t3_model_canary_go2_clear_v2/val_unseen/val_unseen.json.gz" \
  --pilot "$CONTROL_ROOT/episodes/t3_model_pilot_go2_clear_v2/val_unseen/val_unseen.json.gz" \
  --dev-output "$DEV_ROOT" --heldout-output "$HELDOUT_ROOT" \
  >"$CONTROL_ROOT/episodes/t4_camera_splits.json"

if [[ "$VARIANT" == d435i_* ]]; then
  python3 "$CONTROL_ROOT/scripts/prepare_t4_d435i_camera_heldout.py" \
    --canary "$CONTROL_ROOT/episodes/t3_model_canary_go2_clear_v2/val_unseen/val_unseen.json.gz" \
    --legacy-heldout "$HELDOUT_ROOT/val_unseen/val_unseen.json.gz" \
    --pilot "$CONTROL_ROOT/episodes/t3_model_pilot_go2_clear_v2/val_unseen/val_unseen.json.gz" \
    --heldout-output "$D435I_HELDOUT_ROOT" \
    >"$CONTROL_ROOT/episodes/t4_camera_d435i_heldout.json"
fi

if test "$SPLIT" = dev; then
  DATASET_ROOT="$DEV_ROOT"
elif [[ "$VARIANT" == d435i_* ]]; then
  DATASET_ROOT="$D435I_HELDOUT_ROOT"
else
  DATASET_ROOT="$HELDOUT_ROOT"
fi
RESULT_DIR="$CONTROL_ROOT/results/internnav_t4/t4_1_camera_${SPLIT}_${VARIANT}_attempt_${ATTEMPT}"
test ! -e "$RESULT_DIR/phase_status.json"
mkdir -p "$RESULT_DIR"

OVERLAY="$CONTROL_ROOT/runtime/t4_camera_overlay/${SPLIT}_${VARIANT}_${ATTEMPT}"
mkdir -p "$OVERLAY"
for source in "$CONTROL_ROOT"/scripts/*.py "$CONTROL_ROOT"/scripts/*.sh; do
  test -e "$source" || continue
  ln -sfn "$source" "$OVERLAY/$(basename "$source")"
done
ln -sfn "$CONTROL_ROOT/scripts/build_t4_go2_camera_usd.py" \
  "$OVERLAY/build_go2_internvla_usd.py"
unlink "$OVERLAY/internnav_go2_runtime.py"
python3 "$CONTROL_ROOT/scripts/build_t4_runtime_overlay.py" \
  --source "$CONTROL_ROOT/scripts/internnav_go2_runtime.py" \
  --output "$OVERLAY/internnav_go2_runtime.py" \
  --manifest "$RESULT_DIR/runtime_overlay_manifest.json"
unlink "$OVERLAY/run_go2_continuous_phase.sh"
python3 "$CONTROL_ROOT/scripts/build_t4_camera_phase_overlay.py" \
  --source "$CONTROL_ROOT/scripts/run_go2_continuous_phase.sh" \
  --output "$OVERLAY/run_go2_continuous_phase.sh" \
  --manifest "$RESULT_DIR/phase_overlay_manifest.json"

export INTERNVLA_SCRIPT_ROOT="$OVERLAY"
export INTERNVLA_GO2_DATASET_ROOT="$DATASET_ROOT"
export INTERNVLA_T3_RESULT_DIR="$RESULT_DIR"
export INTERNVLA_T3_TASK_NAME="t4_1_camera_${SPLIT}_${VARIANT}_${ATTEMPT}"
export INTERNVLA_T4_CAMERA_HEIGHT_M="$HEIGHT"
export INTERNVLA_T4_CAMERA_PITCH_DOWN_DEG="$PITCH"
export INTERNVLA_T4_CAMERA_HFOV_DEG="$HFOV"
export INTERNVLA_T4_CAMERA_MODEL="$CAMERA_MODEL"
if test -n "$VFOV"; then
  export INTERNVLA_T4_CAMERA_VFOV_DEG="$VFOV"
else
  unset INTERNVLA_T4_CAMERA_VFOV_DEG
fi
export INTERNVLA_T4_CAMERA_AUDIT="$RESULT_DIR/camera_audit.jsonl"
export INTERNVLA_T3_STATIC_CLEARANCE_GATE_M=0.30
export INTERNVLA_T3_MAX_STEP=16000
export ROS2CLI_NO_DAEMON=1

bash "$OVERLAY/run_go2_continuous_phase.sh" canary_no_obstacle
