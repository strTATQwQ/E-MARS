#!/usr/bin/env bash
set -eo pipefail

# Run on the DGX Spark.  This deliberately uses the model venv's Python so the
# generated Jazzy messages and the frozen InternVLA model share one process.
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
MODEL_PYTHON="${INTERNVLA_MODEL_PYTHON:-$HOME/internnav-t0/venv-model/bin/python}"
RESULT_DIR="${INTERNVLA_MODEL_RESULT_DIR:-$CONTROL_ROOT/results/t1_1_real_model}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-jazzy}"

test -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
test -f "$ROS_WS/install/setup.bash"
test -x "$MODEL_PYTHON"
test -f "$INTERNNAV_ROOT/scripts/eval/configs/h1_internvla_n1_async_cfg.py"
mkdir -p "$RESULT_DIR"

# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
# shellcheck disable=SC1090
source "$ROS_WS/install/setup.bash"
set -u

export INTERNNAV_ROOT
export INTERNVLA_BACKEND="${INTERNVLA_BACKEND:-real}"
# Keep the formal episode/reset barrier pristine while loading the GB10
# checkpoint before Isaac starts its deadline-bound evaluator.
export INTERNVLA_PRELOAD_MODEL="${INTERNVLA_PRELOAD_MODEL:-1}"
export PYTHONUNBUFFERED=1
cd "$INTERNNAV_ROOT"
exec "$MODEL_PYTHON" -m internvla_ros2.model_node "$@" \
  2>&1 | tee "$RESULT_DIR/model_node.log"
