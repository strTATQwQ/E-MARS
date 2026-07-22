#!/usr/bin/env bash
set -euo pipefail

# `gate1` selects the one-episode official minimal-start overlay. With no
# argument this runs the frozen five-episode canary.
PHASE="${1:-canary}"
if [[ "$PHASE" == "pilot" && "${INTERNNAV_T0_ALLOW_PILOT:-0}" == "1" ]]; then
  :
elif [[ "$PHASE" != "gate1" && "$PHASE" != "canary" ]]; then
  echo "usage: $0 [gate1|canary]" >&2
  exit 2
fi

CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
DATASET_ROOT="${INTERNNAV_T0_DATASET_ROOT:-$HOME/internnav-t0/episodes/$PHASE}"
SCENE_ROOT="${INTERNNAV_SCENE_ROOT:-$HOME/InternNav/data/scene_data/mp3d_pe}"
EMBODIMENT_ROOT="${INTERNNAV_EMBODIMENT_ROOT:-$HOME/internnav-t0/assets/Embodiments}"
RESULT_DIR="${INTERNNAV_T0_RESULT_DIR:-$CONTROL_ROOT/results/internnav_t0/$PHASE}"
ISAAC_IMAGE="${INTERNNAV_ISAAC_IMAGE:-crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/internrobotics/internnav@sha256:3decaa3bb3c847009c4130f9fa6321331d8d7bad1ed97a85da6901883e9ac2c4}"
ISAAC_MODE="${INTERNNAV_T0_ISAAC_MODE:-official_container}"
: "${INTERNNAV_SERVER_HOST:?set INTERNNAV_SERVER_HOST to the DGX Spark LAN address}"

python3 "$CONTROL_ROOT/scripts/internnav_t0_preflight.py" \
  --role eval \
  --phase "$PHASE" \
  --internnav-root "$INTERNNAV_ROOT" \
  --control-root "$CONTROL_ROOT" \
  --dataset-root "$DATASET_ROOT"

for path in "$SCENE_ROOT" "$EMBODIMENT_ROOT"; do
  [[ -d "$path" ]] || { echo "required asset directory is absent: $path" >&2; exit 1; }
done
mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/sample_episodes" "$RESULT_DIR/keyframes"

run_official_container() {
  docker run --rm --gpus all --network host --entrypoint bash \
    -e INTERNNAV_ROOT=/workspace \
    -e INTERNNAV_T0_PHASE="$PHASE" \
    -e INTERNNAV_T0_DATASET_ROOT=/t0_dataset \
    -e INTERNNAV_T0_RESULT_DIR=/t0_results \
    -e INTERNNAV_SERVER_HOST="$INTERNNAV_SERVER_HOST" \
    -e MESA_GL_VERSION_OVERRIDE=4.6 \
    -v "$INTERNNAV_ROOT:/workspace:ro" \
    -v "$CONTROL_ROOT:/t0_control:ro" \
    -v "$DATASET_ROOT:/t0_dataset:ro" \
    -v "$SCENE_ROOT:/workspace/data/scene_data/mp3d_pe:ro" \
    -v "$EMBODIMENT_ROOT:/workspace/data/Embodiments:ro" \
    -v "$RESULT_DIR:/t0_results" \
    -v "$RESULT_DIR/logs:/workspace/logs" \
    -v "$RESULT_DIR/sample_episodes:/workspace/data/sample_episodes" \
    "$ISAAC_IMAGE" \
    -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate internutopia && cd /workspace && python scripts/eval/eval.py --config /t0_control/configs/internnav_t0/official_agent_server_cfg.py'
}

run_isaac6_compat() {
  local isaac_python="${INTERNNAV_T0_ISAAC_PYTHON:-$HOME/env_isaacsim/bin/python}"
  local isaac_entrypoint="${INTERNNAV_T0_ISAAC_ENTRYPOINT:-$CONTROL_ROOT/scripts/run_internnav_isaac6_entrypoint.py}"
  local config_path="${INTERNNAV_T0_CONFIG_PATH:-$CONTROL_ROOT/configs/internnav_t0/official_agent_server_cfg.py}"
  local compat_overlay="${INTERNNAV_T0_COMPAT_OVERLAY:-$HOME/internnav-t0/runtime/internutopia_2_2_0}"
  [[ -x "$isaac_python" ]] || { echo "Isaac compatibility Python is absent: $isaac_python" >&2; return 1; }
  [[ -f "$isaac_entrypoint" ]] || { echo "Isaac entrypoint is absent: $isaac_entrypoint" >&2; return 1; }
  [[ -f "$config_path" ]] || { echo "evaluation config is absent: $config_path" >&2; return 1; }
  [[ -d "$compat_overlay/internutopia" ]] || {
    echo "InternUtopia compatibility overlay is absent: $compat_overlay" >&2
    return 1
  }

  # The official container bind-mounts the signed Embodiments payload over
  # /workspace/data/Embodiments.  Reproduce that path mapping in host mode with
  # file symlinks while keeping the downloaded payload outside the git tree.
  while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"$EMBODIMENT_ROOT"/}"
    target_file="$INTERNNAV_ROOT/data/Embodiments/$relative_path"
    mkdir -p "$(dirname "$target_file")"
    if [[ -e "$target_file" || -L "$target_file" ]]; then
      if [[ "$(readlink -f "$target_file")" != "$(readlink -f "$source_file")" ]] && \
         ! cmp -s "$target_file" "$source_file"; then
        echo "refusing to replace a different embodiment asset: $target_file" >&2
        return 1
      fi
    else
      ln -s "$source_file" "$target_file"
    fi
  done < <(find "$EMBODIMENT_ROOT" -type f -print0)
  [[ -f "$INTERNNAV_ROOT/data/Embodiments/vln-pe/h1/h1_internvla.usd" ]] || {
    echo "mapped H1 InternVLA USD is absent" >&2
    return 1
  }

  # Isaac Sim 6 is the minimum installed runtime that starts successfully on
  # the RTX 50 / Blackwell client.  Only the runtime and dependency overlay
  # differ; the upstream evaluator and T0 config remain the executed code.
  cd "$INTERNNAV_ROOT"
  env \
    OMNI_KIT_ACCEPT_EULA=Y \
    MESA_GL_VERSION_OVERRIDE=4.6 \
    PYTHONPATH="$compat_overlay:$INTERNNAV_ROOT" \
    INTERNNAV_ROOT="$INTERNNAV_ROOT" \
    INTERNNAV_T0_PHASE="$PHASE" \
    INTERNNAV_T0_DATASET_ROOT="$DATASET_ROOT" \
    INTERNNAV_T0_RESULT_DIR="$RESULT_DIR" \
    INTERNNAV_SERVER_HOST="$INTERNNAV_SERVER_HOST" \
    "$isaac_python" "$isaac_entrypoint" --config "$config_path"
}

case "$ISAAC_MODE" in
  official_container)
    run_official_container 2>&1 | tee "$RESULT_DIR/eval.log"
    ;;
  isaac6_compat)
    run_isaac6_compat 2>&1 | tee "$RESULT_DIR/eval.log"
    ;;
  *)
    echo "unsupported INTERNNAV_T0_ISAAC_MODE: $ISAAC_MODE" >&2
    exit 2
    ;;
esac
