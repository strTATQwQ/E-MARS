#!/usr/bin/env bash
set -euo pipefail

# T0.0 uses only the released OVON slow-fast runner.  It intentionally refuses
# R2R/RxR manifests, the visual-only OmniNav checkpoint, and missing licensed
# HM3D assets.  Passwords and access tokens are never accepted as arguments.

REPO="${OMNINAV_REPO:-/home/railgun/ai-stack/src/OmniNav}"
MODEL="${OMNINAV_SLOWFAST_MODEL_PATH:-/home/railgun/ai-stack/models/OmniNav_Slowfast}"
DATA_ROOT="${OMNINAV_OVON_DATA_ROOT:-/home/railgun/ai-stack/data/omninav_ovon}"
RUNNER_PYTHON="${OMNINAV_HABITAT_PYTHON:-/home/railgun/ai-stack/venvs/omninav-ovon/bin/python}"
OUTPUT="${OMNINAV_T0_OUTPUT:-$PWD/results/omninav_t0/t0_0_habitat}"
EPISODES="${OMNINAV_T0_EPISODES:-6}"
TIMEOUT_SECONDS="${OMNINAV_T0_TIMEOUT_SECONDS:-3600}"

while (($#)); do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --python) RUNNER_PYTHON="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT"
ERRORS_FILE="$OUTPUT/preflight_errors.txt"
: > "$ERRORS_FILE"
fail() { printf '%s\n' "$1" >> "$ERRORS_FILE"; }

if ! [[ "$EPISODES" =~ ^[0-9]+$ ]] || ((EPISODES < 5 || EPISODES > 10)); then
  fail "episode count must be between 5 and 10"
fi

RUNNER="$REPO/infer_ovon_slowfast/run_nav_ovon_omni.py"
QWEN_UTILS="$REPO/infer_ovon_slowfast/qwen_utils.py"
RUNNER_UTILS="$REPO/infer_ovon_slowfast/utils.py"
[[ -f "$RUNNER" ]] || fail "missing official OVON slow-fast runner"
[[ -f "$QWEN_UTILS" ]] || fail "missing official OVON qwen_utils.py"
[[ -f "$RUNNER_UTILS" ]] || fail "missing official OVON utils.py"
if [[ -f "$RUNNER" ]]; then
  actual="$(sha256sum "$RUNNER" | awk '{print $1}')"
  [[ "$actual" == "308bd370b3e5fcf284fd9579c935751c29c0925c3280e91aeb5a848adf3f44a5" ]] || fail "runner source hash mismatch"
fi
if [[ -f "$QWEN_UTILS" ]]; then
  actual="$(sha256sum "$QWEN_UTILS" | awk '{print $1}')"
  [[ "$actual" == "7ca7aa43d9b38b77634c7439f30421b57a5b374adc88eac3a3b166dbc1673f99" ]] || fail "qwen_utils source hash mismatch"
fi
if [[ -f "$RUNNER" && -f "$RUNNER_UTILS" ]]; then
  if grep -q 'args\.type' "$RUNNER" && ! grep -q "add_argument.*--fast_type" "$RUNNER_UTILS"; then
    fail "official runner reads args.type but parser does not define --fast_type"
  fi
  if grep -q "add_argument('--model_path', type=int" "$RUNNER_UTILS"; then
    fail "official parser declares the checkpoint path as type=int"
  fi
fi

declare -A EXPECTED_SHARDS=(
  [model-00001-of-00002.safetensors]=67c1faef206c3ae15d16e7851ffb2540059f0622e89bb1168ae73d638b367f74
  [model-00002-of-00002.safetensors]=08392f3a27df28478a2774493388424a3ff498e2461994778e08485d6e14b0da
)
for shard in "${!EXPECTED_SHARDS[@]}"; do
  if [[ ! -f "$MODEL/$shard" ]]; then
    fail "missing Slowfast checkpoint shard: $shard"
  else
    actual="$(sha256sum "$MODEL/$shard" | awk '{print $1}')"
    [[ "$actual" == "${EXPECTED_SHARDS[$shard]}" ]] || fail "Slowfast checkpoint shard hash mismatch: $shard"
  fi
done

for path in \
  dataset/hm3d/val \
  dataset/embodied_scan \
  dataset/embodied_bench_data/embodied_bench_data/ovon \
  dataset/embodied_bench_data/embodied_bench_data/our-set/ovon_full_set.json; do
  [[ -e "$DATA_ROOT/$path" ]] || fail "missing OVON prerequisite: $path"
done

if [[ ! -x "$RUNNER_PYTHON" ]]; then
  fail "missing OVON Habitat Python executable"
else
  if ! "$RUNNER_PYTHON" - <<'PY' >"$OUTPUT/python_imports.log" 2>&1
import habitat
import habitat_sim
import torch
import transformers
assert str(getattr(habitat_sim, "__version__", "0.2.3")).startswith("0.2.3")
print("imports_ok", torch.__version__, transformers.__version__)
PY
  then
    fail "Habitat/model Python import contract failed"
  fi
fi

python3 - "$MODEL" "$OUTPUT/checkpoint_contract.json" <<'PY'
import json, pathlib, sys
model = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
required_tokens = ["<input_pos1>", "<input_pos2>", "<input_pos3>", "<input_pos4>", "<input_pos5>", "<input_target>"]
required_keys = ["query_action", "wp_predictor", "wp_predictor_angle", "arrive_predictor", "input_wp_encoder"]
texts = ""
for name in ("added_tokens.json", "special_tokens_map.json", "tokenizer_config.json"):
    path = model / name
    if path.is_file():
        texts += path.read_text(encoding="utf-8")
index_path = model / "model.safetensors.index.json"
keys = list(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]) if index_path.is_file() else []
row = {
    "task_family": "OVON",
    "special_tokens": {token: token in texts for token in required_tokens},
    "action_head_keys": {key: any(key in candidate for candidate in keys) for key in required_keys},
}
row["valid"] = all(row["special_tokens"].values()) and all(row["action_head_keys"].values())
out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if [[ ! -f "$OUTPUT/checkpoint_contract.json" ]] || ! grep -q '"valid": true' "$OUTPUT/checkpoint_contract.json"; then
  fail "processor special-token or action-head contract failed"
fi

if [[ -s "$ERRORS_FILE" ]]; then
  python3 - "$ERRORS_FILE" "$OUTPUT/result.json" <<'PY'
import json, pathlib, sys, time
errors = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "schema_version": 1, "gate": "T0.0", "status": "BLOCKED",
    "task_family": "OVON", "completed_episodes": 0, "successes": 0,
    "errors": errors, "timestamp_s": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "T0.0 BLOCKED; see $OUTPUT/result.json" >&2
  exit 3
fi

# The released point-goal branch references `rot` before assignment.  Do not
# silently switch to A-star because that would bypass the Fast action head.
# A future audited source patch must be supplied explicitly and its hash logged.
if grep -q 'rot=rot' "$RUNNER" && ! grep -q '^rot[[:space:]]*=' "$RUNNER"; then
  fail "official point-goal runner has uninitialized rot; audited patch required"
  python3 - "$ERRORS_FILE" "$OUTPUT/result.json" <<'PY'
import json, pathlib, sys, time
errors = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "schema_version": 1, "gate": "T0.0", "status": "BLOCKED",
    "task_family": "OVON", "completed_episodes": 0, "successes": 0,
    "errors": errors, "timestamp_s": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit 3
fi

echo "Preflight prerequisites passed. Episode materialization and official run are intentionally gated on an audited point-goal source patch." >&2
exit 4
