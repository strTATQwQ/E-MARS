#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 2 || { echo "usage: $0 <fresh-result-dir> <grant-id>" >&2; exit 2; }
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_model_free_sensor_session.sh" \
  --profile completion_sim --result-dir "$1" --grant-id "$2"
