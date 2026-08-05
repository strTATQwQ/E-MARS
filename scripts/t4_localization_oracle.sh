#!/usr/bin/env bash
set -euo pipefail

if test "$#" -ne 2; then
  echo "usage: $0 RESULT_DIR GRANT_ID" >&2
  exit 2
fi
RESULT_DIR="$1"
GRANT_ID="$2"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPOSITORY_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
SHARED_RUNNER="$REPOSITORY_ROOT/scripts/run_t4_completion_localization_oracle.sh"
cd "$REPOSITORY_ROOT"

if test "${INTERNNAV_T4_LOCALIZATION_ORACLE_LEASE_ACK:-}" != 1; then
  echo "coordinator lease acknowledgement missing" >&2
  exit 2
fi
test -n "$GRANT_ID"
test ! -e "$RESULT_DIR"

# Fail before consuming a one-shot result directory when the coordinator has
# not yet integrated the reviewed shared lifecycle runner.
if test ! -x "$SHARED_RUNNER"; then
  echo "shared localization Oracle runner is not integrated" >&2
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 python3 \
  "$REPOSITORY_ROOT/scripts/t4_localization_runtime.py" oracle-preflight \
  --repository-root "$REPOSITORY_ROOT" \
  --result-dir "$RESULT_DIR" \
  --grant-id "$GRANT_ID"

PYTHONDONTWRITEBYTECODE=1 python3 \
  "$REPOSITORY_ROOT/scripts/t4_localization_runtime.py" oracle-ref-check \
  --repository-root "$REPOSITORY_ROOT" \
  --result-dir "$RESULT_DIR"

set +e
"$SHARED_RUNNER" \
  --profile completion_sim \
  --target isaac_simulation \
  --selector-config "$REPOSITORY_ROOT/configs/completion_sim/localization/selector.json" \
  --oracle-config "$REPOSITORY_ROOT/configs/completion_sim/localization/oracle_gate.json" \
  --result-dir "$RESULT_DIR" \
  --grant-id "$GRANT_ID" \
  --grant-claim "$RESULT_DIR/localization_grant_claim.json"
RUNNER_STATUS="$?"
set -e

# A long-running Oracle must also end on the exact granted integration ref.
# The shared runner remains responsible for cleanup before returning.
PYTHONDONTWRITEBYTECODE=1 python3 \
  "$REPOSITORY_ROOT/scripts/t4_localization_runtime.py" oracle-ref-check \
  --repository-root "$REPOSITORY_ROOT" \
  --result-dir "$RESULT_DIR"

exit "$RUNNER_STATUS"
