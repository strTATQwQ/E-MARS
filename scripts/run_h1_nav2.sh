#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-oracle}"
case "$MODE" in
  oracle) exec "$SCRIPT_DIR/run_h1_nav2_oracle.sh" ;;
  canary|pilot) exec "$SCRIPT_DIR/run_h1_nav2_active.sh" "$MODE" ;;
  *) echo "usage: $0 {oracle|canary|pilot}" >&2; exit 2 ;;
esac
