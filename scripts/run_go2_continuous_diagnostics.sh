#!/usr/bin/env bash
set -eo pipefail
exec "$(dirname "$0")/run_go2_continuous_phase.sh" diagnostics
