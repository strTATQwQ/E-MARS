#!/usr/bin/env bash
set -eo pipefail

# Stable deliverable entry point; the longer historical name remains for
# compatibility with already-captured T1.1 evidence.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_internnav_ros2_replay.sh" "$@"
