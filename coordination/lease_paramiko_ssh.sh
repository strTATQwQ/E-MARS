#!/usr/bin/env bash
set -euo pipefail

# Windows Python owns Paramiko and credentials through a repository-external
# helper.  This wrapper only translates WSL paths and forwards ssh-shaped argv.
[[ -n "${ISAAC_HELPER_PYTHON:-}" ]] || {
  printf 'lease SSH bridge: ISAAC_HELPER_PYTHON is required\n' >&2
  exit 64
}
[[ -n "${ISAAC_EXEC_HELPER:-}" && -r "${ISAAC_EXEC_HELPER:-}" ]] || {
  printf 'lease SSH bridge: ISAAC_EXEC_HELPER is required and must be readable\n' >&2
  exit 64
}
command -v wslpath >/dev/null 2>&1 || {
  printf 'lease SSH bridge: wslpath is unavailable\n' >&2
  exit 69
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
adapter_win="$(wslpath -w "$script_dir/lease_paramiko_ssh.py")"
helper_win="$(wslpath -w "$ISAAC_EXEC_HELPER")"
exec "$ISAAC_HELPER_PYTHON" "$adapter_win" --connect-helper "$helper_win" "$@"
