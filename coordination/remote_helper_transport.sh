#!/usr/bin/env bash
# Credential-bearing helpers live outside every repository/worktree.  This
# adapter passes only command/file arguments and never reads or logs secrets.

internnav_helper_requested() {
  [[ -n "${ISAAC_EXEC_HELPER:-}${ISAAC_PUT_HELPER:-}${ISAAC_GET_HELPER:-}" ]]
}

internnav_helper_init() {
  internnav_helper_requested || return 1
  if [[ -z "${ISAAC_EXEC_HELPER:-}" || -z "${ISAAC_PUT_HELPER:-}" || -z "${ISAAC_GET_HELPER:-}" ]]; then
    printf 'remote helper transport: exec/put/get helpers must be provided together\n' >&2
    return 64
  fi
  INTERNNAV_HELPER_PYTHON="${ISAAC_HELPER_PYTHON:-python.exe}"
  command -v "$INTERNNAV_HELPER_PYTHON" >/dev/null 2>&1 || {
    printf 'remote helper transport: helper Python is unavailable\n' >&2
    return 64
  }
  local helper
  for helper in "$ISAAC_EXEC_HELPER" "$ISAAC_PUT_HELPER" "$ISAAC_GET_HELPER"; do
    [[ -f "$helper" && -r "$helper" ]] || {
      printf 'remote helper transport: configured helper is unreadable\n' >&2
      return 64
    }
  done
  INTERNNAV_HELPER_WINDOWS=0
  case "${INTERNNAV_HELPER_PYTHON,,}" in
    *.exe)
      command -v wslpath >/dev/null 2>&1 || {
        printf 'remote helper transport: wslpath is required for Windows Python\n' >&2
        return 64
      }
      INTERNNAV_HELPER_WINDOWS=1
      ;;
  esac
  INTERNNAV_HELPER_EXEC="$(internnav_helper_python_path "$ISAAC_EXEC_HELPER")" || return
  INTERNNAV_HELPER_PUT="$(internnav_helper_python_path "$ISAAC_PUT_HELPER")" || return
  INTERNNAV_HELPER_GET="$(internnav_helper_python_path "$ISAAC_GET_HELPER")" || return
}

internnav_helper_python_path() {
  [[ $# -eq 1 && -n "$1" ]] || return 64
  if [[ "${INTERNNAV_HELPER_WINDOWS:-0}" == 1 ]]; then
    wslpath -w "$1"
  else
    printf '%s\n' "$1"
  fi
}

internnav_helper_python_output_path() {
  [[ $# -eq 1 && -n "$1" ]] || return 64
  if [[ "${INTERNNAV_HELPER_WINDOWS:-0}" != 1 ]]; then
    printf '%s\n' "$1"
    return
  fi
  local parent base python_parent
  parent="$(dirname -- "$1")"
  base="$(basename -- "$1")"
  [[ -d "$parent" ]] || return 66
  [[ -n "$base" && "$base" != . && "$base" != .. && "$base" != *\\* ]] || return 64
  python_parent="$(wslpath -w "$parent")" || return
  case "$python_parent" in
    *\\) printf '%s%s\n' "$python_parent" "$base" ;;
    *) printf '%s\\%s\n' "$python_parent" "$base" ;;
  esac
}

internnav_helper_exec() {
  [[ $# -eq 1 && -n "$1" ]] || return 64
  "$INTERNNAV_HELPER_PYTHON" "$INTERNNAV_HELPER_EXEC" "$1"
}

internnav_helper_put() {
  [[ $# -ge 2 && "$1" == /home/song/* ]] || return 64
  local remote_dir="$1" local_file python_file
  shift
  for local_file in "$@"; do
    [[ -f "$local_file" ]] || return 66
    python_file="$(internnav_helper_python_path "$local_file")" || return
    "$INTERNNAV_HELPER_PYTHON" "$INTERNNAV_HELPER_PUT" \
      "$python_file" "$remote_dir/$(basename "$local_file")"
  done
}

internnav_helper_get() {
  local python_file
  [[ $# -eq 2 && "$1" == /home/song/* && -n "$2" ]] || return 64
  python_file="$(internnav_helper_python_output_path "$2")" || return
  "$INTERNNAV_HELPER_PYTHON" "$INTERNNAV_HELPER_GET" "$1" "$python_file"
}
