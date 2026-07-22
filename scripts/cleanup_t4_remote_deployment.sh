#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: cleanup_t4_remote_deployment.sh <dgx|dgx_b|isaac|isaac_b> ABSOLUTE_DEPLOYMENT_ROOT" >&2
  exit 64
}

[[ $# -eq 2 ]] || usage
role="$1"
target="$2"

case "$role" in
  dgx)
    remote="railgun@10.100.100.128"
    allowed_prefix="/home/railgun/internnav-t1-t2/.t4-deployments/"
    ;;
  dgx_b)
    remote="rail@10.100.120.116"
    allowed_prefix="/home/railgun/internnav-t1-t2/.t4-deployments/"
    ;;
  isaac)
    remote="song@10.100.120.111"
    allowed_prefix="/home/song/internnav-t1-t2/.t4-deployments/"
    ;;
  isaac_b)
    remote="song@10.100.120.111"
    allowed_prefix="/home/song/internnav-t1-t2/.t4-deployments/"
    ;;
  *) usage ;;
esac

case "$target" in
  "$allowed_prefix"?*) ;;
  *) echo "refusing deployment path outside $allowed_prefix" >&2; exit 64 ;;
esac
[[ "$target" != *$'\n'* && "$target" != *$'\r'* ]]

printf -v remote_command '%s\n' \
  'set -euo pipefail' \
  "target=$(printf '%q' "$target")" \
  "allowed_prefix=$(printf '%q' "$allowed_prefix")" \
  'test -d "$target"' \
  'test ! -L "$target"' \
  'test "$(realpath "$target")" = "$target"' \
  'case "$target" in "$allowed_prefix"?*) ;; *) exit 64 ;; esac' \
  'rm -rf -- "$target"' \
  'test ! -e "$target"'

ssh -T -o BatchMode=yes -o ConnectTimeout=8 \
  -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
  -o StrictHostKeyChecking=accept-new "$remote" "$remote_command"
