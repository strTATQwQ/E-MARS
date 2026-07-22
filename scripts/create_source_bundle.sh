#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s REF OUTPUT_TAR_GZ [GZIP_LEVEL]\n' "${0##*/}" >&2
  exit 64
}

[[ $# -ge 2 && $# -le 3 ]] || usage
ref="$1"
output="$2"
gzip_level="${3:-1}"
[[ "$gzip_level" =~ ^[1-9]$ ]] || usage

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
git_command=(git -C "$root")
frontend_git_command=(git -C "$root/frontend")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
  frontend_git_command=(git.exe -C "$(wslpath -w "$root/frontend")")
fi

"${git_command[@]}" cat-file -e "$ref^{commit}"
frontend_entry="$("${git_command[@]}" ls-tree "$ref" frontend | tr -d '\r')"
read -r frontend_mode frontend_kind frontend_sha frontend_path <<<"$frontend_entry"
test "$frontend_mode" = 160000
test "$frontend_kind" = commit
test "$frontend_path" = frontend
test -f "$root/frontend/pyproject.toml" || {
  echo "frontend submodule is not initialized" >&2
  exit 2
}
"${frontend_git_command[@]}" cat-file -e "$frontend_sha^{commit}"

temporary="$(mktemp -d)"
cleanup() {
  rm -rf -- "$temporary"
}
trap cleanup EXIT INT TERM HUP

"${git_command[@]}" archive --format=tar "$ref" | tar -C "$temporary" -xf -
mkdir -p "$temporary/frontend"
"${frontend_git_command[@]}" archive --format=tar "$frontend_sha" |
  tar -C "$temporary/frontend" -xf -

mkdir -p "$(dirname -- "$output")"
commit_time="$("${git_command[@]}" show -s --format=%ct "$ref" | tr -d '\r')"
tar --sort=name --mtime="@$commit_time" --owner=0 --group=0 --numeric-owner \
  -C "$temporary" -cf - . | gzip -n "-$gzip_level" >"$output"
