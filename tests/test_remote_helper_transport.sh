#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../coordination/remote_helper_transport.sh
source "$root/coordination/remote_helper_transport.sh"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/internnav-helper-test.XXXXXX")"
cleanup() {
  case "$tmp" in
    "${TMPDIR:-/tmp}"/internnav-helper-test.*) rm -rf -- "$tmp" ;;
    *) return 99 ;;
  esac
}
trap cleanup EXIT INT TERM HUP

cat >"$tmp/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
script="$1"
shift
exec "$script" "$@"
SH
cat >"$tmp/exec" <<'SH'
#!/usr/bin/env bash
printf 'exec\t%s\n' "$1" >>"$HELPER_TEST_LOG"
[[ "$1" != fail-exit-23 ]] || exit 23
SH
cat >"$tmp/put" <<'SH'
#!/usr/bin/env bash
printf 'put\t%s\t%s\n' "$1" "$2" >>"$HELPER_TEST_LOG"
SH
cat >"$tmp/get" <<'SH'
#!/usr/bin/env bash
printf 'get\t%s\t%s\n' "$1" "$2" >>"$HELPER_TEST_LOG"
SH
chmod +x "$tmp/python" "$tmp/exec" "$tmp/put" "$tmp/get"
export HELPER_TEST_LOG="$tmp/calls.log"

export ISAAC_EXEC_HELPER="$tmp/exec"
unset ISAAC_PUT_HELPER ISAAC_GET_HELPER
if internnav_helper_init 2>/dev/null; then
  printf 'partial helper configuration was accepted\n' >&2
  exit 1
fi

export ISAAC_PUT_HELPER="$tmp/put"
export ISAAC_GET_HELPER="$tmp/get"
export ISAAC_HELPER_PYTHON="$tmp/python"
internnav_helper_init

internnav_helper_exec "printf 'one argument with spaces'"
set +e
internnav_helper_exec fail-exit-23
rc=$?
set -e
[[ "$rc" -eq 23 ]] || exit 1

printf a >"$tmp/a file"
printf b >"$tmp/b"
internnav_helper_put /home/song/stage "$tmp/a file" "$tmp/b"
internnav_helper_get /home/song/stage/result.tgz "$tmp/result.tgz"

grep -Fx $'exec\tprintf '\''one argument with spaces'\''' "$HELPER_TEST_LOG" >/dev/null
grep -Fx $'exec\tfail-exit-23' "$HELPER_TEST_LOG" >/dev/null
grep -Fx $'put\t'"$tmp/a file"$'\t/home/song/stage/a file' "$HELPER_TEST_LOG" >/dev/null
grep -Fx $'put\t'"$tmp/b"$'\t/home/song/stage/b' "$HELPER_TEST_LOG" >/dev/null
grep -Fx $'get\t/home/song/stage/result.tgz\t'"$tmp/result.tgz" "$HELPER_TEST_LOG" >/dev/null

# Windows Python cannot consume WSL paths directly.  Exercise the conversion
# branch with an identity/logging wslpath shim so the fake Linux executables
# remain runnable while every conversion boundary is still asserted.
mkdir -p "$tmp/bin"
cat >"$tmp/bin/wslpath" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 2 && "$1" == -w ]]
printf '%s\n' "$2" >>"$WSLPATH_TEST_LOG"
printf '%s\n' "$2"
SH
chmod +x "$tmp/bin/wslpath"
ln -s "$tmp/python" "$tmp/python.exe"
export WSLPATH_TEST_LOG="$tmp/wslpath.log"
export PATH="$tmp/bin:$PATH"
export ISAAC_HELPER_PYTHON="$tmp/python.exe"
internnav_helper_init
internnav_helper_exec windows-path
internnav_helper_put /home/song/windows-stage "$tmp/a file"
[[ ! -e "$tmp/windows-result.tgz" ]]
internnav_helper_get /home/song/windows-stage/result.tgz "$tmp/windows-result.tgz"
[[ "$(wc -l <"$WSLPATH_TEST_LOG")" -eq 5 ]]
grep -Fx "$tmp/exec" "$WSLPATH_TEST_LOG" >/dev/null
grep -Fx "$tmp/put" "$WSLPATH_TEST_LOG" >/dev/null
grep -Fx "$tmp/get" "$WSLPATH_TEST_LOG" >/dev/null
grep -Fx "$tmp/a file" "$WSLPATH_TEST_LOG" >/dev/null
grep -Fx "$tmp" "$WSLPATH_TEST_LOG" >/dev/null
grep -Fx $'get\t/home/song/windows-stage/result.tgz\t'"$tmp"'\windows-result.tgz' \
  "$HELPER_TEST_LOG" >/dev/null
printf '{"checks":6,"status":"PASS"}\n'
