#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:?output CSV path required}"
STOP_FILE="${OUTPUT}.stop"
mkdir -p "$(dirname "${OUTPUT}")"
rm -f "${STOP_FILE}"
printf 'timestamp,gpu_util_pct,power_w,temperature_c,sm_clock_mhz,load1,ram_used_mib,ram_total_mib,llama_rss_kib\n' > "${OUTPUT}"

while [[ ! -e "${STOP_FILE}" ]]; do
  timestamp="$(date --iso-8601=seconds)"
  gpu="$(nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.current.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  load1="$(cut -d' ' -f1 /proc/loadavg)"
  ram="$(free -m | awk 'NR==2{print $3","$2}')"
  rss="$(ps -C llama-server -o rss= | awk '{s+=$1} END{print s+0}')"
  printf '%s,%s,%s,%s,%s\n' "${timestamp}" "${gpu}" "${load1}" "${ram}" "${rss}" >> "${OUTPUT}"
  sleep 1
done
