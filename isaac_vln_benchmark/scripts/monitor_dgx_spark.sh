#!/usr/bin/env bash
set -eo pipefail

OUT="${1:-/home/railgun/dgx-unitree/test_results/dgx_spark_perf_$(date +%Y%m%d_%H%M%S).csv}"
DURATION_SEC="${DURATION_SEC:-120}"
SAMPLE_SEC="${SAMPLE_SEC:-1}"

mkdir -p "$(dirname "$OUT")"
echo "timestamp,gpu_util_pct,gpu_mem_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_power_w,gpu_temp_c,load1,load5,load15,mem_used_mib,mem_total_mib,internnav_pid,internnav_cpu_pct,internnav_mem_pct,internnav_rss_kib" > "$OUT"

for _ in $(seq 1 "$DURATION_SEC"); do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  gpu_util=""
  gpu_mem_util=""
  gpu_mem_used=""
  gpu_mem_total=""
  gpu_power=""
  gpu_temp=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    IFS=',' read -r gpu_util gpu_mem_util gpu_mem_used gpu_mem_total gpu_power gpu_temp < <(
      nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' '
    )
  fi
  read -r load1 load5 load15 _ < /proc/loadavg
  mem_used="$(free -m | awk '/^Mem:/ {print $3}')"
  mem_total="$(free -m | awk '/^Mem:/ {print $2}')"
  proc_row="$(ps -eo pid,pcpu,pmem,rss,args | awk '/start_server.py/ && !/awk/ {print $1 "," $2 "," $3 "," $4; exit}')"
  if [ -z "$proc_row" ]; then
    proc_row=",,,"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$timestamp" "$gpu_util" "$gpu_mem_util" "$gpu_mem_used" "$gpu_mem_total" "$gpu_power" "$gpu_temp" \
    "$load1" "$load5" "$load15" "$mem_used" "$mem_total" "$proc_row" >> "$OUT"
  sleep "$SAMPLE_SEC"
done

echo "$OUT"
