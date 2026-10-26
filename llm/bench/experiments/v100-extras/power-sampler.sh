#!/usr/bin/env bash
# power-sampler.sh — 1 Hz per-GPU power/clocks/temperature samples (H6).
#
# usage: power-sampler.sh OUT.csv [DURATION_S]
#   DURATION_S omitted or "-" samples until killed (the orchestrator kills it).
# Writes one line per second:
#   epoch,gpu0:power,sm_clk,temp,util,mem;gpu1:...;gpu2:...;gpu3:...
# (fields inside a GPU are comma-separated, GPUs are ';'-separated).
set -uo pipefail
out=${1:?out csv}
dur=${2:--}
: > "$out"
start=$(date +%s)
while :; do
  ts=$(date +%s.%N)
  vals=$(nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used \
           --format=csv,noheader,nounits 2>/dev/null | paste -sd';' -)
  printf '%s,%s\n' "$ts" "$vals" >> "$out"
  if [ "$dur" != "-" ] && [ $(( $(date +%s) - start )) -ge "$dur" ]; then
    break
  fi
  sleep 1
done
