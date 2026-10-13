#!/bin/bash
# Keep the full training run going: start it when RAM allows, resume after any watchdog stop.
G=$(cd "$(dirname "$0")/.." && pwd)
LOG=$G/analysis/grounding/run_full.log
n=0
while ! grep -q "^DONE" "$LOG" 2>/dev/null; do
  until [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 1200 ]; do sleep 10; done
  n=$((n+1))
  echo "$(date +%T) supervisor: launch #$n" >> $G/analysis/grounding/supervisor.log
  systemctl --user reset-failed "globalmm-trainfull-$n.scope" 2>/dev/null
  cd $G/scripts && systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 --unit=globalmm-trainfull-$n \
    $G/venv-gpu/bin/python train_full.py --steps 14800 --tag run_full >> "$LOG" 2>&1
  echo "$(date +%T) supervisor: run #$n exited $?" >> $G/analysis/grounding/supervisor.log
  sleep 30
done
echo "$(date +%T) supervisor: training DONE" >> $G/analysis/grounding/supervisor.log
