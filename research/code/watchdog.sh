#!/bin/bash
# Stop globalmm-* jobs if available RAM gets dangerously low.
LOG=${1:-$(cd "$(dirname "$0")/.." && pwd)/watchdog.log}
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt 750 ]; then
    echo "$(date +%T) KILL avail=${avail}MB" >> "$LOG"
    systemctl --user stop 'globalmm-*' 2>/dev/null
  elif [ "$avail" -lt 2000 ]; then
    echo "$(date +%T) WARN avail=${avail}MB" >> "$LOG"
  fi
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$t" ] && [ "$t" -ge 93 ]; then
    echo "$(date +%T) GPU-HOT ${t}C, stopping jobs" >> "$LOG"
    systemctl --user stop 'globalmm-*' 2>/dev/null
  fi
  sleep 2
done
