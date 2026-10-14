#!/bin/bash
# Keep the audio training run going: start it when RAM allows, resume after any watchdog stop.
G=$(cd "$(dirname "$0")/.." && pwd)
LOG=$G/analysis/audio/a1.log
n=100  # numbering restarted after raising the start threshold
while ! grep -q "^DONE" "$LOG" 2>/dev/null; do
  until [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 3000 ]; do sleep 10; done
  n=$((n+1))
  echo "$(date +%T) supervisor: launch #$n" >> $G/analysis/audio/supervisor.log
  systemctl --user reset-failed "globalmm-audio-$n.scope" 2>/dev/null
  cd $G/scripts && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True systemd-run --user --scope -q \
    -p MemoryMax=3500M -p MemorySwapMax=0 --unit=globalmm-audio-$n \
    $G/venv-gpu/bin/python train_audio.py train --steps 8000 --tag a1 >> "$LOG" 2>&1
  echo "$(date +%T) supervisor: run #$n exited $?" >> $G/analysis/audio/supervisor.log
  sleep 30
done
echo "$(date +%T) supervisor: training DONE" >> $G/analysis/audio/supervisor.log
