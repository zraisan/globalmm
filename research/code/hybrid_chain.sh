#!/bin/bash
# Hybrid experiment, unattended: wait for the decode cache, train hybrid + control (resumable), evaluate.
G=$(cd "$(dirname "$0")/.." && pwd)
L=$G/analysis/hybrid/chain.log
until grep -q "^DONE" $G/analysis/hybrid/prep_train.log 2>/dev/null; do sleep 30; done
run() {  # name, log, python args...  (relaunch until the log says DONE; resumes from checkpoints)
  local name=$1 log=$2; shift 2; local n=0
  while ! grep -q "^DONE" "$log" 2>/dev/null; do
    until [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 3000 ]; do sleep 10; done
    n=$((n+1)); echo "$(date +%T) start $name #$n" >> $L
    systemctl --user reset-failed "globalmm-$name-$n.scope" 2>/dev/null
    cd $G/scripts && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True systemd-run --user --scope -q \
      -p MemoryMax=3500M -p MemorySwapMax=0 --unit=globalmm-$name-$n $G/venv-gpu/bin/python "$@" >> "$log" 2>&1
    echo "$(date +%T) exit $name #$n: $?" >> $L
    [ $n -ge 30 ] && { echo "giving up $name" >> $L; return; }
    sleep 20
  done
}
run hybtrain $G/analysis/hybrid/train_hybrid.log train_hybrid.py --tag hybrid --steps 3000
run hybctrl  $G/analysis/hybrid/train_control.log train_hybrid.py --tag control --steps 3000 --no-words
run hybeval  $G/analysis/hybrid/eval.log eval_hybrid.py hybrid:words control:nowords run_full:nowords
echo "$(date +%T) CHAIN DONE" >> $L
