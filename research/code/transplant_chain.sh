#!/bin/bash
# Trained-transplant experiment, unattended: for k in 0 4 12, train then evaluate (resumable, RAM-gated).
G=$(cd "$(dirname "$0")/.." && pwd)
L=$G/analysis/transplant/chain.log
run() {  # name, log, max_tries, python args...
  local name=$1 log=$2 max=$3; shift 3; local n=0
  while ! grep -q "^DONE" "$log" 2>/dev/null; do
    until [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 3000 ]; do sleep 10; done
    n=$((n+1)); echo "$(date +%T) start $name #$n" >> $L
    systemctl --user reset-failed "globalmm-$name-$n.scope" 2>/dev/null
    cd $G/scripts && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True systemd-run --user --scope -q \
      -p MemoryMax=3500M -p MemorySwapMax=0 --unit=globalmm-$name-$n $G/venv-gpu/bin/python "$@" >> "$log" 2>&1
    echo "$(date +%T) exit $name #$n: $?" >> $L
    [ $n -ge $max ] && { echo "giving up $name" >> $L; return; }
    sleep 20
  done
}
for k in 0 4 12; do   # k=12 runs out of 8 GB at micro-batch 4, so it uses 2 x accum 4 (same batch of 8)
  run tptrain$k $G/analysis/transplant/train_k$k.log 30 train_transplant.py train --k $k --steps 3000 $([ $k -ge 12 ] && echo --bs 2 --accum 4)
  run tpeval$k  $G/analysis/transplant/eval_k$k.log 3 train_transplant.py eval --k $k
done
echo "$(date +%T) CHAIN DONE" >> $L
