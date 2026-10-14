#!/bin/bash
# Equal-footing comparison: both methods from scratch, all 118k train2017 images, 14,800 steps x 8,
# lr 3e-4 (warmup 100, cosine), micro-batch 2 x accum 4; then the same held-out evaluation.
G=$(cd "$(dirname "$0")/.." && pwd)
L=$G/analysis/equal/chain.log
run() {  # name, log, max_tries, python args...  (relaunch until DONE; resumes from checkpoints)
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
run eqtp    $G/analysis/equal/train_transplant.log 40 train_transplant.py train --k 4 --data full --steps 14800 --bs 2 --accum 4 --lr 3e-4 --tag eq_transplant_k4
run eqtpev  $G/analysis/equal/eval_transplant.log 3 train_transplant.py eval --k 4 --tag eq_transplant_k4
run eqbase  $G/analysis/equal/train_baseline.log 40 train_hybrid.py --no-words --init qwen --data full --steps 14800 --bs 2 --accum 4 --lr 3e-4 --tag eq_qwen38
run eqbaseev $G/analysis/equal/eval_baseline.log 3 eval_hybrid.py eq_qwen38:nowords
echo "$(date +%T) EQUAL DONE" >> $L
