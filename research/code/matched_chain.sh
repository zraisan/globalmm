#!/bin/bash
# After the k=12 transplant evaluation: matched-budget baseline of the transfer-learning connector
# (Qwen3.8 tower, fresh connector, same 32k images, 3,000 steps), then evaluate it with the same code.
G=$(cd "$(dirname "$0")/.." && pwd)
L=$G/analysis/transplant/chain.log
until grep -q "^DONE" $G/analysis/transplant/eval_k12.log 2>/dev/null; do sleep 30; done
run() {
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
run matchtrain $G/analysis/transplant/train_matched.log 30 train_hybrid.py --tag matched_qwen38 --steps 3000 --no-words --init qwen --lr 3e-4
run matcheval  $G/analysis/transplant/eval_matched.log 3 eval_hybrid.py matched_qwen38:nowords
echo "$(date +%T) MATCHED DONE" >> $L
