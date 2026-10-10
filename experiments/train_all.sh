#!/usr/bin/env bash
# Trains every checkpoint the experiments need, in priority order. Several hours on an M3.
# Skips anything whose checkpoint already exists, so it can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
have() { [ -f "checkpoints/lam$1_h100_$2_$3.pt" ]; }
run()  { have "$2" "$3" "$1" || { echo "=== $1 lam=$2 $3 ==="; $PY -m genpage.train.run --stage "$1" --lam "$2" --size "$3" ${4:-}; }; }
arm()  { have "$2" S "$1" || { echo "=== arm $1 lam=$2 ==="; $PY -m genpage.train.arms --arm "$1" --lam "$2"; }; }
for lam in 1 0 2; do
  run pretrain $lam S "--epochs 2"
  run wbc      $lam S
done
for a in mtp skeleton parallel_indep parallel_joint; do arm $a 1; done
run wbc_scratch 1 S                 # Table-1 ablation
run pretrain 1 M "--epochs 2"       # latency / scale arm
run wbc      1 M
for lam in 0 2; do for a in parallel_indep parallel_joint; do arm $a $lam; done; done
echo "=== all done ==="
