#!/usr/bin/env bash
# Runs after run_experiments.sh: training-budget control for the skeleton/parallel arms, then
# paired-bootstrap significance on 1000 test pages per lambda, then rebuilds the report tables.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "=== experiments done ===" logs/experiments.log 2>/dev/null; do sleep 20; done
PY="$(pwd)/.venv/bin/python -u"
[ -f checkpoints/lam1_h100_S_pretrain_ctrl3.pt ] || $PY -m genpage.train.run --stage pretrain --lam 1 --size S --epochs 3 --tag _ctrl3
[ -f checkpoints/lam1_h100_S_wbc_ctrl3.pt ]      || $PY -m genpage.train.run --stage wbc --lam 1 --size S --tag _ctrl3
cd experiments
for lam in 1 0 2; do $PY significance.py --lam $lam --n 1000 || echo "!!!! significance lam=$lam failed"; done
$PY make_report.py
echo "=== followups done ==="
