#!/usr/bin/env bash
# Significance (1000 paired pages per lambda), data-scarce GenPage, Jev scenarios (E8b), report.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python -u"
cd "$ROOT"
[ -f checkpoints/lam1_h100_S_pretrain_frac5.pt ] || $PY -m genpage.train.run --stage pretrain --lam 1 --size S --epochs 10 --frac 0.05 --tag _frac5
[ -f checkpoints/lam1_h100_S_wbc_frac5.pt ]      || $PY -m genpage.train.run --stage wbc --lam 1 --size S --epochs 6 --frac 0.05 --tag _frac5
cd "$ROOT/experiments"
for lam in 1 0 2; do $PY significance.py --lam $lam --n 1000 || echo "!!!! significance lam=$lam failed"; done
$PY run_e8b.py || echo "!!!! run_e8b failed"
$PY make_report.py
echo "=== followups3 done ==="
