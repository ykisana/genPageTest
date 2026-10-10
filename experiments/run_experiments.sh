#!/usr/bin/env bash
# Runs every experiment (after training, so the GPU is idle for latency numbers) and builds the report.
set -uo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python -u"
for s in "run_e1.py" "run_e3.py" "run_e4.py" "run_e5.py" "run_e6.py" "run_e7.py" "run_e8.py" "run_e2.py"; do
  echo "################ $s"
  $PY $s || echo "!!!! $s failed"
done
$PY make_report.py
echo "=== experiments done ==="
