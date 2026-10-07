#!/usr/bin/env bash
# Regenerate every Phase 1-3 artifact in hpc_transfer/, in order, with the
# default settings. Run this rather than the individual scripts: the report
# generator reads the CSVs the transfer step writes, so a sensitivity sweep
# left running last would otherwise leak its settings into predictions.md.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO"
cd "$REPO"

python3 hpc_transfer/phase1_best_config.py   "$@"
python3 hpc_transfer/meta_features.py
python3 hpc_transfer/phase3_transfer.py
python3 hpc_transfer/phase3_report.py
