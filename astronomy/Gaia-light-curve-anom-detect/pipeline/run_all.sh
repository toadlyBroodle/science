#!/bin/bash
# CV Hunter Pipeline - run all steps sequentially
# Usage: ./run_all.sh [step_number]
#   No args: run all steps
#   With arg: run from that step onwards (e.g., ./run_all.sh 3)

set -e
cd "$(dirname "$0")"
mkdir -p data ../figs/pipeline

START=${1:-1}

run_step() {
    local num=$1 script=$2
    if [ "$num" -ge "$START" ]; then
        echo ""
        echo "========================================"
        echo "  Running step $num: $script"
        echo "========================================"
        python3 "$script"
    fi
}

run_step 1 01_query_gaia.py
run_step 2 02_detect_anomalies.py
run_step 3 03_crossmatch.py
run_step 4 04_prioritize.py
run_step 5 05_tess_lightcurves.py
run_step 6 06_period_analysis.py
run_step 7 07_deep_investigate.py

echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "========================================"
echo "  Data: data/"
echo "  Plots: ../figs/pipeline/"
