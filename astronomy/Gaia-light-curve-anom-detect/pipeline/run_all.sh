#!/bin/bash
# CV Hunter Pipeline - run all steps sequentially
# Usage: ./run_all.sh [pipeline] [step_number]
#
# Pipelines:
#   v2   - Original variability-only pipeline (default)
#   v3   - Supervised classification: variability + XP spectra
#
# Examples:
#   ./run_all.sh          # run v2 from step 1
#   ./run_all.sh v2 3     # run v2 from step 3
#   ./run_all.sh v3       # run v3 from step 1
#   ./run_all.sh v3 2     # run v3 from step 2

set -e
cd "$(dirname "$0")"

# Parse args
PIPELINE="${1:-v2}"
if [[ "$PIPELINE" =~ ^[0-9]+$ ]]; then
    # First arg is a number — assume v2 with step
    START="$PIPELINE"
    PIPELINE="v2"
else
    START="${2:-1}"
fi

run_step() {
    local num=$1 script=$2 dir=$3
    if [ "$num" -ge "$START" ]; then
        echo ""
        echo "========================================"
        echo "  Running step $num: $script"
        echo "========================================"
        python3 "$dir/$script"
    fi
}

if [ "$PIPELINE" = "v2" ]; then
    echo "=== CV Hunter v2 (variability-only) ==="
    mkdir -p data ../figs/pipeline

    run_step 1 01_query_gaia.py .
    run_step 2 02_detect_anomalies.py .
    run_step 3 03_crossmatch.py .
    run_step 4 04_prioritize.py .
    run_step 5 05_tess_lightcurves.py .
    run_step 6 06_period_analysis.py .
    run_step 7 07_deep_investigate.py .

    echo ""
    echo "========================================"
    echo "  v2 Pipeline complete!"
    echo "========================================"
    echo "  Data:  data/"
    echo "  Plots: ../figs/pipeline/"

elif [ "$PIPELINE" = "v3" ]; then
    echo "=== CV Hunter v3 (supervised classification) ==="
    mkdir -p v3_supervised/data ../figs/v3_supervised

    run_step 1 01_query_sample.py v3_supervised
    run_step 2 02_xp_spectra.py v3_supervised
    run_step 3 03_spectral_features.py v3_supervised
    run_step 4 04_feature_combine.py v3_supervised
    run_step 5 05_anomaly_detect.py v3_supervised
    run_step 6 06_rank_candidates.py v3_supervised
    run_step 7 07_tess_analysis.py v3_supervised

    echo ""
    echo "========================================"
    echo "  v3 Pipeline complete!"
    echo "========================================"
    echo "  Data:  v3_supervised/data/"
    echo "  Plots: ../figs/v3_supervised/"

else
    echo "Unknown pipeline: $PIPELINE"
    echo "Usage: ./run_all.sh [v2|v3] [step_number]"
    exit 1
fi
