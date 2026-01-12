#!/bin/bash
# Master script to run all analysis and generate paper-ready outputs
set -e

echo "========================================"
echo "SALUS Research - Complete Analysis"
echo "========================================"
echo ""

# Step 1: Check if ablation is complete
echo "[1/4] Checking ablation study status..."
if [ -f "results/ablation/ablation_results.csv" ]; then
    echo "  ✓ Ablation results found"
else
    echo "  ⚠ Ablation study still running"
    echo "  Check progress with: ps aux | grep ablate_signals"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Compute missing metrics
echo ""
echo "[2/4] Computing missing metrics (horizons, failure types, latency)..."
python scripts/compute_missing_metrics.py \
    --checkpoint models/salus_predictor_massive.pth \
    --data_path data/massive_collection/20260109_215258/data_20260109_215321.zarr \
    --output_dir results

# Step 3: Create all figures
echo ""
echo "[3/4] Creating all figures..."
python scripts/create_all_figures.py \
    --results_dir results \
    --output_dir figures

# Step 4: Fill results into paper
echo ""
echo "[4/4] Filling results into paper..."
cd paper
python fill_results.py
echo ""

# Step 5: Compile paper
echo "Compiling paper to PDF..."
./compile.sh
echo ""

# Summary
echo "========================================"
echo "✓ Analysis Complete!"
echo "========================================"
echo ""
echo "Generated outputs:"
echo "  • results/missing_metrics.json - Additional metrics"
echo "  • figures/*.pdf - All paper figures"
echo "  • paper/salus_paper.pdf - Final paper"
echo ""
echo "Next steps:"
echo "  1. Review figures: ls -lh figures/"
echo "  2. Check paper: evince paper/salus_paper.pdf"
echo "  3. Review metrics: cat results/missing_metrics.json"
echo ""
