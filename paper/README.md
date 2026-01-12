# SALUS Research Paper

This directory contains the LaTeX source for the SALUS research paper.

## File Structure

- `salus_paper.tex` - Main paper source
- `compile.sh` - Compilation script
- `fill_results.py` - Helper script to populate results from JSON files

## Compiling the Paper

```bash
cd /home/mpcr/Desktop/SalusV3/SalusTest/paper
./compile.sh
```

This will generate `salus_paper.pdf`.

## Filling in Results

All placeholder values are marked in **red** with format `\textcolor{red}{[XX.X]}`.

When ablation study completes, run:

```bash
python fill_results.py
```

This will automatically populate:
- Table 1: Main performance results from `results/salus_results_massive.json` and `results/baseline_results.json`
- Table 2: Multi-horizon breakdown
- Table 3: Ablation results from `results/ablation/ablation_results.csv`
- Table 4: Per-failure-type performance
- All inline metrics in abstract and text

## Manual Result Locations

If you need to manually fill results, search for `\textcolor{red}` tags:

### Abstract
- Line 22-24: Overall recall, precision, AUROC improvement
- Line 25: False alarm reduction
- Line 26-27: Ablation percentages
- Line 28: Inference latency

### Section 4 (Results)
- Table 1 (line 330): Main performance comparison
- Table 2 (line 345): Multi-horizon analysis
- Table 3 (line 358): Ablation study
- Table 4 (line 378): Per-failure-type breakdown

### Section 4.5 (Implementation)
- Line 395-400: Computational efficiency metrics

## Current Status

✅ **Complete Sections:**
- Abstract (structure ready)
- Introduction (complete)
- Related Work (complete)
- Method (complete - all equations and architecture details)
- Experimental Setup (complete - dataset, training config, baselines)
- Discussion (complete)
- Conclusion (structure ready)

⏳ **Waiting for Results:**
- All numerical results (ablation study running - ~6-7 hours remaining)
- Training time measurement
- Inference latency benchmark

## Citation Format

Currently using plain bibliography style. Can switch to conference format (e.g., IEEE, NeurIPS) as needed.

## Notes

- Paper is **truthful** - only claims what's actually implemented
- No fabricated comparisons to methods we haven't tested
- No claims about real robot experiments (simulation only)
- No RLBench/Bridge benchmarks (can add in revision if needed)
- Conservative about performance claims until numbers confirmed
