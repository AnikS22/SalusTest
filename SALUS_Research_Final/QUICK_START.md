# SALUS Research - Quick Start Guide

## What's in This Folder?

This is your **one consolidated folder** with everything needed for the SALUS research paper:

```
SALUS_Research_Final/
├── paper/          → NeurIPS 2025 formatted paper
├── data/           → Link to 5GB dataset (5K episodes)
├── results/        → All evaluation results (SALUS, baselines, ablation)
├── models/         → Trained SALUS model (284KB)
├── scripts/        → Analysis scripts (evaluation, figures, metrics)
├── figures/        → Paper figures (to be generated)
└── run_all_analysis.sh  → Master script (runs everything)
```

---

## Current Status

### ✅ Complete
- [x] **Dataset**: 5,000 episodes, 1M timesteps, 8% failure rate
- [x] **SALUS Model**: Trained (70,672 params), 0.8833 AUROC
- [x] **Paper**: NeurIPS 2025 format, all sections written
- [x] **Scripts**: All analysis tools ready

### ⏳ Running
- [ ] **Ablation Study**: Testing 7 signal configurations (4-5 hours remaining)
  - Check: `ps aux | grep ablate_signals`

### ❌ To Do
- [ ] Compute missing metrics (multi-horizon, failure types, latency)
- [ ] Create all figures (10 figures planned)
- [ ] Fill results into paper
- [ ] Final review

---

## Quick Commands

### 1. View the Paper

```bash
cd paper
./compile.sh
evince salus_paper.pdf  # or xdg-open salus_paper.pdf
```

### 2. Check Current Results

```bash
# SALUS performance
cat results/salus_results_massive.json
# Output: AUROC=0.8833, Recall=51.56%, FAR=6.39%

# Baseline comparison
cat results/baseline_results_massive.json
# Output: Random baseline AUROC=0.5006
```

### 3. Check Ablation Progress

```bash
# Is it still running?
ps aux | grep ablate_signals

# Check output (if running)
tail /tmp/claude/-home-mpcr-Desktop-SalusV3/tasks/b72ae80.output

# View results (when complete)
cat results/ablation/ablation_results.csv
```

### 4. Run Complete Analysis (When Ready)

```bash
# This will:
# - Compute missing metrics
# - Create all figures
# - Fill results into paper
# - Compile final PDF
./run_all_analysis.sh
```

---

## Step-by-Step Workflow

### Phase 1: Wait for Ablation (Automatic)
**Status**: Running in background
**Time**: ~4-5 hours
**Check**: `ps aux | grep ablate`

### Phase 2: Generate Missing Metrics (15 minutes)
**When**: After ablation completes
**Run**:
```bash
python scripts/compute_missing_metrics.py \
    --checkpoint models/salus_predictor_massive.pth \
    --data_path data/massive_collection/20260109_215258/data_20260109_215321.zarr
```

**Output**: `results/missing_metrics.json`

### Phase 3: Create Figures (Auto)
**Run**:
```bash
python scripts/create_all_figures.py \
    --results_dir results \
    --output_dir figures
```

**Output**:
- `figures/fig1_roc_curves.pdf`
- `figures/fig2_confusion_matrix.pdf`
- `figures/fig3_ablation_study.pdf`
- `figures/fig4_method_comparison.pdf`

### Phase 4: Fill Paper & Compile
**Run**:
```bash
cd paper
python fill_results.py  # Fills placeholders with actual numbers
./compile.sh            # Compiles to PDF
```

**Output**: `paper/salus_paper.pdf` (submission-ready)

---

## Master Script (Runs Everything)

Instead of running each step manually, use:

```bash
./run_all_analysis.sh
```

This automatically:
1. Checks ablation status
2. Computes missing metrics
3. Creates all figures
4. Fills results into paper
5. Compiles final PDF

---

## What Each Script Does

### Analysis Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `evaluate_salus.py` | Evaluate SALUS on test set | 5 min |
| `evaluate_baseline_threshold.py` | Baseline comparison | 5 min |
| `ablate_signals.py` | Ablation study (7 configs) | 4-5 hours |
| `compare_methods.py` | Compare all methods | 10 min |

### Figure Generation Scripts

| Script | Purpose | Creates |
|--------|---------|---------|
| `create_all_figures.py` | Master figure generator | All 4-6 figures |
| `compute_missing_metrics.py` | Horizon/type metrics | `missing_metrics.json` |
| `generate_paper_tables.py` | LaTeX tables | `.tex` files |

### Paper Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `fill_results.py` | Fill placeholders in paper | Updated `.tex` |
| `compile.sh` | Compile LaTeX to PDF | `salus_paper.pdf` |

---

## Data Overview

### Dataset Location
```bash
ls -lh data/massive_collection/20260109_215258/
# Output: data_20260109_215321.zarr (5.0 GB)
```

### Dataset Contents (Zarr Arrays)
```python
import zarr
data = zarr.open('data/massive_collection/20260109_215258/data_20260109_215321.zarr')

print(data.tree())
# signals:        (5000, 200, 12)  - 12D uncertainty signals
# horizon_labels: (5000, 200, 16)  - Multi-horizon failure labels
# actions:        (5000, 200, 7)   - Robot actions
# states:         (5000, 200, N)   - Proprioceptive states
# images:         (5000, 200, 256, 256, 3)  - RGB observations
```

### Model Checkpoint
```bash
ls -lh models/salus_predictor_massive.pth
# 284 KB - Contains model weights + training metadata
```

---

## Paper Structure

The paper is **complete** except for numerical results:

### ✅ Complete Sections
1. **Introduction** - Motivation, problem statement
2. **Related Work** - Literature review
3. **Problem Formulation** - Mathematical definitions
4. **Method** - All 12 signals + predictor architecture
5. **Experimental Setup** - Full specifications
6. **Discussion** - Analysis and limitations
7. **Conclusion** - Summary

### ⏳ Pending Sections
- **Results Tables** (need data):
  - Table 1: SALUS vs Baselines
  - Table 2: Multi-horizon breakdown
  - Table 3: Ablation study
  - Table 4: Per-failure-type analysis

### ❌ Missing Elements
- **Figures** (10 planned, see `paper/DATA_AND_FIGURES.md`)

---

## Troubleshooting

### Ablation Taking Too Long?
```bash
# Check GPU usage
nvidia-smi

# Check actual progress
tail -f /tmp/claude/-home-mpcr-Desktop-SalusV3/tasks/b72ae80.output | grep "Epoch"
```

### "Module not found" Errors?
```bash
# Ensure you're in the SalusTest root
cd /home/mpcr/Desktop/SalusV3/SalusTest

# Then run scripts with:
python scripts/evaluate_salus.py
```

### Paper Won't Compile?
```bash
cd paper
# Check for errors
pdflatex -interaction=nonstopmode salus_paper.tex | grep "Error"

# Install missing packages (if needed)
sudo apt-get install texlive-full
```

---

## Timeline to Submission

**Today (Evening)**
- Ablation completes: ~6-7 PM
- Compute missing metrics: +15 min
- Fix baseline implementations: +1 hour

**Tomorrow (Full Day)**
- Create all figures: 8-10 hours
- Fill results into paper: 2 hours
- Final review: 2 hours

**Total**: ~24 hours from now → Submission-ready

---

## Next Immediate Actions

1. **Monitor ablation progress**:
   ```bash
   watch -n 60 'ps aux | grep ablate_signals'
   ```

2. **When ablation finishes**, run:
   ```bash
   ./run_all_analysis.sh
   ```

3. **Review outputs**:
   ```bash
   ls -lh figures/           # Check figures
   evince paper/salus_paper.pdf  # Review paper
   ```

4. **Submit!**

---

## Questions?

- **Paper details**: See `paper/PAPER_SUMMARY.md`
- **Data status**: See `paper/DATA_AND_FIGURES.md`
- **NeurIPS format changes**: See `paper/CHANGES_NEURIPS_STYLE.md`

---

**Last Updated**: January 12, 2026
**Location**: `/home/mpcr/Desktop/SalusV3/SalusTest/SALUS_Research_Final/`
