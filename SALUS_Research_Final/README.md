# SALUS Research - Final Consolidated Folder

This folder contains everything needed for the SALUS research paper submission.

## Structure

```
SALUS_Research_Final/
├── paper/                  # NeurIPS 2025 formatted paper
│   ├── salus_paper.tex    # Main paper (NeurIPS style)
│   ├── neurips_2025.sty   # Conference style file
│   ├── compile.sh         # Compile to PDF
│   ├── fill_results.py    # Auto-fill results from JSON
│   ├── DATA_AND_FIGURES.md # Data status & figure planning
│   └── PAPER_SUMMARY.md   # Complete paper summary
│
├── data/                   # Experimental data
│   └── massive_collection/ -> ../paper_data/massive_collection (5GB)
│       └── 20260109_215258/
│           └── data_20260109_215321.zarr  # 5K episodes, 1M timesteps
│
├── results/               # Evaluation results
│   ├── salus_results_massive.json        # SALUS: 0.8833 AUROC
│   ├── baseline_results_massive.json     # Baseline comparison
│   └── ablation/                          # Ablation study (running)
│       └── ablation_results.csv
│
├── models/                # Trained models
│   └── salus_predictor_massive.pth       # Best checkpoint (70,672 params)
│
├── scripts/               # Analysis & visualization scripts
│   ├── evaluate_salus.py              # SALUS evaluation
│   ├── evaluate_baseline_threshold.py # Baseline comparison
│   ├── ablate_signals.py              # Ablation study
│   ├── compare_methods.py             # Method comparison
│   ├── generate_paper_tables.py       # Generate LaTeX tables
│   ├── train_simple.py                # Training script
│   └── collect_data_franka.py         # Data collection
│
└── figures/               # Paper figures (to be generated)
    └── (empty - figures to be created)

```

## Quick Start

### 1. Compile the Paper

```bash
cd paper
./compile.sh
# Output: salus_paper.pdf
```

### 2. Check Current Results

```bash
cat results/salus_results_massive.json
# AUROC: 0.8833, Recall: 51.56%, False Alarm Rate: 6.39%
```

### 3. Monitor Ablation Study (Currently Running)

```bash
# Check progress
ps aux | grep ablate_signals

# View current results
cat results/ablation/ablation_results.csv
```

### 4. Fill Paper Results (After ablation completes)

```bash
cd paper
python fill_results.py
./compile.sh
```

## Data Summary

### ✅ Complete
- **Dataset**: 5,000 episodes, 1M timesteps, 8% failure rate
- **SALUS Model**: Trained 100 epochs, best val loss 0.0653
- **Evaluation**: AUROC 0.8833 on test set
- **Baseline**: Random baseline (AUROC 0.5006)

### ⏳ Running
- **Ablation Study**: Testing 7 signal configurations (4-5 hours remaining)

### ❌ Missing
- Multi-horizon breakdown (per-horizon metrics)
- Per-failure-type breakdown
- Inference latency benchmark
- Complete baseline implementations (entropy, action variance)

## Paper Status

**Structure**: ✅ Complete (NeurIPS 2025 format)
**Content**: ✅ All technical sections written
**Results**: ⏳ Awaiting final experiments
**Figures**: ❌ 10 figures planned (see paper/DATA_AND_FIGURES.md)

## Next Steps

1. **Wait for ablation to complete** (~4-5 hours)
2. **Fix baseline implementations** (entropy, action variance)
3. **Compute missing metrics** (horizons, failure types, latency)
4. **Create figures** (system overview, ROC curves, etc.)
5. **Fill results into paper** (`python fill_results.py`)
6. **Final review and submit**

## Timeline to Submission

- **Today (evening)**: Ablation completes, fix baselines
- **Tomorrow**: Create figures (8-10 hours), fill results (2 hours)
- **Total**: ~24 hours to submission-ready

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `paper/salus_paper.tex` | NeurIPS 2025 paper | ✅ Complete (needs results) |
| `data/massive_collection/` | Raw data (5GB) | ✅ Complete |
| `results/salus_results_massive.json` | SALUS metrics | ✅ Complete |
| `results/ablation/ablation_results.csv` | Ablation study | ⏳ Running |
| `models/salus_predictor_massive.pth` | Trained model | ✅ Complete |
| `scripts/generate_paper_tables.py` | Create LaTeX tables | ✅ Ready to run |

## Contact

For questions about this research, see paper/README.md

---

**Last Updated**: January 12, 2026
**Research**: SALUS - Safety Assurance via Learning from Uncertainty Signals for VLA Models

---

## System Requirements

### For Running Analysis Scripts
- Python 3.8+
- PyTorch
- NumPy, scikit-learn, matplotlib, seaborn
- IsaacLab (for data collection only)

### For Compiling Paper
- LaTeX distribution (texlive-full or equivalent)
- To install on Ubuntu:
  ```bash
  sudo apt-get install texlive-full
  ```

If LaTeX is not installed, the paper can still be edited in any text editor, and you can compile it online using Overleaf.

