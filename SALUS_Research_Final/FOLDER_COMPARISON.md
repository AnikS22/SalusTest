# SALUS Research Folder Comparison

## Why This Folder Was Created

You had **multiple paper folders** scattered across the codebase with duplicate/old versions. This caused confusion about which was the "final" version.

## What Existed Before

```
SalusTest/
├── paper/                          # NEW (created today) - NeurIPS 2025 format
├── SalusTest/paper/                # OLD (Jan 8) - previous format
├── docs/papers/                    # VERY OLD - pre-consolidation
├── paper_data/                     # DATA (5GB) - scattered
├── results/                        # RESULTS - scattered
├── scripts/                        # SCRIPTS - scattered
└── /home/mpcr/Desktop/SalusV3/checkpoints/  # MODELS - outside folder
```

### Problems:
1. **Multiple paper versions** - Which is correct?
2. **Scattered data** - Data, results, scripts in different places
3. **No clear workflow** - How to go from data → paper?
4. **Missing scripts** - No figure generation, metric computation

---

## What SALUS_Research_Final Contains

### Single Source of Truth

All research materials consolidated into one folder:

```
SALUS_Research_Final/
├── paper/          → BEST VERSION (NeurIPS 2025 format)
│   ├── salus_paper.tex
│   ├── neurips_2025.sty
│   └── compile.sh
│
├── data/           → Link to massive_collection (5GB)
│   └── massive_collection/...
│
├── results/        → All evaluation results
│   ├── salus_results_massive.json
│   ├── baseline_results_massive.json
│   └── ablation/
│
├── models/         → Trained checkpoint
│   └── salus_predictor_massive.pth
│
├── scripts/        → ALL scripts needed
│   ├── evaluate_salus.py
│   ├── create_all_figures.py
│   ├── compute_missing_metrics.py
│   └── ...
│
├── figures/        → Generated figures (empty until created)
│
└── run_all_analysis.sh  → Master workflow script
```

---

## Folder Comparison Table

| Item | Old Location(s) | New Location | Status |
|------|----------------|--------------|--------|
| **Paper (NeurIPS)** | `paper/salus_paper.tex` | `SALUS_Research_Final/paper/` | ✅ Copied |
| **Paper (Old)** | `SalusTest/paper/salus_paper.tex` | ❌ Not copied (outdated) | Archived |
| **Paper (Ancient)** | `docs/papers/*.tex` | ❌ Not copied | Very old |
| **Dataset** | `paper_data/massive_collection/` | `SALUS_Research_Final/data/` | ✅ Linked (symlink) |
| **Results** | `results/*.json` | `SALUS_Research_Final/results/` | ✅ Copied |
| **Model** | `/home/.../checkpoints/*.pth` | `SALUS_Research_Final/models/` | ✅ Copied |
| **Scripts** | `scripts/*.py` (scattered) | `SALUS_Research_Final/scripts/` | ✅ Copied (essential) |
| **Figures** | ❌ Didn't exist | `SALUS_Research_Final/figures/` | ✅ Created (empty) |
| **Master Script** | ❌ Didn't exist | `run_all_analysis.sh` | ✅ Created |

---

## What Was Kept vs. Discarded

### ✅ Kept (Copied to SALUS_Research_Final)

**Paper Files**:
- `paper/salus_paper.tex` (NeurIPS 2025 version - BEST)
- `paper/neurips_2025.sty` (conference style)
- `paper/compile.sh`
- `paper/fill_results.py`
- `paper/DATA_AND_FIGURES.md`
- `paper/PAPER_SUMMARY.md`
- `paper/CHANGES_NEURIPS_STYLE.md`
- `paper/README.md`

**Data**:
- `paper_data/massive_collection/` (5GB dataset - **symlinked**)

**Results**:
- `results/salus_results_massive.json` (SALUS evaluation)
- `results/baseline_results_massive.json` (baseline comparison)
- `results/ablation/` (ablation study folder)

**Models**:
- `/home/mpcr/Desktop/SalusV3/checkpoints/salus_predictor_massive.pth`

**Scripts** (Essential):
- `scripts/evaluate_salus.py`
- `scripts/evaluate_baseline_threshold.py`
- `scripts/ablate_signals.py`
- `scripts/compare_methods.py`
- `scripts/generate_paper_tables.py`
- `scripts/train_simple.py`
- `scripts/collect_data_franka.py`

**New Scripts** (Created):
- `scripts/create_all_figures.py` (generate all paper figures)
- `scripts/compute_missing_metrics.py` (horizons, types, latency)

### ❌ Not Kept (Intentionally Excluded)

**Old Paper Versions**:
- `SalusTest/paper/salus_paper.tex` (outdated format)
- `docs/papers/*.tex` (very old versions)
- Reason: Replaced by NeurIPS 2025 version

**Duplicate Scripts**:
- Multiple data collection variants (kept only `collect_data_franka.py`)
- Reason: Only need the working version

**Experimental/Broken Scripts**:
- `demo_salus_realtime.py` (non-functional VLA integration)
- Reason: Not needed for paper

**Auxiliary Files**:
- `*.aux`, `*.log`, `*.out` (LaTeX build artifacts)
- Reason: Generated files

---

## Why Symlink for Data?

The dataset is **5GB**. Instead of copying it (wasting disk space), we created a **symbolic link**:

```bash
SALUS_Research_Final/data/massive_collection -> ../paper_data/massive_collection
```

**Benefits**:
- No disk space duplication
- Data stays in original location
- Still accessible from consolidated folder

---

## Workflow Before vs. After

### Before (Confusing)
```
1. Find the right paper version (which folder?)
2. Locate data (where is it?)
3. Find evaluation scripts (scattered in scripts/)
4. Manually run each script
5. Copy results to paper
6. Generate figures manually (no scripts)
7. Compile paper (where?)
```

### After (Clear)
```
1. cd SALUS_Research_Final
2. ./run_all_analysis.sh  # Does everything!
3. evince paper/salus_paper.pdf
```

---

## Migration Guide

If you want to update the old folders:

### Archive Old Versions
```bash
cd /home/mpcr/Desktop/SalusV3/SalusTest

# Create archive folder
mkdir _old_versions

# Move old paper folders
mv SalusTest/paper _old_versions/paper_jan8
mv docs/papers _old_versions/papers_old

# Keep the data in place (symlinked from SALUS_Research_Final)
```

### Use Only SALUS_Research_Final
```bash
# All work happens here
cd SALUS_Research_Final

# Paper edits
vim paper/salus_paper.tex

# Run analysis
./run_all_analysis.sh

# Create figures
python scripts/create_all_figures.py

# Compile
cd paper && ./compile.sh
```

---

## File Size Comparison

| Folder | Size | Contents |
|--------|------|----------|
| `paper/` | 100 KB | LaTeX files, scripts |
| `data/` | 5.0 GB | Raw dataset (symlink) |
| `results/` | 24 KB | JSON results |
| `models/` | 284 KB | Trained model |
| `scripts/` | 116 KB | Python scripts |
| `figures/` | 0 KB | (To be generated) |
| **Total** | **~5.1 GB** | **(mostly data)** |

---

## Verification Checklist

To verify everything is correctly consolidated:

```bash
cd SALUS_Research_Final

# 1. Paper compiles?
cd paper && ./compile.sh
# Should create: salus_paper.pdf

# 2. Data accessible?
ls -lh data/massive_collection/20260109_215258/
# Should show: data_20260109_215321.zarr (5.0G)

# 3. Results exist?
cat results/salus_results_massive.json | grep auroc
# Should show: "auroc": 0.8833...

# 4. Model loads?
python -c "import torch; print(torch.load('models/salus_predictor_massive.pth').keys())"
# Should show: dict_keys(['model_state_dict', 'optimizer_state_dict', ...])

# 5. Scripts run?
python scripts/evaluate_salus.py --help
# Should show: usage: evaluate_salus.py [-h] ...
```

---

## Summary

**Before**: 3 paper folders, scattered data, no clear workflow  
**After**: 1 consolidated folder, clear structure, master script

**Old folders can be archived** - everything you need is in `SALUS_Research_Final/`.

---

**Questions?**  
See `QUICK_START.md` for usage instructions.
