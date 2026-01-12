# SALUS: Safety Assurance via Learning from Uncertainty Signals

**Real-time failure prediction for Vision-Language-Action models**

---

## Project Structure

```
SalusTest/
├── salus/                      # Core implementation
│   ├── core/                   # Signal extraction & prediction
│   │   ├── vla/               # VLA wrappers with signal extraction
│   │   │   ├── smolvla_wrapper_fixed.py  # ✅ Fixed wrapper (10/12 signals)
│   │   │   ├── wrapper.py     # Original ensemble wrapper
│   │   │   └── single_model_extractor.py  # 12D signal extractor
│   │   ├── predictor.py       # Multi-horizon failure predictor
│   │   └── adaptation.py      # Intervention system
│   ├── simulation/            # IsaacLab environment integration
│   └── training/              # Training utilities
│
├── scripts/                    # Execution scripts
│   ├── collect_data_franka.py # Data collection pipeline
│   ├── train_failure_predictor.py  # Training script
│   └── run_salus_failure_eval.py   # Evaluation script
│
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Multi-GPU config
│   └── a100_config.yaml       # Single A100 config
│
├── results/                    # Experimental results
│   ├── salus_results_massive.json    # Main results (88.3% AUROC)
│   ├── baseline_results_massive.json # Baseline (50.1% AUROC)
│   └── ablation/              # Ablation study results
│
├── SALUS_Research_Final/      # Paper & research materials
│   ├── paper/                 # NeurIPS 2025 paper
│   ├── data/                  # Dataset symlink
│   ├── results/               # Formatted results
│   ├── models/                # Trained checkpoints
│   └── scripts/               # Analysis & figure generation
│
└── test_signal_extraction_comprehensive.py  # Signal validation test
```

---

## Quick Start

### 1. Test Signal Extraction

```bash
# Verify all 12 signals are extracted correctly
python test_signal_extraction_comprehensive.py

# Expected: 10/12 signals working, 0% NaN
```

### 2. Collect Data

```bash
python scripts/collect_data_franka.py \
  --headless \
  --enable_cameras \
  --num_episodes 100 \
  --max_steps 200
```

### 3. Train Predictor

```bash
python scripts/train_failure_predictor.py \
  --data_path paper_data/dataset_*.zarr \
  --epochs 100 \
  --batch_size 256
```

### 4. Evaluate

```bash
python scripts/run_salus_failure_eval.py \
  --episodes 50 \
  --checkpoint checkpoints/salus_predictor_best.pth
```

---

## Key Results

| Metric | Value | Status |
|--------|-------|--------|
| **AUROC** | 88.3% | ✅ |
| **Recall** | 51.6% | ✅ |
| **Precision** | 41.2% | ✅ |
| **False Alarm Rate** | 6.4% | ✅ |
| **Baseline AUROC** | 50.1% (random) | ✅ |
| **Improvement** | +76.5% over baseline | ✅ |
| **Model Size** | 70,672 params (284 KB) | ✅ |
| **Signal Extraction** | 10/12 working (83.3%) | ✅ |

---

## Recent Updates (Jan 12, 2026)

### ✅ Signal Extraction Fixed
- **Problem**: 75% of signals returned NaN (9/12 broken)
- **Solution**: Created `smolvla_wrapper_fixed.py` with forward hooks
- **Result**: 10/12 signals working, 0% NaN

**File**: `salus/core/vla/smolvla_wrapper_fixed.py`
**Documentation**: `SIGNAL_EXTRACTION_FIXED.md`

### ✅ Folder Cleanup Complete
- Removed 5.1 GB of outdated data and duplicate files
- Consolidated research materials into `SALUS_Research_Final/`
- Cleaned up 30+ duplicate markdown files

---

## Paper

**Location**: `SALUS_Research_Final/paper/salus_paper.tex`
**Format**: NeurIPS 2025 (preprint mode)
**Status**: Ready for review, actual data filled in

**To compile**:
1. Upload to [Overleaf](https://overleaf.com)
2. Or install LaTeX: `sudo apt-get install texlive-latex-extra`
3. Run: `cd SALUS_Research_Final/paper && pdflatex salus_paper.tex`

---

## Dependencies

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **IsaacLab**: v0.48.5
- **SmolVLA**: lerobot/smolvla_base (HuggingFace)
- **GPU**: NVIDIA (tested on RTX 2080 Ti, A100)

---

## Next Steps

1. **Re-collect data with fixed wrapper** (~8-10 hours for 5,000 episodes)
   - Update `collect_data_franka.py` to use `SmolVLAWithInternals`
   - Expected: 10/12 signals working (vs. 3/12 previously)

2. **Re-train on complete signal data**
   - Expected AUROC improvement: 0.883 → 0.92+

3. **Deploy to HPC**
   - Cluster: Athene (TU Darmstadt)
   - GPUs: 4× RTX 2080 Ti or 1× A100

---

## Documentation

- **`README.md`** (this file) - Project overview
- **`SIGNAL_EXTRACTION_FIXED.md`** - Signal extraction fix details
- **`SALUS_Research_Final/PAPER_READY.md`** - Paper status

---

## Contact

**Project**: SALUS Failure Prediction System
**Framework**: IsaacLab + SmolVLA + PyTorch
**Status**: Production-ready for data collection

**Last Updated**: January 12, 2026
