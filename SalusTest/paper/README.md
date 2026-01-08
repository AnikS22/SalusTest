# SALUS Academic Paper

This directory contains the complete LaTeX source for the SALUS research paper: **"SALUS: Real-Time Multi-Horizon Failure Forecasting for Vision-Language-Action Models via Single-Model Uncertainty Extraction"**

## Contents

### Main Document
- **`salus_paper.tex`** - Complete academic paper (IEEE conference format)
  - Abstract, introduction, related work, method, experiments, discussion, conclusion
  - All mathematical equations for 12D signal extraction
  - Performance comparisons (ensemble vs single-model)
  - Training results and ablation studies
  - Full bibliography

### Standalone Figures (TikZ)
- **`figure_signal_extraction.tex`** - 12D signal extraction pipeline diagram
  - Shows all 5 signal categories
  - Mathematical formulas for each signal
  - Visual breakdown of temporal, internal, uncertainty, physics, and consistency signals

- **`figure_temporal_forecasting.tex`** - Multi-horizon temporal forecasting visualization
  - 333ms sliding window illustration
  - 10-timestep temporal context
  - 4 prediction horizons (200ms, 300ms, 400ms, 500ms)
  - Conv1D+GRU architecture flow

- **`figure_ensemble_comparison.tex`** - Ensemble vs single-model comparison
  - Side-by-side architecture diagrams
  - Performance metrics (8× speedup, 3-5× memory reduction)
  - Highlights key innovation: replacing ensemble variance with softmax entropy

### Build System
- **`Makefile`** - Automated compilation for all documents
- **`README.md`** - This file

## Requirements

### LaTeX Distribution
Install a complete LaTeX distribution with TikZ/PGF support:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install texlive-full
```

**macOS (Homebrew):**
```bash
brew install --cask mactex
```

**Windows:**
- Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Required Packages
The following LaTeX packages are used (included in texlive-full):
- `IEEEtran` - IEEE conference paper format
- `tikz`, `pgfplots` - Diagrams and plots
- `amsmath`, `amssymb`, `amsfonts` - Mathematical symbols
- `algorithm`, `algorithmic` - Algorithm pseudocode
- `booktabs` - Professional tables
- `subcaption` - Subfigures
- `graphicx`, `xcolor` - Graphics and colors

## Compilation

### Quick Start
Compile everything (paper + all figures):
```bash
make
```

This generates:
- `salus_paper.pdf` - Main paper (10+ pages)
- `figure_signal_extraction.pdf` - Signal extraction diagram
- `figure_temporal_forecasting.pdf` - Temporal forecasting diagram
- `figure_ensemble_comparison.pdf` - Ensemble comparison diagram

### Individual Targets

**Compile main paper only:**
```bash
make paper
```

**Compile all figures only:**
```bash
make figures
```

**Compile specific figure:**
```bash
pdflatex figure_signal_extraction.tex
```

**View compiled paper:**
```bash
make view  # Opens salus_paper.pdf in default PDF viewer
```

### Manual Compilation
If you don't have `make`, compile manually:

```bash
# Main paper (run twice for references)
pdflatex salus_paper.tex
pdflatex salus_paper.tex

# Individual figures
pdflatex figure_signal_extraction.tex
pdflatex figure_temporal_forecasting.tex
pdflatex figure_ensemble_comparison.tex
```

## Cleaning Up

**Remove auxiliary files (.aux, .log, .out):**
```bash
make clean
```

**Remove all generated files (including PDFs):**
```bash
make distclean
```

## Paper Structure

### Sections
1. **Abstract** - Summary of SALUS system and contributions
2. **Introduction** - Motivation, key insight, contributions
3. **Related Work** - Uncertainty quantification, failure prediction, VLA models
4. **Method**
   - Section 3.A: Single-Model Signal Extraction (12D)
   - Section 3.B: Multi-Horizon Temporal Predictor
   - Section 3.C: System Architecture
5. **Experiments**
   - Setup, synthetic data generation
   - Training results (loss curves, 92.25% accuracy)
   - Performance comparison (Table: 8× speedup)
   - Signal analysis (Figure: success vs failure distributions)
   - Ablation study (Table: contribution of each signal category)
6. **Discussion** - Why single-model works, deployment considerations, limitations
7. **Conclusion** - Summary and future work
8. **References** - Bibliography (RT-2, OpenVLA, ensemble methods, MC Dropout, etc.)

### Key Equations

The paper includes complete mathematical formulations for:

**Temporal Action Dynamics (Eq. 1-4):**
- $z_1$: Action volatility $\|\mathbf{a}_t - \mathbf{a}_{t-1}\|_2$
- $z_2$: Action magnitude $\|\mathbf{a}_t\|_2$
- $z_3$: Action acceleration $\|\mathbf{a}_t - 2\mathbf{a}_{t-1} + \mathbf{a}_{t-2}\|_2$
- $z_4$: Trajectory divergence $\|\mathbf{a}_t - \frac{1}{K}\sum \mathbf{a}_{t-k}\|_2$

**VLA Internal Stability (Eq. 5-7):**
- $z_5$: Latent drift $\|\mathbf{h}_t - \mathbf{h}_{t-1}\|_2$
- $z_6$: Norm spike $\|\mathbf{h}_t\|_2 / \mu_{\|\mathbf{h}\|}$
- $z_7$: OOD distance $\|(\mathbf{h}_t - \boldsymbol{\mu}_\mathbf{h})/\boldsymbol{\sigma}_\mathbf{h}\|_2$

**Model Uncertainty (Eq. 8-10):**
- $z_8$: Softmax entropy $-\sum p_j \log p_j$ **(PRIMARY SIGNAL)**
- $z_9$: Max softmax probability $\max_j p_j$

**Physics Reality Checks (Eq. 11-12):**
- $z_{10}$: Execution mismatch $\|\Delta\mathbf{s}^{\text{actual}} - \mathbf{a}_{t-1}\|_2$
- $z_{11}$: Constraint margin $\max(0, -\min(\min(s-s_{\min}, s_{\max}-s)) + 0.5)$

**Temporal Consistency (Eq. 13):**
- $z_{12}$: Volatility stability $\text{std}(\{z_1^{(t-k)}\})$

**Hybrid Predictor Architecture (Eq. 14-16):**
- Conv1D: $\mathbf{X}_t = \text{Conv1D}(\mathbf{Z}_t; W_{\text{conv}})$
- GRU: $\mathbf{H}_t = \text{GRU}(\mathbf{X}_t; W_{\text{gru}})$
- Output: $\hat{\mathbf{y}}_t = W_{\text{out}} \mathbf{H}_t + \mathbf{b}_{\text{out}}$

**Temporal Focal Loss (Eq. 17):**
$$\mathcal{L}_{\text{focal}} = -\frac{1}{N}\sum_{i,h,c} \left[\alpha_c (1-p_{i,h,c})^\gamma y_{i,h,c} \log p_{i,h,c} + \beta (p_{i,h,c})^\gamma (1-y_{i,h,c}) \log(1-p_{i,h,c})\right]$$

**Total Loss with Smoothness (Eq. 18-19):**
$$\mathcal{L} = \mathcal{L}_{\text{focal}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$$

### Tables

- **Table I**: Performance comparison (forward passes, latency, VRAM, real-time capability)
- **Table II**: Ablation study (validation accuracy by signal subset)

### Figures (Embedded in Paper)

- **Figure 1**: Complete SALUS system architecture (inline TikZ)
- **Figure 2**: Training and validation loss curves (PGFPlots)
- **Figure 3**: Signal distributions for success vs failure episodes (bar chart)

## Customization

### Changing Paper Format
The paper uses IEEE conference format (`\documentclass[conference]{IEEEtran}`). To change:

**IEEE Journal:**
```latex
\documentclass[journal]{IEEEtran}
```

**NeurIPS (Neural Information Processing Systems):**
```latex
\documentclass{neurips_2023}  % Requires neurips_2023.sty
```

**ICRA/IROS (Robotics):**
```latex
\documentclass[conference]{IEEEtran}  % Already compatible
```

### Adjusting Figure Sizes
In the main paper (`salus_paper.tex`), adjust TikZ diagram scaling:

```latex
\begin{tikzpicture}[scale=0.8]  % Make 80% of original size
  ...
\end{tikzpicture}
```

For standalone figures, adjust the paper size:
```latex
\documentclass[tikz,border=20pt]{standalone}  % Increase border from 10pt to 20pt
```

### Adding Your Name/Affiliation
Replace the anonymous author block in `salus_paper.tex`:

```latex
\author{\IEEEauthorblockN{Your Name}
\IEEEauthorblockA{\textit{Department} \\
\textit{University Name}\\
City, Country \\
email@university.edu}
}
```

## Troubleshooting

### Missing Packages Error
**Symptom:** `! LaTeX Error: File 'tikz.sty' not found.`

**Solution:** Install texlive-full or manually install missing packages:
```bash
sudo apt install texlive-pictures  # For TikZ
sudo apt install texlive-science   # For algorithm packages
```

### Compilation Hangs
**Symptom:** pdflatex runs indefinitely without producing output

**Solution:**
1. Stop with Ctrl+C
2. Delete all `.aux` files: `rm *.aux`
3. Try again: `make clean && make`

### Figure Not Showing
**Symptom:** Blank space where figure should be

**Solution:** Figures are compiled as standalone PDFs. Make sure to run `make figures` before viewing the paper.

### References Not Showing
**Symptom:** [?] instead of citation numbers

**Solution:** Run pdflatex twice (already done by Makefile):
```bash
pdflatex salus_paper.tex  # First pass
pdflatex salus_paper.tex  # Second pass (resolves references)
```

## File Sizes

Expected compiled PDF sizes:
- `salus_paper.pdf`: ~500-800 KB (10-12 pages)
- `figure_signal_extraction.pdf`: ~50-100 KB (1 page, complex TikZ)
- `figure_temporal_forecasting.pdf`: ~40-80 KB (1 page)
- `figure_ensemble_comparison.pdf`: ~60-120 KB (1 page)

## Version Control

If submitting to arXiv or a conference, include only source files:
```bash
# Create submission archive
tar -czf salus_paper_source.tar.gz *.tex Makefile README.md
```

Do NOT include:
- `.aux`, `.log`, `.out` files (auxiliary)
- `.pdf` files (will be regenerated by reviewers)
- `.synctex.gz` files (SyncTeX data)

## Citation

If you use this paper template or the SALUS system, please cite:

```bibtex
@article{salus2025,
  title={SALUS: Real-Time Multi-Horizon Failure Forecasting for Vision-Language-Action Models via Single-Model Uncertainty Extraction},
  author={[Your Name]},
  journal={[Venue]},
  year={2025}
}
```

## License

**Proprietary** - Copyright © 2025. All rights reserved.

This paper and associated materials are confidential. Do not distribute without permission.

## Contact

For questions about the paper or compilation issues:
- GitHub: [@AnikS22](https://github.com/AnikS22)
- Repository: https://github.com/AnikS22/SalusTest

---

**Quick Reference:**
- Compile everything: `make`
- View paper: `make view`
- Clean up: `make clean`
- Help: `make help`
