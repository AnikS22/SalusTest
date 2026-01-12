# Changes to Match NeurIPS/SafeVLA Style

## Overview
The SALUS paper has been restructured to match the NeurIPS 2025 format and style similar to the SafeVLA paper (@ArXiv-2503.03480v3).

## Major Changes

### 1. Document Class & Style
**Before**: Generic article class with manual formatting
**After**: NeurIPS 2025 style package
```latex
\documentclass{article}
\usepackage[preprint]{neurips_2025}  % Can change to [final] for camera-ready
```

### 2. Abstract Format
**Before**: Standard abstract
**After**: Compressed format with vspace adjustment (-0.8em) matching SafeVLA style

### 3. Section Structure

#### Added "Problem Formulation" Section
New formal mathematical section before Method:
- VLA Policy definition
- Failure Definition
- Failure Prediction Objective with equations
- Multi-Horizon Prediction motivation

**Mathematical notation**:
```latex
f_{\bm{\phi}}: \mathcal{S} \rightarrow [0,1]^{K \times M}
```

#### Method Section Restructured
**Before**: "Method"
**After**: "Method: Multi-Dimensional Uncertainty Monitoring"

Subsections now have more descriptive titles:
- "Uncertainty Signal Extraction" (with \label{sec:signals})
- "Multi-Horizon Failure Predictor" (with \label{sec:predictor})
- "Intervention Strategies"

#### Experiments Section
**Before**: Simple results presentation
**After**: Structured with research questions upfront:

```latex
We evaluate SALUS to answer four key questions:
(I) Can SALUS outperform baselines? (§\ref{sssec:main_results})
(II) How do signal groups contribute? (§\ref{sec:ablation})
(III) How does performance vary across horizons? (§\ref{sssec:horizon_analysis})
(IV) Does SALUS generalize across failure types? (§\ref{sssec:failure_types})
```

### 4. Related Work Format
Changed from subsection headers to **bold inline headers**:
```latex
\noindent\textbf{Vision-Language-Action Models.~}
\noindent\textbf{Uncertainty Estimation.~}
\noindent\textbf{Safe Reinforcement Learning.~}
```

### 5. Itemize Formatting
**Before**: Standard itemize
**After**: Compact itemize matching SafeVLA:
```latex
\begin{itemize}[left=0.0cm, nosep, topsep=0pt, partopsep=0pt]
```

### 6. Experimental Setup Format
Changed to **inline bold headers** instead of subsubsections:
```latex
\noindent\textbf{VLA Model.~} We use SmolVLA...
\noindent\textbf{Simulation Environment.~} We conduct...
\noindent\textbf{Dataset Collection.~} We collect...
```

### 7. Table Formatting
**Before**: Simple tables
**After**: Professional tables with:
- Caption on top (not bottom)
- Bold best results
- Up/down arrows for metrics
- Resizebox for consistent widths
```latex
\textbf{AUROC} $\uparrow$ & \textbf{Recall} $\uparrow$
```

### 8. Mathematical Notation
More formal notation throughout:
- Bold for parameters: $\bm{\theta}$, $\bm{\phi}$
- Proper set notation: $\mathcal{S}$, $\mathcal{A}$, $\mathcal{L}$
- Numbered equations for key formulas
- Inline equations for simple expressions

### 9. Citation Style
**Before**: Verbose inline citations
**After**: Compact cite command:
```latex
vision-language-action models~\cite{rt2,openvla,smolvla}
```

### 10. Key Insights Format
Added **bold labels** for key findings:
```latex
\noindent\textbf{Key Findings:}
\noindent\textbf{Key Insights:}
```

### 11. Section References
Changed to use § symbol:
```latex
(Section~\ref{sec:signals}) → (§\ref{sec:signals})
```

### 12. Discussion Section
Restructured with **inline bold headers**:
```latex
\noindent\textbf{Why Multi-Dimensional Signals Work.~}
\noindent\textbf{Limitations.~}
\noindent\textbf{Future Directions.~}
```

## Files Created/Modified

### New Files
- `neurips_2025.sty` - NeurIPS 2025 style file (copied from SafeVLA)
- `salus_paper_old.tex` - Backup of original paper

### Modified Files
- `salus_paper.tex` - Completely restructured to NeurIPS format
- `compile.sh` - Updated with NeurIPS note
- `CHANGES_NEURIPS_STYLE.md` - This file

## Visual Differences

### Typography
- Uses NeurIPS fonts and spacing
- Compressed vertical spacing
- Professional table formatting
- Consistent mathematical notation

### Structure
- More formal problem formulation
- Clearer experimental narrative (research questions)
- Inline subsection headers in experiments
- Compact itemize lists

### Content Organization
- Related Work uses inline headers (not subsections)
- Method has descriptive section titles
- Experiments structured around questions
- Discussion uses inline headers

## How to Compile

```bash
cd /home/mpcr/Desktop/SalusV3/SalusTest/paper
./compile.sh
```

This will generate `salus_paper.pdf` in NeurIPS 2025 preprint style.

## Switching Between Styles

### Preprint (current)
```latex
\usepackage[preprint]{neurips_2025}
```

### Camera-Ready (for acceptance)
```latex
\usepackage[final]{neurips_2025}
```

### Anonymous Submission
```latex
\usepackage{neurips_2025}  % No option = anonymous
```

## Content Preserved

All truthful content from the original paper is preserved:
- All 12 signal definitions with equations
- Architecture details (70,672 parameters)
- Dataset specifications (5K episodes, 1M timesteps, 8% failure rate)
- Training configuration
- Evaluation metrics
- All placeholders for results

## What's Still Needed

Same as before:
- Ablation study results (currently running)
- Multi-horizon breakdown
- Per-failure-type metrics
- Inference latency benchmark
- Training time

## Advantages of New Format

1. **Professional appearance**: Matches top-tier conference standards
2. **Better organization**: Clear research questions guide reader
3. **Formal math**: Proper problem formulation establishes rigor
4. **Compact presentation**: More content fits in page limits
5. **Consistent style**: Matches conventions of leading robotics papers

## Comparison to SafeVLA Paper

### Similarities (Adopted)
- NeurIPS 2025 style
- Inline subsection headers in experiments
- Compact itemize formatting
- Bold mathematical variables
- Research questions structure
- Table formatting with arrows

### Differences (Intentional)
- SALUS focuses on failure *prediction* vs SafeVLA's safe *training*
- We don't use CMDP (no constrained optimization)
- We monitor pretrained VLAs vs training from scratch
- Different technical approach (signal extraction vs SafeRL)
- Simulation-only (acknowledged) vs real robot experiments

The new format maintains SALUS's unique contribution while adopting professional presentation standards.
