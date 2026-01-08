# SALUS Paper - All Graphs and Figures

## Quick Access to Visual Content

This document provides direct access to all graphs and figures from the SALUS paper in one convenient file.

---

## 📄 Main File: `all_figures.tex`

**Complete visualization document containing:**

### 8 High-Quality Figures

1. **Figure 1: Complete System Architecture**
   - End-to-end SALUS pipeline flowchart
   - VLA → Signal Extractor → Temporal Window → Predictor → Multi-Horizon Output
   - 100ms latency annotation

2. **Figure 2: 12D Signal Extraction Pipeline**
   - Complete breakdown of all 5 signal categories
   - Mathematical formulas for each of the 12 signals
   - Shows: Temporal (4D), Internal (3D), Uncertainty (2D), Physics (2D), Consistency (1D)
   - Highlights PRIMARY uncertainty signals

3. **Figure 3: Multi-Horizon Temporal Forecasting**
   - 333ms sliding window with 10 timesteps
   - Timeline visualization showing past, present, and future
   - 4 prediction horizons: 200ms, 300ms, 400ms, 500ms
   - Temporal context illustration

4. **Figure 4: Training and Validation Curves**
   - Loss over 50 epochs
   - Shows convergence to ~0.15 loss
   - Demonstrates no overfitting (train/val close)
   - Blue = Train, Red = Validation

5. **Figure 5: Signal Distributions (Success vs Failure)**
   - Bar chart comparing 12 signals
   - Green bars = Success episodes (low values)
   - Red bars = Failure episodes (high values)
   - Key insights: z₁ (2.9×), z₈ (2.3×), z₅ (2.7× higher in failures)

6. **Figure 6: Ensemble vs Single-Model Comparison**
   - Side-by-side architecture comparison
   - Top (RED): 5-model ensemble with 8 passes (800ms, 5.6GB)
   - Bottom (GREEN): Single-model with 1 pass (100ms, 1.2GB)
   - Large "8× SPEEDUP" annotation

7. **Figure 7: Performance Comparison Bar Chart**
   - Forward passes: 8 → 1
   - Latency: 800ms → 100ms
   - VRAM: 5.6GB → 1.2GB
   - Real-time capable: NO → YES

8. **Figure 8: Ablation Study Results**
   - Horizontal bar chart showing validation accuracy
   - All signals: 92.25% (best)
   - Temporal only: 82.11%
   - Uncertainty only: 76.45%
   - Demonstrates each category contributes

---

## 🚀 Quick Compilation

### Method 1: Using Make (Recommended)
```bash
cd /home/mpcr/Desktop/Salus\ Test/SalusTest/paper/
make all_figures.pdf
```

### Method 2: Direct Compilation
```bash
cd /home/mpcr/Desktop/Salus\ Test/SalusTest/paper/
pdflatex all_figures.tex
```

### Method 3: Compile Everything
```bash
cd /home/mpcr/Desktop/Salus\ Test/SalusTest/paper/
make  # Compiles paper + all figures including all_figures.pdf
```

---

## 📊 Output

**Generated PDF:** `all_figures.pdf`

**Expected size:** ~800KB - 1.2MB

**Number of pages:** 8-10 pages (one figure per page for clarity)

**Format:**
- Standard US Letter (8.5" × 11")
- High-resolution TikZ vector graphics
- Table of contents for easy navigation

---

## 📖 Using the Figures

### For Presentations
Each figure is on its own page and can be:
- Extracted from the PDF using tools like `pdftk` or Adobe Acrobat
- Directly included in PowerPoint/Keynote (vector quality preserved)
- Used in other LaTeX documents via `\includegraphics`

### Extracting Individual Figures
```bash
# Extract page 2 (Figure 2) to separate PDF
pdftk all_figures.pdf cat 2 output figure2_extracted.pdf

# Or use ghostscript
gs -sDEVICE=pdfwrite -dFirstPage=2 -dLastPage=2 -o figure2.pdf all_figures.pdf
```

### Including in Other Documents
```latex
\includegraphics[width=\textwidth]{all_figures_page2.pdf}
```

---

## 🎨 Customization

### Changing Figure Sizes
Edit `all_figures.tex` and adjust TikZ scaling:

```latex
% Make a figure larger
\begin{tikzpicture}[scale=1.2]  % 120% of original size
...
\end{tikzpicture}

% Make a figure smaller
\resizebox{0.8\textwidth}{!}{  % 80% of text width
\begin{tikzpicture}
...
\end{tikzpicture}
}
```

### Changing Colors
Find color definitions in the figure and modify:

```latex
% Original
fill=blue!30  % 30% blue

% Change to
fill=red!40   % 40% red
```

### Changing Fonts
At the top of `all_figures.tex`:

```latex
\usepackage[default]{lato}  % Use Lato font
% or
\usepackage{times}          % Use Times font
```

---

## 📋 Figure Quality

All figures use **TikZ/PGFPlots** for maximum quality:
- ✅ Vector graphics (infinite zoom, no pixelation)
- ✅ Publication-ready resolution
- ✅ Crisp text rendering
- ✅ Professional appearance
- ✅ Easy to modify and customize

---

## 🔧 Troubleshooting

### Figure Too Large (Doesn't Fit on Page)
Add to the figure:
```latex
\resizebox{\textwidth}{!}{
  \begin{tikzpicture}
  ...
  \end{tikzpicture}
}
```

### Compilation Error: "Dimension too large"
TikZ coordinates exceeded limits. Scale down:
```latex
\begin{tikzpicture}[scale=0.5]  % Reduce to 50%
```

### Text Overlapping in Figures
Increase spacing or reduce font size:
```latex
% Option 1: Smaller font
\node[font=\footnotesize] {...};

% Option 2: More space between elements
\node[minimum height=1.5cm] {...};  % Increase from 1cm
```

### Missing Package Error
Install full TeX Live:
```bash
sudo apt install texlive-full
```

---

## 📐 Technical Specifications

### Required Packages
- `tikz` with libraries: shapes, arrows, positioning, calc, fit, backgrounds, decorations
- `pgfplots` (version 1.18+)
- `amsmath` (mathematical symbols)
- `geometry` (page layout)

### Compilation Time
- **First compilation:** ~10-20 seconds (TikZ rendering)
- **Subsequent compilations:** ~5-10 seconds (cached)

### Memory Usage
- Peak LaTeX memory: ~200-300MB
- Output PDF: ~1MB

---

## 🎯 Comparison: Individual vs Combined

### Individual Figure Files
```
figure_signal_extraction.pdf      (~85KB, 1 page)
figure_temporal_forecasting.pdf   (~72KB, 1 page)
figure_ensemble_comparison.pdf    (~98KB, 1 page)
```

**Pros:** Standalone, easy to include in other documents
**Cons:** Need to compile 3 separate files

### Combined Figure File
```
all_figures.pdf                   (~1MB, 8-10 pages)
```

**Pros:**
- All figures in one place
- Table of contents for navigation
- Includes training curves and analysis charts
- Easy to share/present

**Cons:**
- Larger file size
- Need to extract individual pages if needed

---

## 📤 Sharing

### For Reviewers/Collaborators
Send `all_figures.pdf` alone - it's self-contained and shows all visual content.

### For Paper Submission
Use individual figure PDFs from the main paper compilation.

### For Presentations
Extract specific pages from `all_figures.pdf` or use standalone figure PDFs.

---

## 📞 Quick Help

**View the compiled figures:**
```bash
xdg-open all_figures.pdf  # Linux
open all_figures.pdf      # macOS
start all_figures.pdf     # Windows
```

**Re-compile after changes:**
```bash
make clean
make all_figures.pdf
```

**Check compilation status:**
```bash
./test_compile.sh
```

---

## ✨ Summary

`all_figures.tex` → `all_figures.pdf` provides:

✅ **8 comprehensive figures** covering the entire SALUS system
✅ **High-quality vector graphics** (TikZ/PGFPlots)
✅ **Publication-ready** quality
✅ **Easy to compile** (`make all_figures.pdf`)
✅ **Self-contained** document with table of contents
✅ **Presentation-ready** one figure per page

**Perfect for:**
- Quick visual reference
- Presentations and talks
- Sharing with collaborators
- Understanding SALUS architecture at a glance

---

**Created:** 2026-01-07
**Format:** LaTeX → PDF
**Pages:** 8-10
**Size:** ~1MB
