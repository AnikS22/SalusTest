#!/bin/bash
# Compile SALUS paper (NeurIPS style) to PDF

set -e

echo "Compiling SALUS paper (NeurIPS 2025 style)..."

# Run pdflatex twice for references
pdflatex -interaction=nonstopmode salus_paper.tex
pdflatex -interaction=nonstopmode salus_paper.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.bbl *.blg

echo "✓ Paper compiled successfully: salus_paper.pdf"
echo ""
echo "Note: Using NeurIPS 2025 [preprint] style"
echo "Change to [final] in salus_paper.tex line 5 for camera-ready version"
