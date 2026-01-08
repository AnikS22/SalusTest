#!/bin/bash

# SALUS Paper Compilation Script
# Compiles all figures and main paper

set -e  # Exit on error

echo "================================"
echo "SALUS Paper Compilation"
echo "================================"

# Create output directory
mkdir -p figures/compiled

# Compile all figure files
echo ""
echo "[1/6] Compiling architecture diagram..."
cd figures
pdflatex -interaction=nonstopmode architecture.tex > /dev/null 2>&1 || echo "Warning: architecture.tex compilation had issues"
mv architecture.pdf compiled/ 2>/dev/null || true
cd ..

echo "[2/6] Compiling state machine diagram..."
cd figures
pdflatex -interaction=nonstopmode state_machine.tex > /dev/null 2>&1 || echo "Warning: state_machine.tex compilation had issues"
mv state_machine.pdf compiled/ 2>/dev/null || true
cd ..

echo "[3/6] Compiling risk timeline..."
cd figures
pdflatex -interaction=nonstopmode risk_timeline.tex > /dev/null 2>&1 || echo "Warning: risk_timeline.tex compilation had issues"
mv risk_timeline.pdf compiled/ 2>/dev/null || true
cd ..

echo "[4/6] Compiling calibration curve..."
cd figures
pdflatex -interaction=nonstopmode calibration_curve.tex > /dev/null 2>&1 || echo "Warning: calibration_curve.tex compilation had issues"
mv calibration_curve.pdf compiled/ 2>/dev/null || true
cd ..

echo "[5/6] Compiling robot deployment diagram..."
cd figures
pdflatex -interaction=nonstopmode robot_deployment.tex > /dev/null 2>&1 || echo "Warning: robot_deployment.tex compilation had issues (expected - needs robot_icon.png)"
mv robot_deployment.pdf compiled/ 2>/dev/null || true
cd ..

# Copy compiled figures back to main figures directory
cp figures/compiled/*.pdf figures/ 2>/dev/null || true

# Compile main paper
echo "[6/6] Compiling main paper..."
pdflatex -interaction=nonstopmode salus_full_paper.tex > /dev/null 2>&1 || true
bibtex salus_full_paper > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode salus_full_paper.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode salus_full_paper.tex > /dev/null 2>&1 || true

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
rm -f figures/*.aux figures/*.log

echo ""
echo "================================"
echo "Compilation complete!"
echo "================================"
echo ""
echo "Output: salus_full_paper.pdf"
echo "Figures: figures/*.pdf"
echo ""

# Check if main PDF exists
if [ -f "salus_full_paper.pdf" ]; then
    echo "✓ Paper compiled successfully!"
    echo "  Open with: evince salus_full_paper.pdf"
else
    echo "⚠ Paper compilation may have had issues"
    echo "  Check salus_full_paper.log for details"
fi

echo ""
