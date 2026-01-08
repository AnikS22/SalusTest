#!/bin/bash

# Quick script to compile and view all SALUS graphs

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "SALUS Graphs Viewer"
echo "============================================================"
echo ""

# Check if pdflatex exists
if ! command -v pdflatex &> /dev/null; then
    echo -e "${YELLOW}WARNING: pdflatex not found${NC}"
    echo "Install LaTeX: sudo apt install texlive-full"
    echo ""

    # Check if PDF already exists
    if [ -f "all_figures.pdf" ]; then
        echo -e "${GREEN}Found existing all_figures.pdf${NC}"
        echo "Opening..."
        xdg-open all_figures.pdf 2>/dev/null || open all_figures.pdf 2>/dev/null || echo "Please open all_figures.pdf manually"
        exit 0
    else
        echo "No compiled PDF found. Please install LaTeX and run: make all_figures.pdf"
        exit 1
    fi
fi

# Check if source file exists
if [ ! -f "all_figures.tex" ]; then
    echo -e "${YELLOW}ERROR: all_figures.tex not found${NC}"
    echo "Please run this script from the paper/ directory"
    exit 1
fi

# Compile
echo -e "${BLUE}Compiling all figures...${NC}"
pdflatex -interaction=nonstopmode -halt-on-error all_figures.tex > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful!${NC}"

    # Show PDF info
    if [ -f "all_figures.pdf" ]; then
        SIZE=$(du -h all_figures.pdf | cut -f1)
        echo "  PDF size: $SIZE"
        echo "  Location: $(pwd)/all_figures.pdf"
        echo ""

        # Open PDF
        echo "Opening PDF viewer..."
        if command -v xdg-open &> /dev/null; then
            xdg-open all_figures.pdf &
            echo -e "${GREEN}✓ Opened in default PDF viewer${NC}"
        elif command -v open &> /dev/null; then
            open all_figures.pdf &
            echo -e "${GREEN}✓ Opened in default PDF viewer${NC}"
        else
            echo -e "${YELLOW}Please open all_figures.pdf manually${NC}"
        fi
    else
        echo -e "${YELLOW}WARNING: PDF not found after compilation${NC}"
    fi
else
    echo -e "${YELLOW}✗ Compilation failed${NC}"
    echo ""
    echo "Error details:"
    tail -n 20 all_figures.log | grep -A 3 "^!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Figures Available:"
echo "============================================================"
echo "  1. System Architecture"
echo "  2. 12D Signal Extraction Pipeline"
echo "  3. Multi-Horizon Temporal Forecasting"
echo "  4. Training and Validation Curves"
echo "  5. Signal Distributions (Success vs Failure)"
echo "  6. Ensemble vs Single-Model Comparison"
echo "  7. Performance Comparison Bar Chart"
echo "  8. Ablation Study Results"
echo ""
