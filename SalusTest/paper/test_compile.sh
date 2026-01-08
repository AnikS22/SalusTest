#!/bin/bash

# Test compilation script for SALUS paper
# Verifies all LaTeX documents compile without errors

set -e  # Exit on error

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "SALUS Paper Compilation Test"
echo "============================================================"
echo ""

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo -e "${RED}ERROR: pdflatex not found!${NC}"
    echo ""
    echo "Please install LaTeX:"
    echo "  Ubuntu/Debian: sudo apt install texlive-full"
    echo "  macOS:         brew install --cask mactex"
    echo "  Windows:       Install MiKTeX or TeX Live"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ pdflatex found${NC}"
echo ""

# Function to compile a LaTeX document
compile_document() {
    local file=$1
    local name=$(basename "$file" .tex)

    echo -e "${BLUE}Compiling: $file${NC}"

    # Compile and suppress output, but capture errors
    if pdflatex -interaction=nonstopmode -halt-on-error "$file" > /dev/null 2>&1; then
        if [ -f "${name}.pdf" ]; then
            local size=$(du -h "${name}.pdf" | cut -f1)
            echo -e "${GREEN}✓ SUCCESS: ${name}.pdf (${size})${NC}"
            return 0
        else
            echo -e "${RED}✗ FAILED: PDF not generated${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED: Compilation error${NC}"
        echo ""
        echo "Error details (last 20 lines of log):"
        tail -n 20 "${name}.log" | grep -A 3 "^!"
        return 1
    fi
}

# Test counter
total=0
passed=0
failed=0

echo "Testing LaTeX documents..."
echo ""

# Test main paper
total=$((total + 1))
if compile_document "salus_paper.tex"; then
    passed=$((passed + 1))
    # Run twice for references
    pdflatex -interaction=nonstopmode -halt-on-error "salus_paper.tex" > /dev/null 2>&1
    echo -e "${YELLOW}  (Second pass for references)${NC}"
else
    failed=$((failed + 1))
fi
echo ""

# Test figures
for fig in figure_signal_extraction.tex figure_temporal_forecasting.tex figure_ensemble_comparison.tex; do
    total=$((total + 1))
    if compile_document "$fig"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
    echo ""
done

# Test algorithms
total=$((total + 1))
if compile_document "algorithms.tex"; then
    passed=$((passed + 1))
else
    failed=$((failed + 1))
fi
echo ""

# Summary
echo "============================================================"
echo "Compilation Summary"
echo "============================================================"
echo ""
echo -e "Total tests:  ${total}"
echo -e "Passed:       ${GREEN}${passed}${NC}"
echo -e "Failed:       ${RED}${failed}${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All documents compiled successfully!${NC}"
    echo ""
    echo "Generated PDFs:"
    ls -lh *.pdf 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo "To view the main paper:"
    echo "  make view"
    echo "  or: xdg-open salus_paper.pdf"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some documents failed to compile${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if all LaTeX packages are installed:"
    echo "     sudo apt install texlive-full"
    echo ""
    echo "  2. Review error logs:"
    echo "     cat *.log | grep '!'"
    echo ""
    echo "  3. Try manual compilation:"
    echo "     pdflatex salus_paper.tex"
    echo ""
    exit 1
fi
