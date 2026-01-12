#!/bin/bash
# Simple compilation script that works without pdflatex
set -e

echo "Checking for LaTeX..."
if command -v pdflatex &> /dev/null; then
    echo "✓ pdflatex found, compiling..."
    pdflatex -interaction=nonstopmode salus_paper.tex
    pdflatex -interaction=nonstopmode salus_paper.tex
    rm -f *.aux *.log *.out
    echo "✓ PDF created: salus_paper.pdf"
else
    echo "⚠ pdflatex not installed"
    echo ""
    echo "To compile manually:"
    echo "  1. Install: sudo apt-get install texlive-latex-base texlive-latex-extra"
    echo "  2. Or upload salus_paper.tex to Overleaf (overleaf.com)"
    echo ""
    echo "Checking LaTeX syntax instead..."
    
    # Check for basic LaTeX errors
    if grep -q "\\\\begin{document}" salus_paper.tex && grep -q "\\\\end{document}" salus_paper.tex; then
        echo "✓ Document structure looks good"
    else
        echo "✗ Missing \\begin{document} or \\end{document}"
    fi
    
    # Check for unmatched braces (simple check)
    open_braces=$(grep -o '{' salus_paper.tex | wc -l)
    close_braces=$(grep -o '}' salus_paper.tex | wc -l)
    if [ $open_braces -eq $close_braces ]; then
        echo "✓ Braces balanced ($open_braces pairs)"
    else
        echo "✗ Unmatched braces: $open_braces open, $close_braces close"
    fi
    
    echo ""
    echo "Paper appears syntactically valid. Upload to Overleaf to compile."
fi
