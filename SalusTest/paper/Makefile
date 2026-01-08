# Makefile for SALUS Paper Compilation
# Compiles main paper and all standalone figures

.PHONY: all paper figures clean view help

# Default target
all: paper figures

# Main paper
paper: salus_paper.pdf

salus_paper.pdf: salus_paper.tex
	@echo "Compiling main paper..."
	pdflatex salus_paper.tex
	pdflatex salus_paper.tex  # Second pass for references
	@echo "✓ Main paper compiled: salus_paper.pdf"

# Standalone figures
figures: figure_signal_extraction.pdf figure_temporal_forecasting.pdf figure_ensemble_comparison.pdf all_figures.pdf

figure_signal_extraction.pdf: figure_signal_extraction.tex
	@echo "Compiling signal extraction diagram..."
	pdflatex figure_signal_extraction.tex
	@echo "✓ Figure compiled: figure_signal_extraction.pdf"

figure_temporal_forecasting.pdf: figure_temporal_forecasting.tex
	@echo "Compiling temporal forecasting diagram..."
	pdflatex figure_temporal_forecasting.tex
	@echo "✓ Figure compiled: figure_temporal_forecasting.pdf"

figure_ensemble_comparison.pdf: figure_ensemble_comparison.tex
	@echo "Compiling ensemble comparison diagram..."
	pdflatex figure_ensemble_comparison.tex
	@echo "✓ Figure compiled: figure_ensemble_comparison.pdf"

all_figures.pdf: all_figures.tex
	@echo "Compiling all figures document..."
	pdflatex all_figures.tex
	@echo "✓ All figures compiled: all_figures.pdf (complete visualization document)"

# View compiled paper
view: salus_paper.pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open salus_paper.pdf; \
	elif command -v open > /dev/null; then \
		open salus_paper.pdf; \
	else \
		echo "Please open salus_paper.pdf manually"; \
	fi

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
	@echo "✓ Cleaned"

# Deep clean (removes PDFs too)
distclean: clean
	@echo "Removing compiled PDFs..."
	rm -f *.pdf
	@echo "✓ Deep cleaned"

# Help target
help:
	@echo "SALUS Paper Compilation Makefile"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make all       - Compile paper and all figures (default)"
	@echo "  make paper     - Compile main paper only"
	@echo "  make figures   - Compile all standalone figures"
	@echo "  make view      - Open compiled paper in PDF viewer"
	@echo "  make clean     - Remove auxiliary files (.aux, .log, etc.)"
	@echo "  make distclean - Remove all generated files including PDFs"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (TeX Live or MiKTeX)"
	@echo "  - TikZ/PGF packages"
	@echo "  - Standard LaTeX packages (amsmath, algorithm, etc.)"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Install LaTeX: sudo apt install texlive-full (Ubuntu/Debian)"
	@echo "  2. Run: make"
	@echo "  3. View: make view (or open salus_paper.pdf)"

# List all targets
list:
	@echo "Compilation targets:"
	@echo "  - salus_paper.pdf"
	@echo "  - figure_signal_extraction.pdf"
	@echo "  - figure_temporal_forecasting.pdf"
	@echo "  - figure_ensemble_comparison.pdf"
