# Makefile for Spandrel Paper
# ============================
#
# Build targets:
#   make          - Build PDF with lualatex
#   make pdf      - Same as above
#   make clean    - Remove auxiliary files
#   make distclean - Remove all generated files including PDF
#   make watch    - Continuous compilation (requires latexmk)
#   make arxiv    - Prepare arxiv submission tarball

# Configuration
MAIN = spandrel_paper
LATEX = lualatex
BIBTEX = bibtex
LATEXMK = latexmk
LATEXFLAGS = -interaction=nonstopmode -file-line-error -shell-escape

# Source files
TEXFILES = $(MAIN).tex $(wildcard sections/*.tex)
FIGFILES = $(wildcard figures/*.tex)
BIBFILES = references.bib

# Output
PDF = $(MAIN).pdf

# Auxiliary file extensions
AUXEXT = aux bbl blg log out toc lof lot fls fdb_latexmk synctex.gz \
         nav snm vrb run.xml bcf

.PHONY: all pdf clean distclean watch arxiv

# Default target
all: pdf

# Build PDF
pdf: $(PDF)

$(PDF): $(TEXFILES) $(FIGFILES) $(BIBFILES)
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN)

# Quick build (single pass, no bibliography)
quick:
	$(LATEX) $(LATEXFLAGS) $(MAIN)

# Continuous compilation with latexmk
watch:
	$(LATEXMK) -pvc -pdf -lualatex $(MAIN)

# Clean auxiliary files
clean:
	rm -f $(addprefix $(MAIN)., $(AUXEXT))
	rm -f *.aux *.log *.bbl *.blg *.out *.toc
	rm -f figures/*.aux figures/*.log

# Clean everything including PDF
distclean: clean
	rm -f $(PDF)
	rm -f $(MAIN)-arxiv.tar.gz

# Prepare arxiv submission
arxiv: $(PDF)
	@echo "Preparing arxiv submission..."
	@mkdir -p arxiv-submit
	@cp $(MAIN).tex arxiv-submit/
	@cp $(MAIN).bbl arxiv-submit/
	@cp -r figures arxiv-submit/
	@tar -czvf $(MAIN)-arxiv.tar.gz -C arxiv-submit .
	@rm -rf arxiv-submit
	@echo "Created $(MAIN)-arxiv.tar.gz"

# View PDF (macOS)
view: $(PDF)
	open $(PDF)

# Word count (approximate)
wc:
	@texcount -inc -total $(MAIN).tex 2>/dev/null || \
		echo "Install texcount for word counting"

# Check for common LaTeX issues
check:
	@echo "Checking for common issues..."
	@grep -n "\\\\cite{}" $(TEXFILES) && echo "WARNING: Empty citations found" || true
	@grep -n "\\\\ref{}" $(TEXFILES) && echo "WARNING: Empty references found" || true
	@grep -n "TODO" $(TEXFILES) && echo "WARNING: TODOs found" || true
	@echo "Check complete."

# Help
help:
	@echo "Spandrel Paper Build System"
	@echo "==========================="
	@echo ""
	@echo "Targets:"
	@echo "  make          Build PDF (default)"
	@echo "  make quick    Single-pass compilation"
	@echo "  make watch    Continuous compilation"
	@echo "  make clean    Remove auxiliary files"
	@echo "  make distclean Remove all generated files"
	@echo "  make arxiv    Prepare arxiv tarball"
	@echo "  make view     Open PDF (macOS)"
	@echo "  make wc       Word count"
	@echo "  make check    Check for common issues"
