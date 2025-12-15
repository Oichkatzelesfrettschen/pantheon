# latexmk configuration for Spandrel Paper
# =========================================

# Use lualatex for Unicode and modern font support
$pdf_mode = 4;  # 4 = lualatex
$lualatex = 'lualatex -interaction=nonstopmode -file-line-error -shell-escape %O %S';

# Bibtex
$bibtex_use = 2;

# Clean up additional files
$clean_ext = 'synctex.gz run.xml bcf fdb_latexmk fls nav snm vrb';

# Preview continuously
$preview_continuous_mode = 1;

# Use PDF viewer (macOS)
$pdf_previewer = 'open -a Preview %O %S';

# Recorder for dependency tracking
$recorder = 1;

# Maximum iterations
$max_repeat = 5;

# Show warnings
$warnings_as_errors = 0;
