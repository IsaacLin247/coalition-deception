#!/usr/bin/env bash
# Build the shorter alternative; the original manuscript and build are untouched.
set -euo pipefail
cd "$(dirname "$0")"
COMMON=(-f markdown-implicit_figures+tex_math_dollars --citeproc
        --bibliography references.bib --csl ieee.csl
        --lua-filter=paper_v2_captions.lua --include-in-header=paper_v2_layout.tex
        --include-in-header=caption_layout.tex
        -V geometry:margin=1in -V fontsize=12pt --number-sections
        -M link-citations=true)
pandoc paper_v2.md "${COMMON[@]}" -o paper_v2.pdf --pdf-engine=xelatex -V linkcolor=black
pandoc paper_v2.md "${COMMON[@]}" -s -o paper_v2.tex
echo "built paper_v2.pdf and paper_v2.tex"
