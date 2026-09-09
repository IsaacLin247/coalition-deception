#!/usr/bin/env bash
# Build the paper: markdown -> PDF (xelatex) and a standalone LaTeX source for submission.
#   paper/build.sh
set -euo pipefail
cd "$(dirname "$0")"
SRC=when_credibility_collapses.md
COMMON=(-f markdown-implicit_figures+tex_math_dollars --citeproc --bibliography references.bib --csl ieee.csl
        --lua-filter=numbered_captions.lua --include-in-header=caption_layout.tex
        -V geometry:margin=1in -V fontsize=12pt --number-sections -M link-citations=true)
pandoc "$SRC" "${COMMON[@]}" -o when_credibility_collapses.pdf --pdf-engine=xelatex -V linkcolor=black
pandoc "$SRC" "${COMMON[@]}" -s -o when_credibility_collapses.tex
echo "built when_credibility_collapses.pdf and .tex"
