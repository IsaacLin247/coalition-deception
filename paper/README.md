# Manuscript

[Read the paper](when_credibility_collapses.pdf): *Coalition Deception and Defensive Adaptation in a Social-Deduction Benchmark*, by Isaac Lin and Xi Chen.

This is the latest audited working revision, not a submission-ready final paper. It distinguishes completed corrected experiments from archived, provisional results whose replacements are still being checked. Packaging the source does not complete those experiments or change their scientific status. The publication audit and new findings are described in [cahnges.md](../cahnges.md).

The canonical source is [when_credibility_collapses.md](when_credibility_collapses.md). The standalone TeX and PDF are generated from it. All 19 referenced figure assets are bundled in `figures/`; no files outside this directory are needed to build the manuscript. Appendix C describes the compact public source layout. Historical raw data, checkpoints, draft letters and review files are not part of this paper package.

## Build

Install Pandoc with Lua-filter support and a XeLaTeX distribution providing the packages used in `caption_layout.tex` and Pandoc's standard template. This package was built with Pandoc 3.6.4 and XeTeX from TeX Live 2023. From the repository root, run:

```sh
bash paper/build.sh
```

This regenerates `paper/when_credibility_collapses.pdf` and `paper/when_credibility_collapses.tex`. Scientific code and experiment instructions are in [code/README.md](../code/README.md).

To compile the generated TeX directly, run XeLaTeX from inside `paper/` so its relative figure paths resolve.

The build uses `references.bib`, `ieee.csl`, `numbered_captions.lua` and `caption_layout.tex`. The filter verifies the manuscript's 19 numbered figures and 18 numbered tables, including Appendix Table B1.

## Citation-style attribution

The unmodified `ieee.csl` is the Zotero Citation Style Language project's **IEEE Reference Guide version 11.29.2023** style, updated 27 March 2024. Its original author and contributor information is retained in the file. The style is licensed under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/); see its [style identifier](https://www.zotero.org/styles/ieee). This notice concerns the third-party citation style, not a new license for the manuscript or research assets.
