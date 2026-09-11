# Manuscript

[Read the paper](when_credibility_collapses.pdf): *Coalition Deception and Defensive Adaptation in a Social-Deduction Benchmark*, by Isaac Lin and Xi Chen.

The completed research revision is **53 pages, with 19 figures and 18 tables**. It incorporates all **240 corrected main-study jobs**, **1,255 training stages**, and **303 paired comparisons in 15 declared families**, plus the completed **540,000-episode mechanism study** and **67,500-episode ballot diagnostic**. The paper uses the final corrected game results throughout. The audit corrections and resulting findings are described in [cahnges.md](../cahnges.md).

The findings distinguish coalition victory from innocent ejections, show a targeted vulnerability of hypothesis scoring, and document mixed effects of dependence-aware defenses. Repeated response gains are finite-horizon observations. Statistical intervals are pointwise, Holm correction applies within declared families, and the two mechanism supplements remain descriptive; the manuscript states these limits explicitly.

## Source, build and verification

The canonical source is [when_credibility_collapses.md](when_credibility_collapses.md). Edit that file and rebuild the generated [TeX](when_credibility_collapses.tex) and [PDF](when_credibility_collapses.pdf). All 19 referenced figure assets are bundled under `figures/final/`; the paper builds without external research files or the raw-data archive.

Install Pandoc with Lua-filter support and a XeLaTeX distribution providing the packages used in `caption_layout.tex` and Pandoc's standard template. The checked build used Pandoc 3.6.4 and XeTeX from TeX Live 2023. From the repository root:

```sh
bash paper/build.sh
python paper/verify_manuscript.py
```

The build generates a 12-point manuscript with one-inch margins and uses `references.bib`, `ieee.csl`, `numbered_captions.lua` and `caption_layout.tex`. To compile the generated TeX directly, run XeLaTeX inside `paper/` so relative figure paths resolve.

The verification utility requires Python and Poppler's `pdfinfo` and `pdftotext`. It checks generated table text against the source-value manifest, figure/table counts, figure availability, citation keys, abstract length, TeX structure, PDF captions and text bounds. To retain a validation receipt:

```sh
python paper/verify_manuscript.py --output paper_validation.json
```

The checked-in PDF can be read without these tools. Verification of the underlying numerical analysis is separate and documented in [data/README.md](../data/README.md).

The five custom control, policy-distance, and ballot figures can be regenerated from published summaries with `python paper/plot_final_controls.py --out results/paper_figures`. The remaining empirical figures are produced by the validated analysis renderer. The custom helper records every selected value and input hash; its output was checked pixel-for-pixel against the supplied five PDFs.

## Results and provenance

[final_table_sources.json](data/final_table_sources.json) binds selected table rows and displayed table strings to their original analysis inputs and records source hashes. Its original-workspace input identifiers are provenance references; the files needed to build the paper are self-contained here. Complete seed-level estimates, all planned contrasts, supplemental summaries and validation records are published in the repository's [data directory](../data/README.md).

The [corrected-study-2026-09-11 release](https://github.com/IsaacLin247/coalition-deception/releases/tag/corrected-study-2026-09-11) supplies the original frozen source, all **11,885 retained checkpoints**, raw episode/meeting/voter records, protocols, environment records, and checksum manifests. Follow the release's reconstruction guide to join and verify its split archive before extraction.

The original study uses source fingerprint `691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35`. The smaller portable code package has a distinct fingerprint, `cdaa1c031b4e2e6e690771c9a884ed70ed78653814863f99bb2fe0def367ae68`. Original outputs retain the original study protocol. The portable package preserves the 240 job designs; its fingerprint does not replace the identity of the completed runs. See [packaging provenance](../code/provenance/packaging.json) and the [code guide](../code/README.md).

Published data metadata normalizes machine-specific locations to archive-relative paths. [PUBLICATION_MANIFEST.json](../data/PUBLICATION_MANIFEST.json) records original and published hashes and the path transformations. Numerical values, seed identities, event counts, checkpoint hashes and scientific results are unchanged. Original validation hashes identify the original bytes; the publication manifest connects them to their public copies.

## Submission information

Computational analysis and manuscript integration are complete for author review. Journal submission has not been performed. The authors must supply or confirm affiliations, corresponding-author details, funding, competing interests, contribution statements, the journal's review model, and final submission approval. These are author-supplied facts; this package does not invent them. The manuscript records the assistance used for code review, numerical reanalysis and editing.

## Citation-style attribution

The unmodified `ieee.csl` is the Zotero Citation Style Language project's **IEEE Reference Guide version 11.29.2023** style, updated 27 March 2024. Its original author and contributor information is retained in the file. The style is licensed under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/); see its [style identifier](https://www.zotero.org/styles/ieee). This notice concerns the third-party citation style, not a new license for the manuscript or research assets.
