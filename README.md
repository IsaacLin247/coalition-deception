# Coalition Deception and Defensive Adaptation

Code, completed analysis, and manuscript for **Coalition Deception and Defensive Adaptation in a Social-Deduction Benchmark**, by Isaac Lin and Xi Chen.

**Final corrected analysis — 11 September 2026 UTC:** all **240 jobs**, **1,255 training stages**, and **303 statistical comparisons** are complete and independently verified. Both descriptive supplements are complete: **540,000 mechanism-study episodes** and **67,500 ballot-diagnostic episodes**. The manuscript uses the completed corrected results throughout.

## Read and reproduce

| Item | Contents |
| --- | --- |
| [Paper PDF](paper/when_credibility_collapses.pdf) | Complete research revision with F1–F8, controls, statistical interpretation, and limitations. |
| [Paper source](paper/) | Markdown, generated TeX, bibliography, 19 figures, 18 tables, source-value manifest, and self-contained build. |
| [Code guide](code/README.md) | Training, evaluation, protocols, tests, and original/portable provenance checks. |
| [Final analysis data](data/README.md) | All seed-level results and 303 comparisons, complete figures, supplemental summaries, and validation receipts. |
| [Corrected-study release](https://github.com/IsaacLin247/coalition-deception/releases/tag/corrected-study-2026-09-11) | Original frozen source, all checkpoints and raw records, split archive, reconstruction guide, and checksums. |
| [cahnges.md](cahnges.md) | Audit corrections, final findings, and validation. |

## Main findings

Meeting-only learning increases harmful aligned voting while directly false claims decrease. Spatial adaptation is opponent-dependent. Lower coalition victory can coexist with frequent innocent ejections. The hypothesis defense has a truthful-testimony accuracy cost and a substantial targeted-attack vulnerability. Dependence-aware defense has mixed effects, including increased whole-game innocent ejections in the repeated loop. Finite response sequences do not establish an equilibrium or permanent cycle.

The paper reports effect sizes, seed variability, pointwise intervals, and within-family Holm adjustment. Five-seed exact tests have a minimum raw p-value of 0.0625; larger ten-seed families also face resolution limits. Nonsignificance is not equivalence.

## Install and check

Python 3.10 or later is required:

~~~sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e 'code[rl,env,analysis,dev]'
cd code
python -m pytest
python analysis/verify_packaging.py
python audit/submission/verify_final_inference.py --analysis ../data/final_analysis
~~~

On Windows, activate .venv\Scripts\Activate.ps1 in PowerShell. See the code guide for supplemental tests, training, and analysis of original raw results. Running tests does not start training.

The original and portable source fingerprints differ; their protocols retain identical jobs, seeds and budgets. [Packaging provenance](code/provenance/packaging.json) and the [reanalysis receipt](code/provenance/final_analysis_verification.json) document this relationship. Published metadata uses archive-relative paths; original/published hashes and mappings are in the [publication manifest](data/PUBLICATION_MANIFEST.json).

## Build and verify the paper

~~~sh
bash paper/build.sh
python paper/verify_manuscript.py
~~~

The build requires Pandoc and XeLaTeX. PDF checks additionally use Poppler's pdfinfo and pdftotext. The checked-in PDF requires no tools. The raw-data archive is split to fit GitHub's per-asset limit; follow its RECONSTRUCT.md to join and verify all parts.

## Publication status and attribution

Computational work and manuscript integration are complete for author review. Journal submission has not been performed. Affiliations, corresponding-author details, funding, competing interests, contributions, review model, and final author approval remain author-supplied submission information.

The original [MIT license](LICENSE) and copyright notice are retained. This derives from the [original benchmark](https://github.com/xi10017/when-credibility-collapses). The citation style retains its own license, documented in the paper guide. Citation metadata is in [CITATION.cff](CITATION.cff).
