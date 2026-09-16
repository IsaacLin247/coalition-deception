# Coalition Deception and Defensive Adaptation

Code, completed analysis, and manuscript for **Coalition Deception and Defensive Adaptation in a Social-Deduction Benchmark**, by Isaac Lin and Xi Chen.

**Final corrected analysis — 11 September 2026 UTC:** all **240 jobs**, **1,255 training stages**, and **303 statistical comparisons** are complete and independently verified. Both descriptive supplements are complete: **540,000 mechanism-study episodes** and **67,500 ballot-diagnostic episodes**. The manuscript uses the completed corrected results throughout.

**Additional validation before submission:** a separate [prospective follow-up](validation/README.md) tests hypothesis-defense tie handling and threshold selection, followed by fresh adaptive attacks and a 72-seed response-cycle replication. All 224 jobs are complete and passed independent local reanalysis on 15 September; [scientific interpretation and manuscript integration remain pending](validation/RUN_STATUS.md). The current manuscript reports the earlier 240-job study.

## Read and reproduce

| Item | Contents |
| --- | --- |
| [Paper PDF](paper/when_credibility_collapses.pdf) | Complete research revision with F1–F8, controls, statistical interpretation, and limitations. |
| [Shorter paper v2](paper/paper_v2.pdf) | Separate 21-page single-document edition for comparison; original article unchanged. [Changes and coverage](paper/paper_v2_changes.md). |
| [Paper source](paper/) | Markdown, generated TeX, bibliography, 19 figures, 18 tables, source-value manifest, and self-contained build. |
| [Code guide](code/README.md) | Training, evaluation, protocols, tests, and original/portable provenance checks. |
| [Final analysis data](data/README.md) | All seed-level results and 303 comparisons, complete figures, supplemental summaries, and validation receipts. |
| [Corrected-study release](https://github.com/IsaacLin247/coalition-deception/releases/tag/corrected-study-2026-09-11) | Original frozen source, all checkpoints and raw records, split archive, reconstruction guide, and checksums. |
| [cahnges.md](cahnges.md) | Audit corrections, final findings, and validation. |
| [Additional validation](validation/README.md) | Fixed development/test separation, defense variants, power analysis, and six new primary comparisons. |

## Main findings

Meeting-only training increases false ejection and raw coalition vote agreement; this does not establish communication-driven collusion. The false-claim fraction includes non-informative actions and cannot establish greater truthfulness. Spatial adaptation is opponent-dependent. F3 leaves incident-free innocent ejections unpenalized, and multi-round training optimizes victory rather than avoiding innocent harm. The hypothesis defense has a truthful-testimony accuracy cost and a targeted-attack vulnerability. Dependence-aware defense has mixed effects. Finite response sequences do not establish an equilibrium or permanent cycle. The [v2 review corrections](paper/paper_v2_review_response.md) explain the reward decomposition and interpretation limits.

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

The shorter alternative has its own build and checks:

~~~sh
bash paper/build_v2.sh
python paper/verify_paper_v2.py
~~~

Edit [paper_v2.md](paper/paper_v2.md) for the shorter version; its build generates
`paper_v2.tex` and `paper_v2.pdf` without overwriting the original article.

The build requires Pandoc and XeLaTeX. PDF checks additionally use Poppler's pdfinfo and pdftotext. The checked-in PDF requires no tools. The raw-data archive is split to fit GitHub's per-asset limit; follow its RECONSTRUCT.md to join and verify all parts.

## Publication status and attribution

The corrected study is incorporated in the manuscript. The separate prospective follow-up has completed execution and verification; its scientific interpretation and manuscript integration remain pending. Successful reproduction alone does not establish submission readiness. Journal submission has not been performed. Affiliations, corresponding-author details, funding, competing interests, contributions, review model, and final author approval remain author-supplied submission information.

The original [MIT license](LICENSE) and copyright notice are retained. This derives from the [original benchmark](https://github.com/xi10017/when-credibility-collapses). The citation style retains its own license, documented in the paper guide. Citation metadata is in [CITATION.cff](CITATION.cff).
