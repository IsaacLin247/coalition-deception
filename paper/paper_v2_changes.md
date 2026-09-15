# Paper v2: shorter single-document edition

Created 15 September 2026 for side-by-side comparison. The original manuscript,
TeX, PDF, bibliography, caption layout, and build files are unchanged. Version 2
is a separate editable manuscript, not a replacement for the original.

| Measure | Original | Version 2 |
|:--|--:|--:|
| PDF pages, including references and appendices | 53 | 21 |
| Markdown words* | 14,247 | 5,711 |
| Figures in the document | 19 | 5 |
| Tables in the document | 18 | 3 |
| Abstract words | 224 | 178 |
| Cited references | 19 | 19 |
| Body and caption size | 12 pt | 12 pt |
| Page margins | 1 inch | 1 inch |

*Whitespace-separated source tokens, including metadata, equations, and tables;
the same counting method is used for both files. This is a **59.9% reduction** in
source length and a **60.4% reduction** in pages. The reduction comes from editing
and selecting displays, without reducing the body or caption font size.

## What changed

The original repeatedly introduced, tabulated, plotted, and interpreted the same
outcomes. Version 2 combines the introduction and related work, brings the game,
learning procedure, and evaluation into one Methods section, and groups results
by the scientific question they answer. The separate interpretation and limitations
sections are merged. Repeated audit history and generic cautions are shortened;
qualifications needed to interpret individual findings remain beside those findings.

The main figures retain learning, scripted-rule performance, targeted hypothesis
crossplay, whole-game outcomes, and ten-generation crossplay. They correspond to
the original Figures 1, 2, 18, 4, and 9, renumbered 1–5. Figures 1, 2, and 4 reuse
the original images; Figures 3 and 5 are redrawn from the validated means as one
larger matrix each, improving readability without changing the plotted outcomes. Tables
beside those plots are removed where they repeat the displayed result. Three
tables remain: experimental cohorts, a compact 5+2 ballot diagnostic, and all six
matched dependence-loop comparisons, including the nonsignificant comparison.

All eight experimental themes remain in the same PDF. Information, training-budget,
and reward controls have a concise results subsection. Policy distances, response
gaps, the smaller evidence cohorts, score interventions, and the sensor example
are summarized rather than presented as additional figure sequences. Full grids,
variability, and extended plots remain in the existing public analysis and the
unchanged original article. No separate supplementary manuscript was created.

The Gaussian model remains in the mechanism section. Its complete original
proposition and proof are preserved in Appendix A. Appendix B retains scoring
coefficients, source identity, relevant implementation exceptions, and reproduction
information. The generic PPO objective equation and repeated parameter descriptions
were removed; the algorithm citation and actual training settings remain.

## Scientific content preserved

| Evidence | Location in version 2 |
|:--|:--|
| F1: learning, aligned votes, false claims | Results 3.1; Figure 1 |
| F2: scripted rules and truthful-testimony cost | Results 3.2; Figure 2 |
| F3: single-meeting responses and incident opportunities | Results 3.3 |
| F4: coalition victory versus innocent ejection | Results 3.3; Figure 4 |
| F5: finite response sequences and transfer | Results 3.4; Figure 5 |
| F6: correlated-report model and sensor illustration | Section 4; Appendix A |
| F7: dependence defense and adverse outcomes | Results 3.5; Table 3 |
| F8: targeted hypothesis attacks | Results 3.2; Figure 3 |
| Ballot diagnostics | Results 3.2; Table 2 |
| Information, budget, and reward controls | Results 3.6 |
| Mechanism interventions and their tradeoffs | Section 4 |
| Inclusion, seed-level inference, and multiplicity limits | Methods 2.4 |
| Mathematical assumptions and proof | Appendix A |
| Reproduction and implementation exceptions | Appendix B |

The shorter wording preserves the hypothesis rule's high baseline error, its
heuristic scores and deterministic ties, distinct reward definitions, incident-gated
versus whole-game outcomes, the five-seed and Holm resolution limits, fixed-ballot
counterfactual assumptions, and the limits of finite response training. It does not
turn descriptive differences into significant findings or claim a proven cycle.

Independent wording review also clarified that PPO schedules the **entropy
coefficient**, that no-ejection rewards cover tied ballots as well as abstention,
and that the actual median bound is distinct from the surrogate's amplification
claims. These clarifications do not change the experiments or their results.

## Files and checks

- [Original PDF](when_credibility_collapses.pdf) and [shorter PDF](paper_v2.pdf).
- [Editable v2 Markdown](paper_v2.md) and [standalone v2 TeX](paper_v2.tex).
- [Source map](data/paper_v2_sources.json) and [verification receipt](data/paper_v2_validation.json).

Build and check from the repository root:

```sh
bash paper/build_v2.sh
python paper/verify_paper_v2.py
```

To regenerate the two simplified figures from the existing validated data:

```sh
python paper/plot_paper_v2.py
```

The figure source manifest retains all 130 plotted cells and 1,300 seed records.
The verifier independently reconstructs every plotted mean and sample SD.

The verifier checks original-file hashes, reused figure identities and replotted
source values, retained
numeric table rows against the previously validated sources, preservation of the
mathematical proposition and proof, all 19 citations, caption numbering, 12-point
formatting, and PDF text bounds. Independent scientific wording and visual layout
reviews supplement these automated checks. The original `build.sh` and
`verify_manuscript.py` still operate on the original article.

## Scope of this revision

This is an editorial revision of the existing corrected 240-job study. It makes
no new empirical finding and does **not** incorporate the separate 72-seed
prospective validation. That study's desktop jobs and analysis finished on
12 September; recovery of its local data collection and independent raw-record
verification is underway. Its findings require a separate scientific integration
after verification, which should preserve this shorter structure.
