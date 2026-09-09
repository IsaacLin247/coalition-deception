# Coalition Deception and Defensive Adaptation

Research code and the current manuscript for **Coalition Deception and Defensive
Adaptation in a Social-Deduction Benchmark**, by Isaac Lin and Xi Chen.

This is the cleaned public edition of the project following a publication audit.
It contains the corrected implementation, tests and reproduction instructions,
alongside the latest paper and a record of the audit changes.

**Research status — 9 September 2026:** the full corrected replication is still
running. The manuscript incorporates completed corrected F1/F3 and mechanism
results, and explicitly labels the remaining archived results as provisional.
This repository is a code-and-manuscript release; it is not a completed data
release or a claim of journal acceptance.

## Start here

| Item | Contents |
| --- | --- |
| [Paper PDF](paper/when_credibility_collapses.pdf) | Latest manuscript, including limitations and provisional-result labels. |
| [Paper source](paper/) | Markdown, generated TeX, bibliography, required figures and a self-contained build. |
| [Code](code/README.md) | Environment, policies, training, evaluation, analysis, tests and replication protocols. |
| [cahnges.md](cahnges.md) | Audit corrections, new findings, remaining work and packaging changes. |

The environment models private observations, public testimony, coalition
coordination and voting. It supports scripted and learned policies, defensive
aggregation rules, and repeated coalition/defender adaptation experiments.

## Install and check

Python 3.10 or later is required. From this repository's root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e 'code[rl,env,analysis,dev]'
cd code
python -m pytest
python experiments/submission/run_study.py list
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. The [code guide](code/README.md) documents the complete study and
supplemental evaluations. Full replication requires substantial computation;
listing the plan and running tests do not start model training.

The packaged protocols describe new reproductions with the same planned jobs,
seeds and budgets. Their source hashes identify this cleaned package. The
original replication protocol is retained under
[`code/provenance/`](code/provenance/) to distinguish the provenance of the
ongoing experiments and reported corrected results.

## Build the paper

```sh
cd paper
bash build.sh
```

The build requires Pandoc and XeLaTeX. See [the paper guide](paper/README.md) for
details. The checked-in PDF can be read without installing the build tools.

## License and attribution

The original [MIT license](LICENSE) and copyright notice are retained. This
edition derives from the
[original project](https://github.com/xi10017/when-credibility-collapses).
The bundled citation style retains its own license and attribution, documented
in the paper directory. Citation metadata is in [CITATION.cff](CITATION.cff).
