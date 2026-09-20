# docs/ — what is in here and what to trust

Status tags: **CURRENT** (believe it), **SUPERSEDED** (kept for history, read the
successor), **PROPOSED** (designed, never run — no numbers), **BACKGROUND**
(earlier phase or context).

## Start here

| file | status | what it is |
|---|---|---|
| [handoff_2026_09_20.md](handoff_2026_09_20.md) | CURRENT | **new session starts here**: where things stand, why each direction was taken or dropped, what to run next, and the traps |
| [findings_2026_09_15.md](findings_2026_09_15.md) | CURRENT | newest results: ISSP transfer, hierarchy null, geometry-ranks-disagreement, judge ceiling, router closed |
| [methodology_and_results.md](methodology_and_results.md) | CURRENT | the claim ladder: what is established, what broke, what is pending |
| [methodology.tex](methodology.tex) | CURRENT | the method as a paper section (pdf beside it) |

## Results and status

| file | status | what it is |
|---|---|---|
| [findings_2026_09_15.md](findings_2026_09_15.md) | CURRENT | this session's runs, with caveats and pending jobs |
| [methodology_and_results.md](methodology_and_results.md) | CURRENT | supersedes `results_2026_09.md`; every number sourced to a job log |
| [results_2026_09.md](results_2026_09.md) | SUPERSEDED | September snapshot; still the only write-up of the CAD α-sweep (§2) |
| [reward_gate_failure.md](reward_gate_failure.md) | CURRENT | why the GRPO coverage reward does not rank like the judge (76–96% zeros) |

## Method specifications (LaTeX, compile standalone)

| file | status | what it is |
|---|---|---|
| [methodology.tex](methodology.tex) | CURRENT | the method, the ablation table, the evaluation protocol |
| [established_claims.tex](established_claims.tex) | CURRENT | what the measurements support, what is ruled out, what is untested (split out of `methodology.tex`) |
| [measurement.tex](measurement.tex) | CURRENT | what the numbers identify: attenuation, validity, noise floors, inference |
| [grpo_method.tex](grpo_method.tex) | CURRENT | the RL objective, including the group-diversity variant and its gate |
| [vital_task.tex](vital_task.tex) | CURRENT | the VITAL secondary task: data, scoring, the length cap, valid comparisons |

## Data and graphs

| file | status | what it is |
|---|---|---|
| [data_spec.md](data_spec.md) | CURRENT | the data contract: what any source must provide to be encodable |
| [issp_graph.md](issp_graph.md) | CURRENT | building the ISSP graph and every transformation that made it embeddable |
| [perspectivist_sources.md](perspectivist_sources.md) | CURRENT | beyond US politics: DICES, D3, WildSCOPE, latent perspectives |
| [CULTUREBANK.md](CULTUREBANK.md) | BACKGROUND | CultureBank as a plurality-carrying KG |

## Benchmarks and evaluation

| file | status | what it is |
|---|---|---|
| [overtonbench_eval.txt](overtonbench_eval.txt) | CURRENT | the primary eval: protocol, judge, scoring |
| [noveltybench_vs_overtonbench.md](noveltybench_vs_overtonbench.md) | CURRENT | across-sample vs within-answer diversity; do not conflate |
| [EVALUATION.md](EVALUATION.md) | BACKGROUND | intrinsic embedding metrics (geometry, not coverage) |
| [hivemind_diversity_eval.txt](hivemind_diversity_eval.txt) | CURRENT | INFINITY-CHAT mode-collapse eval design |
| [hivemind_diversity_eval_plan.md](hivemind_diversity_eval_plan.md) | CURRENT | its metric panel |
| [hivemind_diversity_concepts.md](hivemind_diversity_concepts.md) | CURRENT | why circularity matters; the two diversity notions |

## Designs not yet run (no numbers)

| file | status | what it is |
|---|---|---|
| [random_fork_control.md](random_fork_control.md) | PROPOSED | the cheapest test of whether fork CONTENT is load-bearing |
| [cad_experiment.md](cad_experiment.md) | PROPOSED (partly run) | context-aware decoding as a dial; the α-sweep results live in `results_2026_09.md` §2 |
| [untested_test_time_methods.md](untested_test_time_methods.md) | PROPOSED | selection, decode-time, multi-pass and graph-side options |
| [adaptive_injection.md](adaptive_injection.md) | PROPOSED | three routing adaptations — see `findings_2026_09_15.md` §7 before investing |
| [single_answer_coverage_policy_design.md](single_answer_coverage_policy_design.md) | PROPOSED | the single-answer GRPO policy and its go/no-go; the reward failed its gate |
| [grpo_alignment.txt](grpo_alignment.txt) | CURRENT | RL phase design and implementation status |

## Retrieval and architecture

| file | status | what it is |
|---|---|---|
| [scout_design.txt](scout_design.txt) | CURRENT | the divergence scout: relevance gate, OT, fork scoring |
| [embedding_diversity.txt](embedding_diversity.txt) | CURRENT | the encoder and subtree-diversity objective |
| [GATING.md](GATING.md) | BACKGROUND | alternative gating mechanisms for knowledge injection |
| [PLAN.md](PLAN.md) | BACKGROUND | the original GKI + Tree-GRU plan |
| [map_reader.md](map_reader.md) | BACKGROUND | SFT tasks for teaching a model to read hyperbolic coordinates |

## Literature

| file | status | what it is |
|---|---|---|
| [related_work.md](related_work.md) | CURRENT | architecture side: hyperbolic geometry, KG embedding, tree encoders |
| [related_work_rag_diversity.md](related_work_rag_diversity.md) | CURRENT | KG-RAG × diversity, plus the E1–E6 experiment shortlist |
| [HYPERKGR_COMPARISON.md](HYPERKGR_COMPARISON.md) | BACKGROUND | closest prior work, compared claim by claim |

## Earlier phase (KG benchmarks, before the pluralism pivot)

| file | status | what it is |
|---|---|---|
| [WN18RR.md](WN18RR.md) | BACKGROUND | deep-hierarchy benchmark integration |
| [LABEL_LEAKAGE.md](LABEL_LEAKAGE.md) | BACKGROUND | a label leak through question text, and its fix |
| [EXPERIMENTS.md](EXPERIMENTS.md) | BACKGROUND | the older experiment roadmap |

## Planning and raw notes

| file | status | what it is |
|---|---|---|
| [direction.txt](direction.txt) | CURRENT | tiered next-steps list (item 8 = trace-consistency reward) |
| [ideas.txt](ideas.txt) | BACKGROUND | unfiltered idea dump |

## Generated artifacts

Written by code, not by hand — do not edit, and expect them to be overwritten:

- `*.png` — plots from `scripts/analysis/plot_*.py` (`framing_hurts.png`,
  `bars_v11.png`, `overton_v4_v5.png`, `opinionqa_train_metrics.png`, …)
- `*.csv` — analysis outputs (`pole_collapse.csv`, `reward_eval_correlation.csv`,
  `anchor_cov_*.csv`, `coverage_funnel_*.csv`, …)
- `PluralTree_Overview.pdf` — `scripts/analysis/build_overview_pdf.py`
- `*.pdf` beside a `.tex` — `pdflatex <file>.tex`
- `figures/` — LaTeX figure sources (`.tex`/`.svg`/`.png`) plus untracked
  `.aux`/`.log` build litter, which is safe to delete
- `results/` — LaTeX result fragments included by the paper

## Conventions

- **New results go in a dated file** (`findings_YYYY_MM_DD.md`), and
  `methodology_and_results.md` is updated to point at it. Do not rewrite an older
  results file in place: its numbers are the record of what was believed when.
- **Every number names its job** (id or script) and its caveat. Anything
  unmeasured is marked as such.
- **Markdown and text files stay at the top level of `docs/`.** Roughly a hundred
  docstring and job-script references point at `docs/<name>`, and several scripts
  default to reading or writing `docs/<name>.csv`, so moving them into
  subfolders would break those paths for no benefit.
- **Status tags above are part of the contract.** When a design is run, move it
  from PROPOSED and link the results file.
