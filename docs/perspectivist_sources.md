# Beyond US politics: perspectivist sources and pluralism benchmarks

Every number in this project comes from **60 US-political survey questions**
judged against ~60 participants each. Two separate limitations follow, and they
need different fixes:

| limitation | fix | status |
|---|---|---|
| the GRAPH is US political survey data | perspectivist annotation datasets | loader written, no data |
| the EVAL is 60 questions | WildSCOPE / PluralEval | not started |

The second is the one blocking every conclusion: n=60 is why merge_v2 sits at
p=0.075, why 9 of 13 features flip sign between arms, and why the probe lands at
p=0.130.

## What the graph actually requires

Not surveys. Three things:

```
items  x  a partition of people into groups  x  per-group distributions over shared options
```

`parse_atp_dir` is one instantiation. The canonical record is

```python
{"qkey": str, "question": str, "options": [str],
 "attribute": str, "group": str, "dist": [float]}     # dist sums to 1
```

Everything after that record is **domain-agnostic** — MiniLM topic clustering,
the topic→subtopic→item→subgroup hierarchy, hyperbolic embedding, Wasserstein
fork scoring, `positions_from_subtree`. So a new source is a **parser**, not an
architecture change.

`data/loaders/perspectivist.py` is that parser, parameterised by column names
rather than written per dataset.

## Perspectivist annotation datasets

Subjective NLP tasks where every item is rated by many annotators whose
demographics are recorded. That is exactly the required shape, in non-political
domains.

### DICES-350

350 chatbot conversations rated for **safety** by 104 raters across age, gender,
race. Purpose-built for pluralistic alignment.

```python
load_perspectivist("dices350.csv",
                   item_col="item_id", text_col="context",
                   label_col="Q_overall",
                   rater_attr_cols=["rater_age", "rater_gender", "rater_race"],
                   options=["Yes", "No", "Unsure"])   # VERIFY the label set
```

### D3

Social-media comments rated for **offensiveness** by ~4000 raters balanced on
cultural region, gender and age. Much larger on the rater axis than ATP's 58
subpopulations.

**COLUMN NAMES ABOVE ARE UNVERIFIED.** They come from secondary descriptions, not
from opening the files. `parse_perspectivist_csv` fails with the available column
list when a name is wrong, so the first run tells you the truth — but do not
copy these into a writeup before checking.

## Two things that will differ from ATP, and both matter

**`min_group` must drop.** ATP uses 100 respondents per (question, attribute,
group). DICES has ~104 raters *in total*, so a demographic cell on one item is
single digits. The loader defaults to 15 and **raises** when the floor empties
the graph rather than returning an empty list. The cost is noisier per-group
distributions, and that noise propagates straight into the Wasserstein fork
scores. Report achieved group sizes; do not assume they are adequate.

**Label sets are small and often ordinal.** ATP options are survey answers with a
natural order. A 3-point safety scale inferred from data sorts *lexically*, which
would order `"1","10","2"` and destroy the ordering the spectrum rendering and
`pick_personas` depend on. Pass `options=` explicitly for anything ordinal.

## Benchmarks (the eval side)

### WildSCOPE / PluralEval — the one that matters

[ACL 2026](https://aclanthology.org/2026.acl-long.1957/). Reddit crowd
discussions across subjective domains, reported as ~1.2K threads (size not stated
on the abstract page — verify). Decomposes crowd responses into **atomic,
non-overlapping claims** and scores whether a model covers that claim space.

Three reasons it fits:

1. **~20x OvertonBench's 60 questions.** This is the direct fix for the constraint
   behind every null result here.
2. **Same construct, built differently.** Claim coverage vs OvertonScore's cluster
   coverage — the search notes Poole-Dayan et al. formalised Overton Pluralism via
   OvertonBench, so this is the same lineage.
3. **"Sycophancy collapse"** — pluralism degrades once a user's belief is
   revealed. merge_v2's lossless merge might resist that, which would be a second
   claim rather than a replication of the first.

### Evaluating Pluralism through Latent Perspectives

[arXiv:2606.13254](https://arxiv.org/html/2606.13254v1). Unsupervised,
domain-agnostic: extract aspects → cluster into perspective representations →
cluster again into collective perspectives. Needs **no human annotation**, so it
applies to arbitrary question sets — including the 1,492 graph questions that
`reward-labeled-delta` wants to label.

### Domain-specific, for the generality objection

- [Incorporating Diverse Perspectives in Cultural Alignment](https://aclanthology.org/2025.emnlp-main.862.pdf) (EMNLP 2025)
- [Pluralistic Alignment for Healthcare](https://arxiv.org/pdf/2509.10685)

### Venue

[PlurVA-LLM @ AACL 2026](https://www.aclweb.org/portal/content/first-cfp-first-workshop-pluralistic-value-alignment-llms-plurva-llm-aacl-2026)
explicitly calls for "benchmarks and evaluation protocols for pluralistic value
alignment". The routing nulls fit a workshop better than a main conference.

## Order of work

1. **WildSCOPE.** Fixes the power problem, which gates every existing conclusion.
   Needs a loader plus a claim-coverage scorer.
2. **DICES.** Parser is written; needs the file and one verification run. Proves
   the graph is not a politics artefact.
3. **Latent Perspectives.** Only if off-benchmark scoring becomes the bottleneck
   for label generation.

## The prerequisite nobody should skip

**Run the random-fork control first** (`docs/random_fork_control.md`). If
`merge_v2_rand` ties `merge_v2`, fork content is not load-bearing, the graph is
acting as a randomiser, and porting it to a new domain proves nothing. Generality
only matters once the content is shown to matter.
