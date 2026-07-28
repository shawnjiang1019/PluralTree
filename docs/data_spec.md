# PluralTree Data Specification

Data contract for the pluralistic pipeline. Anything that satisfies Plane A can
be encoded, retrieved over, and used as a GRPO reward; Plane B is the held-out
evaluation ground truth; Plane C is the optional path for building Plane A from
unstructured text (the CultureBank route). OpinionQA/ATP is the reference
instance of A + B.

**Hard rule — keep A and B disjoint.** Plane A (training/retrieval/reward) and
Plane B (eval) must not share questions. A question whose text matches a Plane B
item is dropped from training (`alignment/rollout_dataset.load_eval_holdout_texts`).

---

## Plane A — the pluralistic knowledge graph

Feeds: graph encoding (`data/loaders/*`), scout retrieval, GRPO reward
(`alignment/reward.py`). The distribution data here is what makes it *pluralistic*
rather than factual.

### A.1 Hierarchy nodes

A rooted tree, leaves-last topological order. Levels are illustrative (OpinionQA
uses 6); the contract is "≥3 levels ending in distribution-bearing leaves."

| field | type | required | notes |
|---|---|---|---|
| `node_id` | int | yes | 0-based, dense |
| `name` | str | yes | stable key, type-prefixed (`topic:`, `q:`, `ax:`, `op:`) |
| `type` | enum | yes | one of the level names (e.g. topic/subtopic/question/axis/opinion) |
| `text` | str | yes | human-readable description — **used for retrieval embedding** (`entity_text`) |
| `parent_id` | int | yes (except root) | one parent → the tree edge (`children_indices` inverse) |

Constraints: single root; every non-leaf has ≥1 child; every path root→leaf passes
through each level once.

### A.2 Opinion leaf = one subgroup's answer distribution (the core object)

Each leaf is a `(question × attribute × group)` cell. This is the unit the whole
system reasons over.

| field | type | required | notes |
|---|---|---|---|
| `question_key` | str | yes | groups leaves that share an option vocabulary |
| `question_text` | str | yes | the survey/opinion item |
| `options` | list[str] | yes | the answer choices; **identical across all leaves of a question_key** (`opinion_texts` = `"{question} {option}"`) |
| `attribute` | str | yes | the split dimension (e.g. POLIDEOLOGY, MARITAL) |
| `group` | str | yes | the subgroup within the attribute (e.g. "Very liberal") |
| `dist` | list[float] | yes | P(option) for this group; **len == len(options), sums to 1** (`opinion_dist`) |
| `support_n` | int | strong-rec | effective sample size behind `dist` (for smoothing/confidence) |
| `group_weight` | float | strong-rec | population share of `group` — needed to aggregate leaves to a **population** distribution, not an equal-weight mean (see `positions_from_subtree` fix) |

Constraints:
- `sum(dist) == 1 ± 1e-3`; `len(dist) == len(options)`.
- All leaves under one axis share `question_key`, `attribute`, and `options`.
- `dist` should be post-stratification-weighted *within* the group if the source
  sample is non-representative (ATP does this; raw social data cannot — see C).

### A.3 "Covering the middle" requirement

For the content-fix (`distributional`) and the GRPO reward to have a middle to
cover, an axis must expose the *graded spectrum*, not just two poles:

- **≥3 groups per axis** (else there is no non-pole subgroup). Axes with <3 fall
  back to 2-pole injection.
- Groups should span the pole→pole range. Measured by
  `evaluation/intrinsic/subtree_middle.py`: on OpinionQA 68% of non-pole groups
  lie between the poles, 10.9% of axes are bimodal. A new source should be run
  through that diagnostic; if mostly bimodal, the middle isn't in the data and
  only the model-supplies-it route (`expand`/`route`) applies.

### A.4 Derived at build time (not supplied)

`node_id↔name` maps, `children_indices`, `topo_order`, `type_constraints`,
train/val/test triple splits, and the trained Poincaré embeddings + MiniLM
`text_feat` — all produced from A.1–A.2. Not part of the input contract.

---

## Plane B — held-out human evaluation (OvertonBench)

Feeds: `evaluation/overton/judge_overtonbench.py`. This is the *only* source of
truth for whether coverage improved; never used in training.

### B.1 Participant rating records

One row per `(question, participant, reference_model)`; reference `elinorpd/overtonbench`.

| field | type | required | notes |
|---|---|---|---|
| `question_id` | int | yes | the eval question |
| `question` | str | yes | prompt text |
| `user` | str | yes | participant id |
| `freeresponse` | str | yes | the participant's OWN written perspective (few-shot anchor) |
| `cluster_kmeans` | int | yes | the participant's viewpoint cluster — **coverage is computed per cluster** |
| `model` | str | yes | which reference LLM produced the rated response |
| `llm_response` | str | yes | the rated response (few-shot example) |
| `representation_rating` | int 1–5 | yes | "is YOUR perspective represented?" — the label |

Constraints:
- ≥2 ratings per participant (needed for held-out validation and few-shot).
- `cluster_kmeans` assigned for every participant (coverage denominator).
- Enough participants per question to populate its clusters (`MAXU` caps, not floors).

### B.2 What it powers

- **Judge validation** — hold out one rating, predict it from the participant's
  perspective + their other ratings; must beat mean-of-others and be
  length-unbiased *before* any score is trusted.
- **Coverage** — cluster covered iff mean predicted rating ≥ 4;
  `OvertonScore = mean coverage`.
- **coverage@K** — K reference/generated responses per question enable the
  across-sample union metric (`--n_rollouts`, `--k_rollouts`).

---

## Plane C — optional: building Plane A from unstructured text

The CultureBank route (`docs/noveltybench_vs_overtonbench.md`,
`docs/data_spec` §C). Only needed if you don't have tabular survey data. Produces
A.2 leaves from raw items.

### C.1 Raw item

| field | type | required | notes |
|---|---|---|---|
| `item_id` | str | yes | dedup key |
| `text` | str | yes | the comment/post |
| `author_id` | str | strong-rec | **author-level dedup** — one vote per author, not per mention |
| `group` | str | opt | author's demographic/cultural group → enables subgroup cells; without it you only get the population marginal |
| `topic_hint` | str | opt | for hierarchy induction |
| `engagement` | float | opt | upvotes/likes → agreement weighting |
| `timestamp` | str | opt | provenance |

### C.2 Pipeline (raw items → A.2 leaves)

`filter relevance → extract (author, group, stance) → cluster stances into
positions (= options) → author-deduped weighted count per (group, position) →
smooth (Dirichlet) → normalize → attach support_n + agreement`. See
`docs/data_spec` discussion and `data/loaders/opinionqa.py:parse_atp_dir` for the
tabular analog. Output must satisfy A.2 (options shared per question, dist sums
to 1, ≥3 groups/axis for the middle).

### C.3 Validity caveat

C-derived distributions reflect **the platform's posters**, not a population —
no post-stratification. Expect the graph≠human-contestedness gap (corr ~+0.20 on
OpinionQA). A C-built graph must be re-validated against a Plane B set before its
coverage numbers mean anything.

---

## Capability → required data

| Capability | Plane A | Plane B | Plane C |
|---|---|---|---|
| Build + embed the graph | A.1, A.2 (dist, text) | — | — |
| Scout retrieval | A.1 `text`, A.2 | — | — |
| `distributional` injection (cover the middle) | A.2 + A.3 (≥3 groups) | — | — |
| GRPO reward | A.2 `dist`, `options`; A.2 `group_weight` for population aggregation | — | — |
| OvertonScore / coverage@K | — | B.1 (all fields) | — |
| Judge validation (gates everything) | — | B.1 + ≥2 ratings/participant | — |
| Ingest a new text corpus | (produced) | — | C.1 + C.2 |

Minimal to reproduce current results: **A.1 + A.2** (graph/reward) and **B.1**
(eval). `support_n`, `group_weight`, and all of C are enhancements.

---

## Validation checklist (run before trusting a source)

1. **Dist integrity** — every leaf `sum(dist)=1±1e-3`, `len(dist)=len(options)`.
2. **Option sharing** — all leaves of a `question_key` have identical `options`.
3. **Tree integrity** — single root, no cycles, every leaf reaches root through one node per level.
4. **Middle exists** — `subtree_middle.py`: report %contested axes with a graded middle vs bimodal. If mostly bimodal, `distributional` won't help.
5. **A/B disjoint** — no Plane A question text matches a Plane B question (leakage guard).
6. **Judge validity** — Plane B judge beats mean-of-others and is length-unbiased, else no coverage number is meaningful.
7. **(Plane C only)** representativeness — compare C-derived marginals to a known population benchmark; document the skew.
