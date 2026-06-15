# CultureBank — a community-sourced cultural KG with a built-in plurality signal

How [CultureBank](https://aclanthology.org/2024.findings-emnlp.288/) fits PluralTree,
and the concrete ways to use it. It complements the §E4 plurality datasets in
[`EXPERIMENTS.md`](./EXPERIMENTS.md): GlobalOpinionQA is the *clean-tree, full-distribution*
slot; CultureBank is the *messy-DAG, real-overlapping-groups* slot (the natural home for
the C3 multi-parent / E2 pluralistic-existence experiments).

Sources: [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.288/) ·
[arXiv 2404.15238](https://arxiv.org/html/2404.15238v1) ·
[GitHub SALT-NLP/CultureBank](https://github.com/SALT-NLP/CultureBank)

---

## 1. What it is

A cultural knowledge base mined from social media (Shi et al., EMNLP Findings 2024):

- **~12K descriptors from TikTok** (730 cultural groups, 36 topics)
- **~11K descriptors from Reddit** (1,850 cultural groups, 36 topics)
- Built by an LLM pipeline: culture-classify → extract → cluster (Hierarchical
  Agglomerative Clustering over SentenceBERT embeddings, min cluster size 5) →
  summarize → normalize groups/topics.
- Human eval: 84.5% of TikTok entries judged "meaningful" cultural insights (→ ~15% noise).

### Record schema (Table 1 in the paper)

Each descriptor is a **structured record**, not free text:

| Field | Meaning |
|---|---|
| `cultural group` | groups of people with similar cultural backgrounds |
| `context` | setting the behavior takes place in |
| `goal` | what the behavior aims to achieve |
| `actor` | who exhibits the behavior |
| `recipient` | recipient of the action |
| `relation` | relation between actor and recipient |
| `actor's behavior` | behavior of the actor |
| `recipient's behavior` | behavior of the recipient |
| `other description` | anything that doesn't fit the other fields |
| `topic` | one of 36 topics |
| **`agreement`** | **% of the cluster who agree, a one-decimal float in [0,1]** |

**Cultural groups are multi-granularity and overlapping**: nationalities (American,
Norwegian), sub-national regions (Californian), ethnicities (Asian American), social
categories (international student). This is the paper's deliberate move beyond
nationality-only grouping.

---

## 2. Why it fits PluralTree

### 2.1 `agreement` is a built-in plurality signal — no survey needed

The same behavior across different groups carries different agreement levels. That
**spread is empirical E3 variance — measured, not invented**, which is exactly the gap
§E4 flags about the current synthetic plurality.

- **Caveat:** `agreement` is a single scalar per (group, behavior) — effectively a
  2-point `[agree, disagree]` (Bernoulli) distribution, **not** GlobalOpinionQA's full
  multi-option distribution. Thinner distributional target, richer/messier group structure.

### 2.2 Overlapping multi-granularity groups are a natural DAG

"Asian American" overlaps "American" and "Asian"; "Californian" sits under "American."
This is **not a clean tree** — it is the **multi-facet DAG** that §E4 assigns to the
OpinionQA slot (C3 multi-parent + E2 pluralistic existence). CultureBank may be a
*better* fit for the DAG / multi-parent-conditioning experiments than GlobalOpinionQA's
clean geographic tree.

---

## 3. Concrete uses (mapped to existing code)

**A. Drop-in richer KG** — a `load_culturebank()` sibling to `load_culturalbench`
(`data/culturalbench.py`) returning a `CulturalGraph`. Descriptors → leaf nodes (the
templated `actor + behavior + context` text → frozen MiniLM features via
`compute_text_embeddings`); cultural groups → parent nodes. The geographic subset
attaches to the existing `REGION_TO_COUNTRIES` tree; non-geo groups form the DAG layer.
Bigger, culture-themed, community-sourced — vs. the leaky/shallow CulturalBench.

**B. Empirical plurality target — reuse the representativeness metric.**
`evaluation/representativeness.py` already takes ragged per-group distributions. Feed it
`[agreement, 1 − agreement]` as the 2-bin target per (group, behavior); the model
predicts the same 2-bin distribution from the per-parent existence `h_{behavior|group}`.
JS-divergence works unchanged. Instantiates E2/E3 on real data — same machinery, thinner
target than GlobalOpinionQA.

**C. Behavioral / Overton eval — the behavioral-plurality gap.** CultureBank ships
contextualized scenarios. Given a context, score whether the LLM (grounded in retrieved
group-specific descriptors) **surfaces the range of group behaviors** (coverage) weighted
by `agreement` (faithfulness). A *culturally-grounded* Overton benchmark — a better
thematic fit for this project than ValuePrism's moral-values framing, on the project's
own domain. (See the epistemic-vs-behavioral plurality distinction discussed for the
D-track in [`EXPERIMENTS.md`](./EXPERIMENTS.md) §D / §E.)

**D. Link prediction with soft targets.** behavior → group prediction (like
`practiced_in`), with `agreement` as a **weighted/soft label** rather than a hard triple;
type-constrained negatives by group. A natural variant of the current margin loss.

---

## 4. Caveats (vs. CulturalBench / GlobalOpinionQA)

- **Groups need normalization.** Overlapping facets, not a ready tree — requires
  hand-authored group→parent mapping (more work than `REGION_TO_COUNTRIES`), and the
  honest structure is a **DAG**, not a tree.
- **Noisy & non-representative.** Social-media sourced (~15% noise); coverage follows
  online popularity, **not** survey sampling design. Unlike WVS/Pew, the agreement does
  **not** reflect a representative population — frame as *community-expressed* plurality,
  not population-representative.
- **Label-leakage risk returns.** Descriptor text often names the group (the
  CulturalBench `mask_country` problem). Apply the same masking discipline and always
  report the frozen-NN floor — see [`LABEL_LEAKAGE.md`](./LABEL_LEAKAGE.md).
- **Thin distributions.** Bernoulli `agreement`, not full option distributions — good for
  variance/plurality, weaker for "distribution over many answers."

---

## 5. Recommendation

Use CultureBank as the **DAG / multi-facet plurality dataset** (the C3/E2 slot),
**complementing** GlobalOpinionQA (clean-tree, full-distribution slot) — not replacing it.
The two are complementary the way WN18RR and PrimeKG are on the structure side:
GlobalOpinionQA gives clean tree + rich distributions; CultureBank gives real overlapping
groups + a built-in (if thin) agreement signal + a route to a cultural Overton benchmark.

| Dataset | Structure | Plurality target | Role |
|---|---|---|---|
| GlobalOpinionQA | clean tree (World→Region→Country) | full multi-option distribution | flagship distributional/epistemic |
| **CultureBank** | **multi-facet DAG (overlapping groups)** | **2-bin agreement (thin) + scenarios** | **C3/E2 DAG + cultural Overton** |
| OpinionQA | multi-facet DAG (demographic facets) | full distribution per facet | richer distributional DAG (do later) |
