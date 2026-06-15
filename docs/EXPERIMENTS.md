# PluralTree — Experiment Roadmap

This document lays out candidate next experiments and research directions, grouped
by how far they push beyond the current architecture. It is meant as a planning
reference, not a fixed schedule.

**Current status:** The combined Hyperbolic Tree-GRU + GKI model trains end-to-end
on CulturalBench link prediction. Best test results so far (MiniLM encoder,
`d_hidden=128`, 100 epochs): MRR ≈ 0.456, Hits@1 ≈ 0.35, Hits@10 ≈ 0.67. A run with
a stronger encoder (`all-mpnet-base-v2`) and recalibrated curriculum
(`warmup1=400`, `warmup2=1600`, 300 epochs) is the next baseline.

---

## Positioning vs. HyperKGR (closest prior work)

**Liu, *HyperKGR* (EMNLP 2025)** is the nearest published system and overlaps
heavily with our *mechanics*: Poincaré ball, recursive tree in hyperbolic space, GRU
gating, learnable curvature, relation-as-translation, link-prediction MRR/Hits. We
must position against it deliberately.

- **What it does that we don't (yet):** embeds a *per-query message-passing tree*
  (dynamic, query-specific embeddings; a hyperbolic GNN in the NBFNet lineage), with
  a DP=GNN theorem and SOTA results on FB15k-237 / WN18RR / NELL-995 / Family / UMLS,
  in both transductive and inductive settings.
- **What we do that it doesn't:** (1) **external knowledge injection (GKI)** with
  **depth-aware radius gating**; (2) a *fixed semantic hierarchy* encoded once (vs. a
  query-specific computation tree); (3) the **plurality / distributional** directions
  (E1–E3); (4) **learned/soft hierarchy** (E4); (5) the **LLM-reasoning end goal** (D).

**Consequences for the roadmap:**
1. Do **not** frame contributions as "hyperbolic recursive tree + GRU for link
   prediction" — HyperKGR owns that claim. Lead with GKI + plurality + LLM grounding.
2. Treat HyperKGR-class GNN reasoners (NBFNet, RED-GNN, AdaProp) as the **real
   baselines** (sharpens A3). If our MRR can't match them on a fixed KG, the pure-LP
   story is weak — push the value to knowledge injection and downstream LLM use.
3. Its **query-specific embeddings** are a useful template for a context-conditioned
   encoder, which is essentially the bridge to E2 (Pluralistic Leaf Existence) and E4
   (learned attachment).

See `RELATED_WORK.md` (top callout + §2) for the full contrast.

---

## A. Validate the current architecture (do first — low risk, high value)

These confirm that each component earns its place before we build on top of it.
A warning sign already appeared: in the first run, opening the gates *reduced* MRR,
so we cannot yet claim GKI is helping.

### A1. Component ablations
Run the same training while disabling one component at a time:

| Variant | What it tests |
|---------|---------------|
| Gate forced to 0 (no GKI) | Does knowledge injection help at all vs. pure Tree-GRU? |
| Euclidean instead of hyperbolic | Does the Poincaré ball beat flat space on this hierarchy? |
| `HyperbolicGate` instead of `DepthAwareHyperbolicGate` | Does conditioning on radius ρ matter? |
| Each injection point (`pre_agg`, `post_agg`, `post_gru`, `dual`) | Which injection location is best? |

**Deliverable:** a table of MRR / Hits@k per variant. This is the single most
important next step.

### A2. Verify ρ encodes depth
Use `evaluation/gate_analysis.py` to plot:
- Poincaré radius ρ vs. true tree depth (World / Region / Country / Practice)
- Gate activation vs. depth

The entire depth-aware story rests on ρ tracking hierarchy level. If the
correlation is weak, the architecture's premise needs revisiting.

### A3. Stronger baselines
Compare against:
- Off-the-shelf TransE / RotatE
- Frozen sentence-transformer embeddings + cosine nearest-neighbor (no Tree-GRU)

If a frozen encoder + nearest-neighbor matches the full model, the pipeline isn't
adding much — an essential honesty check.

---

## B. Architectural extensions (medium risk)

### B1. Stronger / better frozen input embeddings *(in progress)*
- Swap MiniLM (384-d) → `all-mpnet-base-v2` (768-d). Trivial flag change; `d_input`
  is derived automatically.
- **Enrich entity descriptions** before embedding (e.g. inject hierarchy context
  into the text). Possibly higher leverage than the model swap.
- Keep embeddings **frozen** — do not unfreeze. Unfreezing reintroduces a per-entity
  lookup table (≈500K params on ~1K triples), kills inductive generalization, and
  confounds ablations.

### B2. Multiple, genuinely distinct knowledge sources
Currently the input feature and the knowledge vector are the same MiniLM embedding,
so `KnowledgeSelector` and the broad/precise blending do nothing. Add a second
distinct source (structured country attributes, or pretrained Wikidata KG
embeddings). Only then does the depth-aware blend `β` become meaningful: broad
relational knowledge for abstract nodes, precise factual knowledge for leaves.

### B3. Learnable curvature
`curvature_regularization` and a learnable `c` already exist. Let the model learn
curvature, possibly per-depth or per-layer — different hierarchy levels may prefer
different curvature.

### B4. Symmetric parent–child conditioning (top-down + bottom-up message passing)

**The asymmetry today.** Conditioning is currently one-directional. The bottom-up
pass means a **parent is conditioned on its children** (a country's `h` is an
attention-weighted aggregate of its practices; a region's on its countries; the
world's on everything below). But a **child is *not* conditioned on its parents**: a
practice leaf is encoded from its own text (plus GKI) alone, with `h_agg = 0` — it
never learns which country / region / world it sits under. In the leakage-safe setup
held-out practices are therefore fully isolated leaves with zero structural context,
which likely explains why the task is nearly solvable from question text alone
(no_gki test MRR ≈ 0.94): the hierarchy only shapes the country/region side of the
score, not the practice side.

**The goal: make conditioning symmetric.** Move from "children shape parents" to
"children and parents shape each other," so every node's representation reflects both
its descendants *and* its ancestral context — the tree analogue of a bidirectional
RNN.

**Recipe.**
1. **Up pass** (current): leaves → root, producing `h_v^↑` (node informed by its
   subtree).
2. **Down pass** (new): root → leaves. Each node mixes its own `h_v^↑` with a message
   from its parent's downward state `h_parent^↓` (e.g. a Möbius midpoint / a second
   GRU cell whose "children" slot is fed the parent). A practice then "knows" it is in
   Japan → East Asia → World.
3. **Combine:** final `h_v = combine(h_v^↑, h_v^↓)` (concat+project in tangent space,
   or a gated Möbius blend), keeping everything on the ball.

**Why it's high-leverage.**
- *Accuracy:* gives leaves the structural signal they currently lack, and makes the
  GKI gate genuinely depth-aware on the way down (A2 premise).
- *Research story:* symmetric conditioning is the prerequisite for **E2 (Pluralistic
  Leaf Existence)**. Once a child can be conditioned on a parent, the *same* leaf can
  be conditioned on *different* parents to yield a distinct existence per branch
  (`h_leaf|Japan ≠ h_leaf|UK`), whose spread is the measured variance of **E3**. So
  B4 is the natural bridge from a static tree to the plurality/distributional
  directions — and a way to approach HyperKGR-style query/context conditioning
  without going fully query-dynamic.

---

## C. Scaling and generalization (higher risk, higher payoff)

### C1. A real, deeper hierarchical KG
CulturalBench's tree is shallow (4 levels) and small (1284 nodes). Test on WN18RR
or a Wikidata taxonomy subset — deeper, larger trees are where hyperbolic geometry
should win decisively. Configs for WN18RR / FB15k-237 already referenced.

### C2. Inductive evaluation (a strong, distinctive claim)
Because entity embeddings are *computed* (not stored), the Tree-GRU should
generalize to unseen nodes. Hold out entire subtrees during training, then test
whether the model produces sensible embeddings from text + structure alone. Most
KG embedding methods cannot do this — worth demonstrating explicitly.

### C3. DAG instead of strict tree
Allow multiple parents per node (a practice in multiple countries; a country in
multiple regions). Generalizing the encoder to DAGs widens applicability and is the
technical prerequisite for the pluralistic directions in section E.

---

## D. LLM integration (the motivating end goal)

### D1. Hierarchy-aware retrieval vs. flat RAG
Build a coarse-to-fine retrieval pipeline (walk World → Region → Country → Practice,
collecting context at each level) and measure whether tree-structured context
improves an LLM on a downstream cultural-reasoning QA task vs. flat RAG. This is the
experiment that justifies the whole project.

### D2. Embedding-guided traversal as an LLM tool
Expose a `find_related(query, level, k)` tool backed by hyperbolic nearest-neighbor
search over `h_all`. The LLM calls it mid-reasoning to surface hierarchically
related entities with their tree paths.

### D3. Soft-prompt injection
Project `h_v` into the LLM's token-embedding dimension and prepend as soft prompt
tokens. Riskiest but most novel — tests whether learned hyperbolic structure is
directly consumable by a transformer.

### D4. Uncertainty-calibrated prompting
Once distributional embeddings exist (section E), use variance to hedge language:
low-variance entities asserted as fact, high-variance entities presented as general
tendencies. Reduces hallucination from over-anchoring.

---

## E. Novel research directions — plurality & distributions

These are the conceptually deepest directions and align with the project's name.
Treat as contributions to build toward, after sections A–B are in place.

### E1. Distributional embeddings (parametric route)
Represent each node as a distribution rather than a point:
`h_v ~ WrappedNormal(μ_v, Σ_v)` — a Gaussian in the tangent space wrapped onto the
ball via the exponential map.
- Predict `μ_v`, `Σ_v` per node; sample via reparameterization.
- Score with a distribution divergence (KL / Wasserstein) instead of geodesic
  distance: `score(s,r,o) = -KL(N(μ_s ⊕ r, Σ_s) || N(μ_o, Σ_o))`.
- Benefits: explicit uncertainty, probabilistic **entailment** (containment of one
  distribution in another certifies hierarchy membership), confidence-aware scoring.
- Requires a KL regularizer to prevent variance collapse.

### E2. Pluralistic Leaf Existence (multi-branching)
An entity is not locked into a single branch. A single leaf attaches to multiple
subtrees simultaneously and gets a **distinct, context-refined representation in
each**:
`h_leaf|A ≠ h_leaf|B ≠ h_leaf|C`.

This is stronger than a plain DAG (C3): a DAG gives multiple parents but still one
embedding; pluralistic existence gives a *family* of context-conditioned vectors —
no single canonical representation.
- **Conditioning interpretation:** each existence is `f(x_leaf, context_parent)` —
  the entity's invariant core `x_leaf` modulated by parent context. Conditioning
  operates geometrically (different ρ per branch) and informationally (different
  gated knowledge per branch).
- **Use cases:** polysemy ("tea" in British vs. Japanese vs. medicinal subtrees),
  cross-cultural transfer (shared practice core vs. culture-specific refinement),
  richer LLM grounding (return all contextual existences with their paths).
- **Architectural changes:** allow an entity id at multiple positions in
  `children_indices`; `h_all` becomes `(N_positions, d)` with a position→entity map;
  decide scoring reconciliation (nearest existence / max over existences / mixture);
  add a consistency-vs-divergence regularizer controlling how far existences may
  drift from a shared core.

### E3. Plurality as the source of distribution (E1 × E2)
The cleanest synthesis: the set of conditional existences
`{ h_leaf|p : p ∈ parents }` **are empirical samples** from the entity's
distribution. The spread *is* the uncertainty — measured, not parameterized — and
it carries a concrete meaning: how much the entity's representation depends on
context.
- Recover a parametric distribution cheaply from the samples using existing ops:
  `μ_v = möbius_midpoint(existences)`, `Σ_v = spread in tangent space at μ_v`.
- Variance is interpretable: many diverse parents → high variance → polysemous /
  context-dependent; one or similar parents → low variance → stable.
- Multiple *separated* clusters of existences → genuine polysemy → represent as a
  **mixture**, surface distinct senses to the LLM.
- **Caveats:** a leaf under 2 parents gives only 2 samples — a thin estimate;
  likely combine with the parametric head (E1) as a prior refined by the samples.
  Data thinness makes this a larger-KG direction.

### E4. Plurality benchmarks — GlobalOpinionQA / OpinionQA
E1–E3 are currently *synthetic aspirations*: the plurality is constructed, the
variance invented. These two opinion-survey datasets make plurality **empirical** —
the target is a *distribution over answers per group*, with no single ground truth,
so different groups legitimately differ. This is the natural flagship benchmark for
the plurality track (and it sidesteps the CulturalBench label-leakage failure: the
label is a distribution to represent, not a country name to string-match — see
`LABEL_LEAKAGE.md`).

- **GlobalOpinionQA** (World Values Survey + Pew Global) → a clean **tree**:
  World → Region → Country, each opinion question a leaf whose per-country answer
  distribution is its **per-parent existence** (`dist(q | Japan) ≠ dist(q | UK)`).
  Reuses the existing `REGION_TO_COUNTRIES` scaffold. **Start here.**
- **OpinionQA** (Pew American Trends Panel) → groups defined by *cross-cutting
  facets* (age, sex, race, education, income, religion, politics, region): a
  **multi-facet DAG**, not a tree. Richer, and a genuine stress test of multi-parent
  conditioning (C3) — do *second*.

**Why it fits the architecture (and the recent work).**
- Direct instance of **E2** (per-parent existence) and **E3** (measured variance):
  the spread of a question's distribution across countries *is* the plurality.
- **B4 (top-down conditioning) is the mechanism** — a leaf can only acquire a
  *country-specific* existence if it is conditioned on its parent. So this dataset is
  the proving ground for `--bidirectional`.
- **Distributions over a tree** are exactly what a Tree-Wasserstein distance compares
  (see `EVALUATION.md` Tier-2/4 and the TWD note): "is a country's opinion
  distribution closer to its regional siblings than to a random country?" The
  dual-space split also fits — a hierarchy head for *where a group sits*, a
  distributional head for *what it believes*.

**Evaluation.** Distribution alignment per group, scored with **Wasserstein / JS
divergence** (the papers' "representativeness" metric) — a real-distribution version
of the `STRUCT` geometry checks. Downstream (D-track): retrieve the region/country
subtree, condition the LLM, and measure whether group-conditioned generation aligns
better than flat — i.e., **pluralistic reasoning for an LLM** against an established
benchmark.

**Caveats.**
- *Not* a KG / link-prediction benchmark — repurposed as node-attribute
  distributions over a hierarchy. Complements PrimeKG/WN18RR; does not replace them.
- Shallow hierarchy (~3 levels, ~50–100 countries): a **plurality** test, not a
  depth/scale test (that is PrimeKG's job).
- Representational framing: the task is to *faithfully represent the spread*
  (including disagreement), explicitly **not** to endorse any group's view or treat a
  majority as "correct." Survey snapshots age; coverage is incomplete (not every
  question asked in every country) — that missingness maps onto partial-coverage DAG
  structure but is noise to handle.

---

## Recommended sequencing

1. **Section A** (ablations + ρ-vs-depth + baselines). Do not build on components
   not yet shown to help. The "gates opening hurt MRR" signal makes this urgent.
2. **B1 / B2** — stronger encoder (underway) and a genuinely distinct second
   knowledge source, so the selector and depth-aware blending finally do something.
3. **C2** (inductive eval) — a distinctive, demonstrable claim.
4. **D1** — close the loop back to the motivating LLM goal.
5. **E1 → E2 → E3** — the novel contributions, on a larger KG (C1), with E3 as the
   flagship synthesis (plurality producing distributions). Anchor these on **E4
   (GlobalOpinionQA)** for *empirical* plurality — it instantiates E2/E3 on the
   existing geographic tree and exercises B4 + the distributional eval.

---

## Quick reference: what each direction needs

| Direction | Code touched | Data risk | Novelty |
|-----------|-------------|-----------|---------|
| A1–A3 ablations | configs, train flags | none | validation |
| B1 encoder | flag + download | low | low |
| B2 second source | `knowledge/`, `train.py` | low | medium |
| B3 curvature | `manifolds/`, losses | low | medium |
| B4 top-down | `combined/` encoder | medium | medium |
| C1 larger KG | new data loader | low | medium |
| C2 inductive | eval split logic | medium | high |
| C3 DAG | `tree_builder.py`, encoder | medium | medium |
| D1–D4 LLM | new inference module | medium | high |
| E1 distributional | scoring, losses, cell | high | high |
| E2 pluralistic | `tree_builder.py`, scoring | high | high |
| E3 synthesis | E1 + E2 combined | high | flagship |
