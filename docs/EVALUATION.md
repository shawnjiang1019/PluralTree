# Evaluating PluralTree Embeddings

How we judge whether the embeddings are *good* — not just whether they predict
links. This file exists because the **end goal is geometry**: we want a knowledge
graph encoded so that **subtree similarity** and **path similarity** are
recoverable from distances, so an LLM can be handed diverse, structurally-related
context for reasoning. Link prediction is a training signal and a sanity anc
hor;
it is **not** the primary yardstick.

See also `EXPERIMENTS.md` (roadmap), `LABEL_LEAKAGE.md` (why we distrust a single
high MRR), and `HYPERKGR_COMPARISON.md` (why our static, reusable embeddings are
the point).

---

## 1. Why link prediction alone is the wrong target

Link prediction (LP) optimises one narrow thing — the relational translation

```
score(s, r, o) = −d_H(h_s ⊕_c r, h_o)        # training/scoring.py
```

holds for true triples. A model can post a strong filtered MRR while the **global
metric structure is muddled**, because LP only rewards a *local*,
relation-shifted neighbour — not whether distances and neighbourhoods across the
whole space are faithful to the hierarchy.

Our downstream use is the opposite. Retrieval for an LLM lives or dies on whether
distances and subtree neighbourhoods are meaningful:

- **Subtree similarity** — are nodes in the same subtree close, and is closeness
  graded by how much subtree they share?
- **Path similarity** — does the geometry recover ancestor/descendant relations
  and path overlap?

Neither is what the margin-ranking LP loss rewards. So selecting models on MRR
can quietly degrade exactly the property we need. (This is the classic
Poincaré-embedding observation: hierarchy fidelity and link prediction are
different objectives.)

**Decision:** keep LP as a *training objective* and a *secondary* reported
number; make the structure-fidelity metrics below the *primary* model-selection
criteria.

---

## 2. The structure-fidelity suite

All of these are **intrinsic**: they take the trained `h_all` (the `(N, d_hidden)`
static embeddings from `encode_tree`) plus `children_indices` / `topo_order`, and
need no training loop. Target home: `evaluation/structure_metrics.py`, run under
`torch.no_grad()`. Ordered by closeness to the end goal.

### Tier 1 — Hierarchy faithfulness (the foundation)

| Metric | What it measures | Definition |
|---|---|---|
| **Distortion** | Does the embedding *become* the tree metric? | mean over pairs of `\|d_H(h_u,h_v) − d_tree(u,v)\| / d_tree(u,v)` (sampled pairs at scale) |
| **Reconstruction MAP / mean rank** | Are true neighbours nearest? | for each node, rank all others by `d_H`; average-precision of its true parent/children/ancestors (Nickel–Kiela style) |
| **Depth–radius correlation** | Does `‖h‖` encode depth? | Spearman ρ between `‖h_v‖` and tree depth — **verifies the depth-aware design's core assumption.** Cheapest; run first. |

### Tier 2 — Subtree similarity (the actual ask)

| Metric | What it measures | Definition |
|---|---|---|
| **Same-subtree retrieval AP** | Do subtree-similar things cluster? | pick a cut level `k`; "relevant" = shares an ancestor at level `k`; rank by `d_H`, report AP / silhouette |
| **LCA-depth recovery** | Is graded subtree overlap recoverable? | from geometry (e.g. radius of the Möbius midpoint of `h_u,h_v`) predict the depth of the true lowest common ancestor; correlate with truth |

### Tier 3 — Path similarity

| Metric | What it measures | Definition |
|---|---|---|
| **Ancestor/descendant AUC** | Transitive/path structure | from `(h_u, h_v)` geometry, classify the is-ancestor relation; report AUC |
| **Path-overlap correlation** | Path-level similarity | for path pairs, correlate a geometry-derived similarity with a graph path similarity (shared-node Jaccard or relation-sequence edit distance) |

### Tier 4 — The north star (extrinsic, periodic)

**Retrieval proxy for the LLM use.** Build the real pipeline in miniature: given
a query node, retrieve its top-`k` nearest subtree / path by `d_H`, and score
whether the retrieved set answers a held-out question or matches a gold neighbour
set. This is the extrinsic measure that actually predicts LLM usefulness.
Expensive → run as an occasional checkpoint, not per-epoch.

---

## 3. Link prediction — kept, demoted

Still reported, because it supplies the **relational** semantics (drug↔disease,
disease↔gene, …) that pure hierarchy-reconstruction embeddings lack, and because
a sudden MRR collapse is a fast bug signal.

- Use the **vectorized, leakage-safe** `evaluate_link_prediction`
  (`evaluation/link_prediction.py`): filtered MRR, Hits@{1,3,10}.
- Always pair the number with a **frozen-NN floor** (`scripts/frozen_baseline.py`)
  — a trained MRR only matters relative to "frozen text + cosine NN" with no
  training and no graph. (See `LABEL_LEAKAGE.md`: unmasked CulturalBench scored
  0.96 *for the wrong reason*.)
- On PrimeKG, rank **type-constrained** (a disease object only against diseases)
  via `type_constraints`.

---

## 4. Training toward the geometry (not just the link)

To stop optimising away from the goal, add a **structure-fidelity loss term**
alongside the margin LP loss in `Trainer.train_step`:

```
loss = loss_lp + λ_struct · loss_struct + gate_sparsity_weight · loss_sparse
```

where `loss_struct` pushes ancestors closer than non-ancestors (a hierarchy /
reconstruction term — same shape as a Poincaré-embedding loss, and the same
spirit as the symmetric parent↔child conditioning in `EXPERIMENTS.md` B4). This
makes the *geometry*, not just the translation, carry the hierarchy. Start with a
small `λ_struct` and watch the Tier-1/2 metrics move.

---

## 5. Practical protocol

1. **Select** on Tier-1 + Tier-2 (distortion ↓, reconstruction MAP ↑,
   depth–radius ρ ↑, same-subtree AP ↑). Report MRR as a secondary column.
2. **Always** report the frozen-NN floor next to any trained number.
3. **Sample** pair-based metrics (distortion, retrieval AP) — PrimeKG is ~129K
   nodes, so all-pairs is `O(N²)`; sample anchors + candidates and report with a
   seed.
4. **Run Tier 4** (LLM retrieval proxy) only at milestones.
5. **Per-dataset notes:** CulturalBench's tree is shallow (depth ≈ 3) → Tier-2/3
   are weak there; WN18RR and especially **PrimeKG** (deep, multi-ontology, typed)
   are where these metrics become meaningful.

---

## 6. Status

- [x] `evaluation/structure_metrics.py` — Tier 1–3 + sibling over-smoothing guard
      (intrinsic, no training loop; hyperbolic *and* Euclidean via `manifold=None`)
- [x] printed every run as a `STRUCT |` line (`scripts/train.py`); keys:
      `depth_radius_rho`, `dist_tree_rho`, `recon_map`, `subtree_ap`,
      `ancestor_auc`, `sibling_ratio`
- [ ] structure-fidelity loss term (`λ_struct`) in `Trainer.train_step`
- [ ] Tier-4 retrieval proxy harness
- [x] vectorized filtered LP (`evaluation/link_prediction.py`)
- [x] frozen-NN floor (`scripts/frozen_baseline.py`)
