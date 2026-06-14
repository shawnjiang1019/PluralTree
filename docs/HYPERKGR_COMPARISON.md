# PluralTree ↔ HyperKGR — Detailed Comparison

A paper-grounded comparison of **PluralTree** against the closest prior work,
**HyperKGR** (Lihui Liu, *"HyperKGR: Knowledge Graph Reasoning in Hyperbolic
Space with Graph Neural Network Encoding Symbolic Path"*, EMNLP 2025). See also
the `HyperKGR` codebase walkthrough and the positioning note in
`EXPERIMENTS.md` / `RELATED_WORK.md`.

**Headline:** PluralTree and HyperKGR share a remarkable amount of *mechanics*
but are built on opposite *philosophies* of what an embedding is.

---

## 1. The one fundamental difference

| | HyperKGR | PluralTree |
|---|---|---|
| **What gets embedded** | A **per-query reasoning tree** rooted at the source entity `u`, grown hop-by-hop over arbitrary KG edges | A **fixed semantic hierarchy** (taxonomy: World→Region→Country→Practice, or WN18RR hypernym tree) |
| **Unit of representation** | A **pair** `h_q(u,v)` — "what `v` looks like *from* `u` under relation `q`" | A **per-entity** `h_v` — one vector per node, query-independent |
| **When computed** | **Dynamically, per query** (no entity table; source seeded to zero) | **Once**, then `h_all` is reused for every query |
| **Reasoning** | Dynamic programming over **paths**, query-conditioned | Encoding of **structure**; scoring is nearest-neighbor on a static map |

Everything else flows from this. HyperKGR *reasons over paths for a specific
question*; PluralTree *encodes the shape of a hierarchy once*.

## 2. What they genuinely share (the overlap is large)

This is why HyperKGR is the "closest prior work":

- **Poincaré ball + Möbius addition + exp/log maps** — near-identical hyperbolic
  toolkits.
- **Hyperbolic TransE message**: both compute `proj(h ⊕_c r)` (HyperKGR per edge;
  PluralTree as the scoring translation `h_s ⊕ r`).
- **GRU gating** — HyperKGR fuses hops with a GRU (`h^{l+1}=GRU(h^l, h0)`,
  anti-forgetting); PluralTree's Tree-GRU cell fuses the children-aggregate with
  node features. Both run GRU update/reset gates in tangent space.
- **Attention aggregation** — both weight contributions with a learned
  sigmoid/softmax attention.
- **Link prediction, filtered MRR / Hit@{1,10}** — same task and metrics.
- **The "hyperbolic because trees grow exponentially" motivation** — identical
  justification (the paper's Fig. 1c / Theorem 1; PluralTree's depth-aware
  design).

If someone skimmed only the math, the two would look like cousins.

## 3. What each does that the other doesn't

**HyperKGR has, PluralTree lacks:**
- **Query-conditioned, dynamic embeddings.** The same `v` is represented
  differently per `(u,q)`. PluralTree's `h_v` is fixed.
- **Multi-hop path reasoning** with the **Theorem 1** constraint (aggregate only
  from the source's (i−1)-hop neighborhood, else the GNN injects noise).
  PluralTree does **one** bottom-up pass — it is not path reasoning at all, so
  this theorem does not apply to it.
- **Learnable curvature** as the live default (PluralTree's B3 is still roadmap).
- **Query-relation-conditioned attention** (`α` depends on `q_r`). PluralTree's
  child attention is *not* query-conditioned.

**PluralTree has, HyperKGR lacks:**
- **Gated Knowledge Injection** with **depth-aware radius gating** — injecting
  *external* knowledge, modulated by where a node sits in the tree. HyperKGR has
  no external-knowledge channel (relations are the only learned semantics).
- **Text-grounded input features** (frozen sentence-transformer over glosses /
  questions). HyperKGR seeds the source to **zero** and learns only relation
  vectors — no text. This makes PluralTree inductive *via language*, not just via
  structure.
- **A fixed, reusable hierarchy** → encode once, then cheap scoring. HyperKGR
  re-propagates per query.
- **The plurality / distributional directions (E)** and the **LLM-grounding end
  goal (D)** — entirely outside HyperKGR's scope.

## 4. Axis-by-axis

| Dimension | HyperKGR | PluralTree |
|---|---|---|
| Embeddings | Dynamic, query-specific | Static, computed once |
| Graph used | Per-query subtree of *reasoning paths* | Fixed *taxonomy* edges |
| Message passing | Multi-hop (L layers), outward from source, bidirectional over KG | Single pass, bottom-up only (parent←child; child←parent is roadmap B4) |
| Curvature | Learnable | Fixed (learnable = B3) |
| Attention | Query-relation-conditioned edge attention | Child-aggregation attention (not query-conditioned) |
| External knowledge | None | GKI + depth-aware gate |
| Input features | None (source=0, relations learned) | Frozen text embeddings |
| Scoring / loss | `wᵀh`, multi-class log-loss over all tails | `−d_H(h_s⊕r, h_o)`, margin ranking + type negatives |
| Inductive via | Structure (no entity table) | Text + structure |
| Inference cost | Re-propagate per query (sampling helps) | Encode once → cheap NN lookup |
| Beyond LP | — | Plurality/distributions (E), LLM grounding (D) |

## 5. What this means for PluralTree's positioning

1. **Do not claim to beat HyperKGR at link prediction on a fixed KG.** It (and
   its RED-GNN / NBFNet / AdaProp lineage) is SOTA precisely because of
   *query-conditioned dynamic path reasoning* — the thing PluralTree deliberately
   does not do. The WN18RR run enters HyperKGR's home turf; treat it as **"does
   my static hierarchy beat a frozen-text floor?"**, not "can I beat HyperKGR."

2. **The differentiators are real and orthogonal:** external knowledge injection
   (GKI), text grounding, cheap static reusable embeddings, and the plurality /
   LLM directions. None of these are in HyperKGR. Lead with them.

3. **HyperKGR is a template for the bridge experiments.** To compete on LP, the
   move is to adopt *some* query-conditioning, and the staircase is already in the
   roadmap: **B4 (symmetric / top-down conditioning)** is a soft step toward
   HyperKGR's query-conditioned representations, and **E2 (per-parent
   existences)** is the plurality-flavored analogue of "the same `v` looks
   different depending on context." HyperKGR's query-relation attention and
   learnable curvature can be borrowed wholesale.

4. **A caveat that strengthens the story:** HyperKGR has no notion of *external
   knowledge* or *uncertainty*. PluralTree's GKI (once given a genuinely distinct
   source — roadmap B2) and its measured-variance plurality (E3) are
   contributions HyperKGR structurally cannot make. Novelty lives there, not in
   the shared hyperbolic-GRU mechanics.

## 6. Bottom line

HyperKGR = **dynamic, query-specific path reasoning** in hyperbolic space (a
hyperbolic NBFNet / RED-GNN). PluralTree = **static, text-grounded hierarchy
encoding** with **gated external-knowledge injection**, aimed at **plurality and
LLM grounding**. They overlap heavily in hyperbolic plumbing (Poincaré, Möbius,
GRU, attention, MRR) but answer different questions: *"which paths connect u to
v?"* vs *"where does this entity sit in a knowledge hierarchy, and what external
knowledge refines it?"*