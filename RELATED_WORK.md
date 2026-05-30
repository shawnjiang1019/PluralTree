# PluralTree — Related Work & Reading List

A curated reading list mapped to the PluralTree architecture and the directions in
[`EXPERIMENTS.md`](./EXPERIMENTS.md). Use it as a literature map, not an exhaustive
survey. Each entry notes *why it matters here*.

> **Note:** references are given by author/title/year so they're easy to look up.
> Verify exact venues/details when you pull the papers. This list reflects knowledge
> up to early 2026 — run a fresh literature search for newer hyperbolic-KG and
> KG-augmented-LLM work before writing up.

---

## 0. Read these three first (highest leverage)

1. **Chami et al., *Low-Dimensional Hyperbolic Knowledge Graph Embeddings* (AttH, 2020)**
   — the closest prior work to this project. Hyperbolic KG embeddings with
   attention and learnable curvature; covers the exact MRR / Hits@k filtered-ranking
   evaluation you use. Will calibrate expectations and sharpen baselines (A3) and
   learnable curvature (B3).
2. **Nagano et al., *A Wrapped Normal Distribution on Hyperbolic Space* (2019)**
   — the technical key to the most novel direction (E1/E3): a wrapped Gaussian on
   the ball with a reparameterizable sampler.
3. **Yasunaga et al., *GreaseLM: Graph REASoning Enhanced Language Models* (2022)**
   — the template for joining a KG encoder with an LM for reasoning; motivates the
   whole project (section D).

---

## 1. Hyperbolic geometry foundations
*(underpins the entire `pluraltree/manifolds/` stack and the depth-aware story — A1, A2)*

- **Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations* (2017)**
  — origin of hierarchy-in-the-ball; the radius-encodes-depth intuition the whole
  depth-aware gate rests on (A2).
- **Nickel & Kiela, *Learning Continuous Hierarchies in the Lorentz Model* (2018)**
  — the hyperboloid alternative in `hyperboloid.py`; when Lorentz is numerically
  better than Poincaré.
- **Sala et al., *Representation Tradeoffs for Hyperbolic Embeddings* (2018)**
  — distortion vs. dimension theory; the formal "why hyperbolic beats Euclidean for
  trees" argument behind ablation A1.
- **Ganea, Bécigneul & Hofmann, *Hyperbolic Neural Networks* (2018)**
  — Möbius layers, exp/log maps, gyrovector ops; essentially the math in
  `poincare.py`. Read closely.
- **Chami et al., *Hyperbolic Graph Convolutional Neural Networks* (HGCN, 2019)**
  — message passing / aggregation on the manifold; directly relevant to
  `ChildAggregator` and the top-down extension (B4).
- **Bécigneul & Ganea, *Riemannian Adaptive Optimization Methods* (2019)**
  — theory behind `geoopt`'s RiemannianAdam used in `riemannian_optim.py`.

---

## 2. Knowledge graph embedding & scoring
*(your training task and scoring head — A3, B3)*

- **Bordes et al., *Translating Embeddings for Modeling Multi-relational Data* (TransE, 2013)**
  — your scoring head is a hyperbolic TransE. Essential baseline (A3).
- **Sun et al., *RotatE: Knowledge Graph Embedding by Relational Rotation* (2019)**
  — rotation-based scoring; strong baseline for A3.
- **Balažević et al., *Multi-relational Poincaré Graph Embeddings* (MuRP, 2019)**
  — hyperbolic KG embeddings with relation-specific Möbius transformations;
  precursor to AttH and directly comparable to your relation translation.
- **Chami et al., AttH/RotH (2020)** — see section 0.

---

## 3. Tree-structured / recursive encoders
*(the Tree-GRU lineage — `tree_gru/`)*

- **Tai, Socher & Manning, *Improved Semantic Representations from Tree-Structured
  Long Short-Term Memory Networks* (Tree-LSTM, 2015)**
  — the direct ancestor of your Tree-GRU: recursive bottom-up encoding and child
  aggregation.
- **Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder* (GRU, 2014)**
  — the original gating mechanism the cell adapts to tangent space.

---

## 4. Gating, knowledge injection & retrieval augmentation
*(the GKI side — `gki/`, `combined/`, and B2 multi-source; feeds D1)*

- **Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (adapters, 2019)**
  — gated insertion of new capacity into a frozen backbone; framing for "selectively
  inject external knowledge."
- **Lewis et al., *Retrieval-Augmented Generation* (RAG, 2020)**
  — foundational retrieve-then-condition; informs D1 (hierarchy-aware retrieval).
- **Borgeaud et al., *Improving Language Models by Retrieving from Trillions of
  Tokens* (RETRO, 2022)**
  — chunked cross-attention retrieval; an alternative injection mechanism.
- **Zhang et al., *ERNIE: Enhanced Language Representation with Informative Entities*
  (2019)**
  — injecting KG entity embeddings into a transformer; informs soft-prompt injection
  (D3).

---

## 5. Distributional & region embeddings
*(section E1 parametric route, and E2/E3)*

- **Vilnis & McCallum, *Word Representations via Gaussian Embedding* (2014)**
  — foundational "embed as a distribution, not a point"; KL scoring and entailment
  via containment. Conceptual backbone of E1.
- **He et al., *Learning to Represent Knowledge Graphs with Gaussian Embedding*
  (KG2E, 2015)**
  — Gaussian embeddings applied specifically to KGs.
- **Nagano et al., *A Wrapped Normal Distribution on Hyperbolic Space* (2019)**
  — see section 0; the construction needed for E1/E3.
- **Mathieu et al., *Continuous Hierarchical Representations with Poincaré
  Variational Auto-Encoders* (2019)**
  — hyperbolic geometry + latent distributions; closest existing work to
  "distributions on the ball."
- **Vilnis et al., *Probabilistic Embedding of Knowledge Graphs with Box Lattice
  Measures* (2018)** and **Ren et al., *Query2Box* (2020)**
  — regions instead of points; natural for an entity that "exists across branches"
  as a region — useful for E2 and as an alternative to Gaussians in E1.

---

## 6. Plurality, polysemy & multi-sense representation
*(section E2 — the least-charted, most novel direction)*

> The pluralistic-leaf idea has little direct precedent — a good novelty signal.
> The closest adjacent literature:

- **Neelakantan et al., *Efficient Non-parametric Estimation of Multiple Embeddings
  per Word* (2014)**
  — one entity, multiple context-conditioned vectors; the word-embedding analogue of
  pluralistic existence.
- **Box / region embeddings (section 5)** — represent multi-branch existence as a
  region rather than a point cloud.
- **Mixture-of-experts conditioning** (e.g. Shazeer et al., 2017) — a mechanism for
  "context selects which existence," relevant to the E3 mixture/reconciliation step.

---

## 7. KG-augmented LLM reasoning
*(the motivating end goal — section D)*

- **Yasunaga et al., *QA-GNN: Reasoning with Language Models and Knowledge Graphs
  for Question Answering* (2021)** and ***GreaseLM* (2022)**
  — joint KG-encoder + LM reasoning with structured grounding; the template for
  D1–D3.
- **Lewis et al., RAG (2020)** and **Borgeaud et al., RETRO (2022)** — see section 4.
- **Zhang et al., ERNIE (2019)** — see section 4; informs soft-prompt injection (D3).

---

## How this maps to EXPERIMENTS.md

| EXPERIMENTS.md direction | Primary reading |
|--------------------------|-----------------|
| A1 hyperbolic vs. Euclidean | §1 Sala et al.; Nickel & Kiela 2017 |
| A2 ρ encodes depth | §1 Nickel & Kiela 2017/2018 |
| A3 stronger baselines | §2 TransE, RotatE, MuRP, AttH |
| B3 learnable curvature | §0/§2 Chami AttH |
| B4 top-down message passing | §1 HGCN |
| B2 multiple knowledge sources | §4 adapters, RAG |
| D1–D4 LLM integration | §7 QA-GNN, GreaseLM; §4 RAG, RETRO, ERNIE |
| E1 distributional embeddings | §5 word2gauss, KG2E, Nagano, Poincaré VAE |
| E2 pluralistic existence | §6 multi-sense, box embeddings, MoE |
| E3 plurality → distribution | §5 + §6 combined |
