# Related work: KG-RAG × diversity/pluralism — and what to run next

Companion to [`related_work.md`](./related_work.md), which covers the *architecture*
side (hyperbolic geometry, KG embedding, tree encoders). This file covers the
*retrieval + generation* side, and is the fresh literature search that file asks
for. Survey date: July 2026.

Papers marked ✅ were fetched and verified directly; ⚠️ are reported but
unverified — confirm before citing.

## TL;DR for positioning

Three papers matter most, and two change how this project should be framed:

| paper | why it matters |
|---|---|
| ✅ **DIVERGE** ([2602.00238](https://arxiv.org/abs/2602.00238)) | closest concurrent analog — diversity-enhanced RAG for open-ended QA, ~2× diversity "without noticeable quality degradation". Reports **gains** where we measured **harm**. |
| ✅ **GeoRAG** ([2606.29328](https://arxiv.org/abs/2606.29328)) | same math (Sinkhorn-Wasserstein over distributions) applied to *context selection as coverage*, with a submodular guarantee. Implies our selection rule is the bug. |
| ✅ **ASC** ([2405.13131](https://arxiv.org/abs/2405.13131)) | atomic extract-then-merge across samples beats picking one — the published form of our union finding, with the method our holistic merge lacked. |

---

## 1. Knowledge-graph RAG

- ✅ **GraphRAG** — Edge et al., 2024, [2404.16130](https://arxiv.org/abs/2404.16130).
  LLM-induced entity graph + Leiden community detection → hierarchical community
  summaries, map-reduce answering. The dominant "verbalize a subgraph into the
  prompt" baseline. *We have never compared against it.*
- ✅ **HyperbolicRAG** — Cao, Wang, Li, Zhou, Yang, Nov 2025,
  [2511.18808](https://arxiv.org/abs/2511.18808). Depth-aware node embedding in a
  shared **Poincaré manifold**, unsupervised contrastive regularization across
  abstraction levels, **mutual-ranking fusion** of Euclidean + hyperbolic retrieval
  signals. Closest architecture to ours; evaluated on **factual QA**, not opinion data.
- **HyRAG** — Jin et al., KDD 2026, [2606.03307](https://arxiv.org/abs/2606.03307).
  Hyperbolic indexing motivated by volume-growth mismatch with tree-structured KBs;
  multi-granularity retrieval + dual-path fusion. Same motivation as ours, aimed at
  generalization/hallucination.
- **KAPING** — Baek et al., [2306.04136](https://arxiv.org/abs/2306.04136).
  Zero-shot: retrieve triples by similarity, prepend raw. Essentially our injection
  format — the simplest baseline.
- **G-Retriever** — He et al., NeurIPS 2024, [2402.07630](https://arxiv.org/abs/2402.07630).
  Subgraph retrieval as **Prize-Collecting Steiner Tree** — budget-constrained,
  connected subgraph selection. A principled alternative to "top-k divergent pairs".
- **LightRAG** [2410.05779](https://arxiv.org/abs/2410.05779) · **HippoRAG**
  (NeurIPS 2024; Personalized PageRank over an LLM-extracted KG) ·
  **ToG** [2307.07697](https://arxiv.org/abs/2307.07697) (agentic beam search on the
  KG) · **RoG** [2310.01061](https://arxiv.org/abs/2310.01061) (plan relation paths,
  then retrieve). All target **factual multi-hop QA**.

**Observation:** every KG-RAG system above optimizes *factual sufficiency*. None
retrieves for *viewpoint spread*. That is the gap PluralTree occupies.

## 2. Diversity / pluralism

- **A Roadmap to Pluralistic Alignment** — Sorensen et al., 2024,
  [2402.05070](https://arxiv.org/abs/2402.05070). Defines Overton / steerable /
  distributional pluralism. Our framing derives from this.
- **Benchmarking Overton Pluralism in LLMs** — Poole-Dayan et al., ICLR 2026,
  [2512.01351](https://arxiv.org/abs/2512.01351). OvertonBench/OvertonScore; 1208
  raters, 60 questions, 8 LLMs; automated judge ρ=0.88 vs human; best model
  0.35–0.41. **Our primary eval.**
- ✅ **DIVERGE** — Hu, Tandon, Arora, Jan→Jun 2026,
  [2602.00238](https://arxiv.org/abs/2602.00238). Plug-and-play *agentic* RAG:
  reflection-guided **iterative** exploration of viewpoints + diversity-aware
  retrieval; ~2× diversity, no noticeable quality loss. **The closest concurrent
  work.** Differences from us: agentic/multi-pass rather than single-shot injection,
  no KG structure, and it reports gains where our single-shot pole-injection *hurt*.
- **Modular Pluralism** — Feng et al., [2406.15951](https://arxiv.org/abs/2406.15951).
  Pool of community-specific small LMs feeding a black-box LLM. An
  ensemble-of-models route to Overton coverage — natural comparison to our
  union-of-generations result.
- **Verbalized Sampling** — [2510.01171](https://arxiv.org/abs/2510.01171).
  Attributes mode collapse to *typicality bias* in preference data; fixes it
  training-free by prompting for an **explicit distribution over responses**
  (1.6–2.1× diversity). Directly applicable and cheap.
- **RLHF & diversity** — Kirk et al., ICLR 2024,
  [2310.06452](https://arxiv.org/abs/2310.06452): RLHF improves OOD generalization
  but reduces output diversity vs SFT.
- **NoveltyBench** — [2504.05228](https://arxiv.org/abs/2504.05228). distinct@k /
  utility@k over repeated sampling. See `noveltybench_vs_overtonbench.md`.
- **Distributional alignment** — Meister, Guestrin, Hashimoto,
  [2411.05403](https://arxiv.org/abs/2411.05403): LLMs *describe* opinion
  distributions better than they *sample* from them.
- ⚠️ "The Price of Format: Diversity Collapse in LLMs" (EMNLP 2025 Findings) — ID
  unconfirmed. ⚠️ INFINITY-CHAT arXiv ID unconfirmed.

## 3. Context over-reliance / anchoring — *explains our core failure*

- ✅ **The Distracting Effect** — Amiraz, Cuconasu, Filice, Karnin, May 2025,
  [2505.06914](https://arxiv.org/abs/2505.06914). **Partially-relevant "hard
  distracting" passages mislead RAG generation *more* than clearly irrelevant ones.**
  This is the published mechanism behind our measured
  `corr(relevance, Δcoverage) = −0.26`: our most on-topic forks do the most damage.
- **Context-Aware Decoding** — Shi et al., [2305.14739](https://arxiv.org/abs/2305.14739).
  Contrastive with/without-context decoding. We implemented the *inverse* use
  (α<0 to suppress over-anchoring) in `retrieval/cad.py`.
- **Adaptive Chameleon or Stubborn Sloth** — Xie et al., ICLR 2024 Spotlight,
  [2305.13300](https://arxiv.org/abs/2305.13300). LLMs are highly receptive to
  coherent conflicting context — they *will* be steered by injected poles.
- **Knowledge Conflicts survey** — EMNLP 2024,
  [2403.08319](https://arxiv.org/abs/2403.08319).

## 4. Diversity-aware retrieval — *the missing selection rule*

- ✅ **GeoRAG / "Covering the Unseen"** — Zhang, Jia, Zhu, Jun 2026,
  [2606.29328](https://arxiv.org/abs/2606.29328). Builds a multi-dimensional
  **information-demand distribution** from generated sub-queries, then selects
  context by minimizing **Sinkhorn-Wasserstein distance** between demand and
  coverage. Objective is **demand-weighted facility-location**, monotone submodular
  → **1−1/e greedy guarantee**. Beats MMR, DPP, BGE-Reranker, SMART-RAG (+6.5–9.7
  EM). Same mathematical tool as our scout, used for **coverage of a distribution**
  rather than **the single most divergent pair**.
- **MMR** — Carbonell & Goldstein, SIGIR 1998. Relevance − redundancy greedy.
- **DPPs** — Kulesza & Taskar, [1207.6083](https://arxiv.org/abs/1207.6083).
  Repulsive point process for diverse subset selection; k-DPP fixes subset size.
- **VRSD** — [2407.04573](https://arxiv.org/abs/2407.04573). Recent joint
  similarity/diversity heuristic reported to beat MMR and k-DPP.

## 5. Multi-answer / union generation — *our union finding, published*

- ✅ **Atomic Self-Consistency (ASC)** — Thirukovalluru, Huang, Dhingra, 2024,
  [2405.13131](https://arxiv.org/abs/2405.13131). Extracts **authentic subparts
  ("atoms")** from multiple sampled long-form answers and **merges** them into a
  composite, optimizing *recall of distinct information* rather than picking one
  sample. Significant gains over USC on ASQA/QAMPARI/QUEST/ELI5. **This is the
  method our holistic merge lacked** — we asked for a merge; ASC does atomic
  extraction then union.
- **Universal Self-Consistency** — [2311.17311](https://arxiv.org/abs/2311.17311).
  LLM picks the most consistent sample — the "pick one" baseline our union beats.
- **Aggregating own responses** — [2503.04104](https://arxiv.org/abs/2503.04104).
- **Agreement in representation space** — [2606.12003](https://arxiv.org/abs/2606.12003).
  Clusters sampled generations in embedding space to estimate open-ended agreement —
  essentially our `retrieval/contestedness.py` signal.

## 6. Where our findings sit

| our result | literature status |
|---|---|
| relevant forks hurt most (−0.26) | **explained** by the hard-distractor effect (2505.06914) |
| answer collapses onto injected poles | **novel negative result** — DIVERGE/GeoRAG report only gains |
| graph divergence ⊥ human contestedness (+0.20) | **unreported** |
| prompted self-routing fails (0.072) | **unreported for pluralism**; consistent with the 2026 adaptive-RAG consensus that self-reported retrieval decisions are unreliable |
| union across answers ≫ any single answer | **matches ASC**, but never framed for *viewpoint-cluster* coverage |
| hyperbolic hierarchy + opinion pluralism | **no prior work found** |

---

# Experiments to run

Ordered by (attacks a measured failure) × (cheap) × (publishable).

## E1 — Submodular coverage fork selection *(highest value)*
**Motivation.** Our scout picks the **single max-Wasserstein pair** — two extremes.
GeoRAG shows the right objective is *covering a demand distribution*, solved
greedily with a 1−1/e guarantee. Our own `subtree_middle.py` found 68% of non-pole
subgroups lie *between* the poles, so a max-divergence pair provably misses the
middle by construction.

**Do.** Replace the `score = rel^α · W` argmax with greedy **facility-location /
k-DPP / MMR** selection over the anchor's subgroup distributions, maximizing
coverage of the population distribution rather than pairwise divergence.
Conditions: `scout` (max-W pair) vs `mmr` vs `dpp` vs `submodular`.

**Why it's different from `distributional`.** That condition fixed the *rendering*
(show all subgroups); this fixes the *selection objective*. They are independent and
composable.

**Cost.** CPU-side selection change + one eval run; reuses `positions_from_subtree`.

## E2 — Atomic merge (ASC-style)
**Motivation.** Our `merge` scored 0.503 against a union ceiling of 0.657 — it
*substituted* rather than added. ASC's answer is not "merge two answers" but
**extract atoms, dedupe, union**.

**Do.** Decompose each draft into atomic positions (reuse `split_units`), embed,
cluster to dedupe, then generate conditioned on the *atom list* with an explicit
instruction to articulate each atom fully.

**Cost.** One extra decomposition pass; no new infrastructure.

## E3 — Verbalized sampling
**Motivation.** [2510.01171](https://arxiv.org/abs/2510.01171) traces mode collapse
to typicality bias and fixes it by eliciting an **explicit distribution with
probabilities** — training-free, 1.6–2.1× diversity.

**Do.** A `verbalized` condition: "list the positions people hold on this question
*with their approximate prevalence*, then answer." Contrast with `expand`, which
asked for positions but not a distribution, and failed.

**Cost.** One prompt, one run. Cheapest experiment here.

## E4 — Standard KG-RAG baselines *(required for publication)*
We have never compared against any published KG-RAG injection. Add **KAPING** (raw
triples by similarity) and **GraphRAG-style community summaries** as conditions.
Without these, "our KG retrieval helps/hurts" has no reference point.

## E5 — Hard-distractor characterization *(no generation needed)*
**Motivation.** 2505.06914 predicts partially-relevant context is worst. We measured
exactly that (−0.26) but never framed it.

**Do.** Stratify Δcoverage by fork relevance into irrelevant / partially-relevant /
highly-relevant bands and show the non-monotonicity. Converts an incidental
correlation into a **replication of a known effect in a new domain**, using existing
v6/v7 data.

## E6 — Coverage@K as the primary claim
**Motivation.** Union ≫ single answer is our most robust result (+0.15–0.18, far
above the 0.027 noise floor), it matches ASC, and it is the one number that survives
judge uncertainty.

**Do.** Run `NROLL=5 KROLL=5`; report `coverage@K` and `rollout_gain` alongside
OvertonScore. Reframes the contribution from "injection improves one answer" (false)
to "graph-guided diversity improves *achievable* coverage" (supported).

## E7 — Hyperbolic ablation vs HyperbolicRAG
**Motivation.** HyperbolicRAG uses **mutual-ranking fusion** of Euclidean +
hyperbolic signals; we use hyperbolic alone. Separately, our GraIL result is
confounded (text + structure vs structure-only baselines).

**Do.** (a) Euclidean-only vs hyperbolic-only vs fusion retrieval; (b) the
text-vs-structure ablation. Establishes whether the hyperbolic geometry contributes
anything downstream.

## Blocking dependency
E1–E6 are all scored by the OvertonBench judge, whose validity is unresolved
(aggregate ρ=0.167 vs the paper's 0.88; within-participant discrimination ≈0; noise
floor 0.027). **Run `--human_reliability` (free) and the judge fixes first** —
otherwise none of these can be evaluated, since all plausible effect sizes sit near
the noise floor.

## Suggested order
1. Judge: `--human_reliability` + logprob expected rating *(free/cheap; blocks everything)*
2. **E5** *(no generation — existing data)*
3. **E3** verbalized sampling *(one prompt)*
4. **E1** submodular selection *(the main methodological contribution)*
5. **E6** coverage@K reframing
6. **E2** atomic merge
7. **E4** baselines, **E7** ablations *(needed for a paper, not for the next insight)*
