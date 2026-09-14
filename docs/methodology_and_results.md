# PluralTree — methodology, results, and what to do next

Written 2026-09-13. Supersedes the narrower `results_2026_09.md`. Every number is
from a job log; anything unmeasured is marked. Reads as: what was built → what was
tested → what held, what broke → where to go → what a reviewer will say → what to
run and why.

---

## 1. Methodology

### 1.1 The graph

Pew American Trends Panel (OpinionQA / raw ATP): 1,492 questions, 58
(attribute, group) subpopulations, 75,695 records → 94,423 entities.

```
US Public → topic (12) → subtopic (48) → question (1,492)
          → demographic axis (17,835) → subgroup opinion leaf (75,035)
```

Each leaf holds a subgroup's answer distribution over the question's options.
**Topic and subtopic layers are induced by k-means over MiniLM question
embeddings**, not native. Embedded with a Tree-GRU + gated knowledge injection
in a Poincaré ball (c=0.5). Intrinsic link prediction: MRR ≈ 0.456 (historical);
fresh runs this month land at 0.25–0.26 — unresolved discrepancy, see §6.

### 1.2 The divergence scout

1. Embed the question (MiniLM), score every node's relevance by cosine.
2. Find anchors lexically / by relevance; gate child branches at `tau=0.25`.
3. For each anchor, score every child pair as `rel^α · W`, where `W` is the
   hyperbolic Wasserstein between the two sibling subtrees' embedding point
   clouds (relevance-weighted mass).
4. Keep the top-`k=5` forks. Render each as a contrast block (two poles + driver
   pairs, or the full subgroup spectrum) and inject into the prompt.

### 1.3 Conditions

| condition | what it does |
|---|---|
| `baseline` | no retrieval, plain instruction |
| `scout` / `distributional` | one pass, forks injected (2-pole / full spectrum) |
| `route` / `expand` | model decides whether to use forks / asked for breadth |
| `merge` (v1) | plain + injected draft → extractive merge |
| **`merge_v2`** | 3 drafts (plain / scout / distributional) → one merge; structural guard (length, deep-paragraph count) with concatenation fallback |
| `persona_merge` | merge_v2 machinery; drafts conditioned on subgroup vantage points (poles first) |
| `merge_v2_rand` | control: forks swapped for structurally matched forks from an unrelated anchor |
| `merge_v2_sem` | merge_v2 + semantic retention guard (deduped draft units must survive at cos ≥ 0.55, ≥ 80%) |
| `merge_v2_divrand` | ablation: uniform random sibling pair instead of max-W, same candidate pool, volume-matched |
| `merge_v2_flat` | ablation: no hierarchy — MiniLM similarity over opinion leaves, same tau gate, same fork count |
| `cad<α>` | context-aware decoding: `logits = (1+α)·logits(y|forks,x) − α·logits(y|x)` |
| `g2` / `g2_graph` | logit-level diversity steering; graph variant aims the guide at a subgroup vantage point |

### 1.4 Evaluation

**OvertonBench** (primary): 60 questions, ~7.75 human viewpoint clusters each
(Polis-style k-means over participants' agree/disagree votes on peer statements).
An LLM judge predicts each participant's 1–5 "did this represent me" rating; a
cluster is covered at mean ≥ 4. OvertonScore = covered / n_clusters, mean over
questions. Judge validated at ρ = 0.88 — **for ranking 8 reference models**, not
per-question or per-cluster. Generator and judge: Qwen2.5-72B-Instruct-AWQ
unless stated. Paired bootstrap over questions for all deltas.

**INFINITY-CHAT** (secondary): across-sample diversity (Vendi, mean cosine)
over 8–50 samples per query. Different construct.

**Union / oracle / gain**: per question, union of covered-cluster sets across
conditions; `gain = union − best single`. Measures *what an arm adds*, not how it
scores. `union ≥ oracle ≥ best` by construction.

---

## 2. Results

### 2.1 What held

| finding | evidence |
|---|---|
| merge_v2 beats baseline, replicated | v10 +0.0389; v11 +0.0498 (p=0.014); v12 +0.0475 (p=0.024); Euclidean-embedding run +0.0707 (p=0.014) |
| merge_v2 fixes merge v1's dilution | v1 below baseline; control p=0.002 |
| Merging beats selecting | best-of-k 0.5470 > selector 0.5243 > 0.5123; learned selector +0.013, p=1.0 |
| The model reads the injected block | 61% of injected positions surface at cos ≥ 0.35; median 0.415 (correlational, no chance floor) |
| Anti-anchoring is real and monotone | CAD at 7B: α = −0.5 / −0.25 / 0 / +0.25 / +0.5 → 0.352 / 0.327 / **0.099** / 0.047 / 0.042; `base7b` 0.394 |
| The merge preserves draft content | semantic retention median 1.000, mean 0.966 (cos 0.55); structural fallback fires 25–38% at 7B |
| G2 lifts diversity | INFINITY-CHAT, vanilla G2 vs baseline: 19/20 wins (diversity panel, 7B) |
| ATP graph reaches VITAL situations | 489/500 resolved (97.8%), median relevance 0.400 vs 0.490 — **but under our template; VITAL-template gate unread** |

### 2.2 What broke

**Geometry (Tier 1, item 1).** Fresh embeddings, same seed, `NROLL=1`, 72B judge:

| | baseline | merge_v2 | delta | 95% CI | p |
|---|---|---|---|---|---|
| hyperbolic c=0.5 | 0.5089 | 0.5225 | **+0.0136** | [−0.041, +0.067] | 0.615 |
| Euclidean c=0 | 0.4951 | 0.5658 | **+0.0707** | [+0.012, +0.134] | 0.014 |
| difference | | | +0.0571 | [−0.009, +0.132] | 0.094 |

Hyperbolic fit the hierarchy far better — `dist_tree_rho` 0.64 vs 0.29,
`ancestor_auc` 0.958 vs 0.870, `sibling_ratio` **0.65 vs 0.08** — and produced no
coverage gain. Sibling separation is the property the scout depends on; it
improved 8× and coverage got worse. Baseline-to-baseline noise across the two
arms (a quantity that cannot depend on curvature) is 0.0138 — the same size as
the hyperbolic delta. **Defensible claim: no evidence the geometry contributes to
coverage. Not defensible: that it hurts (p=0.094).**

**Scale.** 7B generator, 7B judge, `NROLL=1`:

| run | baseline | merge_v2 | delta |
|---|---|---|---|
| scale_7b | 0.4277 | 0.1727 | **−0.2550** |
| scale_7b_sem | 0.3913 | 0.2230 | −0.1683 |

Two of merge_v2's three drafts are the injected condition, which CAD measured at
0.099 on this model. Retention says the merge keeps them faithfully. So the
collapse is **draft quality, not merge fidelity** — the merge averages a good
draft with two collapsed ones (≈ (0.43+0.10+0.10)/3 ≈ 0.21; observed 0.17–0.22).
This retracts the earlier "quarantine" reading: the guard does not protect the
good draft. 72B re-judge (job 2701171) pending.

**Semantic guard.** Fired 1/60. `merge_v2_sem` 0.2170 vs `merge_v2` 0.2230. The
merge is not substituting content at this measure; the guard has nothing to
catch. Retention at stricter cosine (0.65–0.85) not yet recomputed.

**Routing.** Closed. Five independent nulls; oracle caps +0.0415; best gate
+0.006; `route` 0.072 vs baseline 0.507; probe p=0.130; random k-fold gave AUC
1.000 on a pure topic confound vs 0.007 leave-one-topic-out.

**Reward (GRPO gate).** Within-question pairwise concordance with the judge:
0.152 at headline (t=0.50, d=60), best cell 0.301, chance 0.500. Low threshold →
few ties, `conc|sep` 0.417 (confident and wrong); high threshold → 73% ties
(silent). No (t, d) escapes both, so recalibration is ruled out. Diagnosis: the
reward scores ~19.9 **survey positions**, the judge scores ~7.75 **participant
clusters**. Cluster-target arm crashed after building 465 targets; untested.
Judge noise at the per-(question, condition) level — the ceiling on any
concordance — has never been measured. GRPO stays blocked.

**Random-fork control — invalid for three runs.** The v12 tie (+0.0154, p=0.42)
compared arms differing in fork content *and* fork count (1.00 vs 4.85) *and*
merge algorithm (fell through to merge v1 via a dispatch gap). It tested nothing.
Fixed; rerun pending in `sel_ablation`.

**persona_merge.** Same score as merge_v2 (0.6423 vs 0.6460 coverage@K), 2.4×
the union gain (+0.0917 vs +0.0389); 21/180 rows ran with zero personas. The
drafts find different clusters; the score does not move.

**Position-statement artifact.** 53% of rows are stored fallback (`"<question>
<option>"`, `template_match_rate` 0.467). Rewriting *lowers* cosine (median
−0.124) — harmless for the reward, fatal for `g2_graph`'s first smoke, which was
steering toward survey questions. Fixed to steer on subgroup vantage points.

### 2.3 Pending

| job | settles |
|---|---|
| `sel_ablation` | max-W vs random sibling pair; hierarchy vs flat. Last untested Tier 1 item |
| `geom_*_r3` | geometry at 3 rollouts (protocol match; unlikely to move p=0.094 — heterogeneity is between questions) |
| `judge_only` 2701171 | 7B collapse under the 72B judge |
| `hivemind_div` | merge_v2 on INFINITY-CHAT; 71/100 queries resolve, median relevance 0.326 |
| anchor_cov 2706825 | VITAL resolution under **its own** prompt (the 97.8% used ours, which contains "how do people differ") |

---

## 3. The frame that organises this

Three stages; every result belongs to one.

| stage | metric | status |
|---|---|---|
| **retrieve** — find viewpoints | union gain, cluster recovery | *unproven*: content control invalid, selection untested, geometry null |
| **deliver** — get them into the generator | ctx0 vs base, the α curve | *broken*: injection is monotonically harmful, 4× worse at 7B |
| **compress** — fit them into one answer | alone-vs-union gap, retention | *faithful*: merge keeps content; the loss is upstream in draft quality |

Correction to the earlier version of this frame: "compress" was labelled the
bottleneck. Retention says the merge is faithful; the 7B collapse is drafts.

---

## 4. Directions

1. **Stop claiming the geometry is load-bearing.** Report it as: fits the
   hierarchy 2× better, contributes nothing measurable to coverage. Either find
   the mechanism by which sibling separation *should* help and show it doesn't,
   or drop hyperbolic from the headline.
2. **Reframe merge_v2 as an architecture, not a retrieval result** — until the
   selection ablation says otherwise. Its gain replicates; what produces it does
   not yet have a demonstrated cause.
3. **Move the graph out of the prompt.** CAD says the channel is the failure. G2
   (logit-level) and a merge over *uninjected* drafts are the two ways to use the
   graph without injecting it.
4. **Fix the target ontology before touching RL.** Cluster-shaped targets, a
   measured judge-noise ceiling, then concordance. GRPO only after.
5. **Escape n=60 with VITAL** (1,649 items, ~7.25 values each, MIT) — for
   arm-vs-arm comparisons at matched length only. It cannot carry the headline
   claim: GPT-4-generated targets, no human ratings, no validated judge.

---

## 5. What reviewers will say

| objection | current answer | fix |
|---|---|---|
| n=60; effects at p≈0.01–0.09 are not robust | true; benchmark's own ceiling (its authors list expansion as future work, ~$15–25k) | VITAL for ablations; cluster-level analysis (660 rows) as free power |
| No quality measure — coverage rewards listing viewpoints badly | true; nothing distinguishes 5 viewpoints synthesised from 5 drafts stapled | pairwise quality judge (largest unbuilt piece) |
| No length-matched baseline; injected answers are ~5× longer | `longest` at −0.101 is suggestive, not a control | prompt baseline to ~330 words; one job |
| Why hyperbolic, if it doesn't help? | it doesn't, on this evidence | see §4.1; ISSP (native hierarchy) is the last defence |
| The hierarchy is k-means over embeddings — testing an embedding on it is circular | true | ISSP's 11 native modules |
| Judge validated only at model-ranking level, n=8 | true; per-cluster agreement unknown | seed-split re-judge for a self-agreement ceiling |
| merge_v2 collapses at 7B → scale-contingent | true as measured | 72B re-judge pending; report as a limitation, not hide it |
| A control arm was broken for three runs — what else is? | fair | preflight + smoke gate now assert arms actually ablate; document it |
| Key results at NROLL=1 | true for geometry and scale | r3 reruns pending |
| CAD is n=28, half A, no CIs, no α=−1 bracket | true | extend sweep; α=−1 *is* the missing no-context arm |
| MRR 0.456 cited but fresh runs give 0.26 | unresolved | find what produced 0.456; rerun geometry on it if it differs |

---

## 6. Experiments to run, and why

Ordered by (what it settles) × (cost). Items 1–3 are queued.

| # | experiment | why | cost |
|---|---|---|---|
| 1 | **Selection ablation** (`merge_v2_divrand`, `merge_v2_flat`) | The only remaining test of whether the scout does anything. Manipulation check (`mean_w` random < maxw at equal `n_forks`) must pass first or a tie is void | queued |
| 2 | **72B re-judge of the 7B run** | Removes the only confound on the −0.255 collapse | queued |
| 3 | **Geometry at NROLL=3** | Protocol match; expect the conclusion to firm, not flip | queued |
| 4 | **Length-matched baseline** | Cheapest existential threat to the headline. If baseline at ~330 words closes most of +0.04, the result is verbosity | 1 job |
| 5 | **Merge over three *uninjected* drafts** | Separates "merging helps" from "retrieval helps" with no dependence on the graph resolving. Given deliver is broken and compress is faithful, this is the cleanest remaining test of what merge_v2's gain actually is | new condition, 1 job |
| 6 | **Retention at cos 0.65–0.85** | Decides whether the semantic guard is dead or its threshold was too lenient. Offline, from existing drafts | minutes |
| 7 | **Cluster-target reward + judge-noise ceiling** | Tests whether below-chance concordance is a target mismatch; the ceiling makes 0.60 a real bar instead of an arbitrary one | 2 jobs |
| 8 | **`g2_graph` on OvertonBench** | First test of the graph through a channel that never enters the prompt. `g2_graph` vs `g2` is the properly-powered form of real-vs-random forks | 1 job |
| 9 | **Trace-consistency score** | A ground-truth-free reward: fraction of the `<think>` plan the answer executes. Needs no cluster targets, so it works on all graph questions; padding the plan is self-penalising. Test its concordance on v11 before any training | offline |
| 10 | **VITAL, arm-vs-arm only** | 27× the questions for the p=0.094 geometry difference and the selection ablation. Needs a scorer and a driver; not valid for baseline-vs-merge_v2 without #4 | driver + judge |
| 11 | **Max-W vs cheaper divergences** (JS, TV, cosine of means) | Only if #1 is positive. A win over random does not show *Wasserstein* earns its cost | 1 job |
| 12 | **ISSP as a second graph** | Native topic hierarchy; the last fair test of the geometry. Blocked on a GESIS download and a schema round-trip | days |
| 13 | **Pairwise quality judge** | Every number is coverage. Unblocks reading #5, VITAL, and any merge result correctly | build |

**Skip:** recursive self-confidence (= `route`, 0.072), prompt self-refinement
(= `expand`, 0.377), `cad_soft` (routing in a different hat), online learning
(no signal to update on), PERSPECTRA (flattened tree, no license, discriminative).

---

## 7. Data and evals surveyed

| resource | verdict | use |
|---|---|---|
| **VITAL** overton subset | 1,649 items, ~7.25 values, MIT, plain JSON | secondary eval; exclude the two Pew-derived files (contamination) |
| ISSP | 11 native topic modules, harmonised demographics, GESIS `.dta` | second graph, native hierarchy |
| EVS/WVS, ESS, Afrobarometer | usable survey graphs | overlap / Europe-only / no taxonomy |
| DICES-350, D3CODE, Wiki Detox | annotator-disagreement, non-political, ungated | self-contained graph + eval; will not resolve against OvertonBench |
| PERSPECTRA | flat 2-level, no weights, no license, Kialo ToS | no |
| "Benchmarking Overton Pluralism" | *is* OvertonBench | no new questions |
| "Latent Perspectives" | corpus-level, unreleased | no; but its 98%-aspect / 34%-perspective split corroborates our surface-vs-scored dissociation |
