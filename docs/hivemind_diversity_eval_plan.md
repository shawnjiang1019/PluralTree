# Hivemind Output-Diversity Eval — Metric Design

## Context

We built an INFINITY-CHAT mode-collapse eval this session (`data/loaders/infinity_chat.py`,
`evaluation/hivemind/{generate_hivemind,diversity_metrics}.py`,
`jobs/eval/job_hivemind_diversity.sh`). The current eval metric is thin: mean pairwise
MiniLM cosine per pool + `%pairs>0.8`. Two problems surfaced in discussion:

1. **Circularity.** The scout retrieves by MiniLM cosine relevance; measuring output
   diversity *also* with MiniLM shares the representation between the thing being
   optimized (retrieval) and the thing measuring success (eval). A gain could be an
   artifact of the shared encoder.
2. **Single-axis + gameable.** One embedding metric can't distinguish "genuinely diverse"
   from "incoherent/degenerate" text (diversity is trivially maximized by garbage — we
   already hit tag-failure/truncation confounds on OvertonBench). And a single threshold
   (`>0.8`) hides mode structure.

This doc elaborates the eval metric into an **independent, multi-axis diversity panel with
a quality guardrail**, then scopes the implementation.

---

## Part 1 — The eval metric, elaborated

**What we are measuring.** Given a pool of N model responses to one open-ended query,
how much does the pool spread across distinct modes? Mode collapse ⇒ pool concentrates
on one mode ⇒ high self-similarity / few effective modes. We compare this per-condition
(baseline vs scout vs div_only) — the **delta**, not the absolute, is the claim.

**Core principle — evaluation independence.** The eval metric must not depend on the same
representation the retrieval optimizes. Two guards:
- Semantic axis uses a **held-out embedder distinct from the scout's MiniLM**.
- Report **embedding-free lexical metrics** as a cross-check that needs no learned model
  at all — if semantic and lexical axes agree, the gain is real, not encoder-induced.

**Three complementary axes** (report all; a real gain moves more than one):

1. **Semantic** (held-out embedder):
   - `mean_pairwise_cos` — mean upper-triangle cosine of the pool (paper's Fig-4 metric).
   - `pct_pairs_gt80 / gt70` — fraction of pairs above threshold (collapse tail).
   - `vendi_score` — exp(Shannon entropy of the normalized-similarity-matrix eigenvalues)
     = **effective number of distinct modes**. Threshold-free headline number; 1.0 = full
     collapse, N = all-distinct. ~10 lines via `numpy.linalg.eigvalsh`, no new deps.
2. **Lexical** (surface, no embedder — zero circularity):
   - `distinct_2 / distinct_3` — unique n-grams ÷ total n-grams across the pool.
   - `self_bleu` — mean BLEU of each response vs the rest (high = repetitive). Report as
     `1 - self_bleu` so higher = more diverse, aligned with the others.
3. **Structural**:
   - captured by `vendi_score` above (effective mode count); optionally a cheap
     cluster-count (agglomerative on the similarity matrix) mirroring the paper's Fig-1
     "2 clusters" demonstration.

**Quality guardrail (anti-gaming).** Diversity is only meaningful at fixed quality.
- **Degeneracy filter (always on):** drop empty, near-empty, and duplicate-token-degenerate
  samples before scoring; carry forward `frac_dropped` per pool. This alone catches the
  truncation/tag-failure confound that bit OvertonBench.
- **Quality score (optional):** a scalar coherence/quality per pool so a diversity gain can
  be reported *at matched quality*. Reuse the OvertonBench judge path
  (`retrieval.answer.chat` + a quality rubric) or perplexity from the served model.

**Aggregation & statistics.**
- Per-query metric → averaged over queries per condition (matches paper).
- **Per-category breakdown** via the INFINITY-CHAT taxonomy label (already threaded through
  `generate_hivemind.py` as `category`) — tests the paper's claim that Brainstorm & Ideation
  is the highest-collapse category, and where injection helps most.
- **Paired significance:** per-query baseline-vs-condition deltas + sign test / Wilcoxon
  (same style as the OvertonBench paired-delta report, 22w/33l). A mean shift without a
  paired test is not evidence.

**Baseline anchors (interpretation aids).**
- Random-pair pool similarity ≈ the paper's 0.1–0.2 floor (fully-diverse reference).
- Full-collapse reference = 1.0 mean-cos / vendi 1.0.

---

## Part 2 — Scope / implementation

**Decisions locked (Q1/Q2):**
- **Eval embedder:** held-out `BAAI/bge-large-en-v1.5` (335M, 1024-d), distinct from the
  scout's MiniLM — breaks circularity, stays offline. Configurable via `--eval_model`;
  MiniLM stays scout-only. (bge-large is symmetric, so no query/passage prefix needed for
  pairwise similarity.) Requires a one-time pre-download into `HF_HOME` on the cluster.
- **Quality guardrail:** degeneracy filters (+`frac_dropped`) ship now; judge-based quality
  scoring is deferred to a later pass (not built here).

Deliverables:
- **`docs/hivemind_diversity_eval.txt`** — this design as a project design doc (project
  convention: `docs/*.txt`, cf. `docs/overtonbench_eval.txt`, `docs/embedding_diversity.txt`).
- **Rewrite `evaluation/hivemind/diversity_metrics.py`** to compute the three-axis panel +
  degeneracy guardrail + paired stats, keeping the existing CSV/table output shape
  (extended columns). Add `--eval_model` (default `BAAI/bge-large-en-v1.5`).
- **`jobs/eval/job_hivemind_diversity.sh`** — add `EVAL_MODEL` env knob; document the
  one-time bge-large pre-download prereq alongside the existing model/dataset prereqs.

New functions in `diversity_metrics.py` (pure, testable without an endpoint):
- `vendi_score(sim_matrix) -> float`  (eigvalsh entropy → effective #modes)
- `distinct_n(responses, n) -> float`, `self_bleu(responses) -> float`  (stdlib n-grams)
- `degeneracy_filter(responses) -> (kept, frac_dropped)`  (empty / near-empty / token-degenerate)
- `pool_panel(responses, embedder) -> dict`  (all three axes for one pool)
- `paired_sign_test(per_query_by_condition) -> dict`  (baseline vs each condition)

Reuse (do not reinvent):
- Held-out embedder loaded the same way `retrieval.scout` loads MiniLM
  (`SentenceTransformer`, cached), just a **different model name** (`--eval_model`).
- Paired-delta reporting pattern from the OvertonBench scoring path.
- Existing per-`(condition, query_id)` pooling + `category` threading already in the file.

Deferred (explicitly not in this pass): judge-based per-pool quality scalar (matched-quality
reporting). Hook left open — the degeneracy filter already carries `frac_dropped` so the
later judge stage can attach to the same panel rows.

Explicitly **out of scope** (keep eval independent of retrieval): the intrinsic
Wasserstein/`branch_divergence` machinery is the retrieval-side signal and must not be
reused as the eval metric.

---

## Verification

- **Unit (local, no GPU/endpoint):** synthetic pools — a hand-diverse pool vs a
  near-duplicate pool — must give `vendi≈N / low cos / high distinct` vs `vendi≈1 / high
  cos / low distinct`. Extends the synthetic smoke test already run this session
  (diverse 0.10 vs collapsed 0.98).
- **Degeneracy filter:** inject empty/truncated samples; confirm `frac_dropped` rises and
  they are excluded.
- **End-to-end (cluster):** `NQ=5 NS=8 CONDS=baseline sbatch jobs/eval/job_hivemind_diversity.sh`;
  confirm the panel CSV + per-category table + paired-stats block print.
- **Independence check:** semantic (held-out) and lexical axes should agree in direction on
  the smoke run; a large disagreement flags an encoder artifact.

---

## Implementation status (completed this session)

All deliverables built and verified:
- `evaluation/hivemind/diversity_metrics.py` — rewritten with the full panel; `--eval_model`
  default `BAAI/bge-large-en-v1.5`.
- `docs/hivemind_diversity_eval.txt` — design doc.
- `jobs/eval/job_hivemind_diversity.sh` — `EVAL_MODEL` knob + pre-download prereq.

Synthetic verification (MiniLM as local eval embedder):

| metric | baseline (diverse) | scout (collapsed) |
|---|---|---|
| `mean_cos` (↓ better) | ~0.00 | 0.89 |
| `vendi` (↑, ≈#modes) | 4.5 (≈N) | 1.4 (≈1 mode) |
| `distinct_2` (↑) | 1.00 | 0.68 |
| `inv_self_bleu` (↑) | 1.00 | 0.85 |

Degeneracy filter dropped empty + token-degenerate samples (`frac_dropped=0.33`); all three
axes agreed in direction (independence check passes).
