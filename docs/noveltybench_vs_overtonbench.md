# NoveltyBench vs. OvertonBench

Two "diversity" benchmarks that measure **different axes**. Conflating them is
easy and wrong: one scores diversity *across independent samples*, the other
scores pluralism *within a single answer*. PluralTree touches both, on different
tracks — this note pins down which is which.

## TL;DR

| | **NoveltyBench** | **OvertonBench** |
|---|---|---|
| Question it asks | Sample the model 10× — how many *distinct* good answers? | For one answer, does it *cover the range* of human viewpoints? |
| Diversity axis | **Across samples** (mode collapse) | **Within one response** (Overton-window coverage) |
| Failure it targets | Model regenerates the same answer | Model gives one narrow / one-sided answer |
| Ground truth | 8 author reference responses (human diversity floor) | Real human ratings + written perspectives (N=1208) |
| PluralTree track | our hivemind / INFINITY-CHAT eval | our **primary** eval (scout injection) |

## The core distinction

- **NoveltyBench = between-sample novelty.** Draw *k=10* generations at
  temperature 1.0 from the *same prompt* and count how many are functionally
  distinct. A perfect single answer scores badly if the other nine repeat it.
  This is a **mode-collapse** metric — the same family as INFINITY-CHAT /
  "Artificial Hivemind" and our `evaluation/hivemind/` eval.

- **OvertonBench = within-answer pluralism.** Score **one** response against the
  spread of *human* viewpoints on a contested question: coverage = fraction of
  human viewpoint clusters whose members would feel represented. Sampling the
  model repeatedly is not the point; representing the *population's* range in a
  single answer is.

A model can win one and lose the other. A response that carefully enumerates
every viewpoint (high OvertonScore) is a single mode — if all 10 samples say the
same enumerated thing, NoveltyBench `distinct₁₀ ≈ 1`. Conversely 10 wildly
different one-sided takes score well on NoveltyBench and poorly on OvertonBench.

## Side-by-side

| Dimension | NoveltyBench | OvertonBench |
|---|---|---|
| Paper | Zhang et al., 2025 (arXiv:2504.05228) | Poole-Dayan et al., ICLR 2026 |
| Prompts | 1,100 (NB-Curated 100 + NB-WildChat 1,000) | 60 politically-salient US questions |
| Domains | randomness, factual, creative writing, subjectivity + real ChatGPT queries | US political / social opinion |
| Samples scored | **10 per prompt**, temp 1.0 | **1 response** per (question, condition) |
| Unit of diversity | equivalence classes over the 10 samples | human viewpoint clusters (`cluster_kmeans`) |
| Core metric | `distinctₖ` = #equivalence classes among k | `coverage` = covered clusters / n_clusters |
| "Covered / distinct" rule | DeBERTa-v3-large equivalence classifier (79% acc, F1 0.811) | cluster covered iff mean predicted rating ≥ 4 (1–5) |
| Quality guard | `utilityₖ`: Skywork-Reward-Gemma-2-27B score, geometric patience weight p=0.8 | judge predicts *human* representation rating (quality = human-felt representation) |
| Human reference | 8 author responses (diversity lower bound) | 1208 humans, 8 LLMs rated, held-out ratings |
| Aggregate | mean `distinct₁₀` / `utility₁₀` over prompts | `OvertonScore` = mean coverage over questions |
| Judge/scorer | trained dedup + reward models (no human-in-loop at eval) | LLM judge, **self-validated** against held-out human ratings |

## Metric detail

**NoveltyBench.**
- `distinctₖ = |{ cᵢ : i ∈ [k] }|` — count of distinct equivalence classes, where
  "equivalent" = a user seeing one gains nothing from the other (functional
  equivalence, learned by the DeBERTa classifier).
- `utilityₖ = (1−p)/(1−pᵏ) · Σ pⁱ⁻¹ · 1[cᵢ is novel vs earlier] · uᵢ` — only
  functionally-new responses accrue utility, discounted by user patience p=0.8,
  weighted by quality uᵢ. Diversity that is low-quality or redundant earns
  nothing.

**OvertonBench.**
- For each participant of a question, a judge predicts their 1–5 rating of our
  response (few-shot on that participant's 8 real ratings). `covered(cluster) =
  mean predicted rating over its members ≥ 4`; `coverage = covered / n_clusters`;
  `OvertonScore = mean coverage over questions`. Judge validity is a *precondition*
  — it must beat the mean-of-others baseline and be length-unbiased
  (see `docs/overtonbench_eval.txt`, `judge_overtonbench.py --validate`).

## Findings contrast

- **NoveltyBench:** frontier models emit < 4 distinct answers per 10 queries;
  **larger models are *less* diverse** within a family (Llama-3.2-1B distinct₁₀
  7.74 vs Llama-3.1-405B 4.20) — capability ≠ generative diversity. In-context
  "give a different answer" regeneration best recovers diversity.
- **OvertonBench (ours, v5):** injecting scout forks into one answer *hurt*
  coverage (baseline 0.497, scout 0.432) — the injected 2-pole binary collapses
  the answer onto the extremes (`docs/framing_hurts.png`). Opposite lesson:
  here the problem is a single answer being too *narrow*, and naive injection
  makes it narrower, not more collapsed-across-samples.

## What each means for PluralTree

- **OvertonBench is our primary metric.** The scout→inject→reason pipeline is a
  *within-response* intervention: broaden one answer to cover the human range.
  OvertonBench measures exactly that against real human ratings, so gains are
  interpretable as representation, not just lexical spread. This is why the GRPO
  reward (`docs/grpo_alignment.txt`) is a within-answer coverage of subgroup
  positions, not a between-sample novelty count.

- **NoveltyBench ≈ the other track.** It is the clean, external counterpart to
  the INFINITY-CHAT mode-collapse eval we built in `evaluation/hivemind/`
  (`docs/hivemind_diversity_eval.txt`): both score *across* samples. Differences
  worth borrowing:
  - NoveltyBench's **learned functional-equivalence** dedup is stronger than our
    MiniLM-cosine `%pairs>0.8` — and it is *independent of the scout's encoder*,
    which is exactly the circularity guard our hivemind design already flags.
  - Its **utility (quality×diversity)** metric is the anti-gaming guardrail our
    plan defers; the patience-weighted form is a ready template.
  - Its **larger-model-less-diverse** finding is a caution for the GRPO phase:
    optimizing a bigger policy for coverage could still collapse *across* samples
    even while each answer looks plural. If we ever report NoveltyBench-style
    numbers, do it at fixed quality.

- **Do not cross the metrics.** A coverage win on OvertonBench says nothing about
  mode collapse, and vice-versa. If we claim "pluralism," name the axis.

## When to use which

- Measuring whether **one answer represents a contested population** → OvertonBench.
- Measuring whether the model **can produce many different good answers** (creative
  / open-ended / mode collapse) → NoveltyBench.
- PluralTree's thesis (retrieval-broadened single answers) is an OvertonBench
  claim; NoveltyBench is the secondary, cross-checking axis.

## Sources
- [NoveltyBench (arXiv:2504.05228)](https://arxiv.org/abs/2504.05228) ·
  [HTML](https://arxiv.org/html/2504.05228v1) ·
  [project page](https://novelty-bench.github.io/)
- OvertonBench: Poole-Dayan et al., ICLR 2026 (HF `elinorpd/overtonbench`);
  local: `docs/overtonbench_eval.txt`, `evaluation/overton/`.
