# Hivemind Diversity Eval — Concepts & Rationale

Companion to `docs/hivemind_diversity_eval.txt` (design) and
`docs/hivemind_diversity_eval_plan.md` (plan). Captures the *why* behind the eval.

## Purpose

Check two things about generation on open-ended queries:
1. Does the **generator collapse** — give near-identical answers to a question that has
   many valid ones?
2. Does **scout-injected divergence** reduce that collapse?

The graph-free `baseline` alone answers (1); comparing `baseline` vs `scout`/`div_only`
answers (2).

## The generator

The frozen LLM served by vLLM (default Qwen2.5-72B-Instruct-AWQ). It is **not trained or
modified** by this eval. The only thing that differs across conditions is the **prompt**:

- `baseline` — just the query.
- `scout` / `div_only` — the query **plus** divergent perspectives the scout retrieved from
  the knowledge graph, injected as context.

So the eval measures whether prepending scout context changes what a fixed model produces.

## Mode collapse

Ask the model the same open-ended question N times; it keeps returning essentially the same
answer, and different models converge on the same answer too. There is no single correct
answer to "write a metaphor about time," yet ~50 samples nearly all say "time is a river."
The output distribution has collapsed onto one mode instead of spreading over the valid ones.

Measured concretely: sample N=50 per query, embed the pool, quantify self-similarity.
- High self-similarity / few effective modes (Vendi ≈ 1) = collapsed.
- Low self-similarity / many modes (Vendi ≈ N) = diverse.

## Why circularity matters (the key methodological point)

**The trap.** The scout selects which forks to inject by **MiniLM cosine** — it searches for
content that is *far apart in MiniLM space*. If the eval also measures diversity as *distance
in MiniLM space*, it scores outputs with the exact yardstick the retrieval was built to
stretch. Retrieval and metric then share one geometry: the metric partly measures "did the
injection move things in MiniLM space," not "did the model produce genuinely diverse answers."

**The failure it hides.** Injected forks might nudge the model toward different *words MiniLM
weights heavily* while the answers stay substantively the same idea. A held-out encoder or a
human calls them near-duplicates; MiniLM — the space the forks were selected to spread —
scores them far apart. You would report a diversity gain that does not exist. The bias runs
one direction: it systematically **flatters the scout**.

**Framing.** This is train/test leakage. The thing you optimize (retrieval objective) and the
thing you evaluate (metric) must be independent, or the result is circular — you cannot claim
"scout increases diversity" if diversity is *defined by the signal scout maximizes*. It is the
first objection a reviewer raises.

**The fix.**
- **Held-out embedder** (`bge-large`, distinct from MiniLM): different geometry, not the target
  of retrieval, so a gain there reflects genuine semantic spread — not the scout gaming its own
  metric.
- **Lexical axis** (distinct-n, self-BLEU): uses *no* learned encoder, so no embedding trick
  can game it.
- If all three independent axes move together, the diversity is real.

**Scope.** Circularity only bites the `scout` / `div_only` conditions (they use the scout).
The `baseline` number has no retrieval and shares nothing, so the graph-free baseline collapse
result is clean regardless.

## Two diversity notions — keep separate

- **Intrinsic graph divergence** (`evaluation/intrinsic/branch_divergence.py`): Wasserstein
  over child-subtree *embeddings* — the **retrieval-side** signal (how the scout finds forks).
- **Extrinsic output diversity** (this eval): self-similarity of *generated text* — the
  **eval-side** signal (did behavior actually diversify).

Never reuse the intrinsic Wasserstein machinery as the eval metric — that would measure the
retrieval signal against itself (the same circularity, one level up).
