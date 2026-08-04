# Test-time methods not yet tried

Written because "test-time methods aren't enough" is currently **overstated**. The
evidence is one benchmark, one weakly-validated judge, and a single family of
intervention. This catalogues what the evidence actually covers and what it does
not, so the GRPO phase is chosen against a known map rather than an assumed one.

## What has actually been tested

The OvertonBench condition sweep (v4 → v6): 60 US-political questions, paired
per-question, unweighted cluster coverage.

| condition | intervention | result |
|---|---|---|
| `baseline` | none | 0.507 |
| `scout` | top-k divergent forks in prompt (tau=0.25, alpha=1.0) | ≤ baseline |
| `div_only` | divergence with no relevance gating (tau=0, alpha=0) | ≤ baseline |
| `route` | commit to one position | 0.072 (collapse) |
| `expand` | broaden the answer set | ≤ baseline |
| `merge` | answer twice, then merge | slightly < baseline |

Plus the union analysis on the same responses: **union 0.687 > oracle 0.622 >
best single answer 0.507**.

**Every one of these varies what text goes into the prompt.** That is one family.
The sweep licenses "prompt-level injection variants did not beat baseline" and
nothing broader.

Two caveats that limit how hard any of these numbers can be pushed:

- **Judge validity.** Aggregate rho = 0.167 against the source paper's 0.88;
  within-participant rho = 0.059 (chance). A negative result from a weak
  instrument is weak evidence in both directions.
- **Noise floor 0.027**, from two independent baseline draws. Several condition
  deltas plausibly sit inside it.

## In flight (queued, no numbers yet)

- **`distributional`** — inject the full subgroup distribution rather than the
  selected fork. Motivated by the finding that OpinionQA's middle is a real
  graded spectrum, not noise between poles. Built, never evaluated.
- **G2** (arXiv:2511.00432) — `logits = z + alpha(z+ - z-)` with an entropy gate
  `alpha = theta if H(z) >= beta else 0`, plus Center Selection. First
  decode-time method attempted. `retrieval/g2.py`, `jobs/eval/job_g2_diversity.sh`.

Neither has returned. Calling test-time insufficient before they do is not
supported.

## Untested

Ordered by how directly each attacks **the union gap** — the fact that coverage
already exists across samples (0.687) but not within one answer (0.507). A method
that converts across-sample coverage into within-answer coverage is attacking the
measured problem; one that only adds context is repeating what already failed.

### 1. Sampling + selection — highest priority

The union result says the good answers are *already in the sampling
distribution*. Selection is the cheapest way to exploit that, and it needs no
training and no new model.

- **Best-of-K with a coverage selector.** Sample K, score each with
  `coverage_reward` (already implemented), return the argmax. Directly converts
  the 0.687 that exists across samples into a single returned answer.
  *Cost:* K x inference. *Needs:* nothing new — `alignment/reward.py` works today.
  *Note:* this doubles as the Stage-0b reachability diagnostic for GRPO, so it
  pays for itself either way.
- **MBR decoding against a diversity/coverage utility.** Sample K, return the
  candidate maximizing expected utility against the rest. Standard MBR, swapped
  utility.
- **NoveltyBench-style sample-then-partition.** Cluster K samples, return a
  representative set. Changes the output contract from one answer to a set — see
  the compute-matching note below.

**Compute matching is the catch.** Best-of-K spends K times the inference of
`baseline`. A win over single-sample baseline is not a fair comparison; the
honest control is baseline-with-K-samples-and-random-selection, or reporting at
matched token budget. The `merge` condition was an attempt at this and lost, so
the naive version of this critique is already partly answered.

### 2. Decode-time / logit-level

Changes the distribution being sampled from rather than what conditions it. G2
covers part of this space; these do not.

- **CAD** (arXiv:2305.14739) — `(1+alpha) * logits_ctx - alpha * logits_plain`.
  With `alpha > 0` it amplifies context reliance, which is the direct test of
  whether injected forks are being *underused* rather than *unhelpful*. With
  `alpha < 0` it suppresses context. Cheap: one extra forward pass, and
  `retrieval/g2.py` already runs multiple KV caches in lockstep, so the
  machinery exists.
- **DoLa / contrastive decoding across layers** — contrasts early vs late layer
  logits. Untested here, and unclear it targets pluralism specifically.
- **Entropy-gated temperature.** Raise temperature only at high-entropy
  positions. Cheaper cousin of G2's gate; isolates whether the gate or the
  contrastive term is doing G2's work.

### 3. Multi-pass / iterative

- **Critique-and-revise.** Generate, then ask the model which positions are
  missing relative to the injected forks, then revise. Targets the elicitation
  gap directly and is a genuinely different mechanism from single-pass prompting.
- **Multi-persona / role-conditioned generation, then merge.** Generate once per
  demographic axis, merge. `merge` merged two *unconditioned* answers; this
  conditions each pass on a different subgroup first. Plausibly the strongest
  untested prompt-family variant, since it manufactures the across-sample
  diversity that the union result shows is useful.

### 4. Graph-side search (retrieval still frozen, but used harder)

- **Multi-anchor retrieval.** Currently one anchor per question. Union over
  several anchors' subtrees widens the position set.
- **Beam search over forks** rather than top-k by score.
- **Iterative anchor refinement** — re-run the scout conditioned on the draft
  answer to find what it missed.

### 5. Controls that were never run

Cheap, and their absence is a real hole in the negative result.

- **Temperature / top-p sweep.** No number for baseline at varied temperature.
  If diversity moves with temperature alone, several conclusions shift.
- **Repetition / presence penalty.**
- **Length-matched prompting.** Injected answers run ~330 words vs baseline ~69.
  A length-matched baseline is a missing control everywhere in this project.

## Priority

1. **Best-of-K + coverage selector** — exploits the union gap directly, reuses
   existing code, and doubles as the GRPO reachability diagnostic.
2. **Temperature sweep** — hours of work, and it conditions the interpretation of
   everything else.
3. **CAD with alpha > 0** — decides whether injected context is underused or
   unhelpful, which is the question the whole injection line rests on.
4. **Multi-persona then merge** — best remaining prompt-family idea.
5. Everything else.

## Relationship to the GRPO phase

These are not mutually exclusive with training. The scout is frozen during RL, so
any decode-time or selection method here composes with a trained policy. Best-of-K
in particular is the same computation as GRPO's rollout scoring — a strong
best-of-K result is direct evidence RL has headroom, and a flat one is evidence
it does not.
