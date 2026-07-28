# Adaptive injection: three adaptations from the 2026 adaptive-RAG literature

## The problem

Deciding **when** to inject retrieved perspectives. Two attempts failed:

| attempt | mechanism | result |
|---|---|---|
| `route_signal` | threshold a **graph** signal (W, calibrated z, relevance, driver-match) | no signal separates helped from hurt. `relevance` is *anti*-correlated (−0.26). Root cause: graph divergence = demographic subgroup splits, which correlates only **+0.20** with human contestedness |
| `route` (v6) | **ask the model** in `<think>` whether the question is contested | **0.072** vs baseline 0.479 |

The prize is real: always-baseline 0.497, always-scout 0.443, **oracle 0.622**.

The 2026 adaptive-RAG literature reaches the same conclusion independently — a
retrieval decision taken *solely from the LLM's output* is "superficial", being
itself hallucination-prone and an unreliable indicator of what the model needs.
SOTA has moved to internal states ([CtrlA](https://arxiv.org/abs/2405.18727),
[SeaKR](https://arxiv.org/pdf/2406.19215)), token-level uncertainty (FLARE,
DRAGIN, [DTR](https://arxiv.org/pdf/2601.03908)), and trained routing (Self-RAG,
[SIGIR'26](https://arxiv.org/pdf/2604.26649)).

**Caveat that shapes all three adaptations:** those methods trigger on
*knowledge sufficiency* ("do I know this fact?"). Ours is a **diversity** gap
("do people disagree?"). A confident answer to a contested question is exactly
the case where we *do* want to pluralize — so entropy-style triggers do not
transfer unmodified.

---

## 1. Self-consistency contestedness — `retrieval/contestedness.py`

Don't ask the model and don't read the graph: **measure**. Sample K answers under
a **commit-forcing** prompt and see whether the model lands on the same stance
each time. Wide spread = the model itself is unsettled = contested.

- `PROBE_INSTRUCTION` forces one stance per sample. This is load-bearing: sampled
  under a pluralism prompt the model enumerates everything every time, so spread
  goes *low* on contested questions — backwards.
- `stance_spread` → mean pairwise distance + Vendi (effective number of stances),
  on a **held-out mpnet** embedder so the gate is independent of the MiniLM
  retrieval it gates.
- `contestedness_score` ∈ [0,1] combines "how far apart" with "how many modes".
- `evaluate_gate` scores the gate **post-hoc against a finished run** — we
  already have per-question baseline/scout coverage, so no generation is needed
  to know whether the signal would have worked. Same design as `route_signal`.

Runnable now: `jobs/eval/job_contestedness.sh`. Cost = 60 questions × K samples.

## 2. Contestedness probe — `alignment/probe.py`

CtrlA extracts *honesty* and *confidence* directions from activations and
triggers retrieval off the confidence monitor. Same trick, different quantity:
a linear probe for a **contestedness direction** in hidden states — reading what
the model internally represents, before it collapses into a token (which is
where `route` lost the signal).

- `HiddenExtractor` — mean-pooled hidden states at layer L (needs local HF
  weights; the OpenAI-compatible endpoint does not expose them).
- `LinearProbe` — torch logistic regression; `.direction` is the unit vector,
  reusable for activation steering.
- `cross_val_auc` — **the only number that means anything.** With ~60 questions a
  probe fits the training set perfectly; the self-test deliberately shows the
  noise case at train 0.90 / cv 0.60.

**Label source matters:** do *not* train on graph divergence — you inherit its
+0.20 ceiling. Use adaptation 1's score as a weak label (it measures the model's
own stance spread), or external controversy data.

## 3. Context-aware decoding — `retrieval/cad.py`

Stop deciding *whether*; control *how much*. CAD (Shi et al., 2023):

```
logits = (1 + α)·logits(y | context, x) − α·logits(y | x)
```

- `α = 0` → today's behavior; `α > 0` → amplify the forks; **`α < 0` → suppress
  them**, interpolating back toward the no-context model. That anti-anchoring
  direction is unreachable by prompting, and it targets the measured failure
  directly: injection over-anchors on the two poles (`docs/framing_hurts.png` —
  on-pole similarity 0.334→0.389, corr(attraction, Δcoverage) = −0.31).
- `alpha_from_contestedness` is the **synthesis**: α is set continuously from
  adaptation 1's score, giving *soft* routing instead of the binary gate every
  previous attempt tried and failed to learn.

Needs local HF weights (two forward passes per token, stepped in lockstep with
separate KV caches).

---

## How they compose

```
contestedness score  ──► binary gate        (adaptation 1, cheap, runnable now)
        │
        ├────────────► weak label for probe (adaptation 2, needs hidden states)
        │
        └────────────► α for CAD            (adaptation 3, soft routing)
```

Adaptation 1 is the only one that runs against the served endpoint. 2 and 3 need
local HF weights, so they target the GRPO base (Qwen2.5-7B-Instruct), not the
72B AWQ eval model.

## Order of work

1. **Run adaptation 1 and evaluate the gate post-hoc.** If `corr(score, Δ)` is
   flat and helped/hurt do not separate, self-consistency is dead too — and the
   honest conclusion is that per-question routing is not the lever.
2. Only if (1) shows signal: train the probe (2) as the cheap inference-time
   version, and/or sweep α in (3).
3. `cad.py` is worth a sweep **regardless** of (1), because it attacks
   over-anchoring rather than routing — it is the one adaptation that does not
   depend on predicting contestedness at all.

## Standing caveat

Everything here is scored by the OvertonBench judge, whose validity is still
open: within-participant discrimination is ≈0 (mean ρ +0.059, median 0.000, 16%
degenerate) and aggregate ρ=0.357 vs the paper's 0.88 (confounded by AGGU=5 —
rerun pending). Until the judge is certified, a gate that "improves" OvertonScore
by 0.03 is not evidence.
