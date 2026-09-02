# CAD: turning injection into a dial

**Status: designed, not run.** `retrieval/cad.py` exists and self-tests; no eval
arm, no job, no numbers.

## The method

Context-aware decoding (Shi et al., [arXiv:2305.14739](https://arxiv.org/abs/2305.14739))
contrasts the with-context and without-context next-token distributions:

```
logits = (1 + alpha) * logits(y | context, x)  -  alpha * logits(y | x)
```

| alpha | effect |
|---|---|
| `0` | ordinary with-context decoding — what every current condition does |
| `> 0` | **amplify** the injected context's influence |
| `< 0` | **suppress** it, interpolating back toward the no-context model |

Two forward passes per token over the same model, one with the fork in the
prompt and one without.

`alpha < 0` is the part that matters. There is no prompt that means "pay less
attention to the context I just gave you" — the instruction is itself context.
CAD reaches that region; prompting cannot.

## Why this experiment, from the measured evidence

Four results point here, and no other untested method sits at their intersection.

**1. Prompt-level injection fails regardless of content.** `scout` 0.3927,
`distributional` 0.3941 — a gap of +0.0014 against a 0.027 noise floor. Changing
*what* is injected changes nothing. That rejects the content hypothesis and
leaves the mechanism.

**2. The failure mode is over-anchoring, not irrelevance.** `docs/framing_hurts.png`:
on-pole similarity 0.334 -> 0.389 under injection, and
`corr(attraction, dcoverage) = -0.31`. The model collapses onto the two injected
poles and loses the positions it would have produced unaided. Suppressing context
influence is the direct counter, and it is exactly what `alpha < 0` does.

**3. Logit-level beat prompt-level once already.** G2 on INFINITY-CHAT: 19/20
paired wins on `vendi` and `mean_cos`, 5 of 7 metrics at p <= 0.0004. Same model,
same queries — asking for diversity failed, steering the logits worked. CAD is the
same intervention point applied to coverage instead of diversity.

**4. The relationship we need to exploit is a gradient, not a threshold.**
`baseline_gate.py`: injection helps where the baseline was weak (r = -0.433,
p < 0.001), but every discrete rule collapses — best independent LOO gain
**+0.0060**, 52 of 60 questions unchanged, best threshold 0.80 (i.e. "inject on
50 of 60"). Five signal sources failed to produce a usable switch.

`alpha_from_contestedness()` is the continuous version of that decision. Consensus
questions decode with context suppressed, contested ones amplified, everything
between is interpolated. **A dial is the right shape for a gradient; a switch is
not.** That is the strongest argument for this experiment and it is not available
to any prompt-level method.

## What exists

`retrieval/cad.py`:

- `cad_logits(logits_ctx, logits_plain, alpha)` — the combination rule
- `alpha_from_contestedness(score, cfg)` — soft routing, mapping a
  self-consistency score in [0,1] onto `[alpha_consensus, alpha_contested]`,
  default `[-0.5, +1.0]`
- `cad_generate_ids` / `cad_generate` — two KV caches stepped in lockstep
- `_selftest()` — synthetic, verifies the algebra

Missing: an eval condition in `evaluation/overton/eval_overtonbench.py`, a job,
and a 7B baseline to compare against (below).

## Arms

All at the same model, one alpha sweep plus the soft-routed arm.

| arm | context | alpha | tests |
|---|---|---|---|
| `base7b` | none | — | the reference; **must be regenerated, see below** |
| `ctx0` | fork | 0.0 | reproduces `scout` at 7B; isolates the model change |
| `cad_neg` | fork | -0.5, -0.25 | **anti-anchoring — the hypothesis** |
| `cad_pos` | fork | +0.25, +0.5, +1.0 | is the fork *underused* rather than unhelpful? |
| `cad_soft` | fork | `alpha_from_contestedness` | the continuous router |

`ctx0` is the load-bearing control. Without it, any CAD-vs-baseline difference
confounds the decoding change with the model change.

Sweep alpha on a **held-out half** of the questions and report the other half.
With 60 questions and 6 alpha values, picking the best cell post hoc is the same
mistake `bestofk` flagged in its own sweep output.

## The model problem, stated plainly

CAD needs logits, so it needs local HF weights. Every OvertonBench number you have
is **Qwen2.5-72B-AWQ served through vLLM**, which does not expose logits. The
practical model is **Qwen2.5-7B-Instruct** — the same one `job_probe.sh` and
`job_g2_diversity.sh` already use.

So CAD results are **not comparable to v9/v10**. A 7B baseline has to be
regenerated and rejudged, which means a new baseline, a new oracle, and a new
union. Budget that as part of the experiment rather than discovering it after.

This is not incidental. The probe experiment has the same defect and it was never
called out: the probe reads the *7B's* hidden states while the deltas it is scored
against come from the *72B*. Whatever CAD is compared to must come from the model
CAD runs on.

## What to measure

**Outcome.** OvertonScore via the existing judge. Paired against `ctx0`, not
against `base7b`, with the bootstrap CI on the mean delta — the sign test drops
magnitude and misled us once already (p=0.211 vs bootstrap p=0.075 on the same
data).

**Mechanism, and do not skip this.** The hypothesis is specifically that
`alpha < 0` reduces pole attraction. Measure on-pole similarity directly, the
`framing_hurts` quantity: injected 0.389 vs baseline 0.334. If coverage improves
while attraction is unchanged, the improvement came from somewhere else and the
stated mechanism is wrong.

An outcome without the mechanism is the weaker result. Both together is a claim.

## How to read it

- **`cad_neg` beats `ctx0` and attraction falls** — the over-anchoring diagnosis
  is right, and prompt-level injection was failing for a fixable reason. The
  strongest available outcome.
- **`cad_pos` beats `ctx0`** — the fork was *underused*, not unhelpful. Opposite
  diagnosis, equally publishable, and it would revive the retrieval contribution.
- **`cad_soft` beats every fixed alpha** — the continuous router works where the
  discrete one did not. This is the outcome that would reopen routing, and the
  only one that would.
- **Nothing separates** — injection fails at the decoding level too, which closes
  the test-time family and makes the train-time argument by elimination rather
  than assumption.

Every branch is informative. That is the argument for running it.

## Cost and dependencies

Two forward passes per token, no batching across the pair, ~1024 new tokens. On
one A100 with a 7B in bf16 this is roughly G2's cost profile — budget ~1 min per
answer, so 60 questions x 7 arms is a few hours plus the 7B baseline regeneration.

`cad_soft` additionally needs a contestedness score per question, which costs
K=8 samples each. Those already exist for the 60 eval questions in
`contestedness_labels.json` from `job_probe.sh` — same model, same prompt, so they
are reusable. **Do not regenerate them.**

## Failure modes to watch

- **Degenerate text at large |alpha|.** Contrastive decoding is unstable when the
  two distributions disagree sharply; `alpha = 1.0` may produce fluent nonsense
  that the judge scores as zero coverage. Read samples before trusting the number.
- **`alpha < 0` collapsing to baseline.** If suppression works too well the answer
  is just the no-context answer, and the arm has measured nothing. `ctx0` and
  `base7b` bracket this — `cad_neg` landing exactly on `base7b` is a null, not a
  win.
- **Two KV caches drifting.** `cad_generate_ids` must step both in lockstep on the
  *same* sampled token. The `BatchEncoding` unwrap bug that hit `g2.py` and
  `cad.py` is fixed; the lockstep property is covered by `_selftest`.
- **Judge validity.** rho = 0.059 within-participant. A negative result here is
  weak evidence in both directions, the same caveat that limits every OvertonBench
  conclusion in this project.
