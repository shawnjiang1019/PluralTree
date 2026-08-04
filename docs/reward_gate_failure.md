# Stage 0 gate failure: why the GRPO reward does not rank like the judge

Run: `jobs/eval/job_reward_correlation.sh` (job 66879256), v6 and v5,
`d ∈ {0,30,60,90,120,150}`, gate 0.60.

**Verdict: FAIL on both versions.** Within-question pairwise concordance at the
headline `d=60` was **0.070** (v6, route excluded) and **0.092** (v5) against a
chance level of 0.500. No training was launched — the gate's exit code stranded
`grpo_smoke` and everything behind it, as designed.

---

## 1. The headline number is not what it looks like

`_concordance` counts a **reward tie as a disagreement**. That is deliberate — a
reward that cannot separate two rollouts gives GRPO no gradient — but it means a
reward that is *zero almost everywhere* scores near 0.0 rather than near 0.5.
0.070 is therefore mostly "undefined," not "anti-correlated."

Concordance tracks `1 − frac_zero` almost exactly (v6, route excluded):

| `d` | frac_zero | concordance |
|---|---|---|
| 0 | 0.76 | 0.158 |
| 30 | 0.83 | 0.123 |
| 60 | 0.89 | 0.070 |
| 90 | 0.93 | 0.061 |
| 120 | 0.95 | 0.044 |
| 150 | 0.96 | 0.026 |

The judge, on the same responses: mean 0.432, `frac_zero` **0.23**. The judge
discriminates fine. The reward does not fire at all.

## 2. The actual finding

**At `d=0` — with the depth gate entirely disabled — 76% of responses matched
zero positions.** Not one unit in those responses reached cosine 0.50 against any
position, while the judge scored the same responses at 0.43 mean coverage.

`v1` (also depth-blind) is 78% zero. So the failure is **upstream of every v2
change**, at `mentioned = sim.max(axis=0) >= match_thr`.

## 3. What this rules out

- **Not the depth gate.** `d=0` already fails. The `d` sweep is measuring the
  sparsity gradient, nothing else.
- **Not the v1→v2 redesign.** Depth gating, uniform weighting, and the
  multiplicative form were never exercised — the signal is gone before they
  apply. `d=0` scoring higher than `d=60` here is **not** evidence against the
  depth gate; it is the same sparsity effect. **The v2 fixes remain
  unevaluated.**
- **Not version-specific.** v6 and v5 agree (0.070 / 0.092). Note the two v5
  blocks are numerically identical — v5 has no `route` condition, so
  `--exclude route` was a no-op there and the run yielded 3 distinct
  measurements, not 4.

## 4. Candidate causes, in likely order

**(a) The match target is the wrong shape.** `Position.embed_text` is
`"<question> <option>"` — a *question-shaped* string — compared against
*answer-shaped* response prose. mpnet is symmetric and this is an asymmetric
comparison, which depresses cosine across the board. Separately, every option of
an item shares the question stem, so positions are near-identical to each other,
which also makes the argmax assignment in `position_depths` close to arbitrary.

**(b) `match_thr = 0.50` was never fitted.** Chosen, not measured. If the real
cosines live at 0.30–0.45 the threshold sits above the entire signal.
`min_depth_words` was flagged provisional in `RewardConfig` and repeatedly
scrutinized; the threshold sitting *upstream* of it never was. Wrong ordering.

**(c) Construct gap.** The reward matches ATP **survey options** (pollster
language: "About right", "Stricter gun laws"). The judge scores coverage of
**human viewpoint clusters** (open-ended perspectives in people's own words).
Different target sets — previously measured at corr +0.20 between graph
structure and human cluster structure. No threshold tuning closes this one.

(a) and (b) are mechanical and fixable. (c) is a design premise.

## 5. Is there signal underneath?

Weaker than it first appears. "Picks the judge's best condition" runs above
chance, but only convincingly in the easy case:

| block | n_cond | chance | observed | ≈ SE | z |
|---|---|---|---|---|---|
| v6 all (incl. `route`) | 4 | 0.250 | 0.40–0.50 | 0.06 | ~2.5 |
| v6 excl. `route` | 3 | 0.333 | 0.42–0.48 | 0.068 | ~1.6 |
| v5 | 3 | 0.333 | 0.38–0.45 | 0.063 | ~1.2 |

The clear above-chance result is the block containing `route`, where the reward
only has to notice a 0.4 collapse. **In the near-tie regime — which is what a
GRPO group actually looks like — best-pick is not distinguishable from chance.**

Pooled correlations are also ~0 (v6 excl. route, `d=60`: −0.028).

So "the objective is fine, only the matcher is broken" is a *hypothesis*, not a
finding. It has not been established.

## 6. The tension in the gate itself

The reward was designed to be **causally independent** of the judge — that is the
argument for graph-grounding, and why it can train on unlimited questions without
leaking the benchmark. The gate demands it **agree** with the judge. Independence
and agreement pull against each other, so a low score is genuinely ambiguous
between "the reward is broken" and "the reward measures something else, as
intended, and the judge is noisy anyway" (within-participant rho = 0.059).

This does not make the gate wrong. If you train on the reward and evaluate on the
judge, you need *some* agreement or training cannot move the metric — arithmetic,
independent of which instrument is more valid. But it does mean the fix might be
"make the reward's positions resemble the judge's clusters," which is a design
change, not a calibration.

## 7. Why the self-test did not catch this

`_selftest` builds positions as `embed_text=f"{opt} {opt} {opt} {opt}"`, with the
comment "a real embedder does this via token salience; the stub needs it made
explicit." The stub was constructed so that matching would succeed, and the depth
gate was then verified on top of it. That validates the **logic** and says nothing
about whether `match_thr` is calibrated for mpnet on real text. Any offline test
with a synthetic embedder has this blind spot.

## 8. Diagnostics added for the re-run

In `scripts/analysis/reward_eval_correlation.py`:

- **`_concordance_split`** → `tie_rate` and concordance computed only over pairs
  the reward actually separated. Distinguishes "orders wrongly" from "cannot
  order," which need opposite fixes.
- **cosine distribution** — percentiles of `pos_best` (what `mentioned`
  thresholds) and `unit_best` (what `precision` thresholds), plus the fraction
  clearing `match_thr`.
- **units/response**, including `frac_with_ZERO_units` — `split_units` discards
  any unit under 8 words, and a response left with no units scores 0 regardless
  of content. Third candidate cause, previously invisible.

`coverage_rewards_sweep` now also returns the `sim` matrix so callers can inspect
the distribution the threshold slices.

## 9. Decision rule

Re-run `sbatch jobs/eval/job_reward_correlation.sh` (~15 min), then:

| observation | diagnosis | action |
|---|---|---|
| high `tie_rate`, `conc\|separated` ≳ 0.5 | too sparse | recalibrate `match_thr`; drop the question stem from `embed_text` |
| low `tie_rate`, `conc\|separated` < 0.5 | orders wrongly | redesign the objective — cause (c) |
| `pos_best` p75 < `match_thr` | threshold above the signal | recalibrate first, re-measure before anything else |
| `frac_with_ZERO_units` high | unit segmentation | lower the 8-word floor in `split_units` |

Do not tune `min_depth_words` until `mentioned` fires at a sane rate. It is
downstream of the actual failure.

## 10. Consequence for the phase

GRPO is not available until this is fixed — 76–96% zeros produce no gradient
whatever the cause. The motivation for the phase is untouched: the union gap
(0.687 across samples vs 0.507 in any single answer) is a within-run comparison,
robust to judge calibration, and unaddressed by any prompting condition.

That same gap also motivates **best-of-K with a coverage selector**, which needs
only a good *ranker*, not a trainable gradient. If the reward turns out fixable
as a ranker but never good enough to train on, that path remains open — see
`docs/untested_test_time_methods.md`.
