# Results snapshot — September 2026

What the last round of experiments established, what it invalidated, and what is
still running. Numbers are copied from job logs; anything not measured is marked
as such.

**One-line state.** The pipeline has three stages — retrieve, deliver, compress —
and the evidence now says two of them are broken, neither of which is retrieval.

---

## 1. The three-stage decomposition

Every result below belongs to exactly one stage, and each stage already has its
own metric. Keeping them separate is what makes the diagnosis legible.

| stage | what it does | metric | status |
|---|---|---|---|
| **retrieve** | find the viewpoints (graph, scout, Wasserstein forks) | union gain | looks better than the headline score suggests |
| **deliver** | get them into the generator (prompt injection today) | `ctx0` vs `base7b`, the α curve | **actively harmful** |
| **compress** | fit them into one answer (the merge) | alone-vs-union gap | **lossy** |

This matters for reading every null result about the graph. The merge_v2_rand
tie, the +0.0154 content effect at p=0.42, the flat routing — all are measured
*through* stages 2 and 3. Retrieval cannot be evaluated until the channel stops
destroying the content and the merge stops discarding it.

---

## 2. CAD — the delivery channel is the failure

`evaluation/overton/eval_cad.py`, Qwen2.5-7B, half A, n=28, judged with
`--dump_clusters`. See `docs/cad_experiment.md` for the design.

    logits = (1 + α)·logits(y | forks, x) − α·logits(y | x)

| α | arm | OvertonScore |
|---|---|---|
| — | `base7b` (no forks, baseline instruction) | **0.3942** |
| −0.5 | `cad-0.5` | 0.3520 |
| −0.25 | `cad-0.25` | 0.3271 |
| 0 | `ctx0` (plain injection) | 0.0992 |
| +0.25 | `cad0.25` | 0.0465 |
| +0.5 | `cad0.5` | 0.0417 |

**Monotone across all five α.** Suppressing the injected context helps;
amplifying it destroys the answer. No exceptions. This is a dose-response, not a
two-condition comparison, so it cannot be attributed to one prompt's quirks —
the strongest causal evidence in the project.

**Injection at 7B is catastrophic.** 0.3942 → 0.0992, a 75% collapse. At 72B the
same injection cost 0.497 → 0.393. `DRAFT_SPECS` had already recorded the 72B
version as a standalone-coverage ordering (baseline 0.4967 > scout 0.3927); CAD
shows these are one phenomenon with a size-dependent magnitude.

**Negative α recovers ~85% of the damage but never beats `base7b`.** CAD is a
diagnostic that worked, not a method that works.

**The content is not the problem — the delivery is.** Union with `base7b`:
`cad-0.5` **+0.1167** against `ctx0`'s +0.0179. Suppressed-context answers cover
clusters the plain answer misses. If the forks were noise, suppressing their
influence would converge on `base7b` and add nothing.

**Fluency and coverage came apart.** `<answer>` tag compliance *rose* with α
(0/0/12/19/21 across α = −0.5/−0.25/0/+0.25/+0.5), peaking at the worst-scoring
arm. Not degeneration — well-formed answers about the wrong things.

### What CAD reframes

merge_v2's draft 1 is `plain` (no forks, `BASELINE_INSTRUCTION`) and drafts 2–3
are fork-injected under `PLURALISM_INSTRUCTION` — i.e. `base7b` and `ctx0`. CAD
measured merge_v2's own components in isolation. That supports reading merge_v2
as an architecture that **quarantines a harmful channel** (one clean draft, two
exposed, a guard protecting the clean one) rather than as retrieval working.

Falsifiable prediction, untested: merge_v2 should score at or below its plain
draft **at 7B**, where drafts 2–3 collapse to 0.099.

### Caveats

- n=28, half A only, **no confidence intervals** — the paired bootstrap has not
  been run.
- 7B. `base7b`/`ctx0` are the only valid references; v10/v11/v12 are 72B-AWQ.
- **`noctx0` is missing**, so `base7b` → `ctx0` still confounds forks with
  instruction. `cad-0.5` heading toward `base7b` *implies* the no-fork pluralism
  arm scores ~0.35–0.40, but that is read off the curve, not measured.
- α = −0.5 is both the best cell and the sweep's edge.

**Both gaps close with one job.** `cad-1.0` is `noctx0` exactly — at α = −1 the
forks cancel algebraically and the logits are the no-context logits, through the
identical code path. Extend the sweep to −0.75 and −1.0.

---

## 3. Reward gate — recalibration is ruled out

`jobs/eval/job_reward_correlation.sh`, v12 responses (= `overton_responses_v11.jsonl`,
see §5), graph targets. 224 responses, 56 questions, 256 condition pairs.

Within-question pairwise concordance, chance = 0.500:

| | concordance |
|---|---|
| `reward_v1` (depth-blind) | 0.191 |
| `reward_v2` best cell (t=0.30, d=0/60) | 0.301 |
| `reward_v2` headline (t=0.50, d=60) | **0.152** |

Every cell is **below chance**. That was already known. What is new is the
decomposition, which closes the escape route:

| t, d | tie_rate | conc\|separated | reading |
|---|---|---|---|
| 0.25, 0 | 0.316 | **0.417** | few ties, below chance → orders **wrongly** |
| 0.50, 60 | 0.727 | 0.557 | mostly ties → too **sparse** |

Lower the threshold and the reward makes confident, wrong calls. Raise it and it
goes silent. The best `conc|separated` anywhere is 0.613 on 62 separated pairs.
**No (t, d) cell escapes both failure modes**, so "the threshold was
miscalibrated" is dead and the objective genuinely disagrees with the judge.

Supporting: picking the judge's best condition lands 0.15–0.29 against a 0.25
chance level (4 conditions) — i.e. chance. Pooled correlations are all ≈ 0.

Note `--exclude route` is a no-op on v12, whose conditions are baseline /
merge_v2 / merge_v2_rand / persona_merge. Both arms print identical numbers, and
the "easy case" arm already *is* the near-tie regime GRPO runs in.

### The target-mismatch hypothesis

The reward scores coverage of **graph positions** (~19.9/question). The judge
scores coverage of **participant clusters**. The cluster-target build printed:

    cluster targets: 60 questions, 465 clusters (7.75/question), dropped 0 with <2

≈2.6 injected positions per scored cluster. Different ontologies, different
granularity.

**The clusters arm crashed** (rc=1) immediately after building targets. The
granularity gap is measured; that it *explains* the below-chance concordance is
still untested. Traceback pending.

### A ceiling nobody has measured

`validate_within` (per-rating agreement with humans) is weak — the noise the
judge's own docstring says OvertonScore averages away. `validate_aggregate`
(ranking the 8 reference models) is what passes, at ρ=0.88 in the paper.

The concordance target sits *between* those levels: per-(question, condition),
averaged over ~7.75 clusters × up to 20 participants. Residual noise there is a
hard ceiling on achievable concordance, and it has never been measured — which
makes the 0.60 gate arbitrary. Fix: re-judge the same responses at a different
`--seed` and compute judge-vs-itself concordance. That is the missing
denominator for every reward number computed so far.

---

## 4. G2 on OvertonBench — pipeline validated, no result yet

`evaluation/overton/eval_g2_overton.py` (new). Moves the graph out of the
answering context entirely: the base stream `z` sees only the question, the
forks condition two guide streams, and `logits = z + α_t(z⁺ − z⁻)` with
`α_t = θ` only where `H(softmax(z)) ≥ β`.

This is the first test of stage 1 (retrieval) with stages 2 and 3 removed —
nothing to anchor on, no merge to lose content in.

Smoke run (2 questions × 3 arms × 2 samples) completed end to end. **Scores are
not reportable** at n=2. Two useful diagnostics:

- The no-priors fix works: `g2_graph #0` shows `gated_off 0.35` where `g2 #0`
  shows 0.00 — sample 0 stays contrastive when a named target carries content.
- `gated_off` 0.21–0.35 means the entropy gate fires on **65–79% of tokens**.
  That is close to intervening everywhere, the regime the paper's ablation says
  costs quality for no diversity gain. Sweep β up (0.2, 0.4) alongside θ.

Note the gate saves no compute: when it closes, both guide forwards still run to
keep the KV caches aligned.

---

## 5. Bugs found — three prior results are affected

**`merge_v2_rand` never ran merge_v2.** It was absent from the condition
dispatch and fell through to `_merge_answer` — merge **v1**: two drafts, no
guard, no concatenation fallback. Every run up to and including v12. The control
therefore differed from merge_v2 in the **merge algorithm** as well as fork
content and fork count (1.00 vs 4.85). Three confounds, not one. **The v12 tie
(0.5440 vs 0.5309) and the inverted union gain (+0.0389 vs +0.0773) cannot be
read as evidence about fork content.** Fixed; `merge_v2_rand` now dispatches to
`_merge_answer_v2`.

**53% of the position-statement artifact is stored fallback.** Its own metadata
reports `template_match_rate: 0.467` and `n_rule.fallback: 2533` of 4755.
`_is_clean` rejects any statement still interrogative, and those rows are written
anyway with the question and option concatenated. So `stmts.get(key)` *hits* and
returns a question — no warning fires. This made `g2_graph` steer toward strings
like *"Do you think abortion should be legal in all or most cases"*. Harmless for
the reward (its rewritten-minus-fallback cosine median is **−0.124**, so
statements buy it nothing and `--backend llm` will not help); fatal for a
steering prompt, which needs a viewpoint to argue rather than a string to match.
Fixed by routing `target_positions` through `pick_personas` + `persona_context`
instead. **The artifact is marked FROZEN and was not rebuilt.**

**G2 dropped the contrast on sample 0.** `g2_generate` forced the guides to base
whenever there were no priors — correct for vanilla G2, wrong for the graph
variant, where the target is meaningful alone and sample 0 seeds the priors for
the whole pool. Fixed.

**There is no `overton_responses_v12.jsonl`.** v12 was a re-judge of
`overton_responses_v11.jsonl` (responses 19:11, v12 scores 19:39 on Sep 1);
confirmed by the condition counts, 4 arms × 180 rows.

---

## 6. Standing results, unchanged

- **merge_v2 > baseline**, replicated: v10 +0.0389, v11 +0.0498 (p=0.0144), v12
  +0.0475 (p=0.0238). Genuine replication is v10 vs v11 (independent generations).
- **The model reads the injection.** `used@0.35` = 0.607 of ~19.9 positions,
  median best-cosine 0.415. Rules out "the model ignores the block" — which is
  what makes CAD's collapse interpretable. Still lacks a chance-matching floor,
  so 0.607 is an upper bound.
- **The merge is the bottleneck.** persona_merge: same score as merge_v2 (0.6423
  vs 0.6460 coverage@K), **2.4× the union gain** (+0.0917 vs +0.0389). Content
  reaches the drafts and dies in the merge. Neither variance (persona_merge has
  the *lowest* sd, 0.2369) nor correlation with baseline explains the ordering.
- **Routing is closed** — five independent nulls, oracle caps +0.0415, best gate
  +0.006.
- **21 of 180 persona_merge rows ran with zero personas** (anchor with <3 opinion
  leaves → plain-only fallback), so its numbers are measured at 12% dilution and
  understate the arm.

---

## 7. In flight

| # | job | tests |
|---|---|---|
| 5 | re-judge v11 responses with `--dump_clusters`, then `cluster_overlap.py` | are the uniquely-recovered clusters *minority* views? |
| 6 | v13 with `merge_v2_sem` | does a semantic retention guard beat the structural one? |
| 5b | v13 with fixed `merge_v2_rand` | does fork **content** matter, at matched volume and matched merge? |
| 8 | `job_anchor_coverage.sh` on ValuePrism | does the ATP graph resolve anchors outside US politics? |
| 9 | reward correlation, `--targets clusters` | is the below-chance concordance a target mismatch? |
| 4 | `job_g2_overton.sh` full half-A | does the graph help through a non-prompt channel? |

On `cluster_overlap` over the v12 dump: read the `merge_v2` and `persona_merge`
rows only. `merge_v2_rand` in that file carries both the v1-merge and 1-fork
bugs.

---

## 8. Open holes

**Quality is never measured.** Every number in this document is coverage, which
rewards listing viewpoints regardless of how well they are argued. G2's paper
reports Quality beside Distinct precisely because diversity methods trade against
it — and `route` (0.072) / `expand` (0.377) are that trade, measured here. The
pairwise quality judge is still unwritten and remains the largest hole.

**No length-matched baseline.** Injected answers run ~330 words against
baseline's ~69. The `longest` condition scoring −0.101 argues against a pure
length artifact but is not the same control: nobody has prompted the *baseline*
to 330 words. This is the cheapest existential threat to the headline result.

**No temperature sweep.** If diversity moves on temperature alone, several
conclusions shift.

**n=60.** Behind every marginal p-value. Resolving the +0.0154 content effect
needs ~365 questions — which is what makes the ValuePrism gate worth running
before anything else scales.

**GRPO is blocked** on the reward, and the reward is blocked on item 9. Note that
even a passing gate does not hand you targets: cluster targets exist only for
OvertonBench's 60 questions, while `job_grpo_align.sh` trains over all usable
graph questions.
