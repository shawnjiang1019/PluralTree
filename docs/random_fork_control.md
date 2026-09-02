# The random-fork control: does the graph supply content, or just variance?

**Status: designed, not run.** One eval arm. This is the cheapest experiment that
can falsify the project's central claim, and it has never been run.

## The question

merge_v2 beats baseline (+0.0389, bootstrap p=0.075). The graph feeds drafts 2
and 3. The natural reading is that retrieved survey positions supply content the
model would not have produced.

**Three results say that reading is not established.**

1. **Graph content hurts a single answer.** `scout` 0.3927 and `distributional`
   0.3941 against baseline 0.4967. Injecting the fork into one answer loses, and
   loses by a lot.
2. **Changing the content changes nothing.** `distributional` (full subgroup
   spectrum) vs `scout` (two poles) differ by **+0.0014** against a 0.027 noise
   floor. Two very different payloads, indistinguishable outcomes.
3. **The gain is in the combining, not the conditioning.** `bestofk` on v10:
   merging (0.5470) beats the best selector over the same pool (0.5243) and
   random@K (0.5123). What merge_v2 exploits is that its drafts *differ*.

So the live alternative hypothesis is: **the graph is a randomizer.** Drafts 2 and
3 differ from draft 1 because they were conditioned on *something*, and any
plausible-looking something would work as well. Under that hypothesis the
divergence scout, the hyperbolic embedding, and the Wasserstein fork selection
contribute nothing that a coin flip would not.

Nothing measured so far distinguishes the two hypotheses.

## Design

One arm, identical to merge_v2 except for which fork gets injected.

| arm | fork source |
|---|---|
| `merge_v2` | top-scoring fork for this question (existing) |
| `merge_v2_rand` | fork from a **random anchor elsewhere in the graph** |

Everything else held fixed: same three draft recipes, same `MERGE_INSTRUCTION_V2`,
same guard and concatenation fallback, same temperature and seed.

### Matching, which is the whole experiment

A sloppy random fork changes prompt length and position count at the same time as
relevance, and then the arm measures nothing. The random anchor must be sampled to
match the real one on:

- **number of positions** (`n_positions` under the anchor) — otherwise draft 3's
  spectrum is a different length
- **tree depth** — position counts and text length both vary with depth
- **rendered token count** to within a tolerance

Sample candidate anchors, keep those matching the real anchor's `n_positions`
exactly and depth within one level, then pick uniformly among them. Log the
achieved match so a failure to match is visible rather than silent.

### The second rung, if the first is ambiguous

`rand_fork` breaks relevance *and* attribution together. A finer control separates
them:

| arm | positions | subgroup labels |
|---|---|---|
| `merge_v2` | correct | correct |
| `merge_v2_shuf` | correct | **shuffled across subgroups** |
| `merge_v2_rand` | wrong | wrong |

`shuf` keeps the real positions and their prevalences but attributes them to the
wrong demographic groups. If `shuf` matches `merge_v2`, the *positions* are doing
the work and the demographic structure — the thing the hierarchy exists to encode
— is decoration.

Run `rand` first. Add `shuf` only if `rand` comes back clearly below `merge_v2`.

## Reading the outcome

Paired against `merge_v2` on the same 60 questions, bootstrap CI on the mean delta
(not the sign test — it discards magnitude, and that misled us at p=0.211 vs
bootstrap p=0.075 on the same data).

- **`rand` ≈ `merge_v2`** — the graph is a randomizer. The divergence scout,
  hyperbolic embedding and Wasserstein selection are not contributing content, and
  the contribution collapses to "sample diverse drafts and merge losslessly."
  That is a real finding, it is publishable, and it is not the paper you wanted.
- **`rand` clearly below `merge_v2`** — content is load-bearing. This would be the
  **first direct evidence the graph earns its place**, and it is worth more than
  the +0.0389 headline, because nothing else in the project isolates retrieval.
- **`rand` below `baseline` too** — irrelevant context actively harms, which is
  consistent with the over-anchoring diagnosis (`docs/framing_hurts.png`: on-pole
  similarity 0.334 -> 0.389, `corr(attraction, dcoverage) = -0.31`) and is
  additional evidence for the CAD experiment.

Note the asymmetry: **only one branch is good news, and the bad branch is
informative enough to be worth the risk.** Running this is how the retrieval claim
stops being an assumption.

## Why this beats attribution analysis

The obvious alternative is to check which injected positions appear in the final
answer — match answer units to injected positions by cosine, using the machinery
`coverage_reward` already has for `unit_best`. That is cheap and it is descriptive.

It also cannot answer the question. An injected position can appear verbatim and
contribute nothing to coverage, because the reward diagnostic indicates graph
positions and judge clusters are **different target sets** — that is the leading
explanation for within-question concordance sitting below chance at every
threshold (best 0.375 against 0.500). "60% of injected positions were used" would
not tell you whether they mattered.

The random-fork arm measures the outcome directly and needs no assumption about
target alignment.

## Cost

One eval run: 60 questions x 4 calls (3 drafts + merge) on the served
Qwen2.5-72B-AWQ, then the judge. Same shape as any `job_overton_eval.sh` arm, so
roughly 6 hours wall clock including judging.

**No new model, no local weights, no logit access** — unlike CAD and G2, this runs
on the existing endpoint and is directly comparable to v9/v10. That comparability
is the reason to run it before the 7B experiments.

## Relation to the other open experiments

- **`persona_merge`** asks whether *better* graph conditioning raises the union.
  This asks whether graph conditioning matters at all. If `rand` ties `merge_v2`,
  `persona_merge` is unlikely to help and should be deprioritised.
- **CAD** tests whether injection fails because of *how* context is used. That
  question is only interesting if the context carries signal, which is what this
  arm establishes.

Run this first. It is one job, it needs nothing new, and both of the other
experiments are conditioned on its answer.
