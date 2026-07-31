# Single-answer coverage policy (GRPO) — design and go/no-go assessment

Companion to [`grpo_alignment.txt`](./grpo_alignment.txt) (implementation status),
[`related_work_rag_diversity.md`](./related_work_rag_diversity.md) (literature,
E1–E7 experiment queue) and [`adaptive_injection.md`](./adaptive_injection.md)
(why graph-signal and self-report routing both failed). Scope: the policy in
`alignment/` — one LoRA adapter, one answer per prompt, reward =
`coverage_reward` (`alignment/reward.py`). Written July 2026.

## 0. Where this sits in the evidence

Five single-answer prompting strategies have now been measured against the
no-retrieval baseline (0.507) on OvertonBench:

| condition | OvertonScore | vs baseline |
|---|---|---|
| baseline (no retrieval) | 0.507 | — |
| merge (2 drafts, extractive union) | 0.503 | tie |
| div_only (pure divergence, no gate) | 0.468 | loss |
| scout (2 max-divergence poles) | 0.444 | loss |
| expand (enumerate beyond the pair) | 0.377 | loss |
| route (self-route contested/consensus) | 0.072 | catastrophic loss |
| noise floor (run-to-run) | 0.027 | — |

Meanwhile the **union** of covered clusters across 2–3 *separately generated*
answers scores 0.657–0.687, and the **oracle** (perfect per-question routing
among those same 2–3 answers) scores 0.622. Union > oracle > every single
answer, including the best one picked in hindsight per question. This
ordering — not just the failures individually — is the central fact the rest
of this document has to explain and design around.

Two more facts bear directly on whether RL is mechanically viable here:
Vendi diversity over 8 same-question rollouts collapses to ~1.4 effective
modes (samples are mostly restatements of each other), and the judge that
would score any of this in eval has unresolved validity (aggregate ρ = 0.167
vs the paper's 0.88, within-participant discrimination ρ ≈ 0.06, and a
generosity gap P(pred≥4)=0.747 vs P(human≥4)=0.650).

---

## 1. Formal specification

**State.** A scout-injected chat prompt, baked at dataset-build time
(`alignment/rollout_dataset.py:build_prompts`). The graph encoder and scout
retrieval are frozen; the policy sees the same fixed context a prompting-only
run would see. `positions_from_subtree(graph, anchor)` (the aggregated
population distribution over the anchor's opinion-leaf subtree) rides along
per example as the reward's ground truth and is **not** shown to the policy.

**Action.** One sampled completion, `<think>…</think><answer>…</answer>`;
only the `<answer>` span is scored (`_completion_text` strips `<think>` via
`retrieval.answer.extract_answer`).

**Reward.** `coverage_reward(response, positions, embed_fn, cfg)`
(`alignment/reward.py:149`):

```
reward = recall_w
        - l_precision * (1 - precision)
        - l_verbose   * max(0, n_expr - target) / target
```
clamped to [0, 1], where `recall_w` is prevalence-weighted recall of graph
positions matched at cosine ≥ 0.50 against a held-out mpnet embedder,
`precision` is the fraction of response units (bullets/sentences, ≥8 words)
that match *some* position, and `target = exp(H(prevalence))` is the
graph-implied effective number of positions.

### 1a. Is prevalence-weighted recall the right target for a per-cluster ≥4 threshold?

No, not cleanly, for two independent reasons — a **level mismatch** and a
**population mismatch**.

*Level mismatch.* OvertonScore is a hard per-cluster gate: a cluster counts
only if its members' *mean* rating clears 4/5. `coverage_reward`'s "expressed"
test is cosine ≥ 0.50 between a response unit and a position's `embed_text`
— a topic/paraphrase-match, not a satisfaction judgment. This is exactly the
gap the repo's own `merge` design comment already names: *"naming a position
is not covering it — the ≥4 bar rewards ARTICULATION DEPTH."* A response that
briefly namedrops six positions can max out `recall_w` (each position's best
unit clears 0.50) while giving every one of them too little depth for any
real participant to rate it ≥4. `coverage_reward` has **no depth term** — it
is indifferent between a one-clause mention and a 150-word treatment of the
same position, as long as both clear the match threshold. This is the reward
being *easier to satisfy* than the eval, which is the failure mode that
matters most: training will find the cheap way to raise `recall_w`.

*Population mismatch.* `positions_from_subtree` aggregates **demographic
subgroup** answer distributions from the graph. OvertonBench clusters are
**k-means clusters over free-response text** — independently derived,
opinion-similarity groupings of actual humans. These are different
partitions of the same population, and the repo has already measured that a
structurally identical pairing (graph divergence vs. human contestedness)
correlates only **+0.20** (`retrieval/contestedness.py:7`,
`docs/adaptive_injection.md`). There is no existing measurement of
`coverage_reward` vs. `OvertonScore` directly — Section 3 specifies one using
files already on disk. Until that number exists, "prevalence-weighted recall
of graph positions" is an *assumed* proxy for "mean-≥4 across human viewpoint
clusters," not a validated one.

### 1b. Does the verbosity penalty help or hurt, given route's 0.072?

These are not the same failure mode, and conflating them would be a mistake.
`route` (0.072) is a **prompt** that asked the model to self-classify
contested-vs-consensus and behave accordingly; the documented failure is
*miscategorization plus poor commit-execution* (`PLURALISM_ROUTE`'s comment:
turning already-good baseline answers into hedged lists). The reward's
verbosity term is structurally different: `target = exp(H(prevalence))` is
computed from **ground truth**, not from the model's self-report — it
sidesteps exactly the two signals already shown not to work for this decision
(graph divergence, +0.20 corr with human contestedness;
self-report, `route` 0.072). As a *smooth gradient replacing a discrete
self-classification*, this is a genuine design improvement over `route`.

But two things are still live risks, both traceable to the *depth* gap in
1a. First, the calibration target is the *graph's* effective-position count,
not the *eval's* effective-cluster count — under the same +0.20 mismatch,
"correct" verbosity by the graph's lights may be wrong by the eval's. Second,
and more concretely: because `recall_w` rewards breadth at zero depth cost,
and `l_verbose` explicitly *penalizes* expressing more positions than
`target`, the joint pressure of these two terms pushes the policy toward
**exactly `target` positions, each stated minimally** — i.e., toward the
terse, label-dense enumeration style that is the closest analog to what made
`route` fail (short, hedge-flavored, low-depth answers). The reward doesn't
know "terse" is bad; nothing in it rewards elaboration. This should be
treated as a concrete, plausible reward-hacking path (enumerated formally in
§6), not just a hypothetical.

### 1c. Is the precision penalty well-motivated?

The intent is sound — without it, `recall_w` is maximized by writing every
conceivable position regardless of truth, so a padding/hallucination guard is
necessary. But the implementation (`split_units` → cosine ≥ 0.50 to *some*
position) scores any unit that doesn't paraphrase a specific option as
"padding," including connective tissue, caveats, and acknowledgment language
("reasonable people weigh this differently…") that plausibly *raises* how
represented a human reader feels without paraphrasing any option. `l_precision
= 0.20` is a moderate weight, but it points in the same direction as 1b: away
from exposition, toward compressed position-restatement. Net assessment:
precision-as-anti-hallucination is well-motivated; precision-as-implemented
likely co-penalizes the depth/empathy content that the eval threshold
plausibly rewards, reinforcing rather than offsetting the verbosity term's
bias.

**Summary of 1a–1c.** The reward is a coherent anti-hacking design *relative
to itself* (the self-test in `reward.py` correctly orders broad-vs-narrow by
contestedness) — but all three terms push toward the same failure surface:
broad, shallow, compressed enumeration. That surface is close to what `route`
already showed scores catastrophically under the real judge. The reward was
built to fix `route`'s decision problem (when to pluralize); it may not have
fixed `route`'s execution problem (pluralizing well).

---

## 2. THE CENTRAL QUESTION — can RL break a ceiling prompting could not?

### The two hypotheses

**H-behavioral.** The good policy — commit tightly on consensus, cover real
disagreement with real depth on contested questions — is *expressible* in
one answer and even has non-trivial probability under the base model's
sampling distribution, but prompting can't reliably steer the model into it:
instructions are followed inconsistently, self-classification is unreliable
(route, 0.072), and the model defaults to whichever mode is highest-density
under pretraining + instruction-tuning. RL reweights the sampling
distribution toward reward; if the good mode exists anywhere in it, GRPO can
raise its probability. Under H-behavioral, this policy is worth building.

**H-structural.** The eval's requirement — every genuinely divergent cluster's
members individually rate representation ≥4 — is not satisfiable by *any*
single bounded-length, single-voice response once the number of genuinely
distinct clusters exceeds ~1–2, because depth is a scarce resource that must
be divided among positions (§1a), and diluted depth plus single-voice framing
structurally caps how "represented" any one cluster can feel. Under
H-structural, no amount of RL on this reward — or any reward shaped like
it — raises OvertonScore beyond a hard ceiling, because the ceiling is a
property of the *action space* (one answer), not of the *policy* generating
actions within it. RL would optimize `coverage_reward` (which is gameable
exactly along the same axis, §1a/§1b) while OvertonScore stays flat or falls.

### The evidence already leans structural, but is not conclusive

Three points from the existing measurements argue for H-structural:

1. **Oracle < union, even picking the best of only 3 fixed strategies per
   question.** `oracle` (0.622) already gets to cherry-pick, per question,
   whichever of baseline/scout/div_only did best — and still falls short of
   `union` (0.657–0.687) by 0.03–0.06. If a single answer *could* reach the
   union ceiling, the best of three quite different single-answer strategies,
   chosen with perfect hindsight, should get close. It doesn't.
2. **`route` was told the exact strategy RL would reward, and it collapsed.**
   `PLURALISM_ROUTE` *is*, in prose, "commit on consensus, cover real
   disagreement with attribution otherwise" — the behavior `coverage_reward`
   is built to elicit. Telling the model this explicitly produced 0.072, not
   an improvement. That is weak evidence against "the model just needs
   correct instructions to already know how" and mild evidence for "executing
   the strategy inside one answer is intrinsically hard," though it's
   confounded by route's specific self-classification step, which RL's
   ground-truth-derived verbosity target does remove (§1b).
3. **Mode collapse (Vendi 1.4 / 8) suggests little exploitable variance sits
   near the current policy.** GRPO's only learning signal is *within-group*
   reward variance (`group_relative_advantage`). If 8 temperature-1.0 samples
   of the same prompt collapse to ~1.4 effective content-modes, most rollouts
   in a training group will be near-paraphrases of each other, `coverage_reward`
   will be near-constant across the group, and advantages will be near-zero
   for most groups — independent of whether the reward is well-aligned. This
   is a *mechanical* obstacle to H-behavioral even if H-behavioral is true:
   if the good mode exists but has near-zero sampling probability and low
   local density, GRPO's local reweighting may never encounter enough
   reward-variance to find it.

Countering H-structural: none of the five failed variants is a *trained*
policy — all are zero-shot prompts on a frozen model, and RL's classical
advantage over prompting is many gradient steps of exploration against a
reward, which can reach distributions temperature sampling and one-shot
instructions cannot. It's also true that two cheaper, more targeted fixes for
the two *specific* diagnosed root causes have not been tried at all:
submodular/coverage-based fork selection (E1, attacking the "two poles
crowd out the middle" mechanism in `docs/framing_hurts.png`) and verbalized
sampling (E3, [2510.01171] — attacking mode collapse directly, training-free,
reported 1.6–2.1× diversity gains). If the ceiling looks structural mainly
because retrieval still anchors on two extremes and sampling still collapses,
that is not evidence against RL in general — it is evidence that the current
five variants are all downstream of the same two unfixed upstream problems.

**This is not yet resolved either way from existing evidence.** The next
subsection is a diagnostic designed to resolve it cheaply.

### Diagnostic: does the good behavior already exist in the base model's output distribution?

RL (KL-regularized GRPO in particular) is a *reweighting* operator on the
reference policy's distribution — it raises the probability of
already-reachable good completions, it does not synthesize completions with
zero density under the reference model. So the diagnostic that separates
H-behavioral from H-structural without spending any training compute is:
**does best-of-K sampling from the frozen base model, scored by the real
OvertonBench judge, approach the union ceiling as K grows, or does it plateau
well below it?**

Concretely, on a modest slice (~15–20 questions, enough for the trend, cheap
enough to not need the diagnostic itself to justify a training run):

1. Generate K ∈ {1, 4, 8, 16, 32} samples per question at temperature 1.0
   from the **untrained** policy base (Qwen2.5-7B-Instruct, or reuse the
   72B eval model if budget allows — the base-model-weakness confound in §6
   argues for running this on 7B specifically, since that's the model that
   will actually be trained) under the `scout` condition prompt (same prompt
   RL would train on).
2. Score every sample with the real judge (`judge_overtonbench.py --score`,
   reusing `overton_responses_v*` machinery with `--n_rollouts K`).
3. Plot `coverage_max(K)` (the best single sample's coverage, per question,
   averaged over questions) against K, alongside `union_coverage@K` (already
   computed by `judge_overtonbench.score`'s `union_coverage` column) and the
   known baseline/union numbers as reference lines.
4. **Read:**
   - If `coverage_max(K)` climbs toward `union_coverage@K` (0.657–0.687) by
     K=32 without clearly plateauing — the good single-answer behavior exists
     in the sampling distribution at low but non-trivial density.
     **H-behavioral supported; build the policy** (RL is the efficient way to
     raise the probability of a rare-but-present mode; best-of-32 at
     inference is not — 32× the cost per query, forever).
   - If `coverage_max(K)` plateaus by K≈8–16, clearly below `union`, and
     close to what `oracle` (0.622) already achieves with only 3 samples of
     very different strategies — the good behavior is not reachable by
     resampling this policy at any practical K. **H-structural supported;
     do not build this policy** — no realistic amount of GRPO reweighting
     recovers what plain sampling can't approach, because GRPO's exploration
     is bounded by the same reference-policy support that best-of-K samples.
   - The mode-collapse fact already measured (Vendi 1.4/8) predicts the
     plateau branch — this diagnostic either confirms that prediction on the
     metric that actually matters (judge-scored coverage, not embedding
     Vendi) or falsifies it.

Cost: ~15–20 questions × 32 samples × judge calls per sample (each judge call
is ~1 short generation) — an inference-only job, no training, on the order of
a few GPU-hours on the 7B generator plus judge-serving cost already paid for
in every prior eval run. This should run **before** any GRPO job, and its
result should gate whether §5's implementation plan is executed at all.

---

## 3. Reward–eval alignment: a concrete, cheap experiment on existing files

The training reward is grounded in the graph; the eval is grounded in human
clusters; the two are known to correlate only +0.20 on a structurally
adjacent quantity (divergence vs. contestedness). Nobody has measured whether
`coverage_reward` itself correlates with `OvertonScore`. This is answerable
today, offline, CPU-only, from files already on disk:
`overton_responses_v5.jsonl` (response text per question/condition) and
`overton_scores_v5.csv` (real judge-scored `coverage` per question/condition).

**What to compute:**

1. For each of the 60 eval questions, resolve the graph anchor the same way
   training does: run `retrieval.scout.scout(question, graph, h_all,
   text_feat, manifold, cfg=ScoutConfig())` and take `forks[0].anchor` (fall
   back to the top `lexical_anchors` node if no fork clears the gate — mirror
   `rollout_dataset.build_prompts`'s `min_positions` skip if it doesn't).
   Compute `positions = positions_from_subtree(graph, anchor)`. This is
   read-only graph traversal — no leakage risk, since nothing is trained
   here, only measured.
2. For every row in `overton_responses_v5.jsonl` (baseline/scout/div_only ×
   60 questions), compute `r, breakdown = coverage_reward(response, positions,
   default_embed_fn(), RewardConfig())` — the exact function and config the
   trainer will use.
3. Join on `(question_id, condition)` with `overton_scores_v5.csv`'s
   `coverage` column (the real, judge-scored OvertonScore per response).
4. Report, at three levels of granularity (weakest signal first, strongest
   last — mirroring `judge_overtonbench.py`'s validate → validate_within →
   validate_aggregate escalation, because pooled correlation is the least
   informative one here too):
   - **Pooled Spearman** of `coverage_reward` vs. `OvertonScore.coverage`
     over all 180 (response, question) pairs. This is the null-hypothesis
     check: is there *any* relationship.
   - **Per-condition Spearman** (does the reward's ranking of *questions*
     agree with the judge's, separately within baseline/scout/div_only) —
     controls for condition-level confounds (e.g. response length differs
     ~5× between baseline and scout).
   - **Within-question rank agreement**: for each question, does
     `coverage_reward` rank the three conditions' responses the same way
     `OvertonScore` does (a 3-item Kendall/sign agreement, analogous to
     `analyze_overtonbench.py`'s paired sign test)? **This is the number that
     actually matters for GRPO**: within a training group, the reward only
     needs to rank same-question rollouts correctly for `group_relative_advantage`
     to point the gradient the right way — it does not need to match
     `OvertonScore`'s absolute scale. A high within-question rank agreement
     with a mediocre pooled correlation would still make this a usable
     training signal; a low within-question agreement would not, regardless
     of the pooled number.
   - **Subcomponent breakdown**: correlate `recall_w`, `precision`, and
     `verbosity_penalty` individually against `OvertonScore.coverage`, and
     against response word count (mirroring `judge_overtonbench.py`'s
     length-bias check) — this directly tests the §1a/§1b concern that
     `recall_w` rewards breadth the judge doesn't reward equivalently, and
     that the verbosity/precision terms may be anti-correlated with
     `OvertonScore` if the judge in fact rewards depth.

**Decision rule:** if within-question rank agreement is materially better
than chance (and better than the pooled number, which will be depressed by
condition-level confounds) the reward is at minimum directionally usable for
GRPO's actual mechanism, independent of whether it matches OvertonScore in
absolute terms. If it is at or near chance, GRPO will optimize a signal
uncorrelated with the eval, and no amount of training compute fixes that —
this is a reward-redesign problem, not a hyperparameter problem, and should
block §5 exactly as hard as an unfavorable §2 diagnostic result.

---

## 4. Literature

**GRPO / group-relative RL mechanics.**
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
  Open Language Models*, [2402.03300](https://arxiv.org/abs/2402.03300) —
  origin of GRPO; KL coefficient β=0.04 (matches `GRPOAlignConfig.kl_coef`
  default exactly), group size 64 for math (much larger than the 8 used
  here — see §5 on why 8 is a mode-collapse risk, not just a compute choice).
- Demystifying GRPO: *Its Policy Gradient is a U-Statistic*,
  [2603.01162](https://arxiv.org/abs/2603.01162) — formal analysis of what
  the GRPO gradient estimator actually computes; relevant to understanding
  why near-constant within-group reward (the mode-collapse risk in §2/§6)
  produces a near-zero, not just noisy, gradient.

**Reward hacking, general.**
- Skalse, Farrugia-Roberts, Russell, Abate, Gleave, *Defining and
  Characterizing Reward Hacking*,
  [2209.13085](https://arxiv.org/abs/2209.13085) — formal result that no
  non-trivial proxy reward is guaranteed unhackable; frames §6's hack
  enumeration as expected rather than a sign of unusually bad reward design.
- *Tackling Length Inflation Without Trade-offs: Group Relative Reward
  Rescaling for RL*, [2603.10535](https://arxiv.org/abs/2603.10535) — GRPO
  specifically develops length inflation under additive length/verbosity
  penalties (the exact penalty *shape* used in `coverage_reward`'s
  `l_verbose` term) because additive penalties create "compensatory
  optimization shortcuts"; directly relevant to whether `l_verbose` as
  implemented will actually hold length down under optimization pressure
  rather than being routed around.
- *Bias Fitting to Mitigate Length Bias of Reward Model in RLHF*,
  [2505.12843](https://arxiv.org/abs/2505.12843) — length-reward
  correlation as a generic RLHF failure mode, relevant background for
  monitoring response length during training (§5 logging).

**Long-form generation with coverage/recall-style rewards.**
- Shi, Kang, Zhou, Weng, Wu, *SPADER: Step-wise Peer Advantage with
  Diversity-Aware Exploration Rewards for Multi-Answer Question Answering*,
  [2606.00593](https://arxiv.org/abs/2606.00593) — RL (critic-free,
  peer-advantage, GRPO-adjacent) for multi-answer QA (QAMPARI/Mintaka/
  WebQSP/QUEST) with a reward that explicitly upweights rare/long-tail
  answers and downweights redundant ones; the closest published analogue to
  "RL reward for covering a set of distinct correct answers" and a candidate
  template for fixing `coverage_reward`'s missing depth/redundancy handling.
- *Reinforced Informativeness Optimization for Long-Form Retrieval-Augmented
  Generation* (RioRAG), [2505.20825](https://arxiv.org/abs/2505.20825) —
  RL with a verifiable "informativeness" (nugget-coverage) reward for
  long-form RAG; same coverage-not-exact-match idea as `coverage_reward`,
  applied to factual nuggets rather than opinion positions.

**Diversity, pluralism, and RL's known effect on diversity (already surveyed
in `related_work_rag_diversity.md`; repeated here because directly load-bearing
for §2/§6).**
- Kirk et al., *Understanding the Effects of RLHF on LLM Generalisation and
  Diversity*, ICLR 2024, [2310.06452](https://arxiv.org/abs/2310.06452) — RLHF
  improves OOD generalization but *reduces* output diversity relative to SFT.
  Directly cautions against expecting GRPO to increase within-answer
  breadth without an explicit diversity term; it tends to sharpen the policy
  toward whatever the reward's mode is, which — per §1a/§1b — may be the
  narrow enumeration mode, not the broad-with-depth one.
- Zhou et al. (Verbalized Sampling), [2510.01171](https://arxiv.org/abs/2510.01171)
  — mode collapse traced to typicality bias in preference data, fixed
  training-free via explicit distributional elicitation (1.6–2.1× diversity).
  Cheaper, untried alternative that attacks the same mode-collapse fact
  (Vendi 1.4/8) driving the §2 mechanical concern about GRPO's gradient
  signal — should be tried before or alongside RL, not instead of measuring
  whether it's needed.
- Poole-Dayan et al., *Benchmarking Overton Pluralism in LLMs*, ICLR 2026,
  [2512.01351](https://arxiv.org/abs/2512.01351) — the eval itself.
- Sorensen et al., *A Roadmap to Pluralistic Alignment*,
  [2402.05070](https://arxiv.org/abs/2402.05070) — the Overton/steerable/
  distributional pluralism framing this whole project (and the
  single-answer-vs-union tension) sits inside.

No published work was found that runs GRPO (or PPO) with a reward
specifically targeting **Overton-style multi-cluster coverage of a single
answer** to human-rated satisfaction thresholds — SPADER and RioRAG are the
closest (RL + coverage/diversity reward for long-form generation), but both
target factual-answer-set recall (QAMPARI-style multi-answer QA, RAG nugget
coverage), not opinion-cluster satisfaction with a human "do I feel
represented" judge. That gap is either a genuine opportunity or, given
§0–§2's evidence, a sign that no one has found this to work yet — this
document cannot distinguish those from literature alone, which is exactly
why §2's diagnostic is proposed as a pre-registered gate rather than an
argument from analogy.

---

## 5. Implementation plan and hyperparameters

**Gate.** Do not run this until §2's diagnostic and §3's correlation check
both return favorably (§2: best-of-K trends toward union without an early
plateau; §3: within-question rank agreement clearly above chance). This
section specifies the run *conditional on* that gate passing.

| knob | value | justification |
|---|---|---|
| base model | Qwen2.5-7B-Instruct (bf16, trainable) | matches `GRPOAlignConfig` default; kept distinct from the 72B AWQ eval model (frozen, quantized, not trainable) — see §6 for the capability-gap risk this creates |
| LoRA rank | 16, α=32, dropout 0.05, `target_modules="all-linear"` | current default; reasonable starting point for a 7B model and a reward with a narrow, structured objective (position coverage) rather than broad capability change — low rank should suffice and limits catastrophic drift risk. Revisit upward (r=32–64) only if the reward plateaus while KL budget remains, not as a first move |
| group size (G) | 8 → **recommend raising to 16 before the first real run** | current default (8) is exactly the group size at which Vendi collapse (1.4/8) was measured — i.e., the group size in production is the one already shown to have thin content diversity. Raising G increases the chance any one group contains a genuinely distinct high-reward sample to anchor the advantage on, at the cost of 2× rollout compute per step. This is the single highest-leverage mitigation for the mechanical risk in §2/§6 short of fixing sampling diversity upstream (verbalized-sampling-style prompting, or higher temperature) |
| temperature | 1.0 (rollouts) | needs to stay ≥1.0 for group spread; do **not** lower this to "clean up" outputs — that directly worsens the Vendi-collapse problem. If reward variance within groups is still too low at G=16, raise temperature before raising G further |
| learning rate | 1e-6 | current default; conservative, appropriate starting point for LoRA GRPO on a reward this noisy (embedding-cosine-thresholded, semi-continuous) — a higher LR on a noisy, low-within-group-variance reward risks amplifying whatever spurious signal survives the mode collapse |
| KL coefficient (β) | 0.04 | matches DeepSeekMath's value exactly (2402.03300) and the current default; given §1's identified hacking surface (shallow breadth), this is a case where KL-to-reference is doing real anti-hacking work, not just anti-drift — do not lower it as a first response to slow reward growth |
| max_new_tokens / max_prompt_tokens | 1024 / 3072 | current defaults; note the reward has no reason to want long completions beyond `target` positions × some depth, so 1024 should not bind unless verbosity-hacking (§6) is occurring, in which case completion-length distribution is itself a hacking signal to log |
| reward config (`match_thr`, `l_precision`, `l_verbose`) | 0.50 / 0.20 / 0.30 | current defaults; **do not treat these as fixed** — §1 identifies `l_verbose`/`l_precision` as jointly biased toward shallow enumeration, so a held-in slice sweep (raise `l_verbose` toward 0, or add a depth term) before the real run is cheap relative to a full GRPO job and should be run alongside §3 |
| training data | all usable `q:` nodes minus the 60-question leakage holdout (`rollout_dataset.build_prompts`) | already correctly implemented with the leakage guard |
| infra | 2×GPU, 8h SLURM job (`jobs/train/job_grpo_align.sh`), LoRA + TRL `GRPOTrainer` | existing scaffold; `use_vllm` colocated rollouts flagged TODO — worth enabling before a full run given G will likely double |

**What to log to catch reward hacking early** (before spending the full 8h
budget):
- **Within-group reward variance over training**, per step — if it trends
  toward 0 (mode collapse *induced by training*, on top of the base-model
  collapse already measured), the gradient signal is dying and further steps
  are wasted compute, not just ineffective.
- **`recall_w`, `precision`, `verbosity_penalty` separately**, not just the
  scalar reward — if `recall_w` climbs while `precision` falls or holds flat
  and mean response length drops, that is the shallow-enumeration hack from
  §1b/§1c actively happening.
- **Response length and unit count (`n_units`, `n_expressed`) over training**
  — a policy converging on `n_expressed ≈ target` with falling per-unit word
  count is the predicted failure mode; catch it before the OvertonBench
  re-eval, which is expensive and slow to iterate on.
- **KL-to-reference**, standard GRPO diagnostic — a sudden drop in fluency
  metrics or qualitative garbling alongside rising reward is the classic
  KL-collapse signature.
- **Periodic OvertonBench spot-checks** (small `--max_questions`, a handful
  of held-in-style questions or a subset of the 60 if the run is far enough
  along that a light peek is worth the eval-set exposure risk — otherwise a
  disjoint small manually-scored sample) to catch reward↔eval divergence
  (§3) manifesting live, not just at the final checkpoint.

---

## 6. Risks

**Reward hacking — concrete, reward-specific hacks available under
`coverage_reward` as currently specified** (not generic RLHF hacking; these
follow directly from §1):
1. **Shallow enumeration.** State each of `target` positions in one clause,
   using vocabulary close to `embed_text` (option label text) to guarantee
   the 0.50 cosine match, then stop. Maximizes `recall_w` at minimal
   `precision` cost and zero `verbosity_penalty`. This is the single most
   likely failure mode given §1's analysis, and closely resembles `route`'s
   hedged-list style that scored 0.072.
2. **Position-text mimicry over genuine articulation.** Because match
   quality is cosine similarity to `embed_text` (`"<question> <option>"`),
   near-quoting the option label scores as well as, or better than, an
   attributed, nuanced explanation of why a group holds that view — the
   reward cannot tell "restated the label" from "explained the position."
3. **Padding just below the precision penalty's bite.** `l_precision = 0.20`
   is a soft weight, not a hard constraint; a policy can trade some precision
   loss for hedging/connective language if that language happens to help
   with the human-facing judge in ways the graph reward doesn't credit —
   or, in the hacking direction, the reverse: strip all connective tissue
   because it's reward-negative under `coverage_reward`, even though it may
   be reward-positive under the real judge (§3 will tell us which).
4. **Gaming the `min_prevalence` floor.** Positions below 0.05 mean
   probability are dropped from `positions_from_subtree` entirely — a
   response that ignores small-prevalence-but-real minority positions pays
   no penalty for it under this reward, even if those clusters exist and are
   ratable in the real eval. This is a scope-type misspecification (Pan et
   al.'s taxonomy, cited via Skalse [2209.13085]): the proxy is correct on
   its restricted domain (prevalence ≥ 0.05) and silent outside it.
5. **Exploiting the embedder gap.** The reward's embedder (mpnet) is
   deliberately different from the scout's (MiniLM) to avoid crediting
   retrieval-shaped text — but nothing prevents the trained policy from
   drifting toward whatever surface form maximizes mpnet-cosine-to-`embed_text`
   specifically, which is a narrower, more exploitable target than "sounds
   like a good pluralistic answer" once thousands of gradient steps optimize
   directly against it (this is different from, and in addition to, hacking
   the scout's retrieval format).

**KL collapse.** Standard GRPO risk, but here compounded by mode collapse:
if within-group reward variance is already thin (Vendi 1.4/8), a few early
steps that happen to raise variance by pushing the policy toward the
shallow-enumeration hack (#1 above) could dominate the gradient
disproportionately, accelerating drift toward that mode specifically rather
than general fluency collapse. Monitor per §5; do not treat KL-to-reference
alone as sufficient — a policy can stay KL-close to the reference while still
drifting hard toward the one reward-hackable mode that reference happens to
support at non-trivial density.

**7B policy vs. 72B AWQ eval model capability gap.** The reward is trained
and optimized on a 7B bf16 base; OvertonBench is scored on responses that, in
prior runs, were generated by the 72B AWQ model. If the trained LoRA adapter
is deployed on the 7B base for eval (as the scaffold implies — there is no
mechanism to transfer a LoRA trained on 7B onto the 72B AWQ model), the
comparison to the 0.507/0.444/... numbers above is confounded by base-model
capability, not just by RL vs. prompting. A fair comparison needs either (a)
a 72B-base RL run (expensive, and AWQ quantization complicates LoRA training
directly on that checkpoint), or (b) a 7B-base *prompting* baseline
(re-measure baseline/scout/route on 7B) so the RL-vs-prompting delta is
measured on a matched base model. Without (b), any post-RL score change is
not attributable to RL specifically.

**Training/eval distribution mismatch.** Training prompts are graph `q:`
nodes (OpinionQA ATP survey items, scout-injected, minus the 60-question
holdout); eval prompts are the 60 OvertonBench questions, which are
"politically salient" by curation and were selected/rated by a different
process (1208 human raters, k-means clustering on free response) than the
graph's demographic-subgroup structure. This is the same population/level
mismatch as §1a/§3, now framed as a train/test gap: even a policy that
perfectly learns to satisfy `coverage_reward` on training questions is
learning to satisfy graph-position coverage on ATP survey items in general,
not Overton-cluster coverage on the specific curated, contested-by-design
question set it will be evaluated on. §3's correlation experiment is the
direct test of whether this gap is small enough to ignore.

---

## 7. Verdict

**Conditional no — do not start the GRPO run yet.** Not because the design
is sloppy (the reward is a genuinely more principled attempt than `route` at
externalizing the pluralize-or-commit decision to ground truth, and the
scaffold is clean, tested, and leakage-safe), but because two cheap,
non-GPU-training experiments that would each independently justify killing
the project *before* an 8-GPU-hour job are still unrun:

1. **§2's best-of-K diagnostic** (does the good single-answer behavior exist
   in the base model's sampling distribution at all, even at low density?).
   The strongest piece of existing evidence — oracle (0.622) still trailing
   union (0.657–0.687) even with hindsight selection among only three
   strategies — leans toward H-structural, i.e., toward "no," but this has
   not been tested directly and cheaply, and it should be, because if it
   comes back "no" it makes the rest of this document moot regardless of how
   well the reward is tuned.
2. **§3's reward↔OvertonScore correlation check** on data already on disk.
   If `coverage_reward` doesn't rank same-question rollouts the way the real
   judge would, GRPO will confidently optimize a number that moves
   OvertonScore by an unknown, possibly negative, amount — this is
   answerable this week with zero new generation, from
   `overton_responses_v5.jsonl` + `overton_scores_v5.csv` + `positions_from_subtree`.

**What would flip this to a yes:** §2 showing `coverage_max(K)` climbing
toward the union ceiling without an early plateau, *and* §3 showing
within-question rank agreement clearly above chance. If both hold, the
reward-shape critiques in §1 (add a depth term; soften or reshape
`l_precision`/`l_verbose` along the lines GR³ [2603.10535] uses for length
control; consider recalibrating `target` against human-cluster counts rather
than graph-option counts) are fixable engineering, not a fundamental
redesign, and the run in §5 (gated, with the logging in place to catch §6's
hacks early) is worth the 8 GPU-hours.

**If §2 comes back structural** (plateau well below union): do not iterate
on the reward. Spend the freed effort on the union-side work the project's
own evidence already supports more strongly — `related_work_rag_diversity.md`'s
E1 (submodular fork selection, attacking the two-poles-crowd-out-the-middle
mechanism directly) and E6 (coverage@K as the primary claim, reframing the
contribution from "one answer improves" — which is false, five ways now — to
"graph-guided diversity improves *achievable* coverage across samples," which
is the one result that's actually 15–18 points above the noise floor).
**A single-answer coverage policy is not obviously the right unit of
optimization for an eval that is itself most cleanly beaten by not answering
with a single answer.** That tension should be resolved by the diagnostic,
not assumed away by building the policy first.
