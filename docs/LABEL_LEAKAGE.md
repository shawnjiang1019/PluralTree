# Label Leakage in Question Text — Diagnosis, Fix, and the Frozen-NN Baseline

This document records a specific experiment: discovering and removing a **label
leak through the input text** in the CulturalBench practice→country link-prediction
task, and the baseline used to prove it.

It complements `EXPERIMENTS.md` (roadmap) — this is the post-mortem of why the
early A1 numbers were misleadingly high, and what we changed.

---

## 1. The symptom

After fixing the earlier *graph* leakage (structural triples in every split;
held-out practices aggregated into their country's embedding — see
`data/culturalbench.py`, `leakage_safe`), the leakage-safe A1 ablation still
produced suspiciously high test MRR for the two best variants:

| variant | test MRR |
|---------|----------|
| pre_agg | 0.954 |
| no_gki  | 0.937 |
| post_gru | 0.441 |
| dual | 0.378 |
| plain_gate | 0.241 |
| baseline (post_agg, depth_aware) | 0.207 |

`no_gki` ≈ `pre_agg` ≈ 0.94 was the red flag: a model with **knowledge injection
disabled** scored as well as the best injecting model. That means the graph /
GKI / hierarchy were contributing almost nothing — something else was carrying
the score.

## 2. The diagnosis

Two independent code audits (one on the eval/ranking path, one on the data
construction path) reached the same verdict:

- **No classic leakage.** The `leakage_safe` split is sound: val/test practices
  are disjoint and never added to `children_indices`, so held-out leaves stay
  isolated and never see their country. Filtered ranking is the textbook setting
  and is a no-op here (each practice has exactly one country, so nothing
  competing is filtered out). `all_triples` is used only as the filter set, never
  fed to the encoder.

- **The label is in the input text.** The candidate set for `practiced_in` is only
  the ~44 countries (`type_constraints`), and **~89% of question texts literally
  name the country or its demonym**: *"In **Japanese** culture…"*, *"…for
  **Spanish** people?"*, *"In the **Netherlands**…"*. The practice node's feature
  is the raw `prompt_question` (`data/culturalbench.py`), so the answer is sitting
  in the input. The task had collapsed to **string matching**, not cultural
  inference.

This is **label leakage through the input features** — distinct from graph
leakage. No amount of architecture work could be evaluated meaningfully while it
was present.

## 3. The baseline that proves it — Frozen-NN (roadmap A3)

**Frozen-NN = Frozen sentence-transformer embeddings + Nearest-Neighbor
retrieval.** It is the dumbest possible model:

- **Frozen:** each entity is its raw sentence-transformer embedding; the encoder
  weights are never updated. No training.
- **NN:** to predict a practice's country, rank the ~44 country embeddings by
  cosine similarity to the practice embedding. No Tree-GRU, no GKI, no relation,
  no hierarchy.

It is a *control*: "what score do you get with zero learning and zero structure?"
If the full trained model can't beat it, the architecture isn't justified.

Implementation: `scripts/frozen_baseline.py`. It reuses the same type-constrained
candidate set and filtered-ranking protocol as the trained model, so the numbers
are directly comparable.

```
python scripts/frozen_baseline.py                      # masked text (default)
python scripts/frozen_baseline.py --keep_country_text  # leaky text (A/B)
python scripts/frozen_baseline.py --embed_model all-mpnet-base-v2
```

## 4. The fix — mask the country in the input text

We strip country names and demonyms from each question before it becomes the
node feature, so the model must infer the country from the cultural *content*:

> "In Japanese culture, how do people greet elders?"
> → "In [COUNTRY] culture, how do people greet elders?"

- `data/culturalbench.py`: `COUNTRY_ALIASES` (name + demonym + common variants
  for all 44 countries), `mask_country_text()`, and a `mask_country=True`
  parameter on `load_culturalbench` (default **on**).
- `[COUNTRY]` is a single constant token, so it reveals *that* a country is
  referenced but never *which* one. Multi-word names are matched longest-first
  ("South Korea" before "Korea"); matching is word-boundary and case-insensitive.
  Bare ambiguous tokens (e.g. "US") are deliberately excluded to avoid clobbering
  ordinary words ("us").
- `scripts/train.py`: `--keep_country_text` flag reproduces the leaky text for
  A/B; the run banner and `RESULT` line now report `mask_country=...`.

About 1086/1227 (~88%) of practice texts contain a masked mention.

## 5. The evidence

Frozen-NN, MiniLM, identical candidate set and protocol — masked vs. unmasked:

| Setup (no training, no graph) | test MRR | H@1 | H@10 |
|-------------------------------|---------|-----|------|
| **Unmasked** (old leaky text) | **0.9615** | 0.9435 | 0.9919 |
| **Masked** (the fix)          | **0.2494** | 0.1694 | 0.4032 |

Interpretation:

1. **The 0.94 was the leak.** A no-training, no-graph baseline scores **0.96** on
   the unmasked text — *higher* than the trained `no_gki` (0.937) and `pre_agg`
   (0.954). The entire Tree-GRU + GKI pipeline was adding nothing over "the
   country name is in the question."
2. **The fix removes it.** Masking drops frozen-NN to **0.25** (random over 44
   candidates ≈ 0.10), so the task is now genuinely hard.
3. **There is now headroom.** The masked frozen-NN score is the **floor** the
   trained model must beat for the hierarchy / GKI to be justified.

## 6. What this changes

- **A1 must be re-run on masked text** (now the default). The question becomes the
  right one: *on a non-trivial task, does the hierarchy/GKI beat the ~0.25 frozen
  floor?* Re-running `jobs/job_a1_*.sh` uses masked text automatically.
- Always report the **frozen-NN floor for the same encoder** alongside trained
  numbers. Get the mpnet floor with
  `python scripts/frozen_baseline.py --embed_model all-mpnet-base-v2`.
- If trained variants land near the floor, that points to the architectural
  levers (B4 symmetric conditioning, B2 a genuinely distinct knowledge source)
  rather than more tuning.

## 7. Residual caveats

`[COUNTRY]` masking removes the explicit name, but weaker giveaways may remain in
the text (cities, languages, dishes, currencies, festivals). The 0.25 floor
suggests the residual signal is modest, but a stricter benchmark would scrub
those next. This is a known limitation, not a blocker.

---

**Files:** `data/culturalbench.py` (masking), `scripts/train.py`
(`--keep_country_text`), `scripts/frozen_baseline.py` (A3 baseline).
