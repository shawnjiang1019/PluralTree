# SFT Data Generation Plan (revised)

Goal: produce SFT data that makes the Reasoner **use the injected Poincaré embeddings as
genuine reasoning tools**, not decorative tokens it ignores in favour of the text captions.

The single principle everything follows from:

> **Caption–latent asymmetry.** If the caption already states a property, the latent is
> redundant for that property and the model will rationally read the text instead. The
> latent only becomes load-bearing when it is the *sole* source of some information the
> task needs. Every design choice below enforces this, and a causal ablation (not a
> text-overlap heuristic) is what proves it.

Implemented in: `pluraltree/sft/scout.py`, `pluraltree/sft/verbalize.py`,
`scripts/generate_reasoner_sft.py`.

---

## Phase 0: Poincaré embedding export

Make the trained embeddings injection-ready, once:

- Encode the full graph (WN18RR/CultureBank) with the Tree-GRU encoder → `h_all` on the ball.
- Per node store: `h_node` (vector), `ρ = √c·‖h‖` (depth proxy), `N_node` (parent/sibling/child ids).
- **Producer gap:** the trainer does not currently save `h_all`. Add an export step
  (`encode_tree()` under `torch.no_grad()` → `torch.save`) that feeds `--embeddings`.
- **Why:** `ρ`, neighbours, and subtree shape are the inputs the Scout and verbalizer need;
  precompute so Phase 2 is cheap.

---

## Phase 1: Map Reader (teach latent → structure decoding)

The "insurance policy": before reasoning, the model must learn to *read* a latent. This
only works if the embedding is the **sole input** — the caption is the **target**, never a
co-input (otherwise the model shortcuts caption → answer and never reads the latent).

### Step 1A — Embedding → Caption (supervised decode)
- **Input:** `h_node` only. **Target:** a compact structural caption.
- **Caption (symbolic, geometry-bearing):** `Depth: 3, ρ: 0.87, Branching: [3,2], Role: junction, |children|: 3`.
- **Why:** a clean latent → description dictionary. Correct as a *decode* task because the
  caption is the output, not the input.

### Step 1B — Structural-role QA (embedding-only input)
- **Input:** `h_node` + a templated question (`"depth?"`, `"leaf/junction/root?"`, `"#children?"`).
  **No caption in the input** — that would leak the answer.
- **Target:** the structural answer (`"junction at depth 3 with 3 children"`).
- **Why:** forces `latent → structural fact`. (The original draft fed the caption *and* the
  embedding as input; that leaks — fixed here to embedding-only.)

> Map Reader makes the model *able* to read latents. It does **not** by itself force the
> Reasoner to read them at Phase 2 — see the caption-starving rule below.

---

## Phase 2: Scout & Reasoner data

### Step 2A — Queries + gold answers
- Source: existing QA dataset (JSONL `{question, answer, anchor?}`), or programmatic from the KG.
- Prefer **verifiable** answers (enables the correctness gate and RL outcome reward).
- Include structural-query variants alongside semantic ones.

### Step 2B — DPP Scout (`scout.py`)
- Retrieve `K` structural cousins per query: structurally isomorphic, semantically distant.
- **Two distinct signals — do not conflate:**
  - **Structural match = a shape fingerprint** (relative-depth histogram + degree histogram +
    size), *not* hyperbolic distance. Two identically-shaped subtrees in different domains are
    **far apart** in geodesic distance — that is exactly the case we want, so distance cannot
    measure isomorphism.
  - **Diversity = geodesic distance + domain check.** Same-domain (shared level-1 ancestor)
    candidates are pinned to high similarity so the DPP suppresses co-selecting them.
- Selection: greedy MAP over `L = diag(q)·S·diag(q)` (quality `q` = structural match,
  similarity `S` = semantic closeness). The DPP log-det scores the *set* — the non-scalar
  diversity signal.

### Step 2C — Verbalize subtrees (`verbalize.py`), **asymmetric**
- Compact, symbolic captions. **Critical change vs the draft:** to preserve asymmetry, the
  Phase-2 caption is **coarse** — domain label + name + a qualitative abstraction tag
  (`broad`/`intermediate`/`specific`). It **omits the quantitative geometry** (no exact `ρ`,
  no depth integer, no branching vector). Those remain available *only* through the latent.
- Provide a `--caption_detail {full, coarse, none}` knob to emit curricula (full for warm-up,
  coarse/none for forcing latent use).
- Relation labels (`hypernym`, …) require a relation-aware tree; `children_indices` currently
  drops relation type — add if WN18RR relation semantics are needed in captions.

### Step 2D — Distill candidate traces (Claude, rationalization)
- Prompt with query + gold answer + **coarse** cousin captions.
- Trace format = **observe-then-reason** (the in-trace map-reading step):
```
<think>
<observe>
  Latent 1: <decode ρ / depth / branching FROM THE EMBEDDING — these are not in the caption>
  Latent 2: ...
</observe>
<reason>
  Path 1 (<domain>): structural analogy — holds because ...
  Path 2 (<domain>): analogy BREAKS because ...
  Contrast: holds — ...; breaks — ...
</reason>
</think>
<answer>...</answer>
```
- Because the caption is coarse, the only way to fill `<observe>` with `ρ`/depth/branching is
  to read the latent — so the map-reading step is a genuine read-out, not a copy.

### Step 2E — Filter → gold traces
- Keep: correct answer, references all `K` domains, explicit holds/breaks, faithful to structure.
- **Replace the broken "latent-only property" text filter.** The draft required citing a
  property "only in the embedding and not in the caption" while the caption *contained* `ρ`/
  depth/branching — self-contradictory and unverifiable. Two sound replacements:
  1. **Faithfulness check:** score each `<observe>` value against ground truth
     (`verbalize_subtree`'s true `ρ`/depth/neighbours). Reject hallucinated observations.
  2. **Causal ablation (the real proof):** see Verification below.

### Step 2F — Emit record
```json
{
  "query": "...",
  "answer": "...",
  "anchor": "<entity>",
  "node_ids": ["n123", "n234", "n345"],
  "domains": ["...", "...", "..."],
  "caption_coarse": ["...", "...", "..."],
  "gold_trace": "<think>...</think><answer>...</answer>"
}
```
- Store **`node_ids`, not embedding vectors** — look up `h_all` at train time. Dumping float
  vectors into JSONL is heavy and brittle.

---

## Caption-starving (the rule that actually forces latent use)

Map Reader + observe-block are necessary but not sufficient: if the caption is present at
reasoning time, the model can still copy it. So:

- **SFT (warm-up):** coarse caption present — teaches the format.
- **SFT (curriculum) / RL:** degrade then **drop** the caption (`--caption_detail none`). The
  latent becomes the sole source of cousin-specific facts; the reward depends on getting them
  right → the embedding is load-bearing.

---

## Verification (acceptance test — not optional)

Text filters cannot prove latent usage. The only proof is causal:

- **Zero-latent:** inject zeros. Score should drop.
- **Shuffle:** swap cousin A's latent with cousin B's, captions fixed. `<observe>` and the
  answer should change; if not, the model is reading text/priors, not the latent.
- **Drop-caption:** remove captions, keep latents; performance should hold.

The metric is the **delta** `score(true) − score(zeroed/shuffled)`. A near-zero delta is the
failure signal — gate magnitude and probes are necessary but not sufficient.

---

## Data flow

```
[Phase 0] trained encoder ─→ export h_all (+ρ, neighbours)

[Phase 1: Map Reader]  h_node ─→ caption (1A) ;  h_node + question ─→ structural answer (1B)

[Phase 2: Scout & Reasoner]
Queries + gold answers
  └─ DPP Scout(h_all): fingerprint match  +  geodesic/domain diversity ─→ K cousin node_ids
       └─ verbalize (COARSE: domain + abstraction, no ρ/depth numbers)
            └─ distill(query, coarse captions, answer, observe-then-reason)
                 └─ filter (correct + all K domains + holds/breaks + observe faithfulness)
                      └─ gold trace
Emit: {query, node_ids, domains, caption_coarse, gold_trace, answer}
```

---

## Training sequence

| Stage | Data | Caption | Goal |
|---|---|---|---|
| SFT 1 (Map Reader) | Phase 1 | embedding-only input | learn to *read* latents |
| SFT 2 (Reasoner, warm-up) | Phase 2 | coarse | learn the observe-then-reason format |
| SFT 2 (Reasoner, curriculum) | Phase 2 | degrade → none | shift reliance onto the latent |
| RL (GRPO) | verifiable answers | none | optimise path selection; reward observe-faithfulness + answer |

> Map Reader first, Explorer second — but the embeddings only become genuine reasoning tools
> once the caption is **starved** and latent usage is **causally verified**. The earlier draft
> kept the caption (and its geometry) everywhere, which would have left the embeddings
> decorative despite the Map Reader stage.
