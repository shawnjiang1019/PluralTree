# Map Reader: Phase-1 Training Pipeline

Teach the LM to **read an injected Poincaré latent** before it is asked to reason
over latents (Phase 2). Three steps: export embeddings, generate data, train —
then a validation protocol that distinguishes latent-reading from memorization.

> **Scope.** Phase 1 is *necessary, not sufficient* for Phase 2. Reading one
> latent's geometry does not confer multi-latent reasoning; a strong Map Reader
> must not be mistaken for Phase-2 readiness.

---

## Step 0 — Export embeddings

Produce `h_all`: `(N, d_hidden)` entity hidden states on the Poincaré ball.

**Method.** A Hyperbolic Tree-GRU + Gated Knowledge Injection (GKI) encoder, *not*
a free lookup table. Embeddings are encoder-produced (inductive), so unseen
entities can also be embedded.

- **Inputs:** frozen sentence-transformer text features per entity (`all-MiniLM-L6-v2`);
  tree structure (`children_indices`, `topo_order`).
- **Encoder:** bottom-up recursion (leaves → root). Each node combines its text
  feature, its children's aggregated hidden states (Möbius midpoint), and a
  depth-aware gated knowledge injection, via a hyperbolic GRU cell.
- **Objective:** margin-ranking **link prediction**, `relu(margin + score_neg − score_pos)`,
  scoring the relational translation `h_s ⊕ r ≈ h_o`. Optional structure-fidelity
  loss (`--lambda_struct`) pushes a node's parent closer than a non-ancestor.
- **Optimization:** Riemannian Adam, with warmup phases gating GKI.

```powershell
python scripts/train.py --dataset wn18rr --data_dir data/wn18rr `
    --inductive_holdout 0.1 --save_embeddings runs/h_all.pt --device cuda
```

### Validation gate (do not proceed if these fail)

The pipeline rests entirely on `h_all`; emergence of geometry must be *measured*,
not assumed. `train.py` already reports the needed quantities:

- **`RESULT` line** — held-out link-prediction MRR / Hits@k. Must be clearly above
  chance, else relational structure is not encoded.
- **`STRUCT` line** — `depth_radius_rho = Spearman(‖h‖, depth)`. Must be strongly
  positive, else the norm does not track depth and `ρ`/depth probes are vacuous.

Hold out entities (`--inductive_holdout`) at this step so a node-level split is
available downstream (Step 1, `--split`).

> **What the latent plausibly contains.** Link prediction preserves relational
> and hierarchical structure, so `ρ` and depth are recoverable. There is **no a
> priori reason** it encodes `|children|`, `domain`, or `role`. Whether each fact
> is in the latent is an empirical question answered by the per-fact ablation
> below — not assumed by construction.

---

## Step 1 — Generate Map Reader data

Deterministic supervision derived from the tree + `h_all`. No LLM, no GPU.

Each record stores `node_id` (latent looked up at train time), a text `prompt`,
a `target`, and a `fact` type. Two kinds:

- **1A `decode`** (one per node): input = latent only; target = full structural
  caption: `Depth, ρ, branching, role, |children|, domain`.
- **1B `qa`** (k sampled probes per node): input = latent + templated question;
  target = a single structural answer. The prompt contains **no** geometry.

| Fact | Source | Latent-readable? |
|---|---|---|
| `ρ` (radius) | embedding (`= √c·‖h‖`) | yes — but a shallow norm readout; sanity check only |
| depth | embedding (norm-emergent) | likely; also near-trivial |
| role, branching | tree | **unknown — to be tested** |
| \|children\| | tree | **unknown — to be tested** |
| domain | tree | **likely not — to be tested** |

`abstraction` is `ρ`-thresholded, hence **not** an independent signal; it is not
emitted as a separate fact.

Split entities into train/val so reading can be validated on **held-out nodes**
(the encoder is inductive, so their latents are still meaningful):

```powershell
python scripts/generate_map_reader_sft.py --dataset wn18rr --data_dir data/wn18rr `
    --embeddings runs/h_all.pt --split runs/holdout.json `
    --out data/map_reader_sft.jsonl
```

Knobs: `--max_nodes`, `--qa_per_node`, `--no_decode`, `--no_qa`. `--embeddings`
is required (the `ρ` target needs the trained vectors).

---

## Step 2 — Supervised fine-tuning (QLoRA + LatentBridge)

Frozen 4-bit LM (7–8B). Trains **only** the LoRA adapter and the `LatentBridge`.

- **LatentBridge:** `log_map_zero(h)` → MLP → `n_tokens` soft tokens in LM
  embedding space (`pluraltree/sft/latent_bridge.py`). `n_tokens` is the bridge's
  information **bandwidth** — sweep and report it (default 4).
- **Injection:** soft tokens prepended to the text via `inputs_embeds`.
- **Loss:** next-token cross-entropy on the target span only (`-100` over soft
  tokens and prompt).

```
[ soft tokens(h_all[node_id]) ] + "<prompt>\nAnswer:" + " <target><eos>"
└─ label = -100 ───────────────┘ └─ label = -100 ───┘ └─ loss here ─────┘
```

```powershell
pip install transformers peft bitsandbytes accelerate
python scripts/train_map_reader.py --base Qwen/Qwen2.5-7B-Instruct `
    --embeddings runs/h_all.pt --data data/map_reader_sft.jsonl --out runs/map_reader
```

Outputs: LoRA adapter + `latent_bridge.pt` + `bridge_config.json` in `runs/map_reader/`.
Numerical stability near the ball boundary (`‖h‖ → 1/√c`) is handled in the
manifold (`project_to_ball`, `BOUNDARY_EPS`, `safe_norm`, `safe_artanh`); the
bridge inherits it via `log_map_zero`.

---

## Validation protocol

Caption / QA loss alone is **insufficient**: 1A lets the model memorize a
`node_id → answer` prior. All evaluation is on **held-out nodes** (Step 1 split),
per fact, with these conditions:

| Condition | Latent | Caption text | Purpose |
|---|---|---|---|
| **true** | node's latent | — | the model |
| **zero** | zeros | — | does the latent matter at all? |
| **shuffle** | another node's latent | — | does the *specific* latent matter? |
| **text-only** | none | — | lower bound: prompt/prior guessing |
| **facts-in-prompt** | none | answer in prompt | upper bound: LM ceiling |

**Metric.** Per fact `f`, QA exact-match accuracy `acc_f`. Report:

- `Δ_zero(f)  = acc_f(true) − acc_f(zero)`
- `Δ_shuffle(f) = acc_f(true) − acc_f(shuffle)`

with bootstrap confidence intervals (no hand-set threshold). Read the result
**per fact**, not in aggregate:

- Facts the latent encodes (`ρ`, depth) → large positive Δ.
- Facts it does not (likely `domain`) → Δ ≈ 0; the model can only memorize them,
  so they do **not** support a reading claim.

Sanity: `acc(true)` must sit clearly **above** text-only; the gap to
facts-in-prompt quantifies the bridge's information loss. A near-zero
`Δ_shuffle` on the non-norm facts is the failure signal.
