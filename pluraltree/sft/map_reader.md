# Map Reader: Phase-1 Training Pipeline

Teach the LM to **read an injected Poincaré latent** before it is asked to reason
over latents (Phase 2). Three steps: export embeddings, generate data, train.

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

The geometry (`ρ = √c·‖h‖`, depth-faithfulness, neighborhood structure) is an
emergent property of this objective.

```powershell
python scripts/train.py --dataset wn18rr --data_dir data/wn18rr `
    --save_embeddings runs/h_all.pt --device cuda
```

---

## Step 1 — Generate Map Reader data

Deterministic supervision derived from the tree + `h_all`. No LLM, no GPU.

Each record stores `node_id` (latent looked up at train time), a text `prompt`,
and a `target`. Two kinds:

- **1A `decode`** (one per node): input = latent only; target = full structural
  caption: `Depth, ρ, branching, role, |children|, domain`.
- **1B `qa`** (k sampled probes per node): input = latent + templated question;
  target = the single structural answer. The prompt contains **no** geometry, so
  the answer is derivable only from the latent.

| Fact | Source |
|---|---|
| depth, role, \|children\|, branching, domain | tree |
| `ρ`, abstraction (broad/intermediate/specific) | embedding |

```powershell
python scripts/generate_map_reader_sft.py --dataset wn18rr --data_dir data/wn18rr `
    --embeddings runs/h_all.pt --out data/map_reader_sft.jsonl
```

Knobs: `--max_nodes`, `--qa_per_node`, `--no_decode`, `--no_qa`. `--embeddings`
is required (the `ρ`/abstraction targets need the trained vectors).

---

## Step 2 — Supervised fine-tuning (QLoRA + LatentBridge)

Frozen 4-bit LM (7–8B). Trains **only** the LoRA adapter and the `LatentBridge`.

- **LatentBridge:** `log_map_zero(h)` → MLP → `n_tokens` soft tokens in LM
  embedding space (`pluraltree/sft/latent_bridge.py`).
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

---

## Validation

Caption cross-entropy alone is insufficient — 1A lets the model memorize a
`node_id` prior. Genuine latent reading is proven by a **causal ablation**:

- **Zero-latent:** inject zeros → accuracy should drop.
- **Shuffle-latent:** swap nodes A and B's latents (prompts fixed) → decoded
  depth/`ρ`/role should swap with them.

The metric is the delta `score(true) − score(zeroed/shuffled)`; a near-zero
delta means the model is reading the prompt, not the latent.
