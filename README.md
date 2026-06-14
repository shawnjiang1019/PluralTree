# PluralTree

PluralTree is a research system for learning hierarchical knowledge graph embeddings using hyperbolic geometry. It combines a **Hyperbolic Tree-GRU** — a recursive neural network that operates in the Poincaré ball — with **Gated Knowledge Injection (GKI)**, a mechanism for selectively incorporating external knowledge into node representations via learned gates.

The system is validated on **CulturalBench**, a dataset of cultural practices organized into a four-level geographic hierarchy (World → Region → Country → Practice), using link prediction as the training task.

**Companion docs** (in [`docs/`](./docs/)): [`EXPERIMENTS.md`](./docs/EXPERIMENTS.md) — the experiment roadmap (validation, ablations, and novel plurality/distribution directions); [`RELATED_WORK.md`](./docs/RELATED_WORK.md) — a reading list mapped to the architecture; [`EVALUATION.md`](./docs/EVALUATION.md) — how we judge embedding quality beyond link prediction.

---

## Core Ideas

### Why Hyperbolic Space?

Standard Euclidean embeddings struggle with hierarchical data: to embed a balanced tree with branching factor `b` and depth `d`, you need exponentially many dimensions. Hyperbolic space grows exponentially with radius, so it can embed trees with low distortion in low dimensions. In the **Poincaré ball** model, the origin represents the abstract root and the boundary represents specific leaf nodes — the radius encodes hierarchical depth.

### What is Gated Knowledge Injection?

GKI selectively blends a node's current hidden state `h` with an external knowledge vector `k` via a learned gate `g`:

```
g = σ(W_g [log₀(h) ; log₀(k) ; ρ] + b)
h' = möbius_midpoint(h, k; 1-g, g)
```

The gate is computed in the tangent space (Euclidean) and the injection uses the Möbius midpoint (manifold-preserving). A gate value near 0 means "ignore the knowledge, keep the current state"; near 1 means "replace with knowledge". Gates start nearly closed during training and gradually open via a curriculum schedule.

### Depth-Aware Gating

The key novel contribution is conditioning gates on the Poincaré ball radius `ρ = √c · ||h||`. Abstract nodes (small `ρ`, near origin) receive broad relational knowledge; specific leaf nodes (large `ρ`, near boundary) receive precise factual knowledge. A blending parameter `β = σ(w_β · ρ + b_β)` continuously transitions between these two knowledge types as a function of position in the tree.

---

## Project Layout

```
PluralTree/
├── pluraltree/                 # Main package
│   ├── manifolds/              # Hyperbolic geometry primitives
│   │   ├── poincare.py         # Poincaré ball: all Möbius ops, exp/log maps, distance
│   │   ├── hyperboloid.py      # Lorentz model (alternative / fallback)
│   │   └── math_utils.py       # Numerical stability: safe_artanh, safe_norm, projection
│   ├── knowledge/              # Knowledge source abstraction
│   │   ├── base.py             # KnowledgeSource abstract base class
│   │   ├── kg_embedding.py     # Embedding lookup table (pretrained or learned)
│   │   ├── projection.py       # d_source → d_hidden, optional manifold projection
│   │   ├── selector.py         # Multi-source attention routing
│   │   └── structured.py       # Entity attribute encoder
│   ├── gki/                    # Standalone Gated Knowledge Injection (Track A)
│   │   ├── gate.py             # EuclideanGate, HyperbolicGate
│   │   └── injector.py         # GKIInjector: retrieval → selection → gating → injection
│   ├── tree_gru/               # Hyperbolic Tree-GRU primitives (Track A)
│   │   ├── cell.py             # HyperbolicTreeGRUCell: single recursion step
│   │   ├── aggregation.py      # ChildAggregator: Möbius weighted midpoint
│   │   └── encoder.py          # TreeEncoder: full bottom-up tree encoding
│   ├── combined/               # GKI + Tree-GRU integration (Track B, default)
│   │   ├── gki_tree_gru_cell.py   # GKITreeGRUCell with configurable injection points
│   │   ├── gki_tree_encoder.py    # GKITreeEncoder: full encoder with curriculum
│   │   ├── depth_aware_gate.py    # DepthAwareHyperbolicGate: radius-conditioned gate
│   │   └── knowledge_schedule.py  # KnowledgeSchedule: 3-phase curriculum
│   └── utils/
│       ├── tree_utils.py       # Topological sort, tree depth, find_root
│       ├── riemannian_optim.py # Split Riemannian/Euclidean optimizer builder
│       └── numerical.py        # Numerical safety utilities
├── data/
│   ├── culturalbench.py        # Load CulturalBench, build CulturalGraph
│   ├── tree_builder.py         # Package CulturalGraph as encoder input tensors
│   ├── negative_sampler.py     # Type-constrained corruption for link prediction
│   └── collate.py              # TripleBatchSampler: batch positives + negatives
├── training/
│   ├── losses.py               # Link prediction loss, gate sparsity, curvature reg
│   ├── scoring.py              # HyperbolicLinkPredictor: TransE-style in hyperbolic space
│   └── trainer.py              # Trainer class: full training loop
├── evaluation/
│   ├── link_prediction.py      # Filtered MRR, Hits@{1,3,10}
│   └── gate_analysis.py        # Gate activation vs depth visualisation
├── scripts/
│   └── train.py                # CLI entry point
├── tests/
│   ├── test_manifolds.py       # Manifold round-trips, distance properties
│   ├── test_gates.py           # Gate shapes and activation ranges
│   ├── test_tree_gru_cell.py   # Cell output shapes, manifold preservation
│   └── test_combined.py        # Full GKI Tree-GRU integration tests
├── configs/                    # YAML configs for different experiment setups
├── jobs/                       # SLURM scripts for Compute Canada (Narval)
│   ├── job.sh                  # Main training job
│   └── job_a1_*.sh             # A1 ablation jobs: baseline, no_gki, plain_gate,
│                               #   pre_agg, post_gru, dual
├── docs/                       # Companion docs (EXPERIMENTS, EVALUATION,
│                               #   RELATED_WORK, GATING, WN18RR, comparisons …)
├── pyproject.toml
└── requirements.txt
```

> **Note on layout:** SLURM scripts live under `jobs/`. Submit them from the repo
> root (e.g. `sbatch jobs/job.sh`) so the `logs/` output paths resolve correctly.

---

## Architecture

### Data Flow

```
CulturalBench (HuggingFace)
        │
        ▼
load_culturalbench()  →  CulturalGraph
        │                 (entity vocab, triples, tree structure, type constraints)
        ▼
compute_text_embeddings()  →  (N, d_input) sentence-transformer embeddings
        │                       (d_input = 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2)
        │
        ├──► KGEmbeddingSource (frozen knowledge source)
        │
        └──► GKITreeEncoder.forward(node_features, children_indices, topo_order)
                    │
                    │  Bottom-up traversal (topological order, leaves first)
                    │
                    ▼
              For each node:
                1. input_proj: x_raw → x_tan (Euclidean projection)
                2. ChildAggregator: {h_child} → h_agg  (Möbius midpoint)
                3. GKI injection (at configured point)
                4. HyperbolicTreeGRUCell: (x_tan, h_agg) → h_v  (on Poincaré ball)
                    │
                    ▼
              h_all: (N, d_hidden)  — one vector per entity, on the Poincaré ball
                    │
                    ▼
              HyperbolicLinkPredictor.score(h_s, r, h_o)
                  score = -d_H(h_s ⊕_c r_embedding, h_o)
                    │
                    ▼
              Margin loss: relu(margin + score_neg - score_pos)
```

### The Hyperbolic Tree-GRU Cell

For a node `v` with children `C`, the cell computes:

```
h_agg  = möbius_midpoint({h_c : c ∈ C}, {α_c})   # attention-weighted, on ball
h_tan  = log_map_zero(h_agg)                        # map to tangent space at origin

z      = σ(W_z [h_tan ; x_v])                      # update gate
r      = σ(W_r [h_tan ; x_v])                      # reset gate
ñ      = tanh(W_n [r ⊙ h_tan ; x_v])              # candidate
h_tan' = (1 - z) ⊙ h_tan + z ⊙ ñ                  # GRU mix

h_v    = exp_map_zero(h_tan')                       # back to Poincaré ball
```

GRU gates are computed in tangent space (Euclidean), which avoids having to define sigmoid on the manifold. The output is projected back onto the ball.

### Knowledge Injection Points

`InjectionPoint` (in `combined/gki_tree_gru_cell.py`) controls where knowledge enters the recursion:

| Point | When | Use case |
|-------|------|----------|
| `PRE_AGGREGATION` | Into each child before combining | Per-leaf refinement |
| `POST_AGGREGATION` | After aggregation, before GRU | Default: enrich aggregated context |
| `POST_GRU` | After full GRU update | Late-stage refinement |
| `DUAL` | Pre-agg and post-GRU | Maximum injection (ablation) |

### Knowledge Curriculum (3 phases)

Gates start nearly closed so the model first learns to encode the tree structure, then gradually learns to leverage external knowledge:

```
Phase 1 (steps 0 → warmup1):    gate bias = -2.0  → sigmoid ≈ 0.12 (nearly closed)
Phase 2 (warmup1 → warmup2):    bias linearly anneals  → gates open progressively
Phase 3 (warmup2+):             gate bias = 0.0   → unbiased gates
```

Default: `warmup1=500`, `warmup2=1500` (configurable via CLI).

### Link Prediction Scoring

`HyperbolicLinkPredictor` implements TransE-style translation in hyperbolic space:

```
score(s, r, o) = -d_H(h_s ⊕_c r, h_o)
```

`r` is a learned relation embedding (a `ManifoldParameter` on the Poincaré ball). Higher score = more plausible triple. The margin ranking loss is:

```
L = mean(relu(margin + score_neg - score_pos))
```

### Optimizer

`build_optimizer` (in `utils/riemannian_optim.py`) splits parameters:

- **Manifold parameters** (relation embeddings, any `ManifoldParameter`): `geoopt.RiemannianAdam` — curvature-aware updates that keep points on the manifold
- **Euclidean parameters** (linear layers, gate weights): standard `Adam`
- Two separate learning rates: `--lr` for Euclidean, `--lr_manifold` for manifold (typically higher)

---

## The Dataset: CulturalBench

`kellycyy/CulturalBench` (Easy split) contains multiple-choice questions about cultural practices across countries. PluralTree uses only the geographic structure and the question text; the answer choices are discarded.

The knowledge graph built from it has four entity types:

```
World (1)
  └── Region (11):  East_Asia, Southeast_Asia, South_Asia, ...
        └── Country (43): China, Japan, India, France, ...
              └── Practice (N): one node per unique question
```

Relations:
- `practiced_in`: practice → country
- `located_in`: country → region
- `part_of`: region → world

Structural triples (country/region/world) are always in the training set. Practice triples are split 80/10/10 train/val/test.

Sentence-transformer embeddings of entity description text serve as both node input features and the frozen knowledge source. The encoder model is selectable via `--embed_model` (default `all-MiniLM-L6-v2`, 384-d; `all-mpnet-base-v2`, 768-d, is a stronger alternative). The hidden dimension `d_input` is derived automatically from the embedding size, so swapping encoders needs no model-code changes.

---

## Installation

```bash
git clone <repo>
cd PluralTree
pip install -r requirements.txt
```

On Compute Canada (Narval), load modules before creating the virtualenv:

```bash
module load python/3.11 gcc cuda/13.2 arrow/24.0.0
python -m venv ~/envs/pluraltree
source ~/envs/pluraltree/bin/activate
pip install -r requirements.txt
```

Pre-download data and model on the login node (compute nodes have no internet):

```bash
python -c "from datasets import load_dataset; load_dataset('kellycyy/CulturalBench', 'CulturalBench-Easy')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# If using the stronger encoder, pre-download it too:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

> On Narval, `pyarrow` is provided by the `arrow` module, **not** pip — load
> `arrow/24.0.0` *before* activating the venv, and do not `pip install pyarrow`.

---

## Training

**Local:**
```bash
python scripts/train.py
python scripts/train.py --d_hidden 128 --n_epochs 100 --device cuda
```

**SLURM (Narval):** submit from the repo root so `logs/` resolves correctly.
```bash
sbatch jobs/job.sh

# A1 ablation matrix (see EXPERIMENTS.md §A1) — 6 separate jobs, each logging
# to logs/a1_<variant>_<jobid>.{out,err}:
for j in baseline no_gki plain_gate pre_agg post_gru dual; do
  sbatch jobs/job_a1_$j.sh
done
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--d_hidden` | 64 | Hidden dimension (also controls relation embedding size) |
| `--n_epochs` | 50 | Training epochs |
| `--batch_size` | 128 | Triples per batch |
| `--n_negative` | 10 | Negative samples per positive triple |
| `--margin` | 1.0 | Margin in ranking loss |
| `--lr` | 1e-3 | Learning rate for Euclidean parameters |
| `--lr_manifold` | 1e-2 | Learning rate for manifold parameters |
| `--warmup1` | 500 | Step at which gate annealing begins |
| `--warmup2` | 1500 | Step at which gates are fully unbiased |
| `--gate_bias` | -2.0 | Initial gate bias (negative = nearly closed) |
| `--injection` | `post_agg` | Injection point: `pre_agg`, `post_agg`, `post_gru`, `dual` |
| `--no_gki` | off | **A1 ablation:** disable all knowledge injection (pure Tree-GRU baseline) |
| `--gate_type` | `depth_aware` | **A1 ablation:** `depth_aware` (radius-conditioned) or `plain` (`HyperbolicGate`) |
| `--embed_model` | `all-MiniLM-L6-v2` | Sentence-transformer encoder (e.g. `all-mpnet-base-v2`) |
| `--curvature` | 1.0 | Poincaré ball curvature `c` |
| `--device` | `cpu` | `cpu` or `cuda` |

**Outputs** (printed during training):
```
step   50 | loss 0.8431 | lp 0.8327 | gate_bias -2.00
[eval] step 200 | MRR 0.1823 | H@1 0.0912 | H@10 0.4102
[best] new best val MRR: 0.1823
Epoch   1/100 | mean loss 0.7214 | 42.3s
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/test_manifolds.py -v      # Manifold round-trips and distance properties
pytest tests/test_combined.py -v       # Full integration tests
```

All tests run on CPU with small synthetic trees and do not require the dataset.

---

## Key Files for Understanding the Codebase

If you are reading the code for the first time, this order is recommended:

1. **`pluraltree/manifolds/poincare.py`** — Start here. All hyperbolic geometry operations. Understanding `mobius_add`, `exp_map_zero`, `log_map_zero`, and `distance` is prerequisite for everything else.

2. **`pluraltree/gki/gate.py`** and **`injector.py`** — The gating mechanism in isolation. `EuclideanGate` is the simplest version; `HyperbolicGate` adds manifold-awareness.

3. **`pluraltree/tree_gru/cell.py`** — The Tree-GRU cell. Shows how children are aggregated and how GRU gates operate in tangent space.

4. **`pluraltree/combined/depth_aware_gate.py`** — The key novel component. Shows how Poincaré radius `ρ` is computed and used to condition gating.

5. **`pluraltree/combined/gki_tree_encoder.py`** — The full encoder. Shows the bottom-up recursion loop and how all pieces fit together.

6. **`data/culturalbench.py`** — The dataset. Shows the geographic hierarchy and how triples are built from it.

7. **`training/trainer.py`** — The training loop. Shows how `encode_tree()`, scoring, loss, and the optimizer all connect.

---

## Conceptual Q&A

**Q: What are the gates learning?**
Each gate decides how much external knowledge to inject at a given node at a given training step. A fully closed gate means the node's representation comes entirely from the tree structure (children aggregation + GRU). A fully open gate means the representation is replaced by the knowledge source. The network learns the optimal blend.

**Q: Why not just concatenate knowledge and hidden state?**
Concatenation would always inject knowledge unconditionally. The gate enables the model to learn when knowledge is helpful (often for leaves where precise factual knowledge matters) and when to trust the tree structure (often for abstract intermediate nodes).

**Q: How does the hyperbolic radius encode depth?**
The Poincaré ball has exponentially more volume near the boundary than near the origin. To embed a balanced tree, children nodes naturally land further from the origin than their parents. After training, `ρ = √c · ||h||` reliably predicts hierarchy level. This is a geometric property of hyperbolic space, not something explicitly supervised.

**Q: What is the knowledge source in the current experiment?**
Sentence-transformer embeddings of entity descriptions (e.g., "Japan — country in East Asia"). These are frozen and serve as both input features and the knowledge vector `k`. The gate learns whether the text embedding of a node is informative given its current hidden state.

**Q: How does this connect to LLMs?**
The trained `h_all` embeddings can be used at inference time to retrieve hierarchically-relevant context for an LLM prompt. Given a query entity, its Poincaré embedding can be used to find nearby entities in the hierarchy (related countries, regional practices), providing structured context that pure retrieval-augmented generation misses.
