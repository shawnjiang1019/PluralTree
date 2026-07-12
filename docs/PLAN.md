# Gated Knowledge Injection (GKI) — Standalone & Combined with Hyperbolic Tree-GRU

## Context

The PluralTree project explores recursive gating in hyperbolic geometry for hierarchical knowledge graphs. This plan covers implementing **Gated Knowledge Injection (GKI)** — a technique for selectively incorporating external knowledge into neural representations via learned gates — in two tracks:

- **Track A**: Standalone GKI (Euclidean and hyperbolic variants)
- **Track B**: GKI combined with the Hyperbolic Tree-GRU, using depth-aware gating conditioned on Poincaré ball radius

---

## Project Layout

```
PluralTree/
├── pyproject.toml
├── ideas.txt
├── configs/
│   ├── base.yaml
│   ├── track_a_euclidean.yaml
│   ├── track_a_hyperbolic.yaml
│   ├── track_b_combined.yaml
│   └── ablations/
├── pluraltree/
│   ├── manifolds/          # Poincaré ball, hyperboloid, math utils
│   ├── knowledge/          # Knowledge source abstraction + multi-source selector
│   ├── gki/                # Track A: standalone GKI (Euclidean + hyperbolic gates)
│   ├── tree_gru/           # Hyperbolic Tree-GRU cell, aggregator, encoder
│   ├── combined/           # Track B: depth-aware GKI + Tree-GRU integration
│   └── utils/              # Tree batching, Riemannian optim wrappers, numerical safety
├── training/
├── evaluation/
├── data/
├── tests/
└── scripts/
```

---

## Mathematical Formulations

### Euclidean GKI (Track A baseline)

```
g = σ(W_g [h ; k] + b_g)
h' = g ⊙ k + (1 - g) ⊙ h
```

### Hyperbolic GKI (Track A, Poincaré ball)

- Compute gate in tangent space at origin: `g = σ(W_g [log₀(h) ; log₀(k)] + b_g)`
- Combine via Möbius weighted midpoint: `h' = möbius_midpoint(h, k; 1-g, g)`

### Combined Depth-Aware GKI + Tree-GRU (Track B)

At each node v in bottom-up recursion:

1. Aggregate children via Möbius weighted midpoint → `h_agg`
2. Compute depth signal: `ρ = √c · ‖h_v‖` (radius encodes hierarchy depth)
3. Blend knowledge: `β = σ(w_β·ρ + b_β)` routes between broad (relational) and precise (factual) knowledge
4. Depth-aware gate: `g = σ(W_g [log₀(h); log₀(k); ρ] + b_g)`
5. Inject: `h' = möbius_midpoint(h, k; 1-g, g)`

**Key insight**: nodes near the origin (abstract) get broad relational knowledge; nodes near the boundary (specific) get precise factual attributes. The exponential volume growth of hyperbolic space provides more representational capacity at leaves where more precise knowledge is injected.

---

## Injection Points (configurable)

| Point | When | Default? |
|-------|------|----------|
| PRE_AGGREGATION | Into each child before combining | No |
| **POST_AGGREGATION** | After child aggregation, before GRU gates | **Yes** |
| POST_GRU | After full GRU update, as refinement | No |
| DUAL | Both pre-agg and post-GRU | Ablation only |

---

## Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `PoincareBall` | `manifolds/poincare.py` | All Möbius operations, exp/log maps, projection |
| `KnowledgeSource` (ABC) | `knowledge/base.py` | Interface for KG embeddings, retrieval, structured attrs |
| `KnowledgeSelector` | `knowledge/selector.py` | Multi-source attention routing |
| `EuclideanGate` / `HyperbolicGate` | `gki/gate.py` | Learned sigmoid gates |
| `GKIInjector` | `gki/injector.py` | Orchestrates retrieval → selection → gating → injection |
| `HyperbolicTreeGRUCell` | `tree_gru/cell.py` | Single recursion step with Möbius GRU |
| `ChildAggregator` | `tree_gru/aggregation.py` | Möbius weighted midpoint over children |
| `DepthAwareHyperbolicGate` | `combined/depth_aware_gate.py` | Gate conditioned on ρ + broad/precise blending |
| `GKITreeGRUCell` | `combined/gki_tree_gru_cell.py` | Tree-GRU cell with configurable GKI injection |
| `KnowledgeSchedule` | `combined/knowledge_schedule.py` | Curriculum: staged source activation + gate bias annealing |

---

## Training Strategy

- **Gate bias init at -2.0** (sigmoid ≈ 0.12): gates start nearly closed, model learns base representations first
- **Knowledge curriculum**: Phase 1 (KG embeddings only, gates closed) → Phase 2 (add structured attrs, anneal gates) → Phase 3 (all sources, unbiased gates)
- **Riemannian optimization**: manifold params via `geoopt.RiemannianAdam`, Euclidean params via standard Adam, split learning rates
- **Numerical stability**: project after every manifold op, safe artanh, hyperboloid fallback when conformal factor > 100, float32 for manifold ops under AMP
- **Loss**: `L_task + α·L_gate_sparsity + β·L_curvature_reg`

---

## Evaluation

- **Benchmarks**: WN18RR (hierarchy-heavy), FB15k-237 (diverse), YAGO3-10 (scale)
- **Metrics**: MRR, Hits@{1,3,10}, distortion, gate activation statistics
- **10 ablations**: no GKI, Euclidean GKI, no depth-aware, single source (×2), each injection point, no curriculum, Euclidean Tree-GRU, fixed curvature
- **Visualization**: gate activation vs radius scatter, source weights vs depth, embedding space t-SNE

---

## Implementation Phases

1. **Project setup** — pyproject.toml, package structure, configs
2. **Manifold primitives** — Poincaré ball ops + exhaustive round-trip tests
3. **Track A Euclidean GKI** — gates, injector, synthetic validation
4. **Track A Hyperbolic GKI** — Möbius midpoint injection, manifold-preserving tests
5. **Knowledge sources** — KG embedding, retrieval, structured, multi-source selector
6. **Tree-GRU** — cell, aggregator, encoder, tree batching utilities
7. **Combined model (Track B)** — depth-aware gate, GKI-Tree-GRU cell, knowledge schedule
8. **Training infrastructure** — trainer, losses, Riemannian optim, data loaders
9. **Evaluation & ablations** — metrics, ablation runner, visualization scripts

---

## Critical Files (implementation priority)

1. `pluraltree/manifolds/poincare.py` — foundation for all hyperbolic ops
2. `pluraltree/gki/injector.py` — central GKI orchestrator (Track A core)
3. `pluraltree/tree_gru/cell.py` — Hyperbolic Tree-GRU cell
4. `pluraltree/combined/depth_aware_gate.py` — novel depth-conditioned gating
5. `pluraltree/combined/gki_tree_gru_cell.py` — Track B integration point

---

## Verification

- **Unit tests**: manifold round-trips, gate output shapes, outputs remain on manifold
- **Synthetic test**: binary tree with known hierarchy, verify gate activations correlate with depth
- **Benchmark test**: WN18RR link prediction, compare MRR across Track A → Track B → ablations
- **Visualization**: `scripts/visualize_gates.py` plots gate activation vs ρ to confirm depth-aware behavior

---

## Phase 10: CulturalBench Data Pipeline

### Goal

Build a knowledge graph from CulturalBench where the model learns link prediction — given `(subject, relation, ?)`, predict the correct object. The learned hidden states then serve as the retrieval index for LLM guidance at inference time.

### Hierarchy structure

```
World
├── Africa
├── Asia
│   ├── East_Asia
│   │   ├── Japan
│   │   │   ├── tea_ceremony      ← leaf (cultural practice)
│   │   │   └── ikebana
│   │   └── China
│   └── South_Asia
│       └── India
├── Europe
└── ...
```

**Levels**: World → Continent → Sub-region → Country → Cultural practice

**Relations**:

| Relation | Example |
|----------|---------|
| `practiced_in` | (tea_ceremony, practiced_in, Japan) |
| `located_in` | (Japan, located_in, East_Asia) |
| `part_of` | (East_Asia, part_of, Asia) |
| `belongs_to_domain` | (tea_ceremony, belongs_to_domain, customs) |
| `similar_to` | (tea_ceremony, similar_to, gongfu_cha) |

### Files to create

**`data/loaders/culturalbench.py`**

- Load CulturalBench from HuggingFace datasets library
- Extract entities: geographic nodes (continent, country) + cultural practice nodes (one per unique question topic)
- Extract relations from question metadata: country tag → `practiced_in`, domain tag → `belongs_to_domain`
- Build geographic containment edges: `located_in`, `part_of` from a static geographic lookup
- Output: entity vocabulary, relation vocabulary, triple list `[(s_id, r_id, o_id), ...]`, train/val/test split (80/10/10)

**`data/tree_builder.py`**

- Takes triple list, builds rooted tree by spanning the geographic hierarchy
- For DAG cases (practice belongs to multiple countries): duplicate the node, one copy per country subtree
- Returns per-tree: `node_features (N, d_input)`, `node_ids (N,)`, `children_indices`, `topo_order`
- Node features: sentence-transformer embeddings (`all-MiniLM-L6-v2`) of entity name + short description

**`data/negative_sampler.py`**

- For each positive triple `(s, r, o)`, corrupt the object with a randomly sampled entity of the same type
- Type-constrained corruption: `practiced_in` corruptions sample from country nodes only, `part_of` from region nodes
- Filtered evaluation: exclude known positives from the negative pool during evaluation

**`data/collate.py`**

- Packages a batch of `(positive_triple, negative_triple, tree)` tuples
- Returns tensors the trainer can consume directly

### Node feature encoding

Text embeddings from `sentence-transformers` serve double duty:

1. As `node_features` fed into the Tree-GRU input projection
2. As the `KnowledgeSource` — the same embeddings are looked up by `KGEmbeddingSource` during GKI injection

This removes the dependency on pretrained KG embeddings (TransE/RotatE) which don't exist for CulturalBench entities. The sentence-transformer embeddings are frozen; the projection layer learns to extract what's useful.

---

## Phase 11: Scoring Head and Training Loop

### Link prediction scoring

Relation embeddings are learned vectors `r ∈ B_c^d`, one per relation type. Score a triple using TransE-style hyperbolic scoring:

```
score(s, r, o) = -d_H(h_s ⊕_c r, h_o)
```

where `⊕_c` is Möbius addition and `d_H` is hyperbolic distance. Higher score = more plausible triple.

**`training/scoring.py`**

```python
class HyperbolicLinkPredictor(nn.Module):
    # relation_embeddings: ManifoldParameter (num_relations, d_hidden) on ball
    # score(h_s, r_id, h_o) → scalar score per triple
    # uses Möbius addition + hyperbolic distance
```

### Training loop

**`training/trainer.py`**

```
for each batch:
    1. Build tree for the batch entities
    2. Forward pass: GKITreeEncoder → h for all nodes
    3. Score positive triples: score(h_s, r, h_o_pos)
    4. Score negative triples: score(h_s, r, h_o_neg)
    5. Link prediction loss (margin-based, from training/losses.py)
    6. Gate sparsity regularization
    7. Backprop through tree + gates + projections
    8. RiemannianAdam step (manifold params) + Adam step (Euclidean params)
    9. Apply KnowledgeSchedule: update gate bias, active sources
    10. Log: loss, gate mean activation, MRR on val set every N steps
```

**Key training decisions**:

- Negative samples per positive: 10–50 (more = better signal, slower)
- Margin: 1.0 (tunable)
- Gate sparsity weight `α`: 0.01 (small, so it doesn't dominate)
- Gradient clipping: max norm 5.0 for Riemannian gradients
- Curriculum warmup: warmup1=2000 steps, warmup2=6000 steps (shorter than WN18RR since CulturalBench is smaller)

---

## Phase 12: Evaluation

**`evaluation/kgc/link_prediction.py`**

For each test triple `(s, r, o)`:
1. Score `(s, r, o)` against all candidate objects
2. Filter out known positives (train + val) from candidates
3. Rank the correct object among the remaining candidates
4. Compute: MRR, Hits@1, Hits@3, Hits@10

```python
def evaluate(model, predictor, test_triples, all_triples, entity_vocab):
    # Returns: {'mrr': float, 'hits@1': float, 'hits@3': float, 'hits@10': float}
```

**`evaluation/gate_analysis.py`**

After training, inspect what the gates learned:

- Plot gate activation `mean(g)` vs hyperbolic radius `ρ` for every node — verify depth-aware behavior
- Plot knowledge source weights `alpha` vs tree depth — verify broad/precise routing
- Identify which relation types cause gates to open vs. close

---

## Phase 13: LLM Guidance Interface

Once the graph is trained, this is how it connects to an LLM.

**`pluraltree/inference/retriever.py`**

```python
class GraphRetriever:
    def __init__(self, encoder, predictor, entity_vocab, relation_vocab):
        ...

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        # 1. Embed query_text with sentence-transformer
        # 2. Find nearest node in hyperbolic space (log_map_zero + cosine, or d_H)
        # 3. Return top_k related entities + their relations + confidence scores
        # 4. Format as context strings for LLM prompt injection
```

At inference time the LLM workflow becomes:

```
LLM receives: "Describe the cultural significance of the tea ceremony"
    ↓
GraphRetriever.query("tea ceremony", top_k=5)
    ↓
Returns: [
    ("tea_ceremony", "practiced_in", "Japan",    score=0.94),
    ("tea_ceremony", "symbolizes",   "harmony",  score=0.87),
    ("tea_ceremony", "belongs_to",   "customs",  score=0.91),
    ("Japan",        "located_in",   "East_Asia", score=0.99),
    ("tea_ceremony", "similar_to",   "gongfu_cha", score=0.72),
]
    ↓
Injected into LLM prompt as structured context
    ↓
LLM generates grounded, relation-aware response
```

The depth-aware gates are what make this useful — the retriever naturally returns both specific facts (leaf nodes, high ρ) and broader context (ancestor nodes, low ρ) depending on how the query maps onto the hyperbolic space.

---

## Updated Implementation Order

| Phase | Files | Depends on |
|-------|-------|------------|
| 10a | `data/loaders/culturalbench.py` | HuggingFace datasets, sentence-transformers |
| 10b | `data/tree_builder.py` | 10a, `utils/tree_utils.py` |
| 10c | `data/negative_sampler.py` | 10a |
| 10d | `data/collate.py` | 10b, 10c |
| 11a | `training/scoring.py` | `manifolds/poincare.py` |
| 11b | `training/trainer.py` | 10d, 11a, `training/losses.py`, `utils/riemannian_optim.py` |
| 12a | `evaluation/kgc/link_prediction.py` | 11a |
| 12b | `evaluation/gate_analysis.py` | trained model |
| 13 | `pluraltree/inference/retriever.py` | trained model, sentence-transformers |
| — | `scripts/train/train.py` | 11b, 12a |

---

## Success Criteria

The pipeline is working correctly when:

1. **Loss decreases** — link prediction loss falls below random baseline within first curriculum phase
2. **Gates open with depth** — `mean(g)` is measurably higher for leaf nodes (cultural practices) than for root nodes (continents)
3. **Knowledge sources route correctly** — text embedding source gets higher weight at leaf nodes, lower at abstract nodes
4. **MRR > random** — filtered MRR on test set exceeds 1/num_entities baseline
5. **Retriever returns sensible results** — querying "tea ceremony" returns Japan-related entities, not random cultural practices from other regions
