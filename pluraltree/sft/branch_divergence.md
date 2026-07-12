# Evaluating Divergence with Wasserstein Distance

Measures whether a parent node's child branches genuinely **fork** (lead to
semantically different regimes) or are **redundant** (collapse to the same
region despite the tree branching). Implemented in `evaluation/branch_divergence.py`.

## What we measure

At a parent `P`, each child `C_i` roots a subtree. Treat that subtree as an
empirical distribution `μ_i` over its nodes' embeddings. The divergence between
two branches is the **Wasserstein (optimal transport) distance** `W(μ_i, μ_j)`.

- **Ground metric = Poincaré geodesic** (`manifold.distance`), the learned
  semantic distance — not raw DAG path length.
- **Distribution = uniform mass** over the subtree's node embeddings.
- Subtrees are capped (`max_nodes=32`) to keep the OT tractable.

Why Wasserstein and not JS-divergence: JS needs the *same support* and ignores
geometry. Wasserstein compares distributions over *different* supports using the
ground metric — exactly "how far apart are these two branches."

## The problem with raw W: spread ≠ divergence

Raw `W` is large for two unrelated reasons:
1. children mean genuinely different things (a real fork), **or**
2. children are the same kind of thing but embedded far apart (scattered
   instances, e.g. `gulf` → {persian gulf, gulf of mexico, ...}).

Raw `W` cannot tell these apart, so scattered categories rank as high as forks.

## Fix: normalize against a null baseline

Rank by divergence **beyond chance**, not absolute `W`:

```
null = mean & std of W over random NON-sibling subtree pairs
z(P) = ( W(P's children) − null_mean ) / null_std
```

- A genuine fork exceeds the null → `z > 0`.
- Scattered instances ≈ random subtrees → `z ≈ 0`.
- Coherent siblings (closer than random) → `z < 0`.

`z` is unitless, so it is comparable across embeddings and curvatures (raw
geodesic scale is not).

## Metrics reported

| Key | Meaning |
|---|---|
| `branch_divergence_mean` | mean raw `W` over sampled sibling sets |
| `branch_divergence_null` | chance-level `W` (random non-sibling pairs) |
| `branch_divergence_rel_mean` | `sibling_mean − null`. **Monoculture indicator**: ≤ 0 means branches are no more divergent than random |
| `branch_divergence_z_max` | most-divergent parent's z-score |

`divergence_anchors` / `relative_divergence_anchors` return the per-parent
ranking (the polarizing "Divergence Anchors").

## Usage

```bash
# off by default in structure metrics (OT is costly); opt in:
compute_structure_metrics(..., branch_divergence=True)

# standalone ranking on a saved embedding:
python -m evaluation.intrinsic.branch_divergence --embeddings EMB.pt \
    --dataset culturalbench --top 20
```

OT solver: exact EMD via `POT` (`ot.emd2`) if installed, else a dependency-free
log-domain Sinkhorn (approximate; fine for ranking).

## Interpreting results

Read `rel_mean` and `z_max` first:
- `rel_mean > 0`, `z_max ≫ 1` → genuine forks exist; top anchors are real.
- `rel_mean ≤ 0`, `z_max ≈ 0` → no forks; siblings cohere (or collapse).

Worked example — WN18RR (a lexical taxonomy):
`null=6.31, sibling_mean=5.39, rel_mean=−0.92, z_max=0.10`.
Siblings are *less* divergent than random pairs → no divergence anchors. Correct
and expected: hyponyms of a shared hypernym cohere. A taxonomy structurally
cannot have value/cultural forks; the metric correctly reports their absence.

The metric only reveals divergence the **embedding actually encodes**. A flat
result on a graph that *should* be plural (e.g. CultureBank) is a finding about
the embedding, not the data.
