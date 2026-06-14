# PluralTree — Gating Methods

Alternative gating mechanisms for Gated Knowledge Injection (GKI), framed as deltas
from the current gate. The gate is the most swappable part of the architecture, and
several of these alternatives map directly onto the roadmap in
[`EXPERIMENTS.md`](./EXPERIMENTS.md). Papers are collected in the
[References](#references) table and cross-referenced from [`RELATED_WORK.md`](./RELATED_WORK.md) §4.

---

## Current gate (the baseline to vary)

Three variants exist today (`pluraltree/gki/gate.py`, `pluraltree/combined/depth_aware_gate.py`):

- **`EuclideanGate`** — `g = σ(W_g [h ; k] + b)`, then `h' = g ⊙ k + (1 − g) ⊙ h`
- **`HyperbolicGate`** — gate computed in tangent space; injection via Möbius midpoint
- **`DepthAwareHyperbolicGate`** — adds the radius feature ρ and a broad/precise blend β

All three are a **convex blend**: produce a per-dimension `g ∈ (0,1)^d` from
`[log₀(h); log₀(k); ρ]`, then `h' = möbius_midpoint(h, k; 1−g, g)`. So `g` can only
*interpolate* between `h` and `k`. **That single constraint is what most alternatives below relax.**

---

## 1. Different gate forms (cheap, A1-style ablations)

Small, drop-in replacements — directly comparable to the existing A1 matrix.

| Method | Form | Delta vs. current |
|--------|------|-------------------|
| **Scalar gate** | one `g ∈ (0,1)` per node (not per-dim) | coarser, far fewer params, highly interpretable ("how much knowledge did node *v* take?") |
| **Highway gate** | `h' = g⊙k + (1−g)⊙h` with a *separate* transform of `k` | decouples carry/transform gates the current coupled form ties together |
| **GLU** (Gated Linear Unit) | `out = (W·k) ⊙ σ(V·k)` | gate and content both from `k`; no convex constraint |
| **FiLM** (feature-wise modulation) | `h' = γ(c)⊙h + β(c)`, `c = [k; ρ]` | **affine** modulation, not a [0,1] blend — `γ` can scale up/down, `β` can shift. Strictly more expressive than interpolation; can "amplify this feature," which a convex gate cannot. |

**Highest-value here: FiLM.** It removes the "can only interpolate" ceiling while
keeping depth conditioning (feed ρ into `c`).

---

## 2. Attention-based gating (medium; unlocks plurality)

Replace the fixed gate with **cross-attention**: `h` is the query, knowledge
candidates `{k_1 … k_m}` are keys/values, and attention weights replace `g`.

- Generalizes to **multiple knowledge sources** (roadmap B2) and, crucially, to
  **multiple candidate parents** — attention weights over candidate parents *are* the
  soft attachment of the learned-hierarchy direction (E4).
- **Hyperbolic Attention Networks** (Gulcehre et al. 2019) does this with hyperbolic
  aggregation (Einstein/Möbius midpoint as the value combiner) — manifold-native, not
  a tangent-space approximation.

This is the bridge from "gating" to E2/E4: the same mechanism that selects knowledge
can select context/parent.

---

## 3. Discrete / stochastic gates (medium; serve sparsity + learned structure)

The current `gate_sparsity_loss` wants gates to turn fully *off*, but a sigmoid never
quite does. These give true on/off:

- **Hard-Concrete / L0 gates** (Louizos et al. 2018) — a stretched concrete
  distribution with real probability mass at exactly 0 and 1, plus a principled L0
  penalty. The *correct* tool for what the sparsity term currently approximates.
- **Gumbel-Softmax gates** (Jang et al. 2017; Maddison et al. 2017) — differentiable
  discrete choice. Use when the gate should *select* (which source, which parent,
  which sense) rather than blend. The mechanism for differentiable hierarchy
  induction (Gumbel attachment, E4).

---

## 4. Hyperbolic-native / geometry-aware gates (research bets)

- **Distance-conditioned gate** — make `g` a function of `d_H(h, k)`: inject more when
  knowledge is geodesically close (consistent) or far (novel), as a design choice.
  Cheap to add, geometrically motivated, clean ablation vs. the learned linear gate.
- **Gyroplane / Möbius gate** — compute the gate via hyperbolic operations end-to-end
  instead of mapping to tangent space first. More faithful to the manifold; riskier
  numerically.

---

## 5. Routing / mixture gates (for many sources / senses)

- **Mixture-of-Experts (MoE) gating** (Shazeer et al. 2017, top-k routing) — once
  there are several knowledge sources or *plural senses* of a node, route each node to
  the relevant expert(s). Natural home for the E2/E3 "multiple separated senses →
  mixture" idea.

---

## Prioritization

| Tier | Method | Serves | Cost |
|------|--------|--------|------|
| **Try first** (drop-in `--gate_type`) | scalar, FiLM, distance-conditioned | gate expressivity ablation | low |
| **Next** | cross-attention gate (hyperbolic) | B2 multi-source, **E4 soft attachment** | medium |
| **For sparsity** | Hard-Concrete / L0 | fixes the sparsity-loss approximation | low–medium |
| **For learned structure** | Gumbel-Softmax routing | E2/E4 plurality & attachment | medium |
| **Research bet** | MoE routing, gyroplane gate | E3 mixtures, manifold fidelity | high |

**Single most informative cheap experiment:** add **FiLM** and a **scalar** gate as
two more `--gate_type` options and run them in the existing A1 matrix. That reveals
whether the convex-blend constraint actually limits the model before investing in
attention/routing.

---

## References

| Method | Paper |
|--------|-------|
| Highway gate | Srivastava, Greff & Schmidhuber, *Highway Networks* (2015) |
| GLU | Dauphin et al., *Language Modeling with Gated Convolutional Networks* (2017) |
| FiLM | Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* (2018) |
| Cross-attention | Vaswani et al., *Attention Is All You Need* (2017) |
| Hyperbolic attention | Gulcehre et al., *Hyperbolic Attention Networks* (2019) |
| Hard-Concrete / L0 | Louizos, Welling & Kingma, *Learning Sparse Neural Networks through L0 Regularization* (2018) |
| Gumbel-Softmax | Jang, Gu & Poole, *Categorical Reparameterization with Gumbel-Softmax* (2017); Maddison, Mnih & Teh, *The Concrete Distribution* (2017) |
| Mixture-of-Experts | Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017) |
| GRU (current cell) | Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder* (2014) |

> References are by author/title/year for easy lookup — verify exact venues when you
> pull them. This list reflects an early-2026 knowledge cutoff.
