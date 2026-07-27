# Finding Intermediate States Between Two Distributions via Wasserstein Distance

## 1. The Core Idea: Wasserstein Geodesics

To find "in-between" distributions when transforming $\mu_0$ into $\mu_1$, use **displacement interpolation** (McCann 1997) — the geodesic in Wasserstein-2 space:

$$\mu_t = (T_t)_\# \pi, \qquad T_t(x,y) = (1-t)x + ty, \quad t \in [0,1]$$

Each mass element travels along a **straight line** from source to target at constant speed.

**Key contrast:** This differs from linear interpolation $(1-t)\mu_0 + t\mu_1$, which just fades one distribution out and the other in (mass appears/disappears). Displacement interpolation actually *moves* mass — a smooth morph rather than a crossfade.

**Geodesic property:** $W_2(\mu_s, \mu_t) = |t - s|\, W_2(\mu_0, \mu_1)$ (constant speed).

### Special cases
- **Discrete:** $\mu_t = \sum_{ij} P_{ij}\, \delta_{(1-t)x_i + t y_j}$, where $P$ is the optimal coupling.
- **Optimal map exists (Brenier):** $\mu_t = \big((1-t)\,\mathrm{id} + t\,T\big)_\# \mu_0$.
- **Gaussians:** closed form with $m_t = (1-t)m_0 + t m_1$ and a matrix-geometric-mean covariance.

**Tooling:** the [POT library](https://pythonot.github.io/) (`ot.emd`, Sinkhorn) computes couplings and interpolations.

---

## 2. Finding the "Most Important" Intermediate States (Discrete Case)

"Important" is ambiguous — four natural notions, each giving different answers.

### Notion 1 — Bifurcation / Topology Change
Track pairwise distances between moving points $z_k(t) = (1-t)x_{i_k} + t y_{j_k}$. Each is a convex quadratic in $t$; its minimizer (closest-approach time):

$$t^*_{kl} = \operatorname{clip}_{[0,1]}\!\left(-\frac{\langle \Delta z_{kl}(0),\, \Delta v_{kl}\rangle}{\|\Delta v_{kl}\|^2}\right)$$

The modes of the histogram of $\{t^*_{kl}\}$ mark structurally important times.

### Notion 2 — Velocity-Field Spread (Max Activity)
Each element has constant velocity $v_{ij} = y_j - x_i$. Kinetic energy is constant, so look at the *spread* relative to the mean flow $\bar v$ — the deformation $\tilde v_k = v_k - \bar v$. Peak spread = maximum shearing/mixing.

### Notion 3 — Diversity / Coverage
Pick $K$ representative frames minimizing $\sum_t \min_k W_2(\mu_t, \mu_{t_k})$. Because $W_2$ is *linear in $t$* along the geodesic, this collapses to **uniform spacing** ($t = k/K$). Only becomes non-trivial when weighted by the change density from Notions 1–2.

### Notion 4 — Metastable States
Local minima of a free-energy functional $F(\mu)$ along the path. These require a different (JKO / gradient-flow) construction, not the OT geodesic itself.

---

## 3. What Each Notion Needs to Compute

**Shared core (all four):** support points + weights, and **one OT solve** for the coupling $P$ (`ot.emd`), giving trajectories $z_k(t)$ and velocities $v_k$.

| Notion | Extra input beyond $P$ | Extra compute |
|---|---|---|
| 1. Bifurcation | none | pairwise closest-approach $t^*$, cheap |
| 2. Activity | none | variance/spread functional, cheap |
| 3. Coverage | $K$ (+ optionally 1/2's output) | trivial |
| 4. Metastable | **density model + chosen $F$ (potential $V$, kernel $W$)** | JKO / gradient-flow solver, expensive |

Notions 1–3 all come from the single OT solve. Only Notion 4 requires modeling choices and a separate dynamics.

---

## 4. What Each Output Means Intuitively

| Notion | Output | Intuition |
|---|---|---|
| **1. Bifurcation** | Special times $t^*$ | Moments where the point cloud reorganizes — clusters merge, split, trajectories cross. The "events." |
| **2. Activity** | Time of peak deformation | The moment of maximum shearing/mixing — where the morph is most turbulent, not just sliding. |
| **3. Coverage** | $K$ representative frames | A storyboard — fewest snapshots summarizing the whole morph (evenly spaced, or clustered on interesting moments if weighted). |
| **4. Metastable** | Times of free-energy minima | Natural "resting states" the system lingers in — physically stable intermediates. |

**Quick contrast:** Notion 1 finds structural *events*, Notion 2 finds peak *motion intensity*, Notion 3 gives a *summary set*, Notion 4 finds *physically stable* intermediates.