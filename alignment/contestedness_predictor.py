"""Predict question contestedness from TEXT — ONE INPUT to the injection router.

Why this exists (all numbers measured on the v5/v6 OvertonBench traces):
  injection helps +0.31 on contested questions and HURTS -0.45 on consensus
  ones; oracle per-question routing scores 0.634 vs 0.497 always-baseline. So
  routing is worth ~0.14 OvertonScore — if a signal exists. Two candidates died:

  1. GRAPH DIVERGENCE (evaluation/overton/route_signal.py) — corr +0.19 with the
     help-delta. Diagnosed cause, recorded in that module's docstring: the scout
     SELECTS the max-Wasserstein fork, so W is near-max on EVERY question;
     max-selection destroyed the between-question variance the router needs.
     Note also that route_signal's `z_level` computes null_mu/null_sd ONCE,
     outside the question loop, so it is a fixed affine map of the raw magnitude
     — rank-identical to it. It was never a second, independent signal.
  2. MODEL SELF-REPORT (PLURALISM_ROUTE in retrieval/answer.py) — the model
     judging its own contestedness inside <think> scored 0.072, i.e. worse than
     always-baseline. A self-reported retrieval decision is hallucination-prone.

So the estimate here comes from NEITHER raw graph magnitude NOR the model.

TARGET (`--target`), computed from the survey leaves — free, no generation, no
judge. For a question, each axis (OpinionQA: one demographic attribute; GOQA:
the country axis) holds >=2 subgroup distributions over the SAME ordered answer
options. Divergence = mean pairwise 1-D Wasserstein on the option INDEX,
normalised to [0,1] by (K-1). Ordinal ground metric on purpose: on a Likert
scale "shifted one notch" and "opposite poles" are different things, and TV/JS
cannot tell them apart.

  z_level   (primary) divergence standardised inside a stratum of (K, aggregate
            entropy bin): is this split UNUSUAL, not is it BIG. Raw magnitude is
            exactly what scored +0.19. The stratum is the fix that route_signal
            lacked: a per-question conditioner. Between-subgroup divergence is
            mechanically bounded by the aggregate marginal (a 95/5 question
            CANNOT have far-apart subgroups) and mechanically scaled by K, so
            raw W mostly reports "how spread is the population", not "do the
            groups disagree". Conditioning on both makes z the residual — the
            part of the split the marginal does not explain. With --null_bins 1
            z degenerates back to a per-K affine map of w_mean, which is
            route_signal's version; the diagnostic corr(z, w_mean) is printed so
            this is visible rather than assumed.
  entropy   entropy of the population-aggregate distribution / log K. A DIFFERENT
            construct: a question everyone is individually torn about scores
            high here with every subgroup identical (no disagreement at all).
  w_mean    raw mean pairwise Wasserstein — the leaf-side analogue of the signal
            that failed at +0.19, kept as the control the others must beat.

INPUT: question TEXT only, deliberately. A predictor keyed on graph features
learns a map from the graph to itself, and wherever those features exist you can
read the target straight off the graph — no predictor needed. Text keying is the
whole point: it produces a contestedness estimate for questions with NO graph
anchor, where the scout cannot operate at all. OvertonBench questions are
free-form user questions, not ATP items; that is the regime.

MODEL: ridge / logistic on FROZEN sentence embeddings. ~4k examples is small, so
no MLP. Default embedder is mpnet, NOT the scout's MiniLM — same argument as
retrieval/contestedness.py: the router must not be a function of the retrieval
it gates. Model class is swappable via MODELS.

SPLITS: grouped by TOPIC, never by question. ATP mints 12 topic clusters and
questions inside one are near-paraphrases ("...favor stricter gun laws" x8), so
a random split scores itself. Also ATP<->GOQA transfer in both directions: US
demographic splits and cross-country splits are two different divergence
structures, and a contestedness notion that survives the crossing is the one
that will survive an unseen user question.

    python -m alignment.contestedness_predictor --selftest
    python -m alignment.contestedness_predictor --dataset opinionqa \
        --target all --scores overton_scores_v6.csv --save contest_ridge.npz

WIRING (route_signal.py, one line at the end of main() — not applied here):
    from alignment.contestedness_predictor import report_route_signal; report_route_signal(rows)
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

MPNET = "sentence-transformers/all-mpnet-base-v2"
TARGETS = ("z_level", "entropy", "w_mean")


# ---------------------------------------------------------------------------
# Records: the dataset-agnostic view the target is computed from
# ---------------------------------------------------------------------------
@dataclass
class QuestionRecord:
    """One survey question, its text, its topic group, and its subgroup splits.

    ``axes`` is one list of subgroup distributions per axis. OpinionQA has ~8
    axes per question (one demographic attribute each); GOQA has exactly one
    (countries). Everything downstream reads only this, so --selftest can build
    fixtures with hand-set distributions and no graph.
    """
    qid: str
    text: str
    topic: str | None
    axes: list[list[list[float]]] = field(default_factory=list)

    @property
    def n_options(self) -> int:
        return len(self.axes[0][0])


# ---------------------------------------------------------------------------
# Divergence between two answer distributions
# ---------------------------------------------------------------------------
def w1_ordinal(p: np.ndarray, q: np.ndarray) -> float:
    """1-D Wasserstein on the option index, normalised to [0, 1] by (K-1).

    Closed form (sum of CDF gaps) — no OT solver needed in 1-D. Normalising by
    K-1 is required before any cross-question comparison: unnormalised, a
    6-option question outranks a 2-option one for free.
    """
    K = len(p)
    if K < 2:
        return 0.0
    return float(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]).sum() / (K - 1))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation — the unordered fallback. Some GOQA items are nominal
    ("which country is the greater threat?"), where option ORDER is meaningless
    and the ordinal metric would invent structure."""
    return float(0.5 * np.abs(p - q).sum())


GROUNDS = {"ordinal": w1_ordinal, "unordered": tv_distance}


def _mean_pairwise(dists: list[np.ndarray], ground) -> float:
    n = len(dists)
    if n < 2:
        return float("nan")
    s = sum(ground(dists[i], dists[j]) for i in range(n) for j in range(i + 1, n))
    return s / (n * (n - 1) / 2)


def question_divergence(rec: QuestionRecord, ground=w1_ordinal,
                        axis_agg: str = "mean") -> float:
    """Mean pairwise subgroup divergence, aggregated over the question's axes.

    axis_agg defaults to MEAN, not max. Max-over-axes is precisely the failure
    diagnosed in route_signal.py: taking the extreme of many candidates pushes
    every question toward the ceiling and flattens the between-question variance
    a router lives on. `max` is kept for ablation only.
    """
    per_axis = [_mean_pairwise([np.asarray(d, float) for d in ax], ground)
                for ax in rec.axes if len(ax) >= 2]
    per_axis = [v for v in per_axis if v == v]
    if not per_axis:
        return float("nan")
    return float(max(per_axis) if axis_agg == "max" else np.mean(per_axis))


def aggregate_dist(rec: QuestionRecord) -> np.ndarray:
    """Population marginal: each axis partitions the SAME population, so each
    axis mean estimates it; average over axes (group sizes are not carried in
    the graph, so uniform weighting is the honest choice)."""
    per_axis = [np.mean([np.asarray(d, float) for d in ax], axis=0)
                for ax in rec.axes if len(ax) >= 1]
    m = np.mean(per_axis, axis=0)
    s = m.sum()
    return m / s if s > 0 else m


def norm_entropy(p: np.ndarray) -> float:
    """H(p)/log K in [0, 1]."""
    K = len(p)
    if K < 2:
        return 0.0
    q = p[p > 0]
    return float(-(q * np.log(q)).sum() / math.log(K))


# ---------------------------------------------------------------------------
# The chance null and the targets
# ---------------------------------------------------------------------------
def _strata(recs: list[QuestionRecord], ent: np.ndarray, n_bins: int) -> list[tuple]:
    """(K, entropy-bin) key per question. Bins are per-K quantiles, so each K
    contributes its own comparison set instead of being ranked against a
    different option count."""
    keys: list[tuple] = [()] * len(recs)
    Ks = np.array([r.n_options for r in recs])
    for K in sorted(set(Ks.tolist())):
        idx = np.flatnonzero(Ks == K)
        if n_bins <= 1 or len(idx) < 2 * n_bins:
            for i in idx:
                keys[i] = (K, 0)
            continue
        edges = np.quantile(ent[idx], np.linspace(0, 1, n_bins + 1)[1:-1])
        for i in idx:
            keys[i] = (K, int(np.searchsorted(edges, ent[i])))
    return keys


def _null_stats(recs: list[QuestionRecord], keys: list[tuple], ground,
                n_null: int, seed: int) -> dict[tuple, tuple[float, float]]:
    """Chance divergence per stratum: (mean, sd) of ground(p, q) over subgroup
    pairs drawn from DIFFERENT questions in the same stratum.

    Non-sibling by construction, exactly like _null_divergence(same_level=True)
    in evaluation/intrinsic/branch_divergence.py. The difference that matters is
    that this null is recomputed per stratum, so z is not one global affine
    rescale of the raw magnitude.
    """
    rng = np.random.default_rng(seed)
    pool: dict[tuple, list[tuple[int, np.ndarray]]] = {}
    for qi, (r, k) in enumerate(zip(recs, keys)):
        for ax in r.axes:
            for d in ax:
                pool.setdefault(k, []).append((qi, np.asarray(d, float)))
    out: dict[tuple, tuple[float, float]] = {}
    for k, items in pool.items():
        if len({q for q, _ in items}) < 2:
            continue
        ws = []
        for _ in range(n_null * 20):
            if len(ws) >= n_null:
                break
            a, b = rng.integers(len(items), size=2)
            if items[a][0] == items[b][0]:          # skip siblings (same question)
                continue
            ws.append(ground(items[a][1], items[b][1]))
        if len(ws) >= 8:
            out[k] = (float(np.mean(ws)), float(np.std(ws)))
    return out


def compute_targets(recs: list[QuestionRecord], *, ground=w1_ordinal,
                    axis_agg: str = "mean", n_bins: int = 4, n_null: int = 400,
                    seed: int = 0) -> dict[str, np.ndarray]:
    """All three targets for a record list. Pure numpy — no graph, no model."""
    w = np.array([question_divergence(r, ground, axis_agg) for r in recs])
    ent = np.array([norm_entropy(aggregate_dist(r)) for r in recs])
    keys = _strata(recs, ent, n_bins)
    null = _null_stats(recs, keys, ground, n_null, seed)
    gm = float(np.nanmean(w))
    gs = float(np.nanstd(w)) or 1.0
    z = np.empty_like(w)
    for i, k in enumerate(keys):
        mu, sd = null.get(k, null.get((k[0], 0), (gm, gs)))
        z[i] = (w[i] - mu) / sd if sd > 1e-9 else 0.0
    return {"z_level": z, "entropy": ent, "w_mean": w}


# ---------------------------------------------------------------------------
# Graph -> records
# ---------------------------------------------------------------------------
def _topic_of(node: int, parents, types, id_to_entity) -> str:
    cur, seen = node, set()
    for _ in range(8):
        nxt = [p for p in parents[cur] if p not in seen]
        if not nxt:
            break
        cur = nxt[0]
        seen.add(cur)
        if types.get(cur) == "topic":
            return id_to_entity[cur]
    return "topic:?"


def extract_questions(graph, dataset: str) -> list[QuestionRecord]:
    """Survey leaves -> QuestionRecords.

    Reads ``opinion_dist`` directly rather than walking children_indices:
    leakage_safe=True hangs only the TRAIN opinions off their axis, and the
    target is a LABEL, not a model input, so it should see every surveyed
    subgroup. (No leakage: the model never sees the graph, only question text.)
    """
    from evaluation.intrinsic.structure_metrics import _parents_from_children

    parents = _parents_from_children(graph.children_indices)
    types, id2e = graph.entity_types, graph.id_to_entity
    buckets: dict[str, dict] = {}
    for oid, dist in graph.opinion_dist.items():
        name = id2e[oid]
        if dataset == "opinionqa":                    # op:{qkey}:{attr}:{group}
            parts = name.split(":", 3)
            if len(parts) != 4:
                continue
            _, qkey, attr, _group = parts
            qnode = graph.entity_vocab.get(f"q:{qkey}")
            if qnode is None:
                continue
            text = graph.entity_text.get(qnode, "")
            topic = _topic_of(qnode, parents, types, id2e)
            axis = attr
        else:                                         # goqa: op_{row}_{country}
            parts = name.split("_", 2)
            if len(parts) != 3:
                continue
            qkey = parts[1]
            text = (graph.entity_text.get(oid, "") or "").rsplit(" [", 1)[0]
            topic, axis = None, "country"
        if not text:
            continue
        b = buckets.setdefault(qkey, {"text": text, "topic": topic, "ax": {}})
        b["ax"].setdefault(axis, []).append([float(p) for p in dist])

    recs = []
    for qkey, b in buckets.items():
        K = len(next(iter(b["ax"].values()))[0])
        axes = [v for v in b["ax"].values()
                if len(v) >= 2 and all(len(d) == K for d in v)]
        if axes:
            recs.append(QuestionRecord(qid=f"{dataset}:{qkey}", text=b["text"],
                                       topic=b["topic"], axes=axes))
    return recs


def drop_eval_holdout(recs: list[QuestionRecord]) -> list[QuestionRecord]:
    """Never train on the 60 OvertonBench eval questions. Reuses the single
    existing guard (alignment/rollout_dataset.py) rather than restating it."""
    from alignment.rollout_dataset import _norm, load_eval_holdout_texts

    hold = load_eval_holdout_texts()
    keep = [r for r in recs if _norm(r.text) not in hold]
    print(f"  eval-holdout guard: dropped {len(recs) - len(keep)} of {len(recs)} "
          f"(exact-normalised matches against {len(hold)} OvertonBench questions)")
    return keep


# ---------------------------------------------------------------------------
# Frozen text embeddings
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str], model_name: str = MPNET,
                cache: str | None = None, batch: int = 64) -> np.ndarray:
    texts = list(texts)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["texts"]) == texts and str(z["model"]) == model_name:
            print(f"  embeddings: reused {cache} {z['X'].shape}")
            return z["X"]
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)
    X = np.asarray(enc.encode(texts, normalize_embeddings=True, batch_size=batch,
                              show_progress_bar=True), dtype=np.float64)
    if cache:
        np.savez(cache, X=X, texts=np.array(texts, dtype=object), model=model_name)
    print(f"  embeddings: {X.shape} from {model_name}")
    return X


# ---------------------------------------------------------------------------
# Models (swappable; numpy only — sklearn is not guaranteed in pluraltree-env)
# ---------------------------------------------------------------------------
class Ridge:
    """Closed-form L2 regression. d<=768, so the (d,d) solve is instant."""
    kind = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha, self.W, self.b = alpha, None, 0.0

    def fit(self, X, y):
        Xc, yc = X - X.mean(0), y - y.mean()
        d = X.shape[1]
        self.W = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ yc)
        self.b = float(y.mean() - X.mean(0) @ self.W)
        return self

    def predict(self, X):
        return X @ self.W + self.b


class Logistic:
    """L2 logistic on a top-tercile 'contested' label. Plain gradient descent:
    IRLS can diverge on near-separable standardised embeddings, and this is the
    secondary model class, not the one the headline numbers are read off."""
    kind = "logistic"

    def __init__(self, alpha: float = 1.0, iters: int = 800, lr: float = 0.5,
                 quantile: float = 2.0 / 3.0):
        self.alpha, self.iters, self.lr, self.q = alpha, iters, lr, quantile
        self.W, self.b, self.thr = None, 0.0, 0.0

    def fit(self, X, y):
        self.thr = float(np.quantile(y, self.q))
        t = (y > self.thr).astype(float)
        n, d = X.shape
        self.W, self.b = np.zeros(d), 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ self.W + self.b, -30, 30)))
            self.W -= self.lr * (X.T @ (p - t) / n + self.alpha * self.W / n)
            self.b -= self.lr * float((p - t).mean())
        return self

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(X @ self.W + self.b, -30, 30)))


MODELS = {"ridge": Ridge, "logistic": Logistic}


@dataclass
class ContestednessPredictor:
    """Frozen embedder + feature standardiser + linear head. Saved as npz so
    route_signal.py (or any caller) can score a question with no graph at all."""
    kind: str = "ridge"
    target: str = "z_level"
    embed_model: str = MPNET
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    W: np.ndarray | None = None
    b: float = 0.0

    @classmethod
    def train(cls, X, y, kind="ridge", alpha=1.0, target="z_level",
              embed_model=MPNET) -> "ContestednessPredictor":
        mu, sd = X.mean(0), X.std(0) + 1e-8
        m = MODELS[kind](alpha=alpha).fit((X - mu) / sd, y)
        return cls(kind=kind, target=target, embed_model=embed_model, mu=mu,
                   sd=sd, W=m.W, b=m.b)

    def score(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mu) / self.sd) @ self.W + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))) if self.kind == "logistic" else z

    def predict(self, texts) -> np.ndarray:
        return self.score(embed_texts(list(texts), self.embed_model))

    def save(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b, mu=self.mu, sd=self.sd,
                 meta=json.dumps({"kind": self.kind, "target": self.target,
                                  "embed_model": self.embed_model}))
        print(f"  saved predictor -> {path}")

    @classmethod
    def load(cls, path: str) -> "ContestednessPredictor":
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        return cls(mu=z["mu"], sd=z["sd"], W=z["W"], b=float(z["b"]), **meta)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xc ** 2).sum() * (yc ** 2).sum()))
    return float(xc @ yc / den) if den else float("nan")


def spearman(x, y) -> float:
    """Rank correlation — routing only needs the ORDER (who is more contested),
    and z scales differ between the two graphs we compare across."""
    return pearson(_rank(np.asarray(x, float)), _rank(np.asarray(y, float)))


# ---------------------------------------------------------------------------
# Topic-grouped cross-validation
# ---------------------------------------------------------------------------
def topic_folds(topics: list[str], n_test: int = 2, seed: int = 0) -> list[list[str]]:
    """Partition topics into held-out blocks of n_test. ATP has 12 clusters ->
    6 folds of 10 train / 2 test. Grouping is by topic, never by question: ATP
    items inside a cluster are near-paraphrases, so a random split leaks."""
    uniq = sorted(set(topics))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(uniq)))
    return [[uniq[i] for i in order[s:s + n_test]]
            for s in range(0, len(uniq), n_test)]


def _pick_alpha(X, y, groups, grid, kind) -> float:
    """Inner topic-grouped selection. Choosing alpha on the outer test fold
    would report a tuned-on-test number."""
    uniq = sorted(set(groups))
    if len(uniq) < 3 or len(grid) == 1:
        return grid[len(grid) // 2]
    inner = {t: i % 3 for i, t in enumerate(uniq)}
    best, best_a = -2.0, grid[0]
    for a in grid:
        rhos = []
        for f in range(3):
            te = np.array([inner[g] == f for g in groups])
            if te.sum() < 5 or (~te).sum() < 20:
                continue
            p = ContestednessPredictor.train(X[~te], y[~te], kind=kind, alpha=a)
            rhos.append(spearman(p.score(X[te]), y[te]))
        r = float(np.nanmean(rhos)) if rhos else float("nan")
        if r == r and r > best:
            best, best_a = r, a
    return best_a


def cv_by_topic(X, y, topics, *, kind="ridge", alphas=(1., 10., 100., 1000.),
                n_test=2, seed=0, verbose=True) -> dict:
    folds = topic_folds(topics, n_test, seed)
    tarr = np.array(topics)
    rhos, ns = [], []
    for fi, held in enumerate(folds):
        te = np.isin(tarr, held)
        if te.sum() < 5 or (~te).sum() < 20:
            continue
        a = _pick_alpha(X[~te], y[~te], tarr[~te].tolist(), list(alphas), kind)
        p = ContestednessPredictor.train(X[~te], y[~te], kind=kind, alpha=a)
        rho = spearman(p.score(X[te]), y[te])
        rhos.append(rho)
        ns.append(int(te.sum()))
        if verbose:
            print(f"    fold {fi}: held-out {held} n={int(te.sum()):4d} "
                  f"alpha={a:<7g} spearman={rho:+.3f}")
    return {"folds": rhos, "n": ns,
            "mean": float(np.nanmean(rhos)) if rhos else float("nan"),
            "sd": float(np.nanstd(rhos)) if rhos else float("nan")}


def kmeans_topics(X: np.ndarray, k: int, seed: int = 0, iters: int = 40) -> list[str]:
    """Pseudo-topics for GOQA, whose hierarchy is geographic and has no topic
    layer. Needed so GOQA folds are grouped like ATP's rather than random."""
    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    C = X[rng.permutation(len(X))[:k]].copy()
    a = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d2 = (X ** 2).sum(1)[:, None] - 2.0 * X @ C.T + (C ** 2).sum(1)[None, :]
        a = np.argmin(d2, axis=1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return [f"kt:{i}" for i in a]


# ---------------------------------------------------------------------------
# Evaluation #2: does the PREDICTION beat w_raw's +0.19 on the help-delta?
# ---------------------------------------------------------------------------
def _best_gate_fn():
    """Reuse route_signal's own gate scorer so numbers are directly comparable
    with the w_raw / z_level / relevance / driver_sim table it prints."""
    try:
        from evaluation.overton.route_signal import _best_gate
        return _best_gate
    except Exception:                                  # standalone / no HF deps
        def _bg(sig, delta, base_cov, scout_cov):
            n = len(sig)
            best, thr = sum(base_cov) / n, None
            for t in [-1e9] + sorted(set(sig)):
                s = sum(scout_cov[i] if sig[i] > t else base_cov[i]
                        for i in range(n)) / n
                if s > best:
                    best, thr = s, t
            return best, thr, sum(base_cov) / n
        return _bg


def score_help_delta(name: str, sig, base_cov, inj_cov) -> dict:
    """Correlation + best-single-threshold gate of one candidate signal against
    the per-question (inject - baseline) coverage delta. Same post-hoc design as
    route_signal.py, so the number lands in the same table as w_raw's +0.19.

    best_gate is an OVERFIT ceiling (one threshold tuned on these 60 questions);
    a signal is only real if helped_mean and hurt_mean clearly separate.
    """
    n = len(sig)
    delta = [i - b for b, i in zip(base_cov, inj_cov)]
    best, thr, base_mean = _best_gate_fn()(list(sig), delta, list(base_cov),
                                           list(inj_cov))
    helped = [s for s, d in zip(sig, delta) if d > 0.01]
    hurt = [s for s, d in zip(sig, delta) if d < -0.01]
    out = {"signal": name, "n": n, "corr": pearson(sig, delta),
           "spearman": spearman(sig, delta), "best_gate": best, "thr": thr,
           "always_base": base_mean, "always_inject": sum(inj_cov) / n,
           "oracle": sum(max(b, i) for b, i in zip(base_cov, inj_cov)) / n,
           "helped_mean": float(np.mean(helped)) if helped else float("nan"),
           "hurt_mean": float(np.mean(hurt)) if hurt else float("nan"),
           "n_helped": len(helped), "n_hurt": len(hurt)}
    thr_s = "none" if out["thr"] is None else f"{out['thr']:.3f}"
    print(f"  {name:<16}corr={out['corr']:+.3f}  rho={out['spearman']:+.3f}  "
          f"gate={out['best_gate']:.4f}@{thr_s}  "
          f"helped={out['helped_mean']:.3f} hurt={out['hurt_mean']:.3f}  "
          f"(base={base_mean:.4f} inject={out['always_inject']:.4f} "
          f"oracle={out['oracle']:.4f})")
    return out


def _cov_from_csv(scores_csv: str, inject_cond="scout", base_cond="baseline"):
    import csv
    from collections import defaultdict
    cov = defaultdict(dict)
    for r in csv.DictReader(open(scores_csv, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])
    return {q: c for q, c in cov.items() if base_cond in c and inject_cond in c}


def predict_contestedness(texts, model_path: str | None = None) -> np.ndarray:
    """Public entry point: question texts -> predicted contestedness.

    ``model_path`` defaults to $CONTEST_MODEL. Needs no graph and no anchor —
    that is the whole reason the predictor is keyed on text.
    """
    path = model_path or os.environ.get("CONTEST_MODEL", "contest_ridge.npz")
    return ContestednessPredictor.load(path).predict(list(texts))


def report_route_signal(rows, model_path: str | None = None,
                        base_key: str = "base", inject_key: str = "scout"):
    """Score the predictor as one more candidate signal inside route_signal.py.

    ``rows`` are that script's per-question dicts (qid / base / scout). Wire it
    with ONE line at the end of route_signal.main():

        from alignment.contestedness_predictor import report_route_signal; report_route_signal(rows)
    """
    path = model_path or os.environ.get("CONTEST_MODEL", "contest_ridge.npz")
    if not os.path.exists(path):
        print(f"\n[contestedness] no predictor at {path} — set CONTEST_MODEL; skipping")
        return None
    from evaluation.overton.eval_overtonbench import load_questions
    qtext = dict(load_questions())
    use = [r for r in rows if r.get("qid") in qtext]
    if not use:
        print("\n[contestedness] no rows matched OvertonBench questions; skipping")
        return None
    sig = ContestednessPredictor.load(path).predict([qtext[r["qid"]] for r in use])
    print(f"\n=== predicted contestedness as a routing signal (n={len(use)}) ===")
    return score_help_delta("pred_contested", list(sig),
                            [r[base_key] for r in use],
                            [r[inject_key] for r in use])


# ---------------------------------------------------------------------------
# Self-test: synthetic fixtures where contested/consensus is known by
# construction. The cluster holds the real graphs and embedders, so this is the
# only thing that can be verified locally.
# ---------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    recs: list[QuestionRecord] = []
    tag: dict[str, str] = {}

    def add(kind, i, axes, topic):
        q = f"{kind}{i}"
        recs.append(QuestionRecord(qid=q, text=q, topic=topic, axes=axes))
        tag[q] = kind

    for i in range(12):
        # A consensus: every subgroup piled on option 2 -> no split, low entropy
        base = np.array([.02, .06, .84, .06, .02])
        jit = [np.abs(base + rng.normal(0, 0.01, 5)) for _ in range(4)]
        add("A", i, [[list(v / v.sum()) for v in jit]], f"t{i % 6}")
        # B contested: subgroups at opposite poles -> real split, high entropy
        add("B", i, [[[.80, .15, .03, .01, .01], [.70, .20, .06, .03, .01],
                      [.01, .03, .06, .20, .70], [.01, .01, .03, .15, .80]]],
            f"t{i % 6}")
        # C spread-but-agreed: everyone individually torn, ZERO disagreement.
        # Separates entropy-of-aggregate from divergence — entropy ranks C top,
        # z_level must rank it down with the consensus questions.
        add("C", i, [[[.2, .2, .2, .2, .2] for _ in range(4)]], f"t{i % 6}")
        # E: a one-notch shift; modest in absolute terms, LARGE for a K=5 item
        add("E", i, [[[.05, .70, .20, .04, .01], [.01, .10, .25, .55, .09]]],
            f"t{i % 6}")
    # D (K=2): raw W is much LARGER than anything above, because 2-option splits
    # are huge for EVERY such question — so no single one is unusual within its
    # stratum. This is the "is it BIG vs is it UNUSUAL" case in miniature.
    for i in range(16):
        d = 0.50 + 0.02 * i                              # 0.50 .. 0.80
        add("D", i, [[[0.5 + d / 2, 0.5 - d / 2], [0.5 - d / 2, 0.5 + d / 2]]],
            f"t{i % 6}")

    tgt = compute_targets(recs, n_bins=2, n_null=400, seed=0)
    z, ent, w = tgt["z_level"], tgt["entropy"], tgt["w_mean"]

    def by(k, arr):
        return float(np.mean([arr[i] for i, r in enumerate(recs) if tag[r.qid] == k]))

    # 1. the basic requirement: contested outranks consensus
    assert by("B", z) > by("A", z) + 1.0, (by("B", z), by("A", z))
    assert by("B", w) > by("A", w), (by("B", w), by("A", w))
    # 2. entropy-of-aggregate is a DIFFERENT construct, not a weaker z
    assert by("C", ent) > by("B", ent) > by("A", ent)
    assert by("C", z) < by("B", z), (by("C", z), by("B", z))
    # 3. the design claim: z is NOT a monotone rescale of raw magnitude.
    #    D (K=2) has the larger RAW split; E (K=5) is the more UNUSUAL one.
    assert by("D", w) > by("E", w), (by("D", w), by("E", w))
    assert by("D", z) < by("E", z), (by("D", z), by("E", z))
    print(f"selftest z_level  A={by('A', z):+.2f} B={by('B', z):+.2f} "
          f"C={by('C', z):+.2f} D={by('D', z):+.2f} E={by('E', z):+.2f}")
    print(f"selftest w_mean   A={by('A', w):.3f} B={by('B', w):.3f} "
          f"C={by('C', w):.3f} D={by('D', w):.3f} E={by('E', w):.3f}"
          f"   <- raw ranks D>E; z inverts it")
    print(f"selftest entropy  A={by('A', ent):.3f} B={by('B', ent):.3f} "
          f"C={by('C', ent):.3f}   <- C is max-entropy with zero disagreement")
    print(f"  rank corr(z, w_mean)={spearman(z, w):+.3f}  "
          f"corr(z, entropy)={spearman(z, ent):+.3f}   "
          f"<- both far from 1.0: z is its own target")

    # --- planted-signal predictor -----------------------------------------
    topics = [r.topic for r in recs]
    y = (z - z.mean()) / (z.std() + 1e-9)
    d = 24
    Xs = rng.normal(0, 1, (len(recs), d))
    Xs[:, 0] = 3.0 * y + rng.normal(0, 0.5, len(recs))   # planted direction
    Xn = rng.normal(0, 1, (len(recs), d))                # pure-noise control
    # Floors differ by class on purpose: logistic thresholds y at the top
    # tercile, so it discards within-class order and cannot reach ridge's rank
    # correlation even on a perfectly planted signal.
    for kind, floor in (("ridge", 0.75), ("logistic", 0.60)):
        got = cv_by_topic(Xs, y, topics, kind=kind, alphas=(1., 10.), n_test=2,
                          verbose=False)
        null = cv_by_topic(Xn, y, topics, kind=kind, alphas=(1., 10.), n_test=2,
                           verbose=False)
        print(f"  {kind:<9} planted rho={got['mean']:+.3f}+-{got['sd']:.3f}  "
              f"noise rho={null['mean']:+.3f}+-{null['sd']:.3f}  "
              f"({len(got['folds'])} topic folds, floor {floor})")
        assert got["mean"] > floor, got
        assert abs(null["mean"]) < 0.35, null     # the harness must not leak y

    # --- save/load round trip ---------------------------------------------
    import tempfile
    p = ContestednessPredictor.train(Xs, y, kind="ridge", alpha=1.0)
    fp = os.path.join(tempfile.mkdtemp(), "p.npz")
    p.save(fp)
    assert np.allclose(p.score(Xs), ContestednessPredictor.load(fp).score(Xs))

    # --- graph extraction (the fiddly name parsing / topic walk) -----------
    class G:
        id_to_entity = ["US_Public", "topic:0", "sub:0.0", "q:AB_W1",
                        "ax:AB_W1:POLPARTY", "op:AB_W1:POLPARTY:Rep",
                        "op:AB_W1:POLPARTY:Dem"]
        entity_vocab = {n: i for i, n in enumerate(id_to_entity)}
        entity_types = {0: "root", 1: "topic", 2: "subtopic", 3: "question",
                        4: "axis", 5: "opinion", 6: "opinion"}
        entity_text = {3: "Should X be legal?"}
        children_indices = [[1], [2], [3], [4], [5, 6], [], []]
        opinion_dist = {5: [0.9, 0.1], 6: [0.1, 0.9]}

    got = extract_questions(G(), "opinionqa")
    assert len(got) == 1 and got[0].topic == "topic:0", got
    assert got[0].text == "Should X be legal?" and len(got[0].axes[0]) == 2
    assert abs(question_divergence(got[0]) - 0.8) < 1e-9

    class G2:
        id_to_entity = ["World", "Europe", "France", "op_7_France", "op_7_Japan"]
        entity_vocab = {n: i for i, n in enumerate(id_to_entity)}
        entity_types = {0: "world", 1: "region", 2: "country", 3: "opinion",
                        4: "opinion"}
        entity_text = {3: "Is trade good? [France]", 4: "Is trade good? [Japan]"}
        children_indices = [[1], [2], [3, 4], [], []]
        opinion_dist = {3: [0.6, 0.4], 4: [0.3, 0.7]}

    got2 = extract_questions(G2(), "globalopinionqa")
    assert len(got2) == 1 and got2[0].text == "Is trade good?", got2
    assert abs(question_divergence(got2[0]) - 0.3) < 1e-9

    print("contestedness_predictor selftest OK "
          "(target ordering, z != raw magnitude, planted-signal recovery, "
          "save/load, graph extraction)")


# ---------------------------------------------------------------------------
def _load(dataset: str, seed: int):
    if dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        g = load_opinionqa(split_seed=seed, leakage_safe=True)
    else:
        from data.loaders.globalopinionqa import load_globalopinionqa
        g = load_globalopinionqa(split_seed=seed, leakage_safe=True)
    recs = drop_eval_holdout(extract_questions(g, dataset))
    print(f"  {dataset}: {len(recs)} questions, "
          f"{len({r.topic for r in recs})} topics, "
          f"K in {sorted({r.n_options for r in recs})}")
    return recs


def _main():
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Text-keyed contestedness predictor")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", choices=["opinionqa", "globalopinionqa"],
                    default="opinionqa")
    ap.add_argument("--cross", action="store_true",
                    help="also load the OTHER graph and report transfer both ways")
    ap.add_argument("--target", default="z_level", choices=list(TARGETS) + ["all"])
    ap.add_argument("--model", default="ridge", choices=list(MODELS))
    ap.add_argument("--ground", default="ordinal", choices=list(GROUNDS))
    ap.add_argument("--axis_agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--null_bins", type=int, default=4,
                    help="entropy strata per K; 1 = route_signal's degenerate "
                         "global null (z collapses to an affine map of w_mean)")
    ap.add_argument("--n_null", type=int, default=400)
    ap.add_argument("--alphas", default="1,10,100,1000")
    ap.add_argument("--embed_model", default=MPNET)
    ap.add_argument("--emb_cache", default=None)
    ap.add_argument("--n_test_topics", type=int, default=2)
    ap.add_argument("--goqa_topics", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores", default=None,
                    help="overton_scores_vN.csv — score the PREDICTION against "
                         "the per-question help-delta (evaluation #2)")
    ap.add_argument("--inject_cond", default="scout")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    ground = GROUNDS[args.ground]
    alphas = tuple(float(a) for a in args.alphas.split(","))
    names = [args.dataset]
    if args.cross:
        names += [o for o in ("opinionqa", "globalopinionqa") if o != args.dataset]

    data = {}
    for ds in names:
        recs = _load(ds, args.seed)
        X = embed_texts([r.text for r in recs], args.embed_model,
                        cache=(f"{args.emb_cache}.{ds}.npz" if args.emb_cache else None))
        topics = [r.topic for r in recs]
        if any(t is None for t in topics):              # GOQA has no topic layer
            topics = kmeans_topics(X, args.goqa_topics, args.seed)
        tg = compute_targets(recs, ground=ground, axis_agg=args.axis_agg,
                             n_bins=args.null_bins, n_null=args.n_null,
                             seed=args.seed)
        print(f"  {ds} target diagnostics: corr(z, w_mean)="
              f"{spearman(tg['z_level'], tg['w_mean']):+.3f}  "
              f"corr(z, entropy)={spearman(tg['z_level'], tg['entropy']):+.3f}  "
              f"corr(w_mean, entropy)={spearman(tg['w_mean'], tg['entropy']):+.3f}")
        data[ds] = (recs, X, topics, tg)

    targets = list(TARGETS) if args.target == "all" else [args.target]
    trained = {}
    for tname in targets:
        print(f"\n=========== target={tname}  model={args.model} ===========")
        for ds, (recs, X, topics, tg) in data.items():
            y = tg[tname]
            ok = np.isfinite(y)
            ys = (y[ok] - y[ok].mean()) / (y[ok].std() + 1e-9)
            tk = [t for t, k in zip(topics, ok) if k]
            print(f"  [{ds}] held-out-topic CV ({args.n_test_topics} topics out, "
                  f"n={int(ok.sum())}):")
            cv = cv_by_topic(X[ok], ys, tk, kind=args.model, alphas=alphas,
                             n_test=args.n_test_topics, seed=args.seed)
            print(f"  [{ds}] mean spearman = {cv['mean']:+.3f} +- {cv['sd']:.3f} "
                  f"over {len(cv['folds'])} topic folds")
            a = _pick_alpha(X[ok], ys, tk, list(alphas), args.model)
            trained[(tname, ds)] = (ContestednessPredictor.train(
                X[ok], ys, kind=args.model, alpha=a, target=tname,
                embed_model=args.embed_model), X[ok], ys)

        if args.cross and len(data) == 2:
            for src, dst in ((names[0], names[1]), (names[1], names[0])):
                p = trained[(tname, src)][0]
                _, Xd, yd = trained[(tname, dst)]
                print(f"  transfer {src} -> {dst}: spearman="
                      f"{spearman(p.score(Xd), yd):+.3f}   (two different "
                      f"divergence structures — the strong test)")

        if args.scores:
            cov = _cov_from_csv(args.scores, args.inject_cond)
            from evaluation.overton.eval_overtonbench import load_questions
            qt = dict(load_questions())
            qids = [q for q in sorted(cov) if q in qt]
            p = trained[(tname, args.dataset)][0]
            sig = p.predict([qt[q] for q in qids])
            print(f"  help-delta vs {args.scores} (n={len(qids)}; these "
                  f"questions were EXCLUDED from training):")
            score_help_delta(f"pred[{tname}]", list(sig),
                             [cov[q]["baseline"] for q in qids],
                             [cov[q][args.inject_cond] for q in qids])
            print("   ^ compare with route_signal.py's w_raw = +0.19; a "
                  "predictor is only worth wiring if it clears that.")

    if args.save:
        trained[(targets[0], args.dataset)][0].save(args.save)


if __name__ == "__main__":
    _main()
