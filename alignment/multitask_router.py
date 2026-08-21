"""Route injection with a two-head model: free contestedness labels, scarce deltas.

THE LABEL ASYMMETRY, which is the entire reason this file exists:

  AUXILIARY   contestedness (`z_level`, alignment/contestedness_predictor.py).
              ~4,000 labels and they are FREE -- computed straight from survey
              distributions (ATP ~1,492 + GOQA ~2,500). No generation, no judge,
              one embedding pass.
  TARGET      the injection delta, coverage(injected) - coverage(baseline)
              (scripts/analysis/delta_regressor.py). ~60 per scored run, and
              every one costs TWO generations plus TWO judge passes.

Plentiful auxiliary labels + scarce target labels is exactly the case
auxiliary-task pretraining exists for, and the two targets share their causal
axis: injection helps when the question is contested (+0.31) and hurts when it
is not (-0.45). Oracle per-question routing scores 0.634 vs 0.497 always-
baseline -- a +0.137 gap, over 4x the 0.027 noise floor, currently collected by
nothing. This module tries to collect some of it, and the ceiling is that 0.634.

WHY THE COUPLING IS IN THE PENALTY AND NOT IN THE ARCHITECTURE. The obvious
design -- shared linear trunk T (d x k), two linear heads v_aux, v_tgt -- is
VACUOUS for k >= 2. Only g_aux = T v_aux and g_tgt = T v_tgt affect the two
predictions, and for k >= 2 every pair (g_aux, g_tgt) is realisable by some T.
"Sharing a trunk" would then be two independent ridge regressions with extra
notation. k = 1 is the opposite failure: it forces the two heads to be the SAME
signal up to scale, so the target head can only rescale contestedness. So the
sharing here is weight-space, where it actually binds:

  delta-only    w_t = argmin ||Z_t w - y_t||^2 + a ||w||^2
                the baseline to beat: target head alone, no auxiliary labels.
  pretrain-ft   w_a = ridge on the 4,000 auxiliary labels, then
                w_t = argmin ||Z_t w - y_t||^2 + a ||w - s*w_a||^2
                i.e. fine-tune with the PRETRAINED WEIGHTS as the prior mean
                instead of 0. s is the transfer strength, picked on inner folds.
  joint         w_a = u + d_a, w_t = u + d_t, one objective, loss weight `lam`
                on the auxiliary term and `tie` = (task-specific penalty) /
                (shared penalty). tie -> inf fully ties the heads, tie -> 0
                decouples them. Closed form; see fit_joint.

READ THE COMPARISON CORRECTLY. delta-only is NESTED inside pretrain-ft at s=0,
so pretrain-ft can only lose to it through selection noise. The honest control
is therefore pretrain-ft vs pretrain-ft with the AUXILIARY LABELS SHUFFLED:
shuffling leaves s=0 in the grid and leaves the trunk untouched, so it isolates
whether the auxiliary SIGNAL transferred, rather than whether the extra
hyperparameter flatters itself. --shuffle_aux runs it; --selftest asserts it.

ALL THREE MODES SHARE THE SAME TRUNK. The PCA basis is fit on the auxiliary
INPUTS (no labels), so delta-only already gets the auxiliary representation.
What pretrain-ft adds on top is the auxiliary LABELS, which is the claim under
test. 768 raw dimensions against ~60 target labels is hopeless for any of them.

HOLD OUT BY TOPIC, NEVER BY QUESTION. contestedness_predictor's selftest
measured the cost of getting this wrong: with label=topic, random k-fold scores
1.000 and leave-one-topic-out scores 0.007. A random split reports a perfect and
entirely spurious result. --split random reproduces the inflation on purpose,
beside the topic number, so the gap is visible rather than asserted.

LEAKAGE GUARD. The 60 OvertonBench questions are the delta labels' own eval set,
so auxiliary pretraining on them would leak the target's evaluation into the
target's representation. They are dropped from the auxiliary pool by
alignment.rollout_dataset.load_eval_holdout_texts(), via
contestedness_predictor.drop_eval_holdout -- the one existing guard, reused.

CONTAMINATION, inherited and restated. The delta labels ARE OvertonBench. A
router fit on them and scored on them is fit and evaluated on the same 60
questions; CV removes the crudest form and nothing removes the rest (feature
set, cost ratio, the decision to build this). Every delta-head number below is a
CV ESTIMATE of an upper bound and is printed as one.

    python -m alignment.multitask_router --selftest
    OPINIONQA_DIR=... python -m alignment.multitask_router \
        --scores overton_scores_v6.csv --mode pretrain-ft --save mtr_v6.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "scripts", "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reused, not reimplemented. The auxiliary target, its embedder, its topic-fold
# partition and its rank metric all come from the sibling that owns them.
from alignment.contestedness_predictor import (  # noqa: E402
    MPNET, compute_targets, drop_eval_holdout, embed_texts, extract_questions,
    kmeans_topics, spearman, topic_folds)
# The routing metric, the cost-asymmetry threshold and the two exact anchors come
# from the sibling that owns THEM, so the headline lands on the same scale as
# delta_regressor's own table.
from delta_regressor import (  # noqa: E402
    Row, anchors, build_rows, decision_threshold, route_metrics)
from evaluation.overton.route_signal import _corr  # noqa: E402

MODES = ("delta-only", "pretrain-ft", "joint")


# ---------------------------------------------------------------------------
# Shared encoder: frozen embeddings -> standardise -> PCA trunk
# ---------------------------------------------------------------------------
@dataclass
class Trunk:
    """Standardiser + PCA basis, fit on the AUXILIARY inputs only.

    Two reasons this is not optional. (1) ~60 target labels against 768 frozen
    embedding dimensions cannot identify anything; the auxiliary pool is the only
    place a representation can come from. (2) It makes the joint solve a 3p x 3p
    system instead of 3*768, which is what lets nested CV run at all.

    Fit on INPUTS, no labels, and the auxiliary pool has already had the 60
    OvertonBench questions removed -- so this step cannot leak the delta labels'
    evaluation set into the representation.
    """
    mu: np.ndarray
    sd: np.ndarray
    P: np.ndarray                      # (d, p) top-p right singular vectors
    var_kept: float = float("nan")

    @classmethod
    def fit(cls, X: np.ndarray, n_comp: int) -> "Trunk":
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - mu) / sd
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        p = min(n_comp, Vt.shape[0])
        kept = float((s[:p] ** 2).sum() / (s ** 2).sum()) if s.size else float("nan")
        return cls(mu=mu, sd=sd, P=Vt[:p].T, var_kept=kept)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.P

    @property
    def p(self) -> int:
        return self.P.shape[1]


# ---------------------------------------------------------------------------
# Heads. Closed-form ridge with a PRIOR MEAN -- the prior is what "pretrained"
# means for a linear model (L2-SP: shrink toward w_aux, not toward 0).
# ---------------------------------------------------------------------------
def _ridge(Z: np.ndarray, y: np.ndarray, alpha: float,
           prior: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    zm, ym = Z.mean(0), float(y.mean())
    Zc = Z - zm
    p = Z.shape[1]
    rhs = Zc.T @ (y - ym)
    if prior is not None:
        rhs = rhs + alpha * prior
    w = np.linalg.solve(Zc.T @ Zc + alpha * np.eye(p), rhs)
    return w, float(ym - zm @ w)


@dataclass
class AuxStats:
    """Sufficient statistics of the auxiliary task, computed ONCE.

    The auxiliary set is never folded (its labels are independent of the delta
    labels), so its Gram matrix is constant across every fold and every
    hyperparameter. Caching it turns the joint fit from a 4,000-row least squares
    into a p x p solve, which is the difference between nested CV running in
    seconds and not running.
    """
    S: np.ndarray                      # Zc' Zc      (p, p)
    c: np.ndarray                      # Zc' (y-ym)  (p,)
    zm: np.ndarray
    ym: float
    n: int

    @classmethod
    def build(cls, Z: np.ndarray, y: np.ndarray) -> "AuxStats":
        zm, ym = Z.mean(0), float(y.mean())
        Zc = Z - zm
        return cls(S=Zc.T @ Zc, c=Zc.T @ (y - ym), zm=zm, ym=ym, n=len(y))

    def head(self, alpha: float) -> tuple[np.ndarray, float]:
        w = np.linalg.solve(self.S + alpha * np.eye(len(self.c)), self.c)
        return w, float(self.ym - self.zm @ w)


def fit_delta_only(Zt, yt, *, alpha: float, **_) -> dict:
    """Target head alone. The baseline the multi-task premise has to beat."""
    w, b = _ridge(Zt, yt, alpha)
    return {"w_t": w, "b_t": b}


def fit_pretrain_ft(Zt, yt, *, alpha: float, s: float, w_aux: np.ndarray,
                    **_) -> dict:
    """Fine-tune the target head from the pretrained auxiliary weights.

    The prior is rescaled so `s` is unit-free: w_aux is normalised to give the
    TRAIN fold's projection unit sd, then multiplied by sd(y_t). Without this,
    s would silently absorb the scale gap between a standardised z_level and a
    coverage delta living on +-0.4, and the grid would mean nothing.

    s = 0 reproduces fit_delta_only exactly. That nesting is why the honest
    control for this mode is the SHUFFLED-auxiliary run, not delta-only.
    """
    proj_sd = float(np.std(Zt @ w_aux))
    prior = np.zeros_like(w_aux)
    if s and proj_sd > 1e-12:
        prior = (s * float(np.std(yt)) / proj_sd) * w_aux
    w, b = _ridge(Zt, yt, alpha, prior=prior)
    return {"w_t": w, "b_t": b}


def fit_joint(Zt, yt, *, alpha: float, lam: float, tie: float,
              aux: AuxStats, **_) -> dict:
    """Both heads in one objective, with a loss weight on the auxiliary term.

    w_a = u + d_a, w_t = u + d_t and

        min  lam/n_a ||Z_a w_a - y_a||^2 + (1-lam)/n_t ||Z_t w_t - y_t||^2
             + alpha ||u||^2 + alpha*tie (||d_a||^2 + ||d_t||^2)

    (Evgeniou-Pontil regularised multi-task least squares.) The heads are tied
    through u and separated by tie: tie -> inf collapses both heads onto the
    shared vector, tie -> 0 makes them independent ridges. Normal equations are
    built from Gram blocks, so the 4,000 auxiliary rows never materialise here.
    """
    p = len(aux.c)
    ztm, ytm = Zt.mean(0), float(yt.mean())
    Ztc = Zt - ztm
    St, ct = Ztc.T @ Ztc, Ztc.T @ (yt - ytm)
    wa2 = lam / max(aux.n, 1)
    wt2 = (1.0 - lam) / max(len(yt), 1)
    Sa, ca = wa2 * aux.S, wa2 * aux.c
    Sb, cb = wt2 * St, wt2 * ct

    A = np.zeros((3 * p, 3 * p))
    A[:p, :p] = Sa + Sb
    A[:p, p:2 * p] = A[p:2 * p, :p] = A[p:2 * p, p:2 * p] = Sa
    A[:p, 2 * p:] = A[2 * p:, :p] = A[2 * p:, 2 * p:] = Sb
    A[:p, :p] += alpha * np.eye(p)
    A[p:2 * p, p:2 * p] += alpha * tie * np.eye(p)
    A[2 * p:, 2 * p:] += alpha * tie * np.eye(p)
    rhs = np.concatenate([ca + cb, ca, cb])
    th = np.linalg.solve(A, rhs)
    u, d_a, d_t = th[:p], th[p:2 * p], th[2 * p:]
    w_t = u + d_t
    w_a = u + d_a
    return {"w_t": w_t, "b_t": float(ytm - ztm @ w_t),
            "w_a": w_a, "b_a": float(aux.ym - aux.zm @ w_a),
            "shared_frac": float(np.linalg.norm(u) /
                                 (np.linalg.norm(u) + np.linalg.norm(d_t) + 1e-12))}


FITTERS = {"delta-only": fit_delta_only, "pretrain-ft": fit_pretrain_ft,
           "joint": fit_joint}


def mode_grid(mode: str, args) -> list[dict]:
    """Hyperparameter grid per mode. Every grid contains the delta-only
    solution, so no mode is handicapped by having fewer knobs."""
    if mode == "delta-only":
        return [{"alpha": a} for a in args.alphas]
    if mode == "pretrain-ft":
        return [{"alpha": a, "s": s} for a in args.alphas for s in args.shrinks]
    return [{"alpha": a, "lam": l, "tie": t} for a in args.alphas
            for l in args.joint_lams for t in args.joint_ties]


# ---------------------------------------------------------------------------
# Folds: by TOPIC, with the random split kept as the inflation comparator
# ---------------------------------------------------------------------------
def question_folds(qids, topic_of, *, split="topic", n_test=2, seed=0):
    """qid -> fold index, plus the fold count.

    `topic` blocks near-paraphrases together; contestedness_predictor measured
    what happens without it (random k-fold 1.000 vs leave-one-topic-out 0.007 on
    a label that IS the topic). `random` uses the SAME number of folds so the
    only thing that changes between the two numbers is the grouping.
    """
    qids = list(qids)
    blocks = topic_folds([topic_of[q] for q in qids], n_test, seed)
    if split == "topic":
        fold_of = {t: i for i, blk in enumerate(blocks) for t in blk}
        return {q: fold_of[topic_of[q]] for q in qids}, len(blocks)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(qids))
    return {qids[perm[i]]: i % len(blocks) for i in range(len(qids))}, len(blocks)


def _inner_select(mode, grid, Zt, yt, qids, topic_of, aux_ctx, seed):
    """Pick hyperparameters on TRAIN-fold topics only, by MSE.

    Selecting on the outer fold would report a number tuned on its own test set;
    MSE rather than the routing metric because an inner fold can hold as few as
    a handful of questions and the routing metric over a handful of questions is
    noise. Same argument, same choice, as delta_regressor.cv_predict.
    """
    if len(grid) == 1:
        return grid[0]
    uniq = sorted({topic_of[q] for q in qids})
    if len(uniq) < 3:
        return grid[len(grid) // 2]
    inner = {t: i % 3 for i, t in enumerate(uniq)}
    f = np.array([inner[topic_of[q]] for q in qids])
    best, best_mse = grid[0], float("inf")
    for params in grid:
        err = []
        for k in range(3):
            te, tr = f == k, f != k
            if not te.any() or tr.sum() < 5:
                continue
            out = FITTERS[mode](Zt[tr], yt[tr], **params, **aux_ctx)
            err.extend(((Zt[te] @ out["w_t"] + out["b_t"] - yt[te]) ** 2).tolist())
        m = float(np.mean(err)) if err else float("inf")
        if m < best_mse:
            best, best_mse = params, m
    return best


def cv_oof(mode, Zt, yt, rows, topic_of, aux_ctx, args, *, split="topic"):
    """Out-of-fold delta predictions for one mode. Returns (oof, params used).

    The auxiliary head is fit OUTSIDE this loop on purpose: auxiliary labels are
    survey divergences, not coverage deltas, so no held-out delta enters it. What
    the guard has to prevent is the auxiliary pool containing the held-out
    QUESTIONS, and drop_eval_holdout already removed all 60 of them.
    """
    qids = sorted({r.qid for r in rows})
    fold_of, n_folds = question_folds(qids, topic_of, split=split,
                                      n_test=args.n_test_topics, seed=args.seed)
    f = np.array([fold_of[r.qid] for r in rows])
    q = np.array([r.qid for r in rows])
    grid = mode_grid(mode, args)
    oof = np.zeros(len(rows))
    used = []
    for k in range(n_folds):
        te, tr = f == k, f != k
        if not te.any():
            continue
        if tr.sum() < Zt.shape[1] // 4 + 3:
            oof[te] = yt[tr].mean() if tr.any() else 0.0
            continue
        params = _inner_select(mode, grid, Zt[tr], yt[tr], q[tr].tolist(),
                               topic_of, aux_ctx, args.seed)
        out = FITTERS[mode](Zt[tr], yt[tr], **params, **aux_ctx)
        oof[te] = Zt[te] @ out["w_t"] + out["b_t"]
        used.append(params)
    return oof, used, n_folds


# ---------------------------------------------------------------------------
# Evaluation: delta_regressor's metric, verbatim
# ---------------------------------------------------------------------------
def evaluate(rows, oof, args, label, n_folds, split="topic", quiet=False) -> dict:
    y = np.array([r.delta for r in rows])
    resid_sd = float(np.std(y - oof))
    thr = decision_threshold(resid_sd, args.cost_ratio)
    m = route_metrics(rows, oof, thr)
    m.update(corr=_corr(list(oof), list(y)), resid_sd=resid_sd, label=label,
             split=split, n_folds=n_folds)
    if not quiet:
        print(f"\n=== {label}   [{n_folds}-fold, held out by {split.upper()}, "
              f"CV ESTIMATE] ===")
        print(f"  OOF corr={m['corr']:+.3f}  resid_sd={resid_sd:.3f}  "
              f"thr={thr:+.4f}  inject_rate={m['inject_rate']:.2f}   (NOT the headline)")
        print(f"  always_base={m['always_base']:.4f}  oracle={m['oracle']:.4f}  "
              f"ROUTED={m['achieved']:.4f}")
        print(f"  >>> ORACLE GAP RECOVERED = {m['recovered']:+.3f}")
    return m


def length_features(rows, texts_by_qid) -> np.ndarray:
    """LENGTH-ONLY control, in the spirit of delta_regressor's --features
    length_only. Length has already reversed one conclusion in this project: if
    it recovers the gap, the result is about verbosity and the embeddings are
    decoration. Uses the baseline ANSWER when responses were loaded, else the
    question text, so the control exists on either path.
    """
    try:
        from alignment.reward import split_units
    except Exception:                                     # noqa: BLE001
        def split_units(t):
            return [s for s in re.split(r"(?<=[.!?])\s+", t or "") if s.strip()]
    X = []
    for r in rows:
        t = r.feats.get("_base_text") or texts_by_qid.get(r.qid, "")
        w = (t or "").split()
        u = split_units(t or "")
        X.append([float(len(w)), float(len(t or "")), float(len(u)),
                  float(len(w) / len(u)) if u else 0.0])
    return np.asarray(X, dtype=float)


# ---------------------------------------------------------------------------
# The saved router
# ---------------------------------------------------------------------------
@dataclass
class MultiTaskRouter:
    """Frozen embedder + trunk + two heads + the decision threshold.

    `route` is the point of the whole file: a per-question inject/skip decision
    whose ceiling is the 0.634 oracle, taken with the cost-asymmetric threshold
    rather than at 0 (a wrong inject costs ~0.45, a wrong skip forgoes ~0.31).
    """
    trunk: Trunk
    w_t: np.ndarray
    b_t: float
    w_a: np.ndarray | None = None
    b_a: float = 0.0
    thr: float = 0.0
    resid_sd: float = float("nan")
    mode: str = "pretrain-ft"
    embed_model: str = MPNET

    def predict_delta(self, texts) -> np.ndarray:
        Z = self.trunk.transform(embed_texts(list(texts), self.embed_model))
        return Z @ self.w_t + self.b_t

    def predict_contestedness(self, texts) -> np.ndarray:
        if self.w_a is None:
            raise ValueError("no auxiliary head (mode=delta-only)")
        Z = self.trunk.transform(embed_texts(list(texts), self.embed_model))
        return Z @ self.w_a + self.b_a

    def route(self, texts) -> np.ndarray:
        return self.predict_delta(texts) > self.thr

    def save(self, path: str) -> None:
        np.savez(path, mu=self.trunk.mu, sd=self.trunk.sd, P=self.trunk.P,
                 w_t=self.w_t, w_a=(self.w_a if self.w_a is not None
                                    else np.zeros(0)),
                 meta=json.dumps({"b_t": self.b_t, "b_a": self.b_a,
                                  "thr": self.thr, "resid_sd": self.resid_sd,
                                  "mode": self.mode,
                                  "embed_model": self.embed_model,
                                  "var_kept": self.trunk.var_kept}))
        print(f"  saved router -> {path}")

    @classmethod
    def load(cls, path: str) -> "MultiTaskRouter":
        z = np.load(path, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        wa = z["w_a"]
        return cls(trunk=Trunk(mu=z["mu"], sd=z["sd"], P=z["P"],
                               var_kept=m.get("var_kept", float("nan"))),
                   w_t=z["w_t"], b_t=m["b_t"],
                   w_a=(wa if wa.size else None), b_a=m["b_a"], thr=m["thr"],
                   resid_sd=m["resid_sd"], mode=m["mode"],
                   embed_model=m["embed_model"])


# ---------------------------------------------------------------------------
# Real-path data loading
# ---------------------------------------------------------------------------
def load_aux(datasets, args):
    """~4,000 free labels: texts, standardised z_level, dataset-prefixed topics.

    z_level is computed and standardised PER DATASET before pooling. Its strata
    are (K, entropy-bin) within one graph, so an ATP z and a GOQA z are not on a
    common scale; pooling the raw values would make the auxiliary target partly a
    dataset indicator.
    """
    texts, ys, topics = [], [], []
    for ds in datasets:
        if ds == "opinionqa":
            from data.loaders.opinionqa import load_opinionqa
            g = load_opinionqa(split_seed=args.seed, leakage_safe=True)
        else:
            from data.loaders.globalopinionqa import load_globalopinionqa
            g = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
        # THE LEAKAGE GUARD: the 60 OvertonBench questions are the delta labels'
        # own eval set. Pretraining on them would put the target's evaluation
        # into the shared representation.
        recs = drop_eval_holdout(extract_questions(g, ds))
        tg = compute_targets(recs, n_bins=args.null_bins, n_null=args.n_null,
                             seed=args.seed)
        y = tg["z_level"]
        ok = np.isfinite(y)
        y = (y[ok] - y[ok].mean()) / (y[ok].std() + 1e-9)
        rs = [r for r, k in zip(recs, ok) if k]
        tp = [r.topic for r in rs]
        if any(t is None for t in tp):                    # GOQA has no topic layer
            tp = kmeans_topics(
                embed_texts([r.text for r in rs], args.embed_model,
                            cache=(f"{args.emb_cache}.{ds}.npz"
                                   if args.emb_cache else None)),
                args.goqa_topics, args.seed)
        texts += [r.text for r in rs]
        ys.append(y)
        topics += [f"{ds}:{t}" for t in tp]
        print(f"  aux[{ds}]: {len(rs)} questions, {len(set(tp))} topics")
    return texts, np.concatenate(ys), topics


def load_target(args):
    """Delta labels + question text. Rows come from delta_regressor.build_rows,
    so the (run, question, condition) semantics and the run tagging are theirs."""
    from evaluation.overton.eval_overtonbench import load_questions
    inject = {c.strip() for c in args.inject_conds.split(",") if c.strip()}
    rows, meta = build_rows(args.scores, args.responses, args.baseline_cond, inject)
    qtext = dict(load_questions())
    keep = []
    for r in rows:
        if r.qid not in qtext:
            continue
        rb = r.feats.pop("_resp_base", {}) or {}
        r.feats.pop("_resp_inj", None)
        r.feats["_base_text"] = rb.get("response") or ""
        keep.append(r)
    print("=== label provenance ===")
    for tag, m in meta.items():
        print(f"  {tag:<6} scores={os.path.basename(m['scores'])} "
              f"n_q={m['n_q']} conditions={m['conditions']}")
    if len(meta) > 1:
        print("  POOLED across runs: the delta is assumed to mean the same thing "
              "under different prompt variants. Pooling is a CHOICE.")
    print(f"  {len(keep)} delta labels over {len({r.qid for r in keep})} questions"
          f"   <- this is the scarce side of the asymmetry")
    return keep, qtext


# ---------------------------------------------------------------------------
# Self-test. Graphs and embedders are cluster-side, so a planted fixture is the
# only thing verifiable locally. It plants ONE latent factor read by both tasks.
# ---------------------------------------------------------------------------
def _fixture(seed=0, n_aux=1500, n_tgt=60, d=128, n_topics=10, n_nuis=8):
    """Synthetic (aux, target) pair sharing a planted latent contestedness axis.

    Built to mirror the three things that make the real problem hard:
      * the shared axis is NOT the dominant direction of the embedding -- eight
        nuisance factors carry 3x its amplitude, so 60 target labels cannot find
        it alone but 1,500 auxiliary labels pin it down;
      * each topic has its OWN direction and its own delta offset, so a random
        split can memorise topic identity and a topic split cannot -- the
        inflation this file refuses to report;
      * the delta spread matches the measured +0.31 / -0.45 regime.
    """
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.normal(0, 1, (d, 1 + n_nuis + n_topics)))[0]
    v_s, V_n, V_t = V[:, 0], V[:, 1:1 + n_nuis], V[:, 1 + n_nuis:]
    s_topic = rng.normal(0, 1, n_topics)
    off = rng.normal(0, 0.30, n_topics)          # per-topic delta offset

    def draw(n):
        k = rng.integers(0, n_topics, n)
        s = 0.7 * s_topic[k] + 0.7 * rng.normal(0, 1, n)
        X = (np.outer(s, v_s)
             + 5.0 * rng.normal(0, 1, (n, n_nuis)) @ V_n.T
             + 1.2 * V_t[:, k].T
             + 0.5 * rng.normal(0, 1, (n, d)))
        return X, s, k

    Xa, sa, _ = draw(n_aux)
    ya = sa + rng.normal(0, 0.25, n_aux)         # auxiliary: reads the latent
    ta = [f"t{i}" for i in rng.integers(0, n_topics, n_aux)]

    Xt, st, kt = draw(n_tgt)
    delta = 0.30 * st + off[kt] + rng.normal(0, 0.10, n_tgt)
    base = np.clip(rng.normal(0.50, 0.12, n_tgt), 0.02, 0.95)
    rows = [Row(run="vS", qid=i + 1, condition="scout", base=float(base[i]),
                inj=float(np.clip(base[i] + delta[i], 0.0, 1.0)))
            for i in range(n_tgt)]
    topic_of = {i + 1: f"t{kt[i]}" for i in range(n_tgt)}
    return (Xa, ya, ta), (Xt, rows, topic_of)


class _A:
    """Minimal args for the fixture runs."""
    alphas = [0.3, 1.0, 3.0, 10.0, 30.0]
    shrinks = [0.0, 0.25, 0.5, 0.75, 1.0]
    joint_lams = [0.3, 0.6, 0.9]
    joint_ties = [1.0, 10.0]
    n_test_topics = 2
    cost_ratio = 0.45 / 0.31
    seed = 0


_ARMS = ("delta-only", "pretrain-ft", "joint",
         "pretrain-ft[shuffled aux]", "joint[shuffled aux]",
         "delta-only [RANDOM split]")


def _run_fixture(seed: int, n_comp: int, verbose: bool = False) -> dict:
    """One fixture draw, all arms, leave-one-topic-out. Returns label -> metrics."""
    (Xa, ya, _ta), (Xt, rows, topic_of) = _fixture(seed=seed)
    args = _A()
    args.seed = seed
    trunk = Trunk.fit(Xa, n_comp)                 # AUX INPUTS ONLY, no labels
    Za, Zt = trunk.transform(Xa), trunk.transform(Xt)
    yt = np.array([r.delta for r in rows])

    if verbose:
        print(f"  aux n={len(ya)}   target n={len(rows)}   "
              f"asymmetry {len(ya) / len(rows):.0f}:1   (the real one is ~4000:60)")
        print(f"  trunk: {Xa.shape[1]} -> {trunk.p} dims, "
              f"{trunk.var_kept:.3f} of input variance, fit on AUX INPUTS")
        anchors(rows)

    aux = AuxStats.build(Za, ya)
    w_aux, _ = aux.head(alpha=10.0)
    # shuffled auxiliary labels: the trunk is untouched and s=0 stays in the
    # grid, so ONLY the transferred signal is destroyed.
    rng = np.random.default_rng(seed + 991)
    aux_sh = AuxStats.build(Za, ya[rng.permutation(len(ya))])
    w_sh, _ = aux_sh.head(alpha=10.0)

    plan = {"delta-only": ("delta-only", {}, "topic"),
            "pretrain-ft": ("pretrain-ft", {"w_aux": w_aux}, "topic"),
            "joint": ("joint", {"aux": aux}, "topic"),
            "pretrain-ft[shuffled aux]": ("pretrain-ft", {"w_aux": w_sh}, "topic"),
            "joint[shuffled aux]": ("joint", {"aux": aux_sh}, "topic"),
            # the inflation demo, on the arm with no auxiliary confound
            "delta-only [RANDOM split]": ("delta-only", {}, "random")}
    got = {}
    for label in _ARMS:
        mode, ctx, split = plan[label]
        oof, _, nf = cv_oof(mode, Zt, yt, rows, topic_of, ctx, args, split=split)
        got[label] = evaluate(rows, oof, args, label, nf, split, quiet=not verbose)
    return got


def _selftest(n_rep: int = 12, n_comp: int = 32) -> bool:
    """REPLICATED fixture draws, asserted on the mean.

    Not a style choice. The target set is 60 questions in 5 folds, so each fold's
    routing number is read off ~12 questions -- delta_regressor.py's own note is
    that the metric there "moves by more than the whole +0.137 gap on resampling".
    A single draw therefore decides nothing, and asserting on one would make this
    test a coin flip rather than a check on the machinery. Per-draw win counts are
    printed beside the means so the spread stays visible.
    """
    print("=== SELFTEST: synthetic fixtures, one latent factor shared by both "
          "tasks ===")
    print(f"  {n_rep} independent fixture draws; assertions are on the MEAN "
          f"because n=60 in 5 folds is noise-dominated per draw")
    reps = [_run_fixture(s, n_comp, verbose=(s == 0)) for s in range(n_rep)]
    rec = {k: np.array([r[k]["recovered"] for r in reps]) for k in _ARMS}

    print(f"\n  {'arm':<28}{'mean recov':>12}{'sd':>8}{'min':>8}{'max':>8}"
          f"{'>delta-only':>13}")
    for k in _ARMS:
        wins = int((rec[k] > rec["delta-only"]).sum())
        print(f"  {k:<28}{rec[k].mean():>+12.3f}{rec[k].std():>8.3f}"
              f"{rec[k].min():>+8.3f}{rec[k].max():>+8.3f}{wins:>10}/{n_rep}")

    ok = True

    def check(cond, good, bad):
        nonlocal ok
        print(("  [ok] " + good) if cond else ("  [FAIL] " + bad))
        ok = ok and bool(cond)

    d0 = rec["delta-only"].mean()
    dp = rec["pretrain-ft"].mean()
    sh = rec["pretrain-ft[shuffled aux]"].mean()
    rd = rec["delta-only [RANDOM split]"].mean()
    wins = int((rec["pretrain-ft"] > rec["delta-only"]).sum())
    print("\n=== assertions ===")
    # 1. the multi-task premise, on data where it is true by construction.
    #    PAIRED as well as averaged: the two arms see the SAME draw, so the
    #    per-draw win count is the stronger statement and the draw-to-draw sd
    #    (~0.23) does not enter it.
    check(dp > d0 + 0.10 and wins >= (2 * n_rep + 2) // 3,
          f"pretrain-ft {dp:+.3f} beats delta-only {d0:+.3f} by >0.10 on the mean "
          f"and on {wins}/{n_rep} paired draws, when the tasks share a latent factor",
          f"pretrain-ft {dp:+.3f} vs delta-only {d0:+.3f} ({wins}/{n_rep} paired "
          f"wins) -- the transfer machinery is broken")
    # 2. THE CONTROL. delta-only is NESTED inside pretrain-ft at s=0, so the real
    #    question is whether the auxiliary SIGNAL transferred, not whether an
    #    extra hyperparameter flatters itself. Shuffling answers exactly that.
    check(sh <= d0 + 0.05,
          f"shuffled-aux pretraining does NOT help ({sh:+.3f} vs delta-only "
          f"{d0:+.3f}): assertion 1 is the auxiliary signal, not the extra knob",
          f"shuffled-aux pretraining 'helped' ({sh:+.3f} vs {d0:+.3f}) -- "
          f"pretraining is flattering itself and assertion 1 means nothing")
    # 3. the split. Every number above is leave-one-topic-out; this is what a
    #    random split would have reported on the same data.
    check(rd > d0 + 0.05,
          f"random split INFLATES to {rd:+.3f} vs {d0:+.3f} held out by topic -- "
          f"topic grouping is load-bearing, not decoration",
          f"random split {rd:+.3f} showed no inflation over {d0:+.3f}; the "
          f"fixture's topic structure is too weak to demonstrate the guard")

    best = max(MODES, key=lambda m: rec[m].mean())
    print(f"\n  winner on the fixture: {best} ({rec[best].mean():+.3f} of the "
          f"planted gap, vs delta-only {d0:+.3f})")
    print("  SCOPE: this verifies the MACHINERY and the controls on data where the "
          "premise is TRUE BY CONSTRUCTION. It says nothing about whether real "
          "contestedness and the real delta share structure -- only a cluster run "
          "on real labels can say that, and if pretrain-ft loses there the "
          "multi-task premise is wrong and the PR should say so.")
    print("\nSELFTEST: " + ("PASS" if ok else "FAIL"))
    return ok


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-task router: free contestedness labels -> scarce delta labels")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reps", type=int, default=12,
                    help="--selftest only: independent fixture draws. The mean is "
                         "asserted, not any single draw: 60 questions in 5 folds "
                         "is noise-dominated.")
    # target side
    ap.add_argument("--scores", action="append", default=[],
                    help="repeatable overton_scores_vN.csv (the SCARCE labels)")
    ap.add_argument("--responses", action="append", default=[])
    ap.add_argument("--baseline_cond", default="baseline")
    ap.add_argument("--inject_conds", default="scout")
    ap.add_argument("--cost_ratio", type=float, default=0.45 / 0.31,
                    help="|wrong inject| / |wrong skip|; 1.452 = measured 0.45/0.31")
    # auxiliary side
    ap.add_argument("--aux_datasets", default="opinionqa,globalopinionqa",
                    help="pooled for ~4,000 free labels; z_level standardised per "
                         "dataset before pooling")
    ap.add_argument("--null_bins", type=int, default=4)
    ap.add_argument("--n_null", type=int, default=400)
    ap.add_argument("--goqa_topics", type=int, default=12)
    ap.add_argument("--shuffle_aux", action="store_true",
                    help="CONTROL: destroy the auxiliary signal, keep everything "
                         "else. The honest comparator for pretrain-ft.")
    # model
    ap.add_argument("--mode", default="pretrain-ft", choices=list(MODES))
    ap.add_argument("--compare", default="delta-only,pretrain-ft,joint",
                    help="modes scored side by side ('' = only --mode)")
    ap.add_argument("--n_comp", type=int, default=64,
                    help="PCA trunk width. 768 raw dims vs ~60 target labels is "
                         "hopeless; keep this small.")
    ap.add_argument("--alphas", default="0.3,1,3,10,30,100")
    ap.add_argument("--shrinks", default="0,0.25,0.5,0.75,1.0",
                    help="pretrain-ft transfer strength; 0 = delta-only")
    ap.add_argument("--joint_lams", default="0.3,0.6,0.9",
                    help="joint auxiliary loss weight")
    ap.add_argument("--joint_ties", default="1,10",
                    help="joint (task-specific penalty)/(shared penalty)")
    ap.add_argument("--embed_model", default=MPNET)
    ap.add_argument("--emb_cache", default=None)
    ap.add_argument("--target_topics", type=int, default=10,
                    help="k-means pseudo-topics over the eval questions; they have "
                         "no topic layer and near-paraphrases must not split")
    ap.add_argument("--n_test_topics", type=int, default=2)
    ap.add_argument("--split", default="topic", choices=["topic", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None)
    ap.add_argument("--out", default=None, help="csv of per-row OOF predictions")
    args = ap.parse_args()
    for k in ("alphas", "shrinks", "joint_lams", "joint_ties"):
        setattr(args, k, [float(v) for v in getattr(args, k).split(",") if v.strip()])

    if args.selftest:
        return 0 if _selftest(n_rep=args.reps) else 1
    if not args.scores:
        ap.error("--scores is required (or --selftest)")
    missing = [p for p in args.scores if not os.path.exists(p)]
    if missing:
        print(f"[warn] missing, skipped: {missing}")
    args.scores = [p for p in args.scores if os.path.exists(p)]
    if not args.scores:
        ap.error("none of the --scores files exist")

    # --- the scarce side ---------------------------------------------------
    rows, qtext = load_target(args)
    if not rows:
        print("no (baseline, injected) pairs found")
        return 1
    yt = np.array([r.delta for r in rows])

    # --- the free side -----------------------------------------------------
    print("\n=== auxiliary labels (free: survey distributions, no generation, "
          "no judge) ===")
    aux_texts, ya, aux_topics = load_aux(
        [d.strip() for d in args.aux_datasets.split(",") if d.strip()], args)
    print(f"  aux total {len(ya)} vs target {len(rows)}  -> "
          f"{len(ya) / max(len(rows), 1):.0f}:1 label asymmetry")
    if args.shuffle_aux:
        ya = ya[np.random.default_rng(args.seed + 991).permutation(len(ya))]
        print("  !! --shuffle_aux: auxiliary labels PERMUTED (control run)")

    # --- shared encoder ----------------------------------------------------
    Xa = embed_texts(aux_texts, args.embed_model,
                     cache=(f"{args.emb_cache}.aux.npz" if args.emb_cache else None))
    qids = sorted({r.qid for r in rows})
    Xq = embed_texts([qtext[q] for q in qids], args.embed_model,
                     cache=(f"{args.emb_cache}.tgt.npz" if args.emb_cache else None))
    trunk = Trunk.fit(Xa, args.n_comp)
    print(f"  trunk: {Xa.shape[1]} -> {trunk.p} dims ({trunk.var_kept:.3f} of "
          f"input variance), fit on AUX INPUTS only")
    Za = trunk.transform(Xa)
    Zq = dict(zip(qids, trunk.transform(Xq)))
    Zt = np.array([Zq[r.qid] for r in rows])

    # target-side topics: OvertonBench has no topic layer, so cluster the
    # question embeddings the same way contestedness_predictor clusters GOQA.
    tlab = kmeans_topics(np.array([Zq[q] for q in qids]), args.target_topics,
                         args.seed)
    topic_of = dict(zip(qids, tlab))
    print(f"  target topics: {len(set(tlab))} k-means clusters over {len(qids)} "
          f"questions (held out in blocks of {args.n_test_topics})")

    # --- auxiliary head, and whether it transfers at all -------------------
    aux = AuxStats.build(Za, ya)
    blocks = topic_folds(aux_topics, 2, args.seed)
    tarr = np.array(aux_topics)
    rhos = []
    for held in blocks:
        te = np.isin(tarr, held)
        if te.sum() < 5 or (~te).sum() < 20:
            continue
        w, b = AuxStats.build(Za[~te], ya[~te]).head(10.0)
        rhos.append(spearman(Za[te] @ w + b, ya[te]))
    print(f"  aux head, held-out-TOPIC spearman = "
          f"{float(np.nanmean(rhos)):+.3f} +- {float(np.nanstd(rhos)):.3f} over "
          f"{len(rhos)} folds")
    print("   ^ if this is ~0 the auxiliary head learned nothing and pretraining "
          "has nothing to transfer; stop reading here.")
    w_aux, _ = aux.head(10.0)
    ctx = {"delta-only": {}, "pretrain-ft": {"w_aux": w_aux}, "joint": {"aux": aux}}

    anchors(rows)

    modes = [m.strip() for m in args.compare.split(",") if m.strip()] or [args.mode]
    if args.mode not in modes:
        modes.insert(0, args.mode)
    results = {}
    headline = None
    for m in modes:
        if m not in MODES:
            continue
        oof, used, nf = cv_oof(m, Zt, yt, rows, topic_of, ctx[m], args,
                               split=args.split)
        met = evaluate(rows, oof, args, m, nf, args.split)
        print(f"  params chosen per fold: "
              f"{sorted({tuple(sorted(p.items())) for p in used})}")
        results[m] = met
        if m == args.mode:
            headline = (met, oof)

    # LENGTH-ONLY CONTROL
    Zl = length_features(rows, qtext)
    oof_l, _, nf_l = cv_oof("delta-only", Zl, yt, rows, topic_of, {}, args,
                            split=args.split)
    results["length_only"] = evaluate(rows, oof_l, args, "length_only CONTROL",
                                      nf_l, args.split)

    # THE SPLIT CONTRAST
    other = "random" if args.split == "topic" else "topic"
    oof_o, _, nf_o = cv_oof(args.mode, Zt, yt, rows, topic_of, ctx[args.mode], args,
                            split=other)
    alt = evaluate(rows, oof_o, args, f"{args.mode} [{other.upper()} split]", nf_o,
                   other)

    # --- verdict -----------------------------------------------------------
    print("\n=== summary: fraction of the oracle gap recovered (ALL are CV "
          "ESTIMATES on the eval set) ===")
    print(f"  {'mode':<22}{'recovered':>11}{'routed':>10}{'corr':>9}{'inject':>9}")
    for k, m in results.items():
        print(f"  {k:<22}{m['recovered']:>+11.3f}{m['achieved']:>10.4f}"
              f"{m['corr']:>+9.3f}{m['inject_rate']:>9.2f}")
    print(f"  {args.mode + ' [' + other + ']':<22}{alt['recovered']:>+11.3f}"
          f"{alt['achieved']:>10.4f}{alt['corr']:>+9.3f}{alt['inject_rate']:>9.2f}"
          f"   <- split contrast")

    if "delta-only" in results and "pretrain-ft" in results:
        d0, dp = results["delta-only"]["recovered"], results["pretrain-ft"]["recovered"]
        if dp > d0 + 0.05:
            print(f"\n  MULTI-TASK PREMISE: pretrain-ft {dp:+.3f} > delta-only "
                  f"{d0:+.3f}. Now run --shuffle_aux: delta-only is NESTED inside "
                  f"pretrain-ft at s=0, so only the shuffled control separates a "
                  f"transferred signal from a flattering extra knob.")
        else:
            print(f"\n  MULTI-TASK PREMISE NOT SUPPORTED: pretrain-ft {dp:+.3f} vs "
                  f"delta-only {d0:+.3f}. The ~4,000 free contestedness labels do "
                  f"not transfer to the delta. Report this; do not tune until it "
                  f"flips.")
    if "length_only" in results and headline:
        lo, hi = results["length_only"]["recovered"], headline[0]["recovered"]
        if lo >= 0.7 * hi and lo > 0.05:
            print(f"  !! LENGTH-ONLY recovers {lo:+.3f} vs {hi:+.3f}. The result is "
                  f"about answer length, not routing. Stop here.")
    print("  REMINDER: a router fit on OvertonBench cannot be reported cleanly ON "
          "OvertonBench. Ceiling is the 0.634 oracle; 0.497 is always-baseline.")

    if args.save and headline:
        params = _inner_select(args.mode, mode_grid(args.mode, args), Zt, yt,
                               [r.qid for r in rows], topic_of, ctx[args.mode],
                               args.seed)
        out = FITTERS[args.mode](Zt, yt, **params, **ctx[args.mode])
        MultiTaskRouter(trunk=trunk, w_t=out["w_t"], b_t=out["b_t"],
                        w_a=out.get("w_a", w_aux), b_a=out.get("b_a", 0.0),
                        thr=decision_threshold(headline[0]["resid_sd"],
                                               args.cost_ratio),
                        resid_sd=headline[0]["resid_sd"], mode=args.mode,
                        embed_model=args.embed_model).save(args.save)

    if args.out and headline:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        thr = headline[0]["thr"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "question_id", "condition", "topic", "baseline_cov",
                        "injected_cov", "delta", "pred_delta", "thr", "inject"])
            for r, p in zip(rows, headline[1]):
                w.writerow([r.run, r.qid, r.condition, topic_of[r.qid], r.base,
                            r.inj, r.delta, float(p), thr, int(p > thr)])
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
