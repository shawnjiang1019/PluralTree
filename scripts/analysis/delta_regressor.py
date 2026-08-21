"""Predict the per-question injection delta, then route on it.

THE GAP. On OvertonBench, injecting retrieved viewpoints helps contested
questions (+0.31 coverage) and HURTS consensus ones (-0.45). Per-question oracle
routing scores 0.634 vs 0.497 for always-baseline: a +0.137 headroom, over 4x the
established 0.027 noise floor, currently collected by nothing. Two routers have
already failed on it:
  * graph divergence W          corr +0.19 -- the scout SELECTS max-W forks, so
                                by the time we see W its variance is gone
  * model self-assessment       `route` scored 0.072 -- the model's DECISION was
                                bad. That is not the same claim as "its reasoning
                                carries no signal", which is why the <think> text
                                is read here by a SEPARATE readout.

REGRESSION, NOT CLASSIFICATION. The target is the continuous
`coverage(injected) - coverage(baseline)`. Two reasons:

  1. It preserves MAGNITUDE. A classifier trained on sign(delta) treats a +0.02
     near-tie and a +0.45 rout as the same positive, so it spends capacity
     separating cases where the decision does not matter, and at inference it
     cannot tell a confident call from a coin flip. With 60 questions per run
     that is most of the data: near-ties dominate.
  2. It lets the DECISION THRESHOLD come from the measured cost asymmetry rather
     than from 0. A wrong inject costs ~0.45; a wrong skip forgoes ~0.31. The
     ratio r = 0.45/0.31 ~ 1.45 means a wrong inject is ~1.5x as expensive, so
     the break-even posterior is r/(1+r) ~ 0.59, not 0.5. Optimizing plain binary
     accuracy gets this exactly backwards -- accuracy weights the two errors
     equally and therefore buys cheap correct skips with expensive wrong injects.
     See `decision_threshold`: r is a CLI knob, defaulting to the measured value.

CONTAMINATION, stated up front. The labels are OvertonBench coverage deltas. A
router fit on them and then scored on OvertonBench is fit and evaluated on the
same 60 questions. Cross-validation removes the crudest form of this (the model
never sees the held-out question's label) but NOT the rest: the feature set, the
cost ratio, the alpha grid and the decision to build this at all were all chosen
while looking at these questions. Every headline below is therefore a CV ESTIMATE
of an upper bound, labelled as such, and the only clean number would come from a
fresh contested-question set. Say so wherever it is reported.

WHAT COUNTS. Not R^2. The headline is the fraction of the oracle gap recovered,
    (achieved - always_baseline) / (oracle - always_baseline)
with two anchors that must hold exactly: always-baseline recovers 0.0, and a
router handed the true deltas recovers 1.0.

LENGTH CONTROL. `--features length_only` fits on answer/trace LENGTH alone.
Length has already reversed one conclusion in this project; if length alone
recovers the gap then the result is about verbosity, not routing, and the rest of
the feature set is decoration.

FEATURE COST, and why `causal` is the honest default. Features are tagged:
    pre       available before ANY generation (scout output, graph structure)
    baseline  needs the baseline answer generated first
    post      needs the INJECTED answer generated (the <think> trace lives there)
A `post` router saves no generation -- it is a reranker over two answers already
produced, not a router. It is still worth measuring (it upper-bounds what the
trace knows) but it must not be reported as if it were a routing result. `causal`
= pre + baseline is the set a real deployment could run.

    python scripts/analysis/delta_regressor.py --selftest
    OPINIONQA_DIR=... python scripts/analysis/delta_regressor.py \
        --scores overton_scores_v4.csv --scores overton_scores_v5.csv \
        --features causal --out docs/delta_regressor_preds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root
sys.path.insert(0, _HERE)                                     # sibling scripts

# Reused, not reimplemented: these are route_signal's own statistics, so the
# per-feature correlation and single-threshold ceiling printed here are directly
# comparable to the numbers from the earlier route_signal run (corr +0.19 etc).
from evaluation.overton.route_signal import _best_gate, _corr


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
_VTAG = re.compile(r"(v\d+)")


def run_tag(path: str) -> str:
    """'overton_scores_v5.csv' -> 'v5'. The run tag is carried on EVERY row.

    v4/v5/v6/v8/v9 differ in prompt variant and condition set, so pooling them is
    a modelling CHOICE (more labels, at the price of assuming the delta means the
    same thing across prompt variants). It must be visible in the output, never
    silent -- hence a column, a per-run breakdown, and a printed warning.
    """
    m = _VTAG.search(os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


@dataclass
class Row:
    """One label: (run, question, injected condition) vs that run's baseline."""
    run: str
    qid: int
    condition: str
    base: float
    inj: float
    feats: dict = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.inj - self.base


def load_coverage(path: str) -> dict:
    """qid -> condition -> coverage. Same long-form parse as
    reward_eval_correlation.main and route_signal.main."""
    cov = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])
    return cov


def load_responses(path: str) -> dict:
    """qid -> condition -> response record (response / raw / think / fork_context)."""
    out = defaultdict(dict)
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # rollout>0 rows are extra samples of the same cell; keep the first so
            # one (qid, condition) maps to one response.
            out[int(r["question_id"])].setdefault(r["condition"], r)
    return out


def derive_responses_path(scores_path: str) -> str:
    """overton_scores_v5.csv -> overton_responses_v5.jsonl (same directory)."""
    b = os.path.basename(scores_path).replace("scores", "responses")
    b = os.path.splitext(b)[0] + ".jsonl"
    return os.path.join(os.path.dirname(scores_path) or ".", b)


def build_rows(scores_paths, responses_paths, baseline_cond, inject_conds):
    """One Row per (run, question, injected condition) that has BOTH coverages."""
    rows, meta = [], {}
    for i, sp in enumerate(scores_paths):
        tag = run_tag(sp)
        rp = (responses_paths[i] if responses_paths and i < len(responses_paths)
              else derive_responses_path(sp))
        cov = load_coverage(sp)
        resp = load_responses(rp)
        conds = sorted({c for q in cov.values() for c in q} - {baseline_cond})
        if inject_conds:
            conds = [c for c in conds if c in inject_conds]
        meta[tag] = {"scores": sp, "responses": rp if os.path.exists(rp) else None,
                     "conditions": conds, "n_q": len(cov)}
        for qid, by_cond in cov.items():
            if baseline_cond not in by_cond:
                continue
            for c in conds:
                if c not in by_cond:
                    continue
                r = Row(run=tag, qid=int(qid), condition=c,
                        base=by_cond[baseline_cond], inj=by_cond[c])
                r.feats["_resp_base"] = (resp.get(qid, {}).get(baseline_cond) or {})
                r.feats["_resp_inj"] = (resp.get(qid, {}).get(c) or {})
                rows.append(r)
    return rows, meta


# ---------------------------------------------------------------------------
# Feature registry (pluggable: add a function, name it in a set)
# ---------------------------------------------------------------------------
FEATURES: dict = {}       # name -> {fn, group, cost, doc}


def feature(name: str, group: str, cost: str):
    """cost: 'pre' (no generation), 'baseline' (needs the plain answer),
    'post' (needs the injected answer -- reranker, not router)."""
    def deco(fn):
        FEATURES[name] = {"fn": fn, "group": group, "cost": cost,
                          "doc": (fn.__doc__ or "").strip().splitlines()[0]}
        return fn
    return deco


def features_in(groups) -> list:
    return [n for n, m in FEATURES.items() if m["group"] in groups]


# --- graph side ------------------------------------------------------------
# w_raw / relevance / n_forks are SCOUT OUTPUT: they exist before a token is
# generated, so they are 'pre' even though we recover them here by parsing the
# fork_context that the generation run happened to log.
_FORK_RE = re.compile(r"\[fork (\d+)\][^\n]*?\(divergence=([-\d.]+), relevance=([-\d.]+)\)")


def parse_fork_stats(fork_context: str) -> dict:
    """Top-fork W and relevance + fork count from a logged fork_context block."""
    ms = _FORK_RE.findall(fork_context or "")
    if not ms:
        return {"w_raw": float("nan"), "relevance": float("nan"), "n_forks": 0.0,
                "w_gap": float("nan")}
    ws = [float(m[1]) for m in ms]
    return {"w_raw": ws[0], "relevance": float(ms[0][2]), "n_forks": float(len(ms)),
            # spread between the best and second fork: the scout maximizes W, so
            # the LEVEL of W is saturated (corr +0.19) but its margin may not be.
            "w_gap": (ws[0] - ws[1]) if len(ws) > 1 else 0.0}


@feature("g_w_raw", "graph", "pre")
def _g_w_raw(ctx):
    """Top fork's Wasserstein. Corr +0.19 alone -- kept as the known-weak control."""
    return ctx["fork"]["w_raw"]


@feature("g_w_gap", "graph", "pre")
def _g_w_gap(ctx):
    """W margin of fork 1 over fork 2 (the level is saturated by selection; the
    margin is not necessarily)."""
    return ctx["fork"]["w_gap"]


@feature("g_relevance", "graph", "pre")
def _g_relevance(ctx):
    """Top fork's mean question-relevance (anti-correlated with delta in v5)."""
    return ctx["fork"]["relevance"]


@feature("g_n_forks", "graph", "pre")
def _g_n_forks(ctx):
    """How many forks cleared the relevance gate at all."""
    return ctx["fork"]["n_forks"]


@feature("g_z_level", "graph", "pre")
def _g_z_level(ctx):
    """Same-depth-calibrated divergence z (contestedness, not magnitude). Live path only."""
    return ctx["live"].get("z_level", float("nan"))


@feature("g_driver_sim", "graph", "pre")
def _g_driver_sim(ctx):
    """Cosine(user question, the fork's actual survey question). Live path only."""
    return ctx["live"].get("driver_sim", float("nan"))


@feature("g_n_positions", "graph", "pre")
def _g_n_positions(ctx):
    """Distinct survey positions under the chosen anchor."""
    return ctx["pos"].get("n_positions", float("nan"))


@feature("g_prev_entropy", "graph", "pre")
def _g_prev_entropy(ctx):
    """Shannon entropy (nats) of the position prevalence distribution."""
    return ctx["pos"].get("prev_entropy", float("nan"))


@feature("g_prev_eff", "graph", "pre")
def _g_prev_eff(ctx):
    """exp(entropy) = effective number of positions. ~1 on consensus; the direct
    read of 'is this contested', which is the axis the +0.31/-0.45 split lies on."""
    return ctx["pos"].get("prev_eff", float("nan"))


# --- response side (needs the BASELINE answer only) ------------------------
@feature("r_words", "response", "baseline")
def _r_words(ctx):
    """Baseline answer length in words."""
    return ctx["base_txt_stats"]["words"]


@feature("r_chars", "response", "baseline")
def _r_chars(ctx):
    """Baseline answer length in characters."""
    return ctx["base_txt_stats"]["chars"]


@feature("r_units", "response", "baseline")
def _r_units(ctx):
    """alignment.reward.split_units count on the baseline answer."""
    return ctx["base_txt_stats"]["units"]


@feature("r_unit_words", "response", "baseline")
def _r_unit_words(ctx):
    """Mean words per unit in the baseline answer."""
    return ctx["base_txt_stats"]["unit_words"]


@feature("r_pos_hit", "response", "baseline")
def _r_pos_hit(ctx):
    """How many graph positions the BASELINE already names. If it covers them
    unaided, injection has nothing to add and can only dilute (the -0.45 side)."""
    return ctx["base_txt_stats"].get("pos_hit", float("nan"))


@feature("r_pos_frac", "response", "baseline")
def _r_pos_frac(ctx):
    """Fraction of graph positions the baseline already names."""
    return ctx["base_txt_stats"].get("pos_frac", float("nan"))


# --- trace side (needs the INJECTED answer -- reranker, not router) --------
# `route` scoring 0.072 says the model's DECISION was bad. A separate readout over
# the same <think> text asks whether the reasoning nonetheless carries signal the
# model failed to act on -- a different question, and answerable independently.
_CONTRAST = re.compile(
    r"\b(however|whereas|conversely|in contrast|on the other hand|but |yet |"
    r"differ|disagree|divid|split|oppos|debat|contest|polari[sz])", re.I)
_HEDGE = re.compile(
    r"\b(might|may |could|perhaps|possibly|unclear|uncertain|arguably|"
    r"somewhat|seems|appears|suggests|not directly|do(?:es)? not (?:address|apply)|"
    r"n[o']t relevant|tangential)", re.I)


@feature("t_words", "trace", "post")
def _t_words(ctx):
    """<think> length in words."""
    return ctx["think_stats"]["words"]


@feature("t_pos_named", "trace", "post")
def _t_pos_named(ctx):
    """Distinct graph positions explicitly named inside <think>."""
    return ctx["think_stats"].get("pos_hit", float("nan"))


@feature("t_contrast", "trace", "post")
def _t_contrast(ctx):
    """Contrast-marker count in <think> (however / disagree / divided / ...)."""
    return ctx["think_stats"]["contrast"]


@feature("t_hedge", "trace", "post")
def _t_hedge(ctx):
    """Hedging-marker count in <think>. High hedging co-occurs with 'the retrieved
    forks are off-topic', which is precisely the -0.45 regime."""
    return ctx["think_stats"]["hedge"]


@feature("t_contrast_rate", "trace", "post")
def _t_contrast_rate(ctx):
    """Contrast markers per 100 trace words (length-normalized, so it is not the
    length control in disguise)."""
    w = ctx["think_stats"]["words"]
    return 100.0 * ctx["think_stats"]["contrast"] / w if w else 0.0


@feature("t_hedge_rate", "trace", "post")
def _t_hedge_rate(ctx):
    """Hedging markers per 100 trace words."""
    w = ctx["think_stats"]["words"]
    return 100.0 * ctx["think_stats"]["hedge"] / w if w else 0.0


# --- external: MULTI-TASK SEAM --------------------------------------------
# A sibling is training a TEXT-ONLY contestedness predictor with ~4,000 free
# labels from survey distributions. Delta labels are scarce (60 questions x a
# handful of conditions per run); contestedness labels are plentiful and the two
# targets share their causal axis -- injection helps exactly when the question is
# contested. Consuming its output as ONE MORE FEATURE is the cheapest possible
# coupling: no shared parameters, no joint training, and no dependency in this
# file on its model.
#
# CONTRACT (that model is NOT implemented here):
#   --contestedness FILE.csv, columns: question_id, contestedness[, run]
#   `run` optional; absent => the score is question-level and applies to every run.
# The value must come from a model fit WITHOUT OvertonBench coverage labels, or it
# re-imports the contamination this file is trying to bound.
@feature("x_contested", "external", "pre")
def _x_contested(ctx):
    """Sibling text-only contestedness prediction (seam; NaN when --contestedness absent)."""
    return ctx["external"].get("contestedness", float("nan"))


FEATURE_SETS = {
    "graph": features_in({"graph"}),
    "response": features_in({"response"}),
    "trace": features_in({"trace"}),
    "external": features_in({"external"}),
    # what a deployment could actually run: nothing here needs the injected answer
    "causal": features_in({"graph", "response", "external"}),
    "all": features_in({"graph", "response", "trace", "external"}),
    # CONTROL. Length alone has already reversed a conclusion in this project.
    "length_only": ["r_words", "r_chars", "r_units", "r_unit_words", "t_words"],
}


def load_contestedness(path: str) -> dict:
    """(run, qid) -> score, with (None, qid) as the run-agnostic fallback."""
    out = {}
    if not path:
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = int(r["question_id"])
            out[(r.get("run") or None, qid)] = float(r["contestedness"])
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _mentions(text: str, option: str) -> bool:
    o = (option or "").strip()
    if len(o) < 3:
        return False
    return re.search(r"\b" + re.escape(o) + r"\b", text or "", re.I) is not None


def _text_stats(text, split_units, options) -> dict:
    words = (text or "").split()
    units = split_units(text or "")
    uw = [len(u.split()) for u in units]
    st = {"words": float(len(words)), "chars": float(len(text or "")),
          "units": float(len(units)),
          "unit_words": float(sum(uw) / len(uw)) if uw else 0.0}
    if options:
        hit = sum(1 for o in options if _mentions(text, o))
        st["pos_hit"] = float(hit)
        st["pos_frac"] = float(hit) / len(options)
    return st


def _trace_stats(think, split_units, options) -> dict:
    st = _text_stats(think, split_units, options)
    st["contrast"] = float(len(_CONTRAST.findall(think or "")))
    st["hedge"] = float(len(_HEDGE.findall(think or "")))
    return st


def _entropy(p) -> float:
    p = p / p.sum()
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def attach_features(rows, names, args) -> list:
    """Fill row.feats for every requested feature. Returns names actually usable.

    Everything degrades: no graph -> position features are NaN; no --embeddings ->
    z_level/driver_sim are NaN. NaN and constant columns are DROPPED with a note
    rather than zero-filled, because a zero-filled column looks like a measured 0
    in the correlation table.
    """
    want = set(names)
    need_graph = any(FEATURES[n]["group"] == "graph" for n in want) or \
        any(n in want for n in ("r_pos_hit", "r_pos_frac", "t_pos_named"))
    external = load_contestedness(args.contestedness)

    from alignment.reward import split_units

    graph = index = None
    positions_from_subtree = _effective_positions = None
    parse_anchor_text = resolve_anchor = None
    if need_graph:
        try:
            from alignment.reward import positions_from_subtree, _effective_positions
            from data.loaders.opinionqa import load_opinionqa
            # anchor recovery is reward_eval_correlation's, imported not copied
            from reward_eval_correlation import (build_anchor_index, parse_anchor_text,
                                                 resolve_anchor)
            graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
            index = build_anchor_index(graph)
        except Exception as e:                       # noqa: BLE001
            print(f"  [warn] graph unavailable ({type(e).__name__}: {e}); "
                  f"position/anchor features will be NaN")
            graph = None

    live = {}
    if args.embeddings:
        live = live_graph_features(rows, args)

    pos_cache = {}
    for r in rows:
        rb = r.feats.pop("_resp_base", {})
        ri = r.feats.pop("_resp_inj", {})
        fork = parse_fork_stats(ri.get("fork_context") or "")
        pos, options = {}, []
        if graph is not None:
            key = (r.run, r.qid)
            if key not in pos_cache:
                anchor_txt = parse_anchor_text(ri.get("fork_context") or "")
                anchor = resolve_anchor(anchor_txt, index) if anchor_txt else None
                d = {}
                if anchor is not None:
                    ps = positions_from_subtree(graph, anchor)
                    if len(ps) >= 2:
                        prev = np.array([p.prevalence for p in ps], dtype=float)
                        d = {"n_positions": float(len(ps)),
                             "prev_entropy": _entropy(prev),
                             "prev_eff": float(_effective_positions(prev)),
                             "options": [p.option for p in ps]}
                pos_cache[key] = d
            pos = pos_cache[key]
            options = pos.get("options", [])
        ctx = {
            "fork": fork,
            "pos": pos,
            "live": live.get((r.run, r.qid), {}),
            "base_txt_stats": _text_stats(rb.get("response") or "", split_units, options),
            "think_stats": _trace_stats(ri.get("think") or "", split_units, options),
            "external": {"contestedness": external.get(
                (r.run, r.qid), external.get((None, r.qid), float("nan")))},
        }
        for n in names:
            try:
                v = float(FEATURES[n]["fn"](ctx))
            except Exception:                        # noqa: BLE001
                v = float("nan")
            r.feats[n] = v
    return prune_features(rows, names)


def prune_features(rows, names) -> list:
    """Drop columns that are missing anywhere, or constant. Printed, not silent."""
    keep = []
    for n in names:
        vals = [r.feats.get(n, float("nan")) for r in rows]
        n_nan = sum(1 for v in vals if not np.isfinite(v))
        if n_nan:
            print(f"  drop {n:<16} ({n_nan}/{len(vals)} missing)")
            continue
        if max(vals) - min(vals) < 1e-12:
            print(f"  drop {n:<16} (constant)")
            continue
        keep.append(n)
    return keep


def live_graph_features(rows, args) -> dict:
    """z_level / driver_sim / w_raw from a live scout pass.

    SEAM, and a deliberate one: route_signal.py computes these INLINE inside its
    main() -- there is no function to import, and that file is owned by a sibling
    right now, so it cannot be refactored from here. This replicates its exact
    call sequence (same ScoutConfig, same branch_divergence, same same_level null,
    same driver-text assembly) against the same upstream primitives. If
    route_signal ever exposes e.g. `signals_for_question(...)`, delete this body
    and call it -- two copies of a feature definition WILL drift apart and then
    silently score different things.
    """
    import torch
    from pluraltree.manifolds.poincare import PoincareBall
    from evaluation.intrinsic.branch_divergence import branch_divergence, _null_divergence
    from evaluation.overton.eval_overtonbench import load_questions
    from retrieval.scout import (ScoutConfig, embed_question, node_relevance,
                                 load_or_compute_text_feat, scout)

    if args.dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    else:
        from data.loaders.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)
    cfg = ScoutConfig(tau=args.tau, alpha=1.0)
    null_mu, null_sd = _null_divergence(h_all, graph.children_indices,
                                        manifold=manifold, n_samples=args.n_null,
                                        seed=args.seed, same_level=True)
    print(f"  same-level null W: mean={null_mu:.3f} std={null_sd:.3f}")

    wanted = {r.qid for r in rows}
    per_q = {}
    for qid, question in load_questions():
        if qid not in wanted:
            continue
        q_emb = embed_question(question)
        forks = scout(question, graph, h_all, text_feat, manifold, cfg=cfg, q_emb=q_emb)
        if not forks:
            per_q[qid] = {"w_raw": 0.0, "z_level": 0.0, "driver_sim": 0.0}
            continue
        top = forks[0]
        bd = branch_divergence(top.anchor, h_all, graph.children_indices,
                               manifold=manifold)
        z = (bd["max"] - null_mu) / null_sd if null_sd and null_sd > 0 else 0.0
        drv_ids = [top.branch_a, top.branch_b] + [a for a, _, _ in top.top_pairs]
        drv_txt = " ".join(getattr(graph, "entity_text", {}).get(i, "") for i in drv_ids)
        d_emb = embed_question(drv_txt) if drv_txt.strip() else None
        dsim = float(node_relevance(q_emb, d_emb.unsqueeze(0))[0]) if d_emb is not None else 0.0
        per_q[qid] = {"w_raw": float(top.w), "z_level": float(z), "driver_sim": dsim}
    return {(r.run, r.qid): per_q.get(r.qid, {}) for r in rows}


# ---------------------------------------------------------------------------
# Ridge + question-grouped CV
# ---------------------------------------------------------------------------
def _fit_predict(Xtr, ytr, Xte, alpha):
    """Standardize on TRAIN only, center y on train, closed-form ridge."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    d = Ztr.shape[1]
    w = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(d), Ztr.T @ (ytr - ym))
    return Zte @ w + ym


def _groups(rows, folds, seed):
    """Fold assignment BY QUESTION, never by row.

    A question appears once per pooled run, so a row-level split would put v4's
    label for question 17 in train and v5's in test. Their features are near
    identical, so that is straightforward leakage and it inflates everything
    downstream. n=60 questions per run is also why a single split is meaningless:
    one 80/20 split gives 12 test questions, and the routing metric over 12
    questions moves by more than the whole +0.137 gap on resampling.
    """
    qids = sorted({r.qid for r in rows})
    if folds <= 0 or folds >= len(qids):
        return {q: i for i, q in enumerate(qids)}, len(qids)   # LOQO
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(qids))
    return {qids[perm[i]]: i % folds for i in range(len(qids))}, folds


def cv_predict(rows, names, alphas, folds, seed):
    """Out-of-fold predictions. Nested alpha selection when len(alphas) > 1."""
    X = np.array([[r.feats[n] for n in names] for r in rows], dtype=float)
    y = np.array([r.delta for r in rows], dtype=float)
    fold_of, n_folds = _groups(rows, folds, seed)
    f = np.array([fold_of[r.qid] for r in rows])
    q = np.array([r.qid for r in rows])
    oof = np.zeros(len(rows))
    chosen = []
    for k in range(n_folds):
        te, tr = f == k, f != k
        if not te.any() or tr.sum() < len(names) + 2:
            oof[te] = y[tr].mean() if tr.any() else 0.0
            continue
        a = alphas[0]
        if len(alphas) > 1:
            # INNER CV over training questions only. Selecting alpha on the outer
            # folds would tune the reported number on its own test set.
            inner_q = sorted(set(q[tr].tolist()))
            best, best_mse = alphas[0], float("inf")
            for cand in alphas:
                err = []
                for iq in inner_q:
                    ite = tr & (q == iq)
                    itr = tr & (q != iq)
                    if not ite.any() or itr.sum() < len(names) + 2:
                        continue
                    p = _fit_predict(X[itr], y[itr], X[ite], cand)
                    err.extend(((p - y[ite]) ** 2).tolist())
                m = float(np.mean(err)) if err else float("inf")
                if m < best_mse:
                    best, best_mse = cand, m
            a = best
        chosen.append(a)
        oof[te] = _fit_predict(X[tr], y[tr], X[te], a)
    return oof, y, (chosen or list(alphas[:1])), n_folds


# ---------------------------------------------------------------------------
# The asymmetric decision threshold
# ---------------------------------------------------------------------------
def _inv_norm_cdf(p: float) -> float:
    """Acklam's rational approximation to the standard normal quantile (|err|<1e-9).
    Inlined so this file needs no scipy (not in requirements.txt)."""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return ((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    q = p - 0.5
    r = q * q
    return ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))


def decision_threshold(resid_sd: float, cost_ratio: float) -> float:
    """Predicted-delta threshold above which injecting is worth it.

    Measured costs: a wrong inject loses ~0.45 coverage, a wrong skip forgoes
    ~0.31, so cost_ratio r = 0.45/0.31 ~ 1.45 by default.

    Derivation. With posterior p = P(delta > 0 | prediction):
        E[loss | inject] = (1-p) * L_bad        L_bad  = 0.45
        E[loss | skip]   = p     * L_miss       L_miss = 0.31
    inject iff p > L_bad/(L_bad+L_miss) = r/(1+r)   [= 0.592, NOT 0.5]

    A classifier optimizing accuracy implicitly puts that break-even at 0.5 and so
    systematically over-injects -- it buys cheap correct skips with expensive
    wrong injects. To turn the posterior threshold into a threshold on the
    regression output, take delta | prediction ~ N(pred, sd^2) with sd the
    out-of-fold residual sd; then p = Phi(pred/sd) and
        thr = sd * Phi^-1( r/(1+r) )
    This is exactly why the model has to be a REGRESSOR: a sign classifier emits
    no magnitude for the cost ratio to act on.

    Two properties worth noting:
      * r = 1 (symmetric costs) => thr = 0, i.e. plain "inject if you expect gain"
      * sd = 0 (a perfect predictor) => thr = 0 => routing equals the oracle,
        which is what makes the 1.0 anchor hold EXACTLY.
    """
    if not np.isfinite(resid_sd) or resid_sd <= 0:
        return 0.0
    r = max(cost_ratio, 1e-9)
    return float(resid_sd * _inv_norm_cdf(r / (1.0 + r)))


# ---------------------------------------------------------------------------
# Routing evaluation -- the headline
# ---------------------------------------------------------------------------
def route_metrics(rows, preds, thr) -> dict:
    base = np.array([r.base for r in rows])
    inj = np.array([r.inj for r in rows])
    take = np.asarray(preds) > thr
    achieved = float(np.where(take, inj, base).mean())
    always_base = float(base.mean())
    always_inj = float(inj.mean())
    oracle = float(np.maximum(base, inj).mean())
    gap = oracle - always_base
    return {"n": len(rows), "always_base": always_base, "always_inject": always_inj,
            "oracle": oracle, "achieved": achieved, "gap": gap,
            "recovered": (achieved - always_base) / gap if gap > 1e-12 else 0.0,
            "inject_rate": float(take.mean()), "thr": float(thr)}


def _r2(y, p):
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / ss if ss > 1e-12 else float("nan")


def evaluate(rows, names, args, label):
    """Fit, route, print. Returns (metrics, oof predictions, threshold)."""
    oof, y, alphas_used, n_folds = cv_predict(rows, names, args.alphas, args.folds,
                                              args.seed)
    # resid_sd is computed from ALL out-of-fold residuals, so the threshold sees
    # the whole set once. A small leak, recorded rather than hidden; the effect is
    # a scale, and the cost_ratio sweep below shows how little the answer moves.
    resid_sd = float(np.std(y - oof))
    thr = decision_threshold(resid_sd, args.cost_ratio)
    m = route_metrics(rows, oof, thr)

    n_q = len({r.qid for r in rows})
    cvname = "LOQO" if n_folds == n_q else f"{n_folds}-fold(by question)"
    print(f"\n=== {label}  [{len(names)} features, {cvname} CV ESTIMATE] ===")
    print(f"  features: {', '.join(names)}")
    print(f"  alpha(s) chosen: {sorted(set(alphas_used))}")
    print(f"  OOF corr={_corr(list(oof), list(y)):+.3f}  R2={_r2(y, oof):+.3f}  "
          f"resid_sd={resid_sd:.3f}   (NOT the headline)")
    print(f"  cost_ratio={args.cost_ratio:.3f} -> break-even p="
          f"{args.cost_ratio/(1+args.cost_ratio):.3f} -> thr={thr:+.4f}")
    print(f"  always_baseline={m['always_base']:.4f}  always_inject={m['always_inject']:.4f}"
          f"  oracle={m['oracle']:.4f}  gap={m['gap']:+.4f}")
    print(f"  ROUTED={m['achieved']:.4f}   inject_rate={m['inject_rate']:.2f}")
    print(f"  >>> ORACLE GAP RECOVERED = {m['recovered']:+.3f}  "
          f"(CV estimate, fit on the eval set -- see module docstring)")

    # per-run and per-condition: pooling must never hide which run carries a result
    by_run = defaultdict(list)
    for r, p in zip(rows, oof):
        by_run[r.run].append((r, p))
    if len(by_run) > 1:
        print(f"  {'run':<8}{'n':>5}{'base':>9}{'oracle':>9}{'routed':>9}{'recov':>9}")
        for run in sorted(by_run):
            mm = route_metrics([a for a, _ in by_run[run]], [b for _, b in by_run[run]], thr)
            print(f"  {run:<8}{mm['n']:>5}{mm['always_base']:>9.4f}{mm['oracle']:>9.4f}"
                  f"{mm['achieved']:>9.4f}{mm['recovered']:>+9.3f}")
    by_cond = defaultdict(list)
    for r, p in zip(rows, oof):
        by_cond[(r.run, r.condition)].append((r, p))
    if len(by_cond) > 1:
        print(f"  {'run/cond':<20}{'n':>5}{'base':>9}{'oracle':>9}{'routed':>9}{'recov':>9}")
        for k in sorted(by_cond):
            mm = route_metrics([a for a, _ in by_cond[k]], [b for _, b in by_cond[k]], thr)
            print(f"  {k[0] + '/' + k[1]:<20}{mm['n']:>5}{mm['always_base']:>9.4f}"
                  f"{mm['oracle']:>9.4f}{mm['achieved']:>9.4f}{mm['recovered']:>+9.3f}")
    return m, oof, thr


def anchors(rows) -> None:
    """The two numbers that must come out exact, or the metric is miscoded."""
    zero = route_metrics(rows, np.full(len(rows), -np.inf), 0.0)
    # a perfect predictor has resid_sd 0, so its own threshold is 0 whatever the
    # cost ratio -- which is why this anchor is exactly 1.0 and not "about 1"
    perfect_thr = decision_threshold(0.0, 0.45 / 0.31)
    one = route_metrics(rows, np.array([r.delta for r in rows]), perfect_thr)
    print("\n=== anchors ===")
    print(f"  always-baseline router   recovered = {zero['recovered']:+.6f}  (must be 0.0)")
    print(f"  planted perfect router   recovered = {one['recovered']:+.6f}  (must be 1.0)")
    assert abs(zero["recovered"]) < 1e-12, zero
    assert abs(one["recovered"] - 1.0) < 1e-12, one


def cost_sweep(rows, oof, args) -> None:
    """How much does the answer depend on the cost ratio we asserted?"""
    resid_sd = float(np.std(np.array([r.delta for r in rows]) - oof))
    print("\n=== cost_ratio sweep (r = |wrong inject| / |wrong skip|) ===")
    print(f"  {'r':>6}{'break-even p':>14}{'thr':>9}{'inject_rate':>13}{'recovered':>11}")
    for r in sorted({1.0, 1.25, round(args.cost_ratio, 6), 2.0, 3.0}):
        t = decision_threshold(resid_sd, r)
        m = route_metrics(rows, oof, t)
        star = "  <- measured (0.45/0.31)" if abs(r - args.cost_ratio) < 1e-5 else ""
        print(f"  {r:>6.2f}{r/(1+r):>14.3f}{t:>+9.4f}{m['inject_rate']:>13.2f}"
              f"{m['recovered']:>+11.3f}{star}")


def single_feature_table(rows, names) -> None:
    """route_signal's own two statistics, per feature, for comparability.

    _best_gate picks the best threshold ON THESE VERY ROWS: it is an OVERFIT
    ceiling, printed to locate features worth keeping, never as a result.
    """
    delta = [r.delta for r in rows]
    base = [r.base for r in rows]
    inj = [r.inj for r in rows]
    ab = sum(base) / len(base)
    orc = sum(max(b, i) for b, i in zip(base, inj)) / len(base)
    print("\n=== per-feature (route_signal._corr / ._best_gate; gate is an OVERFIT ceiling) ===")
    print(f"  {'feature':<18}{'cost':>9}{'corr(delta)':>13}{'best_gate':>11}{'recov@gate':>12}")
    for n in names:
        sig = [r.feats[n] for r in rows]
        c = _corr(sig, delta)
        score, thr, _ = _best_gate(sig, delta, base, inj)
        rec = (score - ab) / (orc - ab) if orc - ab > 1e-12 else float("nan")
        print(f"  {n:<18}{FEATURES[n]['cost']:>9}{c:>+13.3f}{score:>11.4f}{rec:>+12.3f}")


# ---------------------------------------------------------------------------
# Self-test on a synthetic fixture with a PLANTED routing signal
# ---------------------------------------------------------------------------
def _synthetic(n_q=60, runs=("vA", "vB"), seed=0):
    """delta is driven by a latent SEMANTIC variable; length is pure noise.

    Mirrors the measured regime: injection helps when latent contestedness is high
    and hurts when it is low, with a spread comparable to +0.31 / -0.45, so a
    length-only feature set has nothing real to find.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for run in runs:
        for qid in range(1, n_q + 1):
            s = rng.uniform(-1, 1)                       # latent contestedness
            base = float(np.clip(rng.normal(0.50, 0.12), 0.02, 0.95))
            d = 0.38 * s + rng.normal(0, 0.10)
            inj = float(np.clip(base + d, 0.0, 1.0))
            r = Row(run=run, qid=qid, condition="scout", base=base, inj=inj)
            r.feats = {
                "sem": s + rng.normal(0, 0.15),          # noisy view of the driver
                "sem2": 0.5 * s + rng.normal(0, 0.40),
                "len_words": rng.normal(180, 45),        # independent of delta
                "len_chars": rng.normal(1100, 260),
                "len_units": rng.normal(6, 2),
            }
            rows.append(r)
    return rows


def selftest(args) -> int:
    print("=== SELFTEST: synthetic fixture, planted SEMANTIC routing signal ===")
    rows = _synthetic(seed=args.seed)
    base = np.array([r.base for r in rows])
    inj = np.array([r.inj for r in rows])
    orc = float(np.maximum(base, inj).mean())
    ab, ai = float(base.mean()), float(inj.mean())
    print(f"n={len(rows)} rows over {len({r.qid for r in rows})} questions x "
          f"{len({r.run for r in rows})} runs")
    print(f"always_base={ab:.4f}  always_inject={ai:.4f}  oracle={orc:.4f}")

    ok = True
    # 1. oracle >= best single condition
    assert orc >= max(ab, ai) - 1e-12, (orc, ab, ai)
    print(f"  [ok] oracle >= best single condition  ({orc:.4f} >= {max(ab, ai):.4f})")

    # 2/3. the exact anchors
    anchors(rows)

    class A:  # minimal args for evaluate()
        alphas = [0.1, 1.0, 10.0]
        folds = 0
        seed = args.seed
        cost_ratio = args.cost_ratio
    m_sem, oof_sem, _ = evaluate(rows, ["sem", "sem2"], A, "SEMANTIC feature set")
    m_len, oof_len, _ = evaluate(rows, ["len_words", "len_chars", "len_units"], A,
                                 "LENGTH-ONLY control")

    # 4. the semantic set must find the planted signal, or the pipeline is broken
    if m_sem["recovered"] < 0.40:
        print(f"  [FAIL] planted semantic signal not recovered "
              f"({m_sem['recovered']:+.3f} < 0.40)")
        ok = False
    else:
        print(f"  [ok] semantic set recovers {m_sem['recovered']:+.3f} of the planted gap")

    # 5. THE CONTROL. Length was planted as noise; if it "recovers" the gap the
    #    evaluation is leaking, not the features working.
    if abs(m_len["recovered"]) > 0.15:
        print(f"  [FAIL] length-only recovered {m_len['recovered']:+.3f} on a signal "
              f"planted as SEMANTIC -- evaluation leaks")
        ok = False
    else:
        print(f"  [ok] length-only does NOT recover ({m_len['recovered']:+.3f}, |.|<=0.15)")

    cost_sweep(rows, oof_sem, A)
    print("\nSELFTEST: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Regress the injection delta, route on it")
    ap.add_argument("--scores", action="append", default=[],
                    help="repeatable: overton_scores_vN.csv. POOLING RUNS IS A "
                         "CHOICE -- v4/v5/v6/v8/v9 differ in prompt variant and "
                         "condition set; every row records its run and per-run "
                         "results are always printed beside the pooled one.")
    ap.add_argument("--responses", action="append", default=[],
                    help="repeatable, paired with --scores by position; default "
                         "derives overton_responses_vN.jsonl from each scores name")
    ap.add_argument("--baseline_cond", default="baseline")
    ap.add_argument("--inject_conds", default="",
                    help="comma list; default = every non-baseline condition present")
    ap.add_argument("--features", default="causal",
                    help=f"named set ({', '.join(sorted(FEATURE_SETS))}) or a comma "
                         f"list of feature names")
    ap.add_argument("--compare", default="causal,graph,response,trace,all,length_only",
                    help="comma list of sets to score side by side ('' = only --features)")
    ap.add_argument("--cost_ratio", type=float, default=0.45 / 0.31,
                    help="|wrong inject| / |wrong skip|. Default 1.452 = the "
                         "measured -0.45 / +0.31 asymmetry.")
    ap.add_argument("--alphas", default="0.03,0.1,0.3,1,3,10,30",
                    help="ridge grid; >1 value triggers NESTED inner-CV selection")
    ap.add_argument("--folds", type=int, default=0,
                    help="0 = leave-one-QUESTION-out (default). n>0 = grouped k-fold.")
    ap.add_argument("--contestedness", default=None,
                    help="SEAM: csv question_id,contestedness[,run] from the "
                         "sibling text-only model. Not implemented here.")
    ap.add_argument("--seed", type=int, default=42, help="graph split seed / fold seed")
    # live graph-feature path (optional; needs the cluster's embeddings)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--n_null", type=int, default=300)
    ap.add_argument("--out", default=None, help="csv of per-row OOF predictions")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    args.alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    if args.selftest:
        return selftest(args)
    if not args.scores:
        ap.error("--scores is required (or --selftest)")

    missing = [p for p in args.scores if not os.path.exists(p)]
    if missing:
        print(f"[warn] missing, skipped: {missing}")
        args.scores = [p for p in args.scores if os.path.exists(p)]
    if not args.scores:
        print("no score files present -- nothing to do")
        return 1

    inject = {c.strip() for c in args.inject_conds.split(",") if c.strip()}
    rows, meta = build_rows(args.scores, args.responses, args.baseline_cond, inject)
    print("=== label provenance ===")
    for tag, m in meta.items():
        rp = os.path.basename(m["responses"]) if m["responses"] else "MISSING"
        print(f"  {tag:<6} scores={os.path.basename(m['scores'])} responses={rp} "
              f"n_q={m['n_q']} conditions={m['conditions']}")
    if len(meta) > 1:
        print("  POOLED across runs: the delta is assumed to mean the same thing "
              "under different prompt variants. Per-run rows are printed below; if "
              "they disagree, the pooled number averages over different tasks.")
    print(f"  {len(rows)} labels over {len({r.qid for r in rows})} distinct questions")
    if not rows:
        print("no (baseline, injected) pairs found")
        return 1

    sets = [s.strip() for s in args.compare.split(",") if s.strip()] or [args.features]
    if args.features not in sets:
        sets.insert(0, args.features)
    wanted = []
    for s in sets:
        wanted += FEATURE_SETS.get(s, [n.strip() for n in s.split(",") if n.strip()])
    wanted = [n for n in dict.fromkeys(wanted) if n in FEATURES]

    print("\n=== feature extraction ===")
    usable = attach_features(rows, wanted, args)
    print(f"  {len(usable)}/{len(wanted)} features usable")
    if not usable:
        print("no usable features")
        return 1

    anchors(rows)
    single_feature_table(rows, usable)

    results, headline = {}, None
    for s in sets:
        names = [n for n in FEATURE_SETS.get(s, [x.strip() for x in s.split(",")])
                 if n in usable]
        if not names:
            print(f"\n=== {s}: no usable features, skipped ===")
            continue
        m, oof, thr = evaluate(rows, names, args, s)
        results[s] = m
        if s == args.features:
            headline = (m, oof, thr)

    if headline:
        cost_sweep(rows, headline[1], args)

    gap = headline[0]["gap"] if headline else 0.0
    print(f"\n=== summary: fraction of the {gap:+.3f} oracle gap recovered "
          f"(ALL numbers are CV estimates on the eval set) ===")
    print(f"  {'feature set':<16}{'recovered':>11}{'routed':>10}{'inject_rate':>13}")
    for s, m in results.items():
        print(f"  {s:<16}{m['recovered']:>+11.3f}{m['achieved']:>10.4f}"
              f"{m['inject_rate']:>13.2f}")
    if "length_only" in results and headline:
        lo, hi = results["length_only"]["recovered"], headline[0]["recovered"]
        if lo >= 0.7 * hi and lo > 0.05:
            print(f"  !! LENGTH-ONLY recovers {lo:+.3f} vs {hi:+.3f} for the headline "
                  f"set. The result is about answer length, not routing. Stop here.")
        else:
            print(f"  length-only {lo:+.3f} vs headline {hi:+.3f}: not merely length.")
    print("  REMINDER: a router fit on OvertonBench cannot then be reported cleanly "
          "ON OvertonBench. These are CV estimates of an upper bound.")

    if args.out and headline:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["run", "question_id", "condition",
                                              "baseline_cov", "injected_cov", "delta",
                                              "pred_delta", "thr", "inject"] + usable)
            w.writeheader()
            for r, p in zip(rows, headline[1]):
                w.writerow({"run": r.run, "question_id": r.qid, "condition": r.condition,
                            "baseline_cov": r.base, "injected_cov": r.inj,
                            "delta": r.delta, "pred_delta": p, "thr": headline[2],
                            "inject": int(p > headline[2]),
                            **{n: r.feats[n] for n in usable}})
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
