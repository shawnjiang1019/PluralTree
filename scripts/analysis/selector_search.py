"""Can a SELECTOR collect the oracle gap that already exists in the v8 responses?

The v8 OvertonBench run (60 questions, 3 conditions) measured:

    baseline        0.4967
    distributional  0.3941
    scout           0.3927
    oracle          0.6344      per question, keep the best condition's answer
    union           0.6730      per question, keep both answers' clusters

Oracle beats baseline by **+0.137**, more than 4x the established noise floor of
0.027 (two independent baseline draws). Every condition individually LOSES to
baseline, yet a perfect per-question router would win by a wide margin -- the
conditions fail on different questions. Nothing in the project has tried to
collect that gap; the whole condition sweep only ever asked "which condition is
best on average", which is the wrong question if the answer is "it depends".

This script asks the cheap version: is there any function of the three candidate
answers, computable offline, that recovers a usable fraction of the +0.137? It
runs on the already-scored responses -- no GPU, no generation, no judge calls --
so a negative result costs minutes and a positive one justifies a real run.

FRACTION OF GAP RECOVERED is the headline, not raw score. (achieved - baseline) /
(oracle - baseline) puts every selector on the same 0..1 scale where 0 = "you may
as well have returned baseline" and 1 = "perfect routing". A selector 0.02 above
baseline is inside the noise floor and recovers ~0.15 of the gap; both facts are
worth seeing at once.

TIES ARE THE POINT for the reward-based selectors. docs/reward_gate_failure.md
measured `coverage_reward` at 0 for 76-96% of responses -- the matcher never
fires, upstream of every v2 design change. A selector built on a reward that is
zero everywhere degenerates to its tie-break, so the tie rate is reported next to
the score: a selector that "ties baseline" because it never separated anything is
a different finding from one that separated answers and chose wrong.

Selectors break ties toward `baseline` by construction. Breaking toward anything
else would be an unmeasured intervention wearing a selector's clothes.

    OPINIONQA_DIR=... python scripts/analysis/selector_search.py \
        --responses overton_responses_v8.jsonl --scores overton_scores_v8.csv

    python scripts/analysis/selector_search.py --selftest   # no graph, no model
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

# Anchor resolution is identical to the reward/judge correlation experiment, so
# IMPORT it rather than copy it -- two divergent copies of _ANCHOR_RE would make
# the two scripts silently score different questions.
from reward_eval_correlation import (build_anchor_index, parse_anchor_text,  # noqa: E402
                                     resolve_anchor)

BASELINE = "baseline"
NOISE_FLOOR = 0.027          # two independent baseline draws; deltas under this
#                              are not distinguishable from run-to-run variance
MATCH_THRS = (0.30, 0.35, 0.40, 0.45, 0.50)   # 0.50 is RewardConfig's unfitted
#                              default and is suspected to sit above the entire
#                              cosine distribution (docs/reward_gate_failure.md 4b)


# ---------------------------------------------------------------------------
# Per-question context handed to every selector
# ---------------------------------------------------------------------------
@dataclass
class QCtx:
    """Everything a selector may look at for one question, plus scratch space.

    `positions`/`P`/`embed_fn` are None when the anchor did not resolve; a
    selector that needs them must fall back rather than crash (see _by_score).

    `scores` and `tied` are written BY the selector and read by the harness. The
    tie rate is the informative statistic for the reward-based selectors, and it
    is invisible from the returned condition alone.
    """
    qid: int
    question: str = ""
    positions: list | None = None
    P: object | None = None            # cached position embeddings (n_pos, d)
    embed_fn: object | None = None
    cfg: object | None = None
    truth: dict | None = None          # judge coverage; SELF-TEST ONLY
    scores: dict = field(default_factory=dict)
    tied: bool = False
    _cache: dict = field(default_factory=dict)

    def reward_and_sim(self, cond: str, text: str):
        """(coverage_reward at the headline depth, sim matrix) -- one embed pass.

        Every embedding-based selector wants some slice of the same (units x
        positions) cosine matrix, and embedding is the only real cost here. One
        pass per (question, condition) is shared across the reward selector and
        all five match_thr variants; recomputing per selector would multiply the
        cost by six for no new information.
        """
        if cond not in self._cache:
            from alignment.reward import coverage_rewards_sweep
            d, t = self.cfg.min_depth_words, self.cfg.match_thr
            # sweep is keyed (match_thr, min_depth_words) -- one cell here
            _r1, r2, sim = coverage_rewards_sweep(text, self.positions,
                                                  self.embed_fn, self.cfg,
                                                  P=self.P, depths=[d], thrs=[t])
            self._cache[cond] = (r2[(t, d)], sim)
        return self._cache[cond]


# ---------------------------------------------------------------------------
# Selector registry -- adding a selector is one decorated function
# ---------------------------------------------------------------------------
@dataclass
class Selector:
    fn: Callable[[str, dict, QCtx], str]
    needs_embed: bool
    doc: str


SELECTORS: dict[str, Selector] = {}


def register(name: str, needs_embed: bool = False):
    def deco(fn):
        SELECTORS[name] = Selector(fn, needs_embed, (fn.__doc__ or "").strip())
        return fn
    return deco


def _by_score(score: Callable[[str, str, str, QCtx], float]) -> Callable:
    """Wrap a per-candidate scoring function `(question, cond, text, ctx)`.

    Ties break toward `baseline`: the floor we are trying to beat is baseline,
    so a selector that cannot separate the candidates must return exactly the
    floor's score, never an accidental lift from an arbitrary ordering. The tie
    is recorded on ctx so the harness can report how often that happened.
    """
    def sel(question: str, candidates: dict, ctx: QCtx) -> str:
        vals = {c: float(score(question, c, candidates[c], ctx))
                for c in candidates}
        ctx.scores = vals
        best = max(vals.values())
        tied = [c for c in candidates if vals[c] >= best - 1e-9]
        ctx.tied = len(tied) > 1
        return BASELINE if BASELINE in tied else sorted(tied)[0]
    return sel


@register("always_baseline")
def always_baseline(question, candidates, ctx):
    """The floor to beat (0.4967). Recovers 0.0 of the gap by definition."""
    ctx.scores, ctx.tied = {}, False
    return BASELINE if BASELINE in candidates else sorted(candidates)[0]


@register("longest")
def longest(question, candidates, ctx):
    """Most words. A LENGTH-ONLY control, not a proposal.

    Injected conditions run ~330 words vs baseline ~69, and length has already
    reversed one conclusion in this project (docs/untested_test_time_methods.md:
    length-matched prompting is a missing control everywhere). If `longest`
    recovers as much gap as a semantic selector, the semantic selector has not
    been shown to use semantics.
    """
    return _by_score(lambda q, k, t, c: len((t or "").split()))(
        question, candidates, ctx)


@register("most_units")
def most_units(question, candidates, ctx):
    """Most units from alignment.reward.split_units.

    Structure rather than raw length: split_units drops anything under 8 words,
    so this counts substantive clauses. It is also the denominator the reward's
    precision term divides by, which makes it the cheapest proxy for "this answer
    has many distinct things in it" that shares the reward's own segmentation.
    """
    from alignment.reward import split_units

    def score(q, k, t, c):
        return len(split_units(t or "", c.cfg.max_units if c.cfg else 40))
    return _by_score(score)(question, candidates, ctx)


@register("coverage_reward", needs_embed=True)
def coverage_reward_sel(question, candidates, ctx):
    """alignment.reward.coverage_reward against positions_from_subtree(anchor).

    This is the selector the GRPO phase would get for free if the reward worked
    as a ranker (docs/reward_gate_failure.md 10: a selector needs only a good
    RANKER, not a trainable gradient). It currently scores 0 for 76-96% of
    responses, so expect it to degenerate to the baseline tie-break -- the tie
    rate below, not the achieved score, is the number that says so.
    """
    if not ctx.positions:
        ctx.scores, ctx.tied = {}, True     # no anchor: cannot rank, so it ties
        return BASELINE
    return _by_score(lambda q, k, t, c: c.reward_and_sim(k, t)[0])(
        question, candidates, ctx)


def _n_mentioned_selector(thr: float):
    def sel(question, candidates, ctx):
        if not ctx.positions:
            ctx.scores, ctx.tied = {}, True
            return BASELINE

        def score(q, k, t, c):
            sim = c.reward_and_sim(k, t)[1]
            if getattr(sim, "size", 0) == 0:
                return 0.0                  # split_units left nothing to match
            return float((sim.max(axis=0) >= thr).sum())
        return _by_score(score)(question, candidates, ctx)
    sel.__doc__ = (f"Positions with ANY unit at cosine >= {thr:.2f}.\n\n"
                   "Drops the depth gate entirely and sweeps the threshold, "
                   "because at d=0 the reward was STILL zero for 76% of "
                   "responses -- the failure is at `mentioned`, upstream of "
                   "depth. match_thr=0.50 was chosen, never fitted; if the real "
                   "cosines live at 0.30-0.45 then only the low end of this "
                   "sweep can rank anything at all.")
    return sel


for _thr in MATCH_THRS:
    register(f"n_positions_mentioned@{_thr:.2f}", needs_embed=True)(
        _n_mentioned_selector(_thr))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value. Ties carry no information and are
    dropped (that is what makes it a SIGN test), so n = wins + losses.

    Exact, not normal-approximated: with 60 questions and a heavy tie rate the
    effective n can fall into the single digits, where the approximation lies.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def evaluate(name: str, sel: Callable, qids: list, cov: dict, resp: dict,
             ctxs: dict) -> dict:
    """Run one selector over every question and score what it picked."""
    picked, ties, w, l, t = [], 0, 0, 0, 0
    choices = {}
    for qid in qids:
        cands = {c: resp[qid][c] for c in cov[qid]}
        ctx = ctxs[qid]
        ctx.scores, ctx.tied = {}, False
        chosen = sel(ctx.question, cands, ctx)
        if chosen not in cov[qid]:           # a selector may not invent an answer
            raise ValueError(f"{name} returned {chosen!r} for q{qid}")
        choices[qid] = chosen
        ties += int(ctx.tied)
        got, base = cov[qid][chosen], cov[qid][BASELINE]
        picked.append(got)
        if got > base + 1e-9:
            w += 1
        elif got < base - 1e-9:
            l += 1
        else:
            t += 1
    return {"selector": name, "achieved": st.mean(picked), "n": len(qids),
            "win": w, "loss": l, "tie": t, "p": sign_test(w, l),
            "tie_rate": ties / len(qids) if qids else float("nan"),
            "choices": choices}


def report(rows: list, baseline: float, oracle: float, union: float | None,
           n_noanchor: int = 0) -> list:
    """Print the table, ranked by achieved score. Returns rows with `frac_gap`."""
    gap = oracle - baseline
    for r in rows:
        r["delta"] = r["achieved"] - baseline
        r["frac_gap"] = r["delta"] / gap if abs(gap) > 1e-12 else float("nan")
    rows.sort(key=lambda r: r["achieved"], reverse=True)

    print(f"\n=== selectors over {rows[0]['n']} questions "
          f"(baseline {baseline:.4f}, oracle {oracle:.4f}, gap {gap:+.4f}) ===")
    print(f"  {'selector':<28}{'score':>8}{'delta':>9}{'frac_gap':>10}"
          f"{'W/L/T':>13}{'sign p':>9}{'tie_rate':>10}")
    for r in rows:
        tr = f"{'-':>10}" if r["tie_rate"] != r["tie_rate"] else f"{r['tie_rate']:>10.3f}"
        wlt = "{}/{}/{}".format(r["win"], r["loss"], r["tie"])
        print(f"  {r['selector']:<28}{r['achieved']:>8.4f}{r['delta']:>+9.4f}"
              f"{r['frac_gap']:>10.3f}{wlt:>13}{r['p']:>9.3f}{tr}")
    print(f"  {'-' * 76}")
    print(f"  {'ORACLE (perfect routing)':<28}{oracle:>8.4f}{gap:>+9.4f}"
          f"{1.0:>10.3f}{'reference':>13}")
    if union is not None:
        print(f"  {'UNION (keep both answers)':<28}{union:>8.4f}"
              f"{union - baseline:>+9.4f}{(union - baseline) / gap:>10.3f}"
              f"{'reference':>13}")
        print("  union >= oracle by construction; it is the ceiling for MERGING "
              "answers,")
        print("  not for selecting one, so no selector here can reach it.")
        # union < oracle is impossible on the same data, so it means --union was
        # carried over from a different run. Say so rather than print a row that
        # quietly falsifies the invariant one line above it.
        if union < oracle - 1e-9:
            print(f"  ** --union {union:.4f} < oracle {oracle:.4f}: impossible "
                  f"on one run. The value passed in does not belong to THESE "
                  f"scores; re-read it from the judge's union table. **")
    if n_noanchor:
        print(f"  ({n_noanchor} questions had no resolvable anchor; embedding "
              f"selectors returned baseline there, which caps their reachable "
              f"frac_gap at {1 - n_noanchor / rows[0]['n']:.2f})")

    # The finding, stated rather than left to be read off the table.
    real = [r for r in rows if r["selector"] != "always_baseline"]
    best = max(real, key=lambda r: r["achieved"], default=None)
    print()
    if best is None or best["delta"] <= 1e-9:
        print("VERDICT: NOTHING BEATS BASELINE. No selector here recovers any of "
              "the oracle gap;")
        print("  the gap is real but none of these functions of the candidate "
              "text locates it.")
    elif best["delta"] < NOISE_FLOOR:
        print(f"VERDICT: best is {best['selector']} at {best['delta']:+.4f}, "
              f"INSIDE the {NOISE_FLOOR} noise floor")
        print(f"  ({best['frac_gap']:.1%} of the gap, sign p={best['p']:.3f}). "
              f"Not a result -- do not build on it.")
    else:
        print(f"VERDICT: best is {best['selector']} at {best['delta']:+.4f} "
              f"({best['frac_gap']:.1%} of the gap),")
        print(f"  above the {NOISE_FLOOR} noise floor, sign p={best['p']:.3f} "
              f"over {best['win'] + best['loss']} non-tied questions.")
    return rows


def oracle_mean(qids, cov) -> float:
    return st.mean(max(cov[q].values()) for q in qids)


def union_mean(qids, clusters: dict, n_clusters: dict) -> float:
    """Cross-condition union of covered clusters.

    NOT recoverable from the scores CSV -- judge_overtonbench keeps the covered
    cluster SETS only in memory and writes coverage fractions. So real runs pass
    the measured value via --union; this path exists for the self-test, which is
    the only place the sets are available.
    """
    return st.mean(len(set().union(*clusters[q].values())) / n_clusters[q]
                   for q in qids)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def load(args):
    """-> (qids, cov, resp, question_text, fork_ctx, n_multi_rollout)"""
    resp, qtext, fork = defaultdict(dict), {}, {}
    for line in open(args.responses, encoding="utf-8"):
        r = json.loads(line)
        qid = int(r["question_id"])
        # First rollout only. Selecting ACROSS ROLLOUTS is best-of-K, a different
        # experiment with a different compute story; this one selects across
        # conditions at fixed 1x sampling per condition.
        if r.get("rollout", 0) != 0 and r["condition"] in resp[qid]:
            continue
        resp[qid][r["condition"]] = r.get("response") or ""
        qtext.setdefault(qid, r.get("question", ""))
        if r.get("fork_context") and qid not in fork:
            fork[qid] = r["fork_context"]

    cov, n_multi = defaultdict(dict), 0
    for r in csv.DictReader(open(args.scores, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])
        n_multi += int(int(r.get("n_rollouts", 1) or 1) > 1)

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    qids = [q for q in sorted(cov)
            if all(c in cov[q] and c in resp.get(q, {}) for c in conds)]
    cov = {q: {c: cov[q][c] for c in conds} for q in qids}
    return qids, cov, resp, qtext, fork, n_multi


def main():
    ap = argparse.ArgumentParser(description="Offline selector search over the "
                                             "OvertonBench condition responses")
    ap.add_argument("--responses", default="overton_responses_v8.jsonl")
    ap.add_argument("--scores", default="overton_scores_v8.csv")
    ap.add_argument("--conditions", default="baseline,distributional,scout")
    ap.add_argument("--selectors", default="",
                    help="comma list to run a subset (default: all registered)")
    ap.add_argument("--union", type=float, default=0.6730,
                    help="measured cross-condition union. Cannot be recomputed "
                         "here -- the scores CSV stores coverage fractions, not "
                         "the covered cluster sets. Negative = omit the row.")
    ap.add_argument("--seed", type=int, default=42, help="graph split seed")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--out", default="docs/selector_search.csv")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic fixture, no graph and no embedder")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    names = [n.strip() for n in args.selectors.split(",") if n.strip()] \
        or list(SELECTORS)
    missing = [n for n in names if n not in SELECTORS]
    if missing:
        ap.error(f"unknown selector(s): {missing}; have {list(SELECTORS)}")

    qids, cov, resp, qtext, fork, n_multi = load(args)
    if not qids:
        ap.error("no question has every requested condition in both files")
    if n_multi:
        print(f"WARNING: {n_multi} score rows have n_rollouts>1. Their coverage "
              f"is a rollout MEAN while the selector sees rollout 0's text; the "
              f"comparison is only exact at 1 rollout per condition.")

    ctxs = {q: QCtx(qid=q, question=qtext.get(q, "")) for q in qids}
    n_noanchor = 0
    if any(SELECTORS[n].needs_embed for n in names):
        from alignment.reward import (RewardConfig, default_embed_fn,
                                      embed_positions, positions_from_subtree)
        from data.loaders.opinionqa import load_opinionqa

        cfg = RewardConfig()
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
        index = build_anchor_index(graph)
        embed_fn = default_embed_fn(args.embedder)
        for q in qids:
            txt = parse_anchor_text(fork.get(q, ""))
            anchor = resolve_anchor(txt, index) if txt else None
            positions = positions_from_subtree(graph, anchor) if anchor is not None else []
            if len(positions) < 2:
                n_noanchor += 1
                continue
            ctxs[q].positions = positions
            ctxs[q].P = embed_positions(positions, embed_fn)
            ctxs[q].embed_fn = embed_fn
            ctxs[q].cfg = cfg
        print(f"anchors resolved for {len(qids) - n_noanchor}/{len(qids)} "
              f"questions (match_thr={cfg.match_thr}, d={cfg.min_depth_words})")

    baseline = st.mean(cov[q][BASELINE] for q in qids)
    oracle = oracle_mean(qids, cov)
    print(f"\nper-condition means over {len(qids)} questions:")
    for c in sorted(cov[qids[0]]):
        print(f"  {c:<18}{st.mean(cov[q][c] for q in qids):.4f}")

    rows = [evaluate(n, SELECTORS[n].fn, qids, cov, resp, ctxs) for n in names]
    rows = report(rows, baseline, oracle,
                  args.union if args.union >= 0 else None, n_noanchor)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cols = ["selector", "achieved", "delta", "frac_gap", "win", "loss",
                "tie", "p", "tie_rate", "n"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------------
# Self-test: no graph, no embedder, no cluster access
# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Synthetic fixture with KNOWN covered-cluster sets.

    The real data lives on the cluster, so the arithmetic that the whole report
    rests on is verified here instead, on numbers whose answers are known by
    construction:

      oracle >= best single      per-question max >= any fixed column
      union  >= oracle           union of sets >= the largest set
      always_baseline            recovers EXACTLY 0.0 of the gap
      planted perfect selector   recovers EXACTLY 1.0

    Ordering identities like the first two hold by construction and would only
    ever break through an indexing bug -- which is exactly the bug that silently
    reports a fake result, so they are asserted rather than assumed.
    """
    import re
    import numpy as np
    from alignment.reward import Position, RewardConfig

    conds = [BASELINE, "distributional", "scout"]
    n_clusters = 4
    # Covered cluster sets chosen so each condition wins some question and no
    # condition dominates -- i.e. the v8 shape: every column loses to baseline on
    # average while the row-wise max beats it.
    fixture = {
        1: {BASELINE: {0, 1}, "distributional": {2}, "scout": {3}},
        2: {BASELINE: {0}, "distributional": {1, 2}, "scout": {0}},
        3: {BASELINE: {0, 1, 2}, "distributional": {0}, "scout": {1}},
        4: {BASELINE: {1}, "distributional": {0}, "scout": {0, 2, 3}},
        5: {BASELINE: {0, 2}, "distributional": {0, 2}, "scout": {0, 2}},
        6: {BASELINE: set(), "distributional": {3}, "scout": {1, 3}},
    }
    qids = sorted(fixture)
    cov = {q: {c: len(fixture[q][c]) / n_clusters for c in conds} for q in qids}
    ncl = {q: n_clusters for q in qids}

    # Responses: one paragraph of the option's token per covered cluster, so
    # split_units keeps them (>=8 words) and a bag-of-words stub embedder matches
    # them. Plus DELIBERATE filler: without it, length is monotone in coverage
    # and `longest` scores a perfect 1.000 -- a fixture that cannot tell a length
    # control from a semantic selector cannot validate this script's main claim.
    # The filler token matches no position, so it moves `longest`/`most_units`
    # and leaves the position-based selectors alone, which is exactly the
    # separation the real experiment is looking for.
    opts = ["alpha", "beta", "gamma", "delta"]

    def para(o, n):
        return " ".join([o] * n) + "."

    def body(q, c):
        pad = (q * 3 + conds.index(c) * 5) % 7
        return "\n".join([para(opts[i], 12 + 3 * i) for i in sorted(fixture[q][c])]
                         + [para("zeta", 9)] * pad) or para("zeta", 9)

    resp = {q: {c: body(q, c) for c in conds} for q in qids}

    vocab: dict[str, int] = {}

    def fake_embed(texts):
        V = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                V[i, vocab.setdefault(w, len(vocab) % 64)] += 1.0
            n = np.linalg.norm(V[i])
            if n:
                V[i] /= n
        return V

    cfg = RewardConfig(min_depth_words=20)
    positions = [Position(option=o, embed_text=f"{o} {o} {o} {o}", prevalence=0.25)
                 for o in opts]
    P = fake_embed([p.embed_text for p in positions])
    ctxs = {q: QCtx(qid=q, question=f"Q{q}?", positions=positions, P=P,
                    embed_fn=fake_embed, cfg=cfg, truth=cov[q]) for q in qids}

    baseline = st.mean(cov[q][BASELINE] for q in qids)
    best_single = max(st.mean(cov[q][c] for q in qids) for c in conds)
    oracle = oracle_mean(qids, cov)
    union = union_mean(qids, fixture, ncl)

    assert oracle >= best_single - 1e-12, (oracle, best_single)
    assert union >= oracle - 1e-12, (union, oracle)
    assert oracle > baseline, (oracle, baseline)          # fixture has a real gap

    # A planted PERFECT selector: reads the judge coverage straight off ctx. Only
    # legal here; it exists to prove the frac_gap denominator is right, since a
    # broken denominator makes every real selector's number meaningless.
    @register("perfect_cheat")
    def _perfect(question, candidates, ctx):
        ctx.scores = dict(ctx.truth)
        best = max(ctx.truth.values())
        tied = [c for c in candidates if ctx.truth[c] >= best - 1e-9]
        ctx.tied = len(tied) > 1
        return tied[0]

    names = list(SELECTORS)
    rows = [evaluate(n, SELECTORS[n].fn, qids, cov, resp, ctxs) for n in names]
    rows = report(rows, baseline, oracle, union)
    by = {r["selector"]: r for r in rows}

    ab = by["always_baseline"]
    assert ab["achieved"] == baseline, (ab["achieved"], baseline)
    assert ab["frac_gap"] == 0.0, ab["frac_gap"]
    assert (ab["win"], ab["loss"], ab["tie"]) == (0, 0, len(qids)), ab

    pf = by["perfect_cheat"]
    assert abs(pf["achieved"] - oracle) < 1e-12, (pf["achieved"], oracle)
    assert abs(pf["frac_gap"] - 1.0) < 1e-12, pf["frac_gap"]
    assert pf["loss"] == 0, pf

    # Every registered selector actually ran and returned a legal condition, and
    # none exceeded the oracle -- a selector scoring above perfect routing means
    # the harness is scoring something other than what it picked.
    for r in rows:
        assert r["n"] == len(qids)
        assert set(r["choices"].values()) <= set(conds), r["selector"]
        assert r["achieved"] <= oracle + 1e-12, r["selector"]

    # The fixture must DISCRIMINATE: irrelevant filler moves the length control
    # off the oracle while the position counter, which the filler cannot touch,
    # stays on it. Without this the earlier asserts would pass on a fixture where
    # every selector trivially wins.
    ln, npos = by["longest"], by["n_positions_mentioned@0.50"]
    assert ln["achieved"] < oracle - 1e-9, ln["achieved"]
    assert abs(npos["frac_gap"] - 1.0) < 1e-12, npos["frac_gap"]

    print("\nselector_search self-test OK")
    print(f"  best single {best_single:.4f} <= oracle {oracle:.4f} "
          f"<= union {union:.4f}")
    print(f"  always_baseline frac_gap {ab['frac_gap']:.1f}, "
          f"perfect_cheat frac_gap {pf['frac_gap']:.1f}")
    print(f"  length control {ln['frac_gap']:.3f} < position counter "
          f"{npos['frac_gap']:.3f} under irrelevant filler")
    print(f"  {len(rows)} selectors ran end to end on the fixture "
          f"(stub embedder, no graph)")


if __name__ == "__main__":
    main()
