"""Self-referential best-of-K: the samples are their own rubric.

THE MEASUREMENT THIS EXPLOITS. On OvertonBench the coverage present in the
UNION across separately generated answers is 0.657-0.687, while any single
answer averages 0.507 (docs/untested_test_time_methods.md; oracle -- best of 3
strategies picked with hindsight -- is 0.622). The good content is already in
the sampling distribution; it is just never inside one response. Selection is
the cheapest way to cash that in, and it needs no training.

WHY NOT ORDINARY BEST-OF-K. Ordinary best-of-K needs a scorer that ranks like
the judge. The project's `coverage_reward` demonstrably does not
(docs/reward_gate_failure.md, scripts/analysis/reward_eval_correlation.py), and
it also needs the graph anchor, a calibrated match_thr and a min_depth_words fit
nobody has fit. This method drops all of that:

    pool_positions   split every one of the K samples into units, embed, and
                     agglomerate the units into clusters. Each cluster is a
                     position the POOL collectively expressed. Clusters
                     supported by fewer than --min_support distinct samples are
                     dropped -- a position only one sample mentions is more
                     likely a hallucination or an idiosyncratic aside than a
                     viewpoint.
    score_candidate  fraction of those pool clusters one sample covers.
    select_best_of_k argmax.

No graph, no judge, no external scorer, no calibrated absolute threshold: the
reference position set is derived from the same K samples being ranked.

KEY RISK -- READ BEFORE TRUSTING ANY NUMBER THIS PRINTS. What is maximized here
is coverage of the pool's CONSENSUS. That is not automatically what the judge
rewards, and the failure is asymmetric in a way that matters for this benchmark:

  * A viewpoint held by a real minority cluster will, if the sampler is
    mode-collapsed (Vendi ~1.4 effective modes over 8 rollouts), appear in few
    samples -- exactly the clusters --min_support deletes. The method can
    therefore be blindest precisely where OvertonScore's unweighted per-cluster
    mean pays the most.
  * A pool cluster is a semantic neighbourhood of TEXT, not a satisfied human
    cluster. OvertonScore requires a cluster's members to rate representation
    >= 4/5, which the repo has already shown is a depth requirement, not a
    mention requirement (`route` named every position and scored 0.072). Nothing
    here measures depth. A sample that name-drops the pool's whole consensus
    can win selection while being the answer the judge likes least.

So --sim_thr and --min_support are swept, not assumed: pass comma lists and read
the sensitivity off the table. The number that decides whether this works is not
the selection score -- it is the ACHIEVED OvertonScore of the selected answers
versus baseline, versus random selection at matched K, and versus oracle, which
this prints whenever --scores is supplied.

    python scripts/analysis/bestofk_selection.py \
        --responses overton_responses_v5.jsonl --scores overton_scores_v5.csv \
        --sim_thr 0.45,0.55,0.65 --min_support 1,2,3

    python scripts/analysis/bestofk_selection.py --selftest    # no model, no data
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alignment.reward import EmbedFn, split_units          # noqa: E402


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class Cluster:
    """One position the pool of K samples collectively expressed."""
    centroid: np.ndarray                       # unit-normalized mean of members
    units: list[str] = field(default_factory=list)
    sources: set[int] = field(default_factory=set)   # response indices that hit it

    @property
    def support(self) -> int:
        """Distinct RESPONSES, not units. One sample repeating itself across
        three bullets is one voice, and must not look like three."""
        return len(self.sources)


def embed_units(responses: list[str], embed_fn: EmbedFn, max_units: int = 40
                ) -> tuple[list[list[str]], list[np.ndarray]]:
    """Split each response and embed ALL units of the pool in ONE call.

    Embedding dominates cost, so this mirrors reward.py's embed-once/score-many
    pattern (embed_positions / coverage_rewards_sweep): the same unit vectors are
    reused for every sim_thr, every min_support, and every candidate scored.
    """
    units = [split_units(r or "", max_units) for r in responses]
    flat = [u for us in units for u in us]
    if not flat:
        return units, [np.zeros((0, 0)) for _ in responses]
    V = np.asarray(embed_fn(flat), dtype=float)
    out, i = [], 0
    for us in units:
        out.append(V[i:i + len(us)])
        i += len(us)
    return units, out


def pool_positions(responses: list[str], embed_fn: EmbedFn, sim_thr: float = 0.55,
                   min_support: int = 2, max_units: int = 40,
                   pre: tuple[list[list[str]], list[np.ndarray]] | None = None
                   ) -> list[Cluster]:
    """Units of all K samples -> clusters, greedily agglomerated by cosine.

    Each unit joins its best-matching existing cluster if that cosine >= sim_thr,
    else opens a new one; the centroid is the running normalized mean. Greedy and
    therefore order-dependent (units are visited response-major) -- that is the
    price of doing this in one pass with no k to choose, and it is why sim_thr is
    swept rather than picked.

    ``pre`` is the (units, vectors) from embed_units; pass it to re-cluster at a
    different sim_thr without re-embedding.
    """
    units, U = pre if pre is not None else embed_units(responses, embed_fn, max_units)
    clusters: list[Cluster] = []
    for r_i, (us, V) in enumerate(zip(units, U)):
        for u_i, u in enumerate(us):
            v = V[u_i]
            best, best_s = -1, sim_thr
            for c_i, c in enumerate(clusters):
                s = float(v @ c.centroid)
                if s >= best_s:
                    best, best_s = c_i, s
            if best < 0:
                clusters.append(Cluster(centroid=v.copy(), units=[u], sources={r_i}))
                continue
            c = clusters[best]
            n = len(c.units)
            cen = (c.centroid * n + v) / (n + 1)
            nrm = np.linalg.norm(cen)
            c.centroid = cen / nrm if nrm else cen
            c.units.append(u)
            c.sources.add(r_i)
    # A position only one sample mentions is likely noise, not a viewpoint. This
    # is also the knob most likely to delete a real minority cluster -- sweep it.
    return [c for c in clusters if c.support >= min_support]


# ---------------------------------------------------------------------------
# Scoring against the pool
# ---------------------------------------------------------------------------
def coverage_matrix(clusters: list[Cluster], U: list[np.ndarray],
                    sim_thr: float) -> np.ndarray:
    """(n_responses, n_clusters) bool: does response i express cluster j?

    Covered = some unit of the response is within sim_thr of the cluster
    centroid. Same threshold that built the cluster, so a response that
    CONTRIBUTED a unit to a cluster essentially always covers it -- self-credit
    is intended (the pool is the rubric), but it means scores are not comparable
    across pools of different K.
    """
    if not clusters:
        return np.zeros((len(U), 0), dtype=bool)
    C = np.stack([c.centroid for c in clusters])
    M = np.zeros((len(U), len(clusters)), dtype=bool)
    for i, V in enumerate(U):
        if V.size:
            M[i] = (V @ C.T).max(axis=0) >= sim_thr
    return M


def score_candidate(response: str, clusters: list[Cluster], embed_fn: EmbedFn,
                    sim_thr: float = 0.55, max_units: int = 40,
                    V: np.ndarray | None = None) -> float:
    """Fraction of pool clusters this single response covers, in [0, 1].

    Unweighted, matching OvertonScore's own `covered / len(cluster_ratings)`
    (evaluation/overton/judge_overtonbench.py) -- weighting by cluster size would
    reintroduce the majority bias RewardConfig.weight="uniform" exists to avoid.
    """
    if not clusters:
        return 0.0
    if V is None:
        V = np.asarray(embed_fn(split_units(response or "", max_units)), dtype=float)
    if not V.size:
        return 0.0
    return float(coverage_matrix(clusters, [V], sim_thr)[0].mean())


def select_best_of_k(responses: list[str], embed_fn: EmbedFn, sim_thr: float = 0.55,
                     min_support: int = 2, max_units: int = 40,
                     clusters: list[Cluster] | None = None,
                     pre: tuple[list[list[str]], list[np.ndarray]] | None = None
                     ) -> tuple[int, list[float]]:
    """(index of the sample covering most of the pool, per-sample scores)."""
    if pre is None:
        pre = embed_units(responses, embed_fn, max_units)
    if clusters is None:
        clusters = pool_positions(responses, embed_fn, sim_thr, min_support,
                                  max_units, pre=pre)
    M = coverage_matrix(clusters, pre[1], sim_thr)
    scores = M.mean(axis=1).tolist() if M.shape[1] else [0.0] * len(responses)
    return (int(np.argmax(scores)) if scores else -1), scores


def select_subset(responses: list[str], k: int, embed_fn: EmbedFn,
                  sim_thr: float = 0.55, min_support: int = 2, max_units: int = 40,
                  clusters: list[Cluster] | None = None,
                  pre: tuple[list[list[str]], list[np.ndarray]] | None = None
                  ) -> tuple[list[int], float]:
    """Greedy max-coverage subset of size k -> (indices, union coverage).

    The multi-answer case exists because union (0.657-0.687) beats even the
    hindsight oracle over single answers (0.622): if more than one answer may be
    returned, maximizing the UNION is the objective, not picking a champion.
    Greedy is (1-1/e)-optimal for monotone submodular coverage, which this is.
    """
    if pre is None:
        pre = embed_units(responses, embed_fn, max_units)
    if clusters is None:
        clusters = pool_positions(responses, embed_fn, sim_thr, min_support,
                                  max_units, pre=pre)
    M = coverage_matrix(clusters, pre[1], sim_thr)
    if not M.shape[1]:
        return list(range(min(k, len(responses)))), 0.0
    chosen: list[int] = []
    have = np.zeros(M.shape[1], dtype=bool)
    for _ in range(min(k, len(responses))):
        gain = (M & ~have).sum(axis=1).astype(float)
        gain[chosen] = -1.0
        i = int(np.argmax(gain))
        if gain[i] < 0:
            break
        chosen.append(i)
        have |= M[i]
    return chosen, float(have.mean())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _qid(r: dict) -> int:
    return int(r.get("question_id", r.get("query_id", -1)))


def _sample(r: dict) -> int:
    """Multi-rollout files disagree on the field name: eval_overtonbench writes
    `rollout`, the hivemind generators write `sample_idx`."""
    return int(r.get("sample_idx", r.get("rollout", 0)))


def load_responses(path: str, exclude: set[str]) -> dict[int, list[dict]]:
    """qid -> [{condition, sample, text}], pooled across conditions AND rollouts.

    Pooling across conditions is deliberate: the union/oracle numbers this method
    targets were measured over baseline+scout+div_only, i.e. across strategies.
    With a single-condition multi-rollout file the pool is the K rollouts.
    """
    by_q: dict[int, list[dict]] = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        cond = r.get("condition", "?")
        if cond in exclude or _sample(r) < 0:      # sample_idx -1 = dry-run row
            continue
        by_q[_qid(r)].append({"condition": cond, "sample": _sample(r),
                              "text": r.get("response") or ""})
    for q in by_q:
        by_q[q].sort(key=lambda c: (c["condition"], c["sample"]))
    return by_q


def load_scores(path: str) -> dict[tuple[int, str], float]:
    """(qid, condition) -> judge coverage. NOTE the granularity: the judge CSV
    stores one row per (question, condition) -- with K rollouts that `coverage`
    is already the MEAN over rollouts, so a per-sample achieved score does not
    exist on disk and selection among samples of one condition cannot be
    credited or debited by this file."""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[(int(r["question_id"]), r["condition"])] = float(r["coverage"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return "  n/a " if x != x else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-referential best-of-K selection")
    ap.add_argument("--responses", default="overton_responses_v5.jsonl")
    ap.add_argument("--scores", default=None,
                    help="judge CSV (condition,question_id,coverage). Without it "
                         "this only reports pool statistics, which prove nothing "
                         "about OvertonScore.")
    ap.add_argument("--sim_thr", default="0.55",
                    help="comma list. Clusters are re-agglomerated per value from "
                         "the SAME embeddings, so the sweep is nearly free.")
    ap.add_argument("--min_support", default="2",
                    help="comma list. Drop clusters expressed by fewer than this "
                         "many distinct samples. 1 = keep everything.")
    ap.add_argument("--k", type=int, default=2, help="subset size for select_subset")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--max_units", type=int, default=40)
    ap.add_argument("--exclude", default="",
                    help="comma-separated conditions to drop from the pool, e.g. "
                         "`route` (0.072 -- a degenerate sample pollutes the pool "
                         "with clusters nothing else supports)")
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--stub_embed", action="store_true",
                    help="deterministic hashing embedder; plumbing smoke test only")
    ap.add_argument("--out", default="docs/bestofk_selection.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    thrs = [float(x) for x in args.sim_thr.split(",") if x.strip()]
    sups = [int(x) for x in args.min_support.split(",") if x.strip()]
    drop = {c.strip() for c in args.exclude.split(",") if c.strip()}

    by_q = load_responses(args.responses, drop)
    qids = sorted(by_q)
    if args.max_questions:
        qids = qids[:args.max_questions]
    qids = [q for q in qids if len(by_q[q]) >= 2]
    cov = load_scores(args.scores) if args.scores else {}

    embed_fn = _stub_embed_fn() if args.stub_embed else \
        __import__("alignment.reward", fromlist=["default_embed_fn"]).default_embed_fn(args.embedder)

    conds_all = sorted({c["condition"] for q in qids for c in by_q[q]})
    multi = any(len({c["sample"] for c in by_q[q]}) > 1 for q in qids)
    print(f"{len(qids)} questions, pool = {conds_all} "
          f"({'multi-rollout' if multi else 'one sample per condition'})")
    if multi and cov:
        print("  NOTE: judge CSV is per (question, condition); achieved scores for "
              "samples of the same condition are identical by construction.")

    # --- embed every pool ONCE, then sweep -----------------------------------
    pre_by_q = {}
    for q in qids:
        pre_by_q[q] = embed_units([c["text"] for c in by_q[q]], embed_fn, args.max_units)

    rows = []
    agg = []
    headline = (thrs[0], sups[0])
    for thr in thrs:
        # agglomeration depends on sim_thr only; min_support just filters columns
        pools = {q: pool_positions([c["text"] for c in by_q[q]], embed_fn, thr,
                                   min_support=1, max_units=args.max_units,
                                   pre=pre_by_q[q]) for q in qids}
        for sup in sups:
            n_cl, sel_s, sub_s = [], [], []
            ach, base, rand, orc, hits, n_j = [], [], [], [], 0, 0
            for q in qids:
                cands = by_q[q]
                clusters = [c for c in pools[q] if c.support >= sup]
                pre = pre_by_q[q]
                idx, scores = select_best_of_k([c["text"] for c in cands], embed_fn,
                                               thr, sup, args.max_units,
                                               clusters=clusters, pre=pre)
                sub, sub_cov = select_subset([c["text"] for c in cands], args.k,
                                             embed_fn, thr, sup, args.max_units,
                                             clusters=clusters, pre=pre)
                n_cl.append(len(clusters))
                sel_s.append(scores[idx] if idx >= 0 else 0.0)
                sub_s.append(sub_cov)
                win = cands[idx] if idx >= 0 else {"condition": "?", "sample": -1}

                have = [cov.get((q, c["condition"])) for c in cands]
                a = cov.get((q, win["condition"]))
                if cov and a is not None and all(h is not None for h in have):
                    n_j += 1
                    ach.append(a)
                    rand.append(st.mean(have))              # E[random pick] at K
                    orc.append(max(have))
                    if (q, "baseline") in cov:
                        base.append(cov[(q, "baseline")])
                    if a >= max(have) - 1e-9:
                        hits += 1
                if (thr, sup) == headline:
                    rows.append({"question_id": q, "sim_thr": thr,
                                 "min_support": sup, "n_candidates": len(cands),
                                 "n_clusters": len(clusters),
                                 "sel_condition": win["condition"],
                                 "sel_sample": win["sample"],
                                 "sel_score": round(scores[idx] if idx >= 0 else 0.0, 4),
                                 "subset_score": round(sub_cov, 4),
                                 "achieved": a if a is not None else "",
                                 "oracle": max(h for h in have if h is not None)
                                           if any(h is not None for h in have) else "",
                                 "sel_units": len(pre[0][idx]) if idx >= 0 else 0})
            agg.append({"sim_thr": thr, "min_support": sup,
                        "clusters": st.mean(n_cl) if n_cl else 0.0,
                        "sel_score": st.mean(sel_s) if sel_s else 0.0,
                        "subset": st.mean(sub_s) if sub_s else 0.0,
                        "achieved": st.mean(ach) if ach else float("nan"),
                        "random": st.mean(rand) if rand else float("nan"),
                        "baseline": st.mean(base) if base else float("nan"),
                        "oracle": st.mean(orc) if orc else float("nan"),
                        "hit": hits / n_j if n_j else float("nan"), "n": n_j})

    # --- per question, headline setting only ---------------------------------
    thr0, sup0 = headline
    print(f"\n=== per question (sim_thr={thr0}, min_support={sup0}) ===")
    print(f"  {'qid':>5}{'K':>4}{'pool':>6}{'best':>8}{'sub@' + str(args.k):>8}"
          f"  {'winner':<22}{'achieved':>9}{'oracle':>8}")
    for r in rows:
        a = r["achieved"]
        o = r["oracle"]
        print(f"  {r['question_id']:>5}{r['n_candidates']:>4}{r['n_clusters']:>6}"
              f"{r['sel_score']:>8.3f}{r['subset_score']:>8.3f}  "
              f"{r['sel_condition'] + '#' + str(r['sel_sample']):<22}"
              f"{(f'{a:.3f}' if a != '' else '    -'):>9}"
              f"{(f'{o:.3f}' if o != '' else '    -'):>8}")

    # --- the sweep -----------------------------------------------------------
    print(f"\n=== sweep: pool geometry (selection score is INTERNAL -- not evidence) ===")
    print(f"  {'sim_thr':>8}{'min_sup':>9}{'clusters/q':>12}{'best-of-K':>11}"
          f"{'subset@' + str(args.k):>11}")
    for a in agg:
        print(f"  {a['sim_thr']:>8.2f}{a['min_support']:>9}{a['clusters']:>12.1f}"
              f"{a['sel_score']:>11.3f}{a['subset']:>11.3f}")
    print("  clusters/q collapsing to ~1 => sim_thr too low (everything merges);")
    print("  exploding to ~units/q => too high (no pooling happened at all).")

    if cov:
        print(f"\n=== ACHIEVED OvertonScore of the selected answers ===")
        print(f"  {'sim_thr':>8}{'min_sup':>9}{'selected':>10}{'random@K':>10}"
              f"{'baseline':>10}{'oracle':>9}{'picks_oracle':>14}{'n':>5}")
        for a in agg:
            print(f"  {a['sim_thr']:>8.2f}{a['min_support']:>9}"
                  f"{_fmt(a['achieved']):>10}{_fmt(a['random']):>10}"
                  f"{_fmt(a['baseline']):>10}{_fmt(a['oracle']):>9}"
                  f"{_fmt(a['hit']):>14}{a['n']:>5}")
        print("  selected vs random@K is THE comparison: random@K is the")
        print("    compute-matched control (K samples, pick one blind). Beating")
        print("    `baseline` while losing to `random@K` means the K samples")
        print("    helped and the selector did not.")
        print("  selected vs oracle bounds what any selector over these same K")
        print("    samples could achieve; oracle vs the 0.657-0.687 union number")
        print("    is the part no single-answer selector can ever reach.")
        best = max(agg, key=lambda a: (a["achieved"] if a["achieved"] == a["achieved"]
                                       else -1))
        print(f"  best cell: sim_thr={best['sim_thr']} min_support={best['min_support']}"
              f" -> {_fmt(best['achieved'])} vs random {_fmt(best['random'])}")
        print("  (that cell is selected POST HOC over the sweep -- treat it as an "
              "upper bound, not a result)")

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------------
# Stub embedder + self-test (no model download, no data, no GPU)
# ---------------------------------------------------------------------------
def _stub_embed_fn(dim: int = 256):
    """Deterministic bag-of-words embedder, collision-free vocabulary.

    Same idea as alignment/train_grpo.py:_hashing_embed_fn, but each distinct
    word gets its own dimension so planted topics cannot alias onto each other
    and the assertions below are exact.
    """
    vocab: dict[str, int] = {}

    def _embed(texts):
        V = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", (t or "").lower()):
                V[i, vocab.setdefault(w, len(vocab) % dim)] += 1.0
            n = np.linalg.norm(V[i])
            if n:
                V[i] /= n
        return V
    return _embed


def _selftest() -> None:
    """Hand-built pool: 3 planted positions, one broad sample and two narrow ones.

    Checks the two things the method must do -- derive the position set from the
    pool alone, and rank the sample that covers most of it first -- plus the
    min_support sensitivity, since that knob deletes real positions as readily as
    noise (see the module docstring's risk note).
    """
    embed = _stub_embed_fn()

    def para(topic: str, n: int = 28) -> str:
        # ~n words on one topic + a shared filler token, so cross-topic cosine is
        # small but nonzero (a degenerate 0.0 would not exercise sim_thr at all)
        return " ".join([topic] * n + ["people", "people"])

    broad = "\n".join(para(t) for t in ("alpha", "beta", "gamma"))
    narrow_a = para("alpha", 40)
    narrow_b = para("beta", 40)
    responses = [broad, narrow_a, narrow_b]

    pre = embed_units(responses, embed)
    assert [len(u) for u in pre[0]] == [3, 1, 1], [len(u) for u in pre[0]]

    # min_support=1: all three planted positions survive -> 3/3, 1/3, 1/3
    cl = pool_positions(responses, embed, sim_thr=0.55, min_support=1, pre=pre)
    idx, scores = select_best_of_k(responses, embed, 0.55, 1, clusters=cl, pre=pre)
    assert len(cl) == 3, [(c.units[0].split()[0], c.support) for c in cl]
    assert idx == 0, (idx, scores)
    assert abs(scores[0] - 1.0) < 1e-9, scores
    assert abs(scores[1] - 1 / 3) < 1e-9 and abs(scores[2] - 1 / 3) < 1e-9, scores
    # score_candidate must agree with the matrix path (it is the single-response API)
    assert abs(score_candidate(narrow_a, cl, embed, 0.55) - 1 / 3) < 1e-9

    # min_support=2: `gamma` is expressed by the broad sample ONLY and is deleted.
    # This is the risk in the docstring made mechanical -- a position that exactly
    # one sample raises is indistinguishable from noise by support alone.
    cl2 = pool_positions(responses, embed, sim_thr=0.55, min_support=2, pre=pre)
    _, s2 = select_best_of_k(responses, embed, 0.55, 2, clusters=cl2, pre=pre)
    assert len(cl2) == 2, len(cl2)
    assert abs(s2[0] - 1.0) < 1e-9 and abs(s2[1] - 0.5) < 1e-9, s2

    # sim_thr sensitivity: at thr=0 every unit merges into one cluster and every
    # sample scores 1.0 -- the selector goes blind, it does not error.
    cl0 = pool_positions(responses, embed, sim_thr=0.0, min_support=1, pre=pre)
    _, s0 = select_best_of_k(responses, embed, 0.0, 1, clusters=cl0, pre=pre)
    assert len(cl0) == 1 and s0 == [1.0, 1.0, 1.0], (len(cl0), s0)

    # multi-answer: with the broad sample removed, no single answer exceeds 1/3,
    # and greedy must pick two DIFFERENT positions -> union 2/3.
    narrow = [narrow_a, narrow_b, para("gamma", 40)]
    pre_n = embed_units(narrow, embed)
    cl_n = pool_positions(narrow, embed, sim_thr=0.55, min_support=1, pre=pre_n)
    sub, union = select_subset(narrow, 2, embed, 0.55, 1, clusters=cl_n, pre=pre_n)
    assert len(cl_n) == 3 and len(set(sub)) == 2, (len(cl_n), sub)
    assert abs(union - 2 / 3) < 1e-9, union

    print("bestofk self-test OK")
    print(f"  pool (min_support=1): {len(cl)} clusters, supports="
          f"{[c.support for c in cl]}")
    print(f"  scores: broad {scores[0]:.3f} > narrow_a {scores[1]:.3f} "
          f"= narrow_b {scores[2]:.3f}  -> selected #{idx}")
    print(f"  min_support=2 drops the singleton position: {len(cl)} -> {len(cl2)} "
          f"clusters, scores {s2[0]:.3f}/{s2[1]:.3f}/{s2[2]:.3f}")
    print(f"  sim_thr=0 collapses the pool to {len(cl0)} cluster, all scores 1.000 "
          f"(selector blind, not crashed)")
    print(f"  subset@2 over 3 disjoint narrow answers: picks {sub}, "
          f"union {union:.3f} (> best single {max(1/3, 0):.3f})")


if __name__ == "__main__":
    main()
