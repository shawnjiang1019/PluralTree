"""Does the GRPO reward rank answers the way the OvertonBench judge does?

This is the experiment that gates the whole RL phase, and it needs no GPU: if
`coverage_reward` cannot order two answers to the SAME question the way the judge
does, then GRPO's advantage -- which is computed strictly within a group of
rollouts sharing one prompt -- is being driven by noise, and training will
optimize something unrelated to the metric.

WITHIN-QUESTION, NOT POOLED. A pooled correlation across all (question,
condition) rows can look healthy purely because some questions are easier than
others, while the within-question ordering is chance. That exact distinction bit
the judge validation itself (pooled rho 0.673 vs within-participant 0.059), so
this script reports both and treats the within-question number as the real one.

PAIRWISE CONCORDANCE is the primary statistic. With only 3-4 conditions per
question a within-question Spearman takes very few values; concordance over
condition PAIRS (does reward order A,B like the judge does?) pools cleanly across
questions and has an unambiguous chance level of 0.5.

Also scores coverage_reward_v1 alongside v2, so the depth/weighting/shape fixes
can be judged on whether they actually improve agreement with the judge.

Positions come from the anchor the scout actually chose, recovered from the
stored `fork_context` -- so this needs no embeddings, only the graph.

    OPINIONQA_DIR=... python scripts/analysis/reward_eval_correlation.py \
        --responses overton_responses_v5.jsonl --scores overton_scores_v5.csv
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# "[fork 1] at 'How POLIDEOLOGY divides opinion on: On balance, do you thin…'"
_ANCHOR_RE = re.compile(r"\[fork \d+\] (?:at|on) '(.+?)'", re.DOTALL)


def parse_anchor_text(fork_context: str) -> str | None:
    """Anchor description of the TOP fork (truncated to 60 chars by describe_node)."""
    m = _ANCHOR_RE.search(fork_context or "")
    if not m:
        return None
    return m.group(1).rstrip("…").rstrip()


def build_anchor_index(graph) -> list[tuple[str, int]]:
    """(entity_text, node_id) for axis nodes, for prefix matching."""
    out = []
    for nid, name in enumerate(graph.id_to_entity):
        if name.startswith("ax:"):
            out.append((graph.entity_text.get(nid, ""), nid))
    return out


def resolve_anchor(prefix: str, index: list[tuple[str, int]]) -> int | None:
    for text, nid in index:
        if text.startswith(prefix):
            return nid
    return None


def _medoid(texts: list[str], embed_fn) -> str:
    """The most representative single text in a cluster.

    coverage_reward takes ONE embed_text per target, so a cluster of k
    perspectives has to collapse to one string. Joining them embeds to their
    centroid-of-words, which is not a viewpoint anyone expressed; the medoid is
    an actual participant's wording and is the closest to all the others.
    """
    import numpy as np

    if len(texts) == 1:
        return texts[0]
    E = np.asarray(embed_fn(texts), dtype=float)
    return texts[int((E @ E.T).mean(axis=1).argmax())]


def cluster_positions(split: str, embed_fn, holdout: bool = False) -> dict:
    """qid -> [Position], one per judge cluster, from participants' own words.

    THE POINT. The reward scores coverage of GRAPH POSITIONS (~19.9/question);
    the judge scores coverage of VIEWPOINT CLUSTERS (~5.65). Those are different
    target sets, which is the leading explanation for concordance sitting below
    chance -- the reward is not miscalibrated, it is measuring something else.
    This builds the reward's targets from the clusters instead, so the two sides
    are finally aimed at the same object.

    Targets are HUMAN-WRITTEN. `cluster_kmeans` ships with OvertonBench and the
    text is each participant's own free response, so this is not fitting a
    judge-generated label -- the judge only rates.

    ``holdout`` keeps every second participant per cluster, so the reward's
    target text is drawn from a subset. Note this REDUCES overlap with what the
    judge rates, it does not eliminate it: the judge draws its own seeded
    ``max_users`` subset from all participants and nothing here constrains that
    draw. Treat the no-holdout number as an upper bound and this one as the
    honest-but-still-optimistic one; a clean split needs the judge to take the
    complement, which is a change to judge_overtonbench.
    """
    from alignment.reward import Position
    from evaluation.overton.judge_overtonbench import load_index

    idx = load_index(split)
    by_q: dict = defaultdict(lambda: defaultdict(list))
    for (qid, _user), e in sorted(idx.items()):
        if (e.get("perspective") or "").strip():
            by_q[qid][e["cluster"]].append(e["perspective"].strip())

    out, n_drop = {}, 0
    for qid, clusters in by_q.items():
        total = sum(len(v) for v in clusters.values()) or 1
        pos = []
        for cl, texts in sorted(clusters.items()):
            use = texts[::2] if holdout else texts
            if not use:
                continue
            # prevalence is the FULL cluster size even under holdout -- how
            # widely a view is held is a property of the population, not of
            # which half we sampled to describe it.
            pos.append(Position(option=f"c{cl}", embed_text=_medoid(use, embed_fn),
                                prevalence=len(texts) / total))
        if len(pos) >= 2:
            out[qid] = pos
        else:
            n_drop += 1
    n_pos = sum(len(v) for v in out.values())
    print(f"cluster targets: {len(out)} questions, {n_pos} clusters "
          f"({n_pos / max(1, len(out)):.2f}/question), dropped {n_drop} with <2"
          + ("  [holdout: every 2nd participant]" if holdout else ""))
    return out


def _concordance(pairs: list[tuple[float, float]]) -> tuple[float, int]:
    """Fraction of (judge_delta, reward_delta) pairs that agree in sign.

    Pairs where the judge is tied carry no information and are dropped; pairs
    where the REWARD is tied count as disagreement (a reward that cannot
    separate two answers gives GRPO nothing).
    """
    used = [(j, r) for j, r in pairs if abs(j) > 1e-9]
    if not used:
        return float("nan"), 0
    agree = sum(1 for j, r in used if (j > 0) == (r > 0) and abs(r) > 1e-9)
    return agree / len(used), len(used)


def _concordance_split(pairs: list[tuple[float, float]]) -> tuple[float, float, int]:
    """(tie_rate, concordance_among_separated, n_separated).

    The headline concordance charges reward ties as disagreements, which is right
    for GRPO but conflates two very different failures: a reward that ORDERS
    WRONGLY vs one that is simply ZERO almost everywhere and cannot order at all.
    Splitting them says which one you have -- and therefore whether to redesign
    the reward or just recalibrate its threshold.
    """
    used = [(j, r) for j, r in pairs if abs(j) > 1e-9]
    if not used:
        return float("nan"), float("nan"), 0
    sep = [(j, r) for j, r in used if abs(r) > 1e-9]
    tie_rate = 1.0 - len(sep) / len(used)
    if not sep:
        return tie_rate, float("nan"), 0
    agree = sum(1 for j, r in sep if (j > 0) == (r > 0))
    return tie_rate, agree / len(sep), len(sep)


def _corr(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Reward vs judge agreement")
    ap.add_argument("--responses", default="overton_responses_v5.jsonl")
    ap.add_argument("--scores", default="overton_scores_v5.csv")
    ap.add_argument("--seed", type=int, default=42, help="graph split seed")
    ap.add_argument("--targets", choices=["graph", "clusters"], default="graph",
                    help="what the reward is scored against. 'graph' = "
                         "positions_from_subtree (~19.9/question, what the reward "
                         "has always used). 'clusters' = the judge's own "
                         "cluster_kmeans groups via participants' free responses "
                         "(~5.65/question) -- the redesign, testing whether the "
                         "below-chance concordance is a target mismatch rather "
                         "than a broken reward.")
    ap.add_argument("--holdout", action="store_true",
                    help="clusters only: build each cluster's target text from "
                         "every 2nd participant, reducing (not eliminating) "
                         "overlap with what the judge rates. See cluster_positions.")
    ap.add_argument("--split", default="full", help="OvertonBench split for --targets clusters")
    ap.add_argument("--min_depth_words", default=None,
                    help="comma list of min_depth_words to sweep, e.g. 0,30,60,90,120. "
                         "All values are scored from ONE embedding pass (d only "
                         "thresholds position_depths; the sim matrix is identical), "
                         "so the sweep costs the same as a single run. Default: "
                         "RewardConfig's value alone.")
    ap.add_argument("--match_thr", default=None,
                    help="comma list of match_thr to sweep, e.g. 0.30,0.35,0.40,0.50. "
                         "Measured: pos_best p50~0.345, p75~0.472, so the 0.50 default "
                         "sits above the 75th percentile of the signal. Swept from the "
                         "same embedding pass as --min_depth_words.")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--exclude", default="",
                    help="comma-separated conditions to drop. Run v6 BOTH ways: "
                         "with `route` included the reward only has to notice a "
                         "0.4 collapse (easy); excluding it leaves the near-ties, "
                         "which is the regime GRPO actually operates in -- every "
                         "rollout in a group comes from the SAME policy and they "
                         "resemble each other far more than baseline resembles route")
    ap.add_argument("--out", default="docs/reward_eval_correlation.csv")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="if >0, exit 2 when headline-d within-question concordance "
                         "falls below this. Makes the Stage-0 gate a real exit code "
                         "so `sbatch --dependency=afterok` cannot chain a training "
                         "run onto a reward that ranks at chance. 0 = report only.")
    args = ap.parse_args()
    drop = {c.strip() for c in args.exclude.split(",") if c.strip()}

    from alignment.reward import (RewardConfig, coverage_rewards_sweep,
                                  default_embed_fn, embed_positions,
                                  positions_from_subtree)
    from data.loaders.opinionqa import load_opinionqa

    # --- load responses + judge scores -------------------------------------
    rows = [json.loads(l) for l in open(args.responses, encoding="utf-8")]
    resp = defaultdict(dict)          # qid -> cond -> response
    ctx = {}                          # qid -> fork_context (from any injected row)
    for r in rows:
        resp[r["question_id"]][r["condition"]] = r.get("response") or ""
        if r.get("fork_context") and r["question_id"] not in ctx:
            ctx[r["question_id"]] = r["fork_context"]

    cov = defaultdict(dict)
    for r in csv.DictReader(open(args.scores, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])

    embed_fn = default_embed_fn(args.embedder)
    cl_pos = {}
    if args.targets == "clusters":
        graph = index = None            # neither the graph nor anchors are needed
        cl_pos = cluster_positions(args.split, embed_fn, args.holdout)
    else:
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
        index = build_anchor_index(graph)
    cfg = RewardConfig()
    depths = ([int(d) for d in args.min_depth_words.split(",") if d.strip()]
              if args.min_depth_words else [cfg.min_depth_words])
    thrs = ([float(t) for t in args.match_thr.split(",") if t.strip()]
            if args.match_thr else [cfg.match_thr])
    t_main = cfg.match_thr if cfg.match_thr in thrs else thrs[0]
    d_main = cfg.min_depth_words if cfg.min_depth_words in depths else depths[0]
    print(f"reward cfg: min_depth_words={depths} (headline d={d_main}) "
          f"match_thr={thrs} (headline t={t_main}) weight={cfg.weight} "
          f"l_precision={cfg.l_precision} l_verbose={cfg.l_verbose}")

    # --- score every response with both rewards -----------------------------
    out_rows, n_noanchor = [], 0
    pairs_v1 = []                       # (judge_delta, reward_delta) within question
    grid = [(t, d) for t in thrs for d in depths]
    pairs_v2 = {k: [] for k in grid}    # ...one list per (match_thr, depth) cell
    pooled_j, pooled_r1 = [], []
    pooled_r2 = {k: [] for k in grid}
    pos_best, unit_best = [], []        # cosine distributions match_thr slices
    pos_best_rw, pos_best_fb = [], []   # ...split by rewritten vs fallback target
    n_units = []                        # units per response (0 => auto-zero reward)
    per_q_pick = {"v1": 0, "n": 0, **{k: 0 for k in grid}}

    for qid in sorted(resp):
        conds = [c for c in resp[qid] if c in cov.get(qid, {}) and c not in drop]
        if len(conds) < 2:
            continue
        if args.targets == "clusters":
            positions = cl_pos.get(qid, [])
            if len(positions) < 2:
                n_noanchor += 1
                continue
            # The rewritten/fallback split is a property of the position-statement
            # artifact, which cluster targets do not use -- participants wrote
            # their own text. Flagged False so the slice stays well-defined and
            # everything lands in pos_best_fb.
            rewritten = [False] * len(positions)
        else:
            anchor_txt = parse_anchor_text(ctx.get(qid, ""))
            anchor = resolve_anchor(anchor_txt, index) if anchor_txt else None
            if anchor is None:
                n_noanchor += 1
                continue
            positions = positions_from_subtree(graph, anchor)
            if len(positions) < 2:
                n_noanchor += 1
                continue

            # Which positions actually got a declarative statement? The artifact
            # only covers ~47% of ATP pairs, so a flat cosine distribution could
            # mean either "rewriting does not help" or "too few were rewritten to
            # show". Comparing against the forced-fallback build separates those,
            # and they need opposite responses (drop the idea vs run --backend llm
            # on the tail).
            fb = positions_from_subtree(graph, anchor, statements={})
            fb_text = {p.option: p.embed_text for p in fb}
            rewritten = [p.embed_text != fb_text.get(p.option) for p in positions]

        # embed the positions ONCE per question (identical for every condition),
        # and get both rewards from a single pass over each response's units
        P = embed_positions(positions, embed_fn)
        scored = {}
        for c in conds:
            r1, r2_grid, sim = coverage_rewards_sweep(resp[qid][c], positions,
                                                      embed_fn, cfg, P=P,
                                                      depths=depths, thrs=thrs)
            if sim.size:
                pb = sim.max(axis=0)
                pos_best.extend(pb.tolist())                # per position
                unit_best.extend(sim.max(axis=1).tolist())  # per unit
                for v, rw in zip(pb.tolist(), rewritten):
                    (pos_best_rw if rw else pos_best_fb).append(v)
                n_units.append(sim.shape[0])
            else:
                n_units.append(0)      # split_units dropped everything (<8 words
                #                        per unit) -> reward is 0 whatever it said
            scored[c] = (r1, r2_grid)
            j = cov[qid][c]
            pooled_j.append(j); pooled_r1.append(r1)
            for k in grid:
                pooled_r2[k].append(r2_grid[k])
            out_rows.append({"question_id": qid, "condition": c, "judge_coverage": j,
                             "reward_v1": r1, "reward_v2": r2_grid[(t_main, d_main)],
                             **{f"reward_v2_t{t}_d{d}": r2_grid[(t, d)]
                                for t, d in grid},
                             "n_positions": len(positions), "anchor": anchor})

        for i in range(len(conds)):
            for k in range(i + 1, len(conds)):
                a, b = conds[i], conds[k]
                jd = cov[qid][a] - cov[qid][b]
                pairs_v1.append((jd, scored[a][0] - scored[b][0]))
                for k in grid:
                    pairs_v2[k].append((jd, scored[a][1][k] - scored[b][1][k]))

        best_j = max(conds, key=lambda c: cov[qid][c])
        if len({cov[qid][c] for c in conds}) > 1:
            per_q_pick["n"] += 1
            if max(conds, key=lambda c: scored[c][0]) == best_j:
                per_q_pick["v1"] += 1
            for k in grid:
                if max(conds, key=lambda c: scored[c][1][k]) == best_j:
                    per_q_pick[k] += 1

    n_q = len({r["question_id"] for r in out_rows})
    print(f"\nscored {len(out_rows)} responses over {n_q} questions "
          f"({n_noanchor} skipped: "
          f"{'fewer than 2 clusters' if args.targets == 'clusters' else 'no resolvable anchor'})")

    # --- the numbers --------------------------------------------------------
    c1, n1 = _concordance(pairs_v1)
    print(f"\n=== WITHIN-QUESTION pairwise concordance (chance = 0.500) ===")
    print(f"  reward_v1 (depth-blind)      {c1:.3f}   over {n1} condition pairs")
    conc = {}
    for k in grid:
        c2, n2 = _concordance(pairs_v2[k])
        conc[k] = c2
        star = " <- headline" if k == (t_main, d_main) else ""
        print(f"  reward_v2  t={k[0]:.2f} d={k[1]:<4}   {c2:.3f}   over {n2} "
              f"condition pairs{star}")
    print("  ^ THIS is what GRPO's advantage consumes. At ~0.5 the reward cannot")
    print("    order rollouts of the same prompt and training optimizes noise.")

    # Decompose: a LOW headline number means either "orders wrongly" or "is zero
    # everywhere and cannot order". Those need opposite fixes -- redesign vs
    # recalibrate -- so never act on the headline alone.
    print(f"\n=== decomposed: ties vs wrong ordering ===")
    print(f"  {'':<22}{'tie_rate':>10}{'conc|separated':>16}{'n_sep':>8}")
    t1, cs1, ns1 = _concordance_split(pairs_v1)
    print(f"  {'reward_v1':<22}{t1:>10.3f}{cs1:>16.3f}{ns1:>8}")
    for k in grid:
        t, cs, ns = _concordance_split(pairs_v2[k])
        lbl = f"reward_v2 t={k[0]:.2f} d={k[1]}"
        print(f"  {lbl:<22}{t:>10.3f}{cs:>16.3f}{ns:>8}")
    print("  tie_rate high + conc|separated ~0.5+ => reward is too SPARSE;")
    print("    lower match_thr (or fix the position embed_text), do not redesign.")
    print("  tie_rate low + conc|separated <0.5  => reward orders WRONGLY;")
    print("    the objective itself disagrees with the judge.")

    # The threshold sweep at the headline depth: the one table that says where
    # match_thr belongs. Rising conc with falling tie_rate = recalibrate and go.
    if len(thrs) > 1:
        print(f"\n=== match_thr sweep at d={d_main} (where should the threshold sit?) ===")
        print(f"  {'thr':>6}{'concordance':>13}{'tie_rate':>10}"
              f"{'conc|sep':>10}{'n_sep':>7}{'pos>=thr':>10}")
        for t in thrs:
            c2, _ = _concordance(pairs_v2[(t, d_main)])
            tr, cs, ns = _concordance_split(pairs_v2[(t, d_main)])
            frac = (sum(1 for v in pos_best if v >= t) / len(pos_best)
                    if pos_best else float("nan"))
            print(f"  {t:>6.2f}{c2:>13.3f}{tr:>10.3f}{cs:>10.3f}{ns:>7}{frac:>10.3f}")

    print(f"\n=== picks the judge's best condition (chance = 1/n_conds) ===")
    if per_q_pick["n"]:
        n = per_q_pick["n"]
        print(f"  reward_v1            {per_q_pick['v1']}/{n} ({per_q_pick['v1']/n:.2f})")
        for k in grid:
            lbl = f"reward_v2 t={k[0]:.2f} d={k[1]}"
            print(f"  {lbl:<22}{per_q_pick[k]}/{n} ({per_q_pick[k]/n:.2f})")

    print(f"\n=== POOLED correlation (for contrast -- NOT the relevant number) ===")
    print(f"  corr(reward_v1, judge)  = {_corr(pooled_r1, pooled_j):+.3f}")
    for k in grid:
        print(f"  corr(v2 t={k[0]:.2f} d={k[1]}, judge) = "
              f"{_corr(pooled_r2[k], pooled_j):+.3f}")
    print("  ^ can look fine purely from between-question difficulty variance")

    print(f"\n=== reward distributions (is the reward even discriminative?) ===")
    dists = [("v1", pooled_r1)] \
            + [(f"v2 t{k[0]:.2f} d{k[1]}", pooled_r2[k]) for k in grid] \
            + [("judge", pooled_j)]
    for name, vals in dists:
        zeros = sum(1 for v in vals if v <= 1e-9) / max(1, len(vals))
        print(f"  {name:<16} mean={st.mean(vals):.3f} sd={st.pstdev(vals):.3f} "
              f"frac_zero={zeros:.2f}")

    # --- where should match_thr actually sit? --------------------------------
    # `mentioned` is pos_best >= match_thr. If the pos_best distribution lies
    # mostly BELOW the threshold, the reward is zero for structural reasons and
    # no amount of depth tuning matters.
    if pos_best:
        qs = [0.10, 0.25, 0.50, 0.75, 0.90]
        def _q(v, p):
            s = sorted(v); i = min(len(s) - 1, int(p * len(s)))
            return s[i]
        print(f"\n=== cosine distributions that match_thr={t_main} slices ===")
        rows = [("pos_best (mentioned?)", pos_best),
                ("unit_best (precision?)", unit_best)]
        # Did the declarative rewrite raise cosine at all? The artifact covers
        # only ~47% of ATP pairs, so a flat overall distribution is ambiguous
        # between "rewriting does not work" and "too few were rewritten".
        if pos_best_rw and pos_best_fb:
            rows += [("  ...rewritten", pos_best_rw),
                     ("  ...fallback", pos_best_fb)]
        for name, vals in rows:
            qtxt = "  ".join(f"p{int(p*100)}={_q(vals, p):.3f}" for p in qs)
            over = sum(1 for v in vals if v >= t_main) / len(vals)
            print(f"  {name:<24} {qtxt}   >=thr: {over:.3f}   n={len(vals)}")
        if pos_best_rw and pos_best_fb:
            d50 = _q(pos_best_rw, 0.50) - _q(pos_best_fb, 0.50)
            print(f"  rewritten - fallback median = {d50:+.3f}")
            print("  ^ near 0 => declarative rewriting does not raise cosine; the")
            print("    register theory is wrong and --backend llm will not help.")
            print("    Clearly positive => coverage is the limit; run the llm tail.")
        print("  If pos_best p75 < match_thr the threshold is the binding")
        print("  constraint -- recalibrate it before touching min_depth_words.")
    if n_units:
        empty = sum(1 for n in n_units if n == 0) / len(n_units)
        print(f"  units/response: mean={st.mean(n_units):.1f} "
              f"min={min(n_units)}  frac_with_ZERO_units={empty:.3f}")
        print("  ^ split_units drops any unit under 8 words; a response left with")
        print("    no units scores 0 no matter what it says.")

    if out_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.out}")

    # --- the gate ------------------------------------------------------------
    # Exit code, not just a printed number: a training job chained with
    # `--dependency=afterok` must not launch on a reward that ranks at chance.
    if args.gate > 0:
        c = conc.get((t_main, d_main), float("nan"))
        best = max(conc, key=lambda k: (conc[k] if conc[k] == conc[k] else -1))
        if not (c >= args.gate):
            print(f"\nGATE: FAIL  concordance {c:.3f} < {args.gate:.3f} at "
                  f"t={t_main} d={d_main}")
            if conc.get(best, 0) >= args.gate:
                print(f"  (t={best[0]:.2f} d={best[1]} would pass at {conc[best]:.3f} "
                      f"-- refit the config before treating that as a pass)")
            sys.exit(2)
        print(f"\nGATE: PASS  concordance {c:.3f} >= {args.gate:.3f} at "
              f"t={t_main} d={d_main}")


if __name__ == "__main__":
    main()
