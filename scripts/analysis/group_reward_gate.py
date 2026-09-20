"""Stage-1 gate for group-diversity GRPO: does the group reward rank rollouts like the judge?

GRPO's advantage is computed within a group of rollouts to one prompt, so the only
property the reward needs is the right WITHIN-GROUP ordering. This measures it on
real rollouts the judge has already scored, one covered-cluster set per rollout
(judge_overtonbench --dump_rollouts).

JUDGE TARGET per rollout i of a question, from its covered clusters C_i:

    t_i = |C_i minus U_{k!=i} C_k|  +  |C_i|
          leave-one-out contribution   own coverage

the across-sample objective plus the rollout's own substance. The reward is
scored against it by within-question pairwise concordance (chance 0.5; a reward
tie counts as disagreement, as in reward_eval_correlation).

THE CEILING. The judge disagrees with itself: re-judged at another --seed it rates
a different participant subset. Judge-vs-judge concordance (--rollouts_b) is the
most a reward can reach, and the pass bar is set against it rather than a fixed
number:

    PASS  confirm-half concordance >= 0.5 + 0.5 * (self - 0.5)
          AND its bootstrap CI lower bound > 0.5
          AND secondary concordance (vs own coverage |C_i| alone) >= 0.5

NO PEEKING. Variants (lambda x depth x mode) are chosen on a random half of the
questions and the gate is read on the other half, so the winner is not both
selected and tested on the same data.

Also prints stage-0 headroom -- union@K minus within-answer coverage per condition.
If there is no across-sample headroom at this model size, there is nothing for the
reward to train toward.

    python scripts/analysis/group_reward_gate.py --responses r.jsonl \\
        --rollouts s_rollouts.csv --rollouts_b s_seed1_rollouts.csv \\
        --condition baseline --gate
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.reward_eval_correlation import _concordance, _concordance_split  # noqa: E402


def load_judge_sets(path: str) -> tuple[dict, dict]:
    """(cond, qid) -> {rollout: set(covered clusters)}, and (cond, qid) -> n_clusters."""
    sets: dict = defaultdict(lambda: defaultdict(set))
    n_cl: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["condition"], int(r["question_id"]))
            roll = int(r["rollout"])
            sets[key].setdefault(roll, set())
            if int(r["covered"]):
                sets[key][roll].add(int(r["cluster"]))
            n_cl[key] = int(r["n_clusters"])
    return sets, n_cl


def load_responses(path: str) -> dict:
    """(cond, qid) -> {rollout: response text}."""
    out: dict = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["condition"], int(r["question_id"]))][int(r.get("rollout", 0))] = r["response"]
    return out


def judge_targets(by_roll: dict[int, set]) -> dict[int, tuple[float, float]]:
    """rollout -> (leave-one-out contribution + own coverage, own coverage)."""
    out = {}
    for i, C in by_roll.items():
        others = set().union(*(c for j, c in by_roll.items() if j != i))
        out[i] = (len(C - others) + len(C), float(len(C)))
    return out


def pairs_of(x: dict[int, float], y: dict[int, float]) -> list[tuple[float, float]]:
    """(x_i - x_j, y_i - y_j) over unordered rollout pairs present in both."""
    ks = sorted(set(x) & set(y))
    return [(x[a] - x[b], y[a] - y[b]) for a, b in itertools.combinations(ks, 2)]


def pooled(qpairs: dict, qids) -> tuple[float, int]:
    return _concordance([p for q in qids for p in qpairs.get(q, [])])


def boot_ci(qpairs: dict, qids: list, n_boot: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        c, n = pooled(qpairs, [rng.choice(qids) for _ in qids])
        if n:
            vals.append(c)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def headroom(sets: dict, n_cl: dict, min_gap: float = 0.03) -> dict:
    by_cond = defaultdict(list)
    for (cond, qid), by_roll in sets.items():
        n = n_cl[(cond, qid)]
        if not n or len(by_roll) < 2:
            continue
        within = st.mean(len(c) for c in by_roll.values()) / n
        union = len(set().union(*by_roll.values())) / n
        by_cond[cond].append((within, union, len(by_roll)))
    print("=== stage-0 headroom: union@K - within-answer coverage ===")
    gaps = {}
    for cond, rows in sorted(by_cond.items()):
        w = st.mean(r[0] for r in rows)
        u = st.mean(r[1] for r in rows)
        k = st.mean(r[2] for r in rows)
        gaps[cond] = u - w
        flag = (f"  <- NO HEADROOM (<= {min_gap}): nothing across samples to "
                f"train toward") if u - w <= min_gap else ""
        print(f"  {cond:<16} within={w:.4f}  union@{k:.0f}={u:.4f}  gap={u - w:+.4f}  "
              f"(n={len(rows)}){flag}")
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser(description="group-diversity reward vs judge gate")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--rollouts", required=True, help="judge --dump_rollouts csv (seed A)")
    ap.add_argument("--rollouts_b", default=None,
                    help="same responses re-judged at another --seed: the ceiling")
    ap.add_argument("--condition", default="baseline", help="whose rollouts form the groups")
    ap.add_argument("--ref_responses", default=None,
                    help="frozen-base samples per question, for mode 'reference'")
    ap.add_argument("--ref_condition", default="baseline")
    ap.add_argument("--lambdas", default="0,0.5,1,2")
    ap.add_argument("--depths", default="0,30")
    ap.add_argument("--sim_thr", type=float, default=0.55)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--stub_embed", action="store_true",
                    help="topic-keyword embedder from group_reward's self-test (tests only)")
    ap.add_argument("--min_headroom", type=float, default=0.03,
                    help="fail when union@K - within-answer is at or below this: "
                         "GRPO reweights the reference policy's own samples, so "
                         "without across-sample spread there is nothing for a "
                         "diversity reward to select. 0.03 is the measured "
                         "per-question noise floor; 0 disables the check.")
    ap.add_argument("--max_flat", type=float, default=0.30,
                    help="fail when this fraction of groups has ~zero reward "
                         "spread. The advantage is a within-group z-score, so a "
                         "flat group contributes no gradient however good the "
                         "reward is -- the mode-collapse failure (Vendi ~1.4/8).")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--out", default="docs/group_reward_gate.csv")
    ap.add_argument("--gate", action="store_true", help="exit 2 unless the gate passes")
    args = ap.parse_args()

    from alignment.group_reward import GroupRewardConfig, _topic_embed, group_rewards
    from scripts.analysis.bestofk_selection import embed_units

    sets_a, n_cl = load_judge_sets(args.rollouts)
    gaps = headroom(sets_a, n_cl, args.min_headroom)
    # Cheapest kill first: without across-sample spread there is nothing for a
    # diversity reward to select, whatever its concordance turns out to be.
    gap0 = gaps.get(args.condition, float("nan"))
    if args.min_headroom and not (gap0 == gap0 and gap0 > args.min_headroom):
        print(f"\nGATE FAILED: headroom {gap0:+.4f} <= {args.min_headroom} for "
              f"{args.condition!r}. The policy's own samples already say the same "
              f"thing, and GRPO can only reweight what the reference policy "
              f"samples -- a diversity reward has nothing to select. Widen the "
              f"sampling distribution (temperature, verbalized sampling) and "
              f"re-measure before training.")
        if args.gate:
            return 2

    sets_b = load_judge_sets(args.rollouts_b)[0] if args.rollouts_b else None
    resp = load_responses(args.responses)
    ref = load_responses(args.ref_responses) if args.ref_responses else None

    qids = sorted(q for (c, q) in sets_a if c == args.condition and (c, q) in resp)
    if len(qids) < 4:
        print(f"only {len(qids)} questions have both responses and judge rows for "
              f"{args.condition!r}")
        return 2
    rng = random.Random(args.split_seed)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    tune, confirm = sorted(shuffled[: len(qids) // 2]), sorted(shuffled[len(qids) // 2:])

    embed_fn = _topic_embed if args.stub_embed else \
        __import__("alignment.reward", fromlist=["default_embed_fn"]).default_embed_fn(args.embedder)

    modes = ["group"] + (["reference"] if ref else [])
    variants = [(m, float(l), int(d)) for m in modes
                for l in args.lambdas.split(",") for d in args.depths.split(",")]
    q_primary = {v: {} for v in variants}
    q_secondary = {v: {} for v in variants}
    q_flat = {v: {} for v in variants}      # question -> 1 when the group is flat
    q_self: dict = {}

    for q in qids:
        by_roll = sets_a[(args.condition, q)]
        rolls = sorted(set(by_roll) & set(resp[(args.condition, q)]))
        if len(rolls) < 2:
            continue
        tgt = judge_targets({i: by_roll[i] for i in rolls})
        t_prim = {i: tgt[i][0] for i in rolls}
        t_own = {i: tgt[i][1] for i in rolls}
        if sets_b is not None and (args.condition, q) in sets_b:
            tb = judge_targets({i: sets_b[(args.condition, q)].get(i, set()) for i in rolls})
            q_self[q] = pairs_of(t_prim, {i: tb[i][0] for i in rolls})

        texts = [resp[(args.condition, q)][i] for i in rolls]
        refs = list(ref.get((args.ref_condition, q), {}).values()) if ref else []
        units, U = embed_units(texts + refs, embed_fn)
        G = len(texts)
        for m, lam, d in variants:
            cfg = GroupRewardConfig(sim_thr=args.sim_thr, min_depth_words=d,
                                    lambda_div=lam, mode=m)
            if m == "group":
                r, _ = group_rewards(texts, embed_fn, cfg, pre=(units[:G], U[:G]))
            elif refs:
                r, _ = group_rewards(texts, embed_fn, cfg, ref=refs, pre=(units, U))
            else:
                continue
            rr = dict(zip(rolls, r))
            q_primary[(m, lam, d)][q] = pairs_of(t_prim, rr)
            q_secondary[(m, lam, d)][q] = pairs_of(t_own, rr)
            # A group whose rewards are all equal z-scores to zero advantage, so
            # it trains nothing. Measured here, before any GPU time.
            q_flat[(m, lam, d)][q] = int(st.pstdev(r) < 1e-9) if len(r) > 1 else 1

    rows = []
    print(f"\n=== variants on the TUNE half ({len(tune)} questions) ===")
    print(f"  {'mode':<10}{'lambda':>7}{'depth':>6}{'conc':>8}{'tie':>7}{'conc|sep':>10}"
          f"{'secondary':>11}{'pairs':>7}")
    for v in variants:
        c, n = pooled(q_primary[v], tune)
        tie, sep, _ = _concordance_split([p for q in tune for p in q_primary[v].get(q, [])])
        s, _ = pooled(q_secondary[v], tune)
        rows.append({"half": "tune", "mode": v[0], "lambda": v[1], "depth": v[2],
                     "conc": c, "tie_rate": tie, "conc_sep": sep, "secondary": s, "pairs": n})
        print(f"  {v[0]:<10}{v[1]:>7.2f}{v[2]:>6}{c:>8.3f}{tie:>7.2f}{sep:>10.3f}{s:>11.3f}{n:>7}")

    tuned = [r for r in rows if r["pairs"] and r["conc"] == r["conc"]]
    if not tuned:
        print("no variant produced any comparable pairs")
        return 2
    best = max(tuned, key=lambda r: r["conc"])
    v = (best["mode"], best["lambda"], best["depth"])
    c, n = pooled(q_primary[v], confirm)
    lo, hi = boot_ci(q_primary[v], confirm)
    s, _ = pooled(q_secondary[v], confirm)
    base_v = ("group", 0.0, v[2])
    c0, _ = pooled(q_primary[base_v], confirm) if base_v in q_primary else (float("nan"), 0)

    print(f"\n=== chosen on tune: mode={v[0]} lambda={v[1]} depth={v[2]} ===")
    print(f"CONFIRM half ({len(confirm)} questions, {n} pairs):")
    print(f"  primary concordance   {c:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  secondary (own cov)   {s:.3f}")
    print(f"  lambda=0 control      {c0:.3f}  (does the novelty term add agreement?)")

    self_c = float("nan")
    if q_self:
        self_c, n_self = pooled(q_self, confirm)
        print(f"  judge vs itself       {self_c:.3f}  ({n_self} pairs) <- ceiling")
    else:
        print("  judge vs itself       not measured (pass --rollouts_b); the gate "
              "cannot pass without its ceiling")

    # Mode collapse: the fraction of groups the reward cannot separate at all.
    flat_all = list(q_flat[v].values())
    flat_rate = st.mean(flat_all) if flat_all else float("nan")
    n_zero = sum(flat_all)
    print(f"  flat groups           {flat_rate:.3f}  ({n_zero}/{len(flat_all)} with "
          f"identical rewards -> zero advantage)")

    # A judge that cannot reproduce its own ordering (self <= 0.5) leaves nothing
    # to align with, and would drop the bar below chance -- fail rather than pass
    # a reward against noise.
    bar = 0.5 + 0.5 * (self_c - 0.5) if self_c == self_c else float("nan")
    judge_ok = self_c == self_c and self_c > 0.5
    gap = gaps.get(args.condition, float("nan"))
    head_ok = (not args.min_headroom) or (gap == gap and gap > args.min_headroom)
    flat_ok = flat_rate == flat_rate and flat_rate <= args.max_flat
    passed = (judge_ok and head_ok and flat_ok and c >= bar and lo > 0.5 and s >= 0.5)
    rows.append({"half": "confirm", "mode": v[0], "lambda": v[1], "depth": v[2],
                 "conc": c, "tie_rate": float("nan"), "conc_sep": float("nan"),
                 "secondary": s, "pairs": n})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    if bar == bar:
        print(f"\nbar = 0.5 + 0.5*(self - 0.5) = {bar:.3f}")
    print("GATE PASSED" if passed else
          "GATE FAILED: do not train on this reward. Reasons: "
          + ", ".join(x for x, bad in [
              ("no judge ceiling", bar != bar),
              (f"judge vs itself {self_c:.3f} <= 0.5", bar == bar and not judge_ok),
              (f"headroom {gap:+.3f} <= {args.min_headroom}", not head_ok),
              (f"flat groups {flat_rate:.2f} > {args.max_flat}", not flat_ok),
              (f"primary {c:.3f} < bar", bar == bar and c < bar),
              (f"CI lower {lo:.3f} <= 0.5", not lo > 0.5),
              (f"secondary {s:.3f} < 0.5", s < 0.5)] if bad))
    return 0 if passed or not args.gate else 2


if __name__ == "__main__":
    sys.exit(main())
