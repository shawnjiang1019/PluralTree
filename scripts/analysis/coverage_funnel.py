"""Where is a viewpoint lost -- retrieval, drafting, merging, or judging?

Every structural ablation ties (curvature p=0.88, flat retrieval p=0.34) and
30-45% of viewpoint clusters are covered by NO arm. Those facts do not say where
the coverage goes missing, and each place implies a different fix. This walks
each (question, cluster) through four stages and reports which one it died at:

    S1 retrieved  a participant statement of that cluster matches a unit of the
                  injected fork block
    S2 drafted    ...matches a unit of some draft
    S3 answered   ...matches a unit of the final answer (optionally with a word
                  floor, the depth rule coverage_reward uses)
    S4 covered    the judge said so (judge_overtonbench --dump_clusters)

    bucket                      reading                         fix it points at
    not S1                      retrieval gap                   graph coverage
    S1 and not S2               the drafts ignored the block    delivery (CAD,
                                                                prompt, VS)
    S2 and not S3               the merge dropped it            merge guard
    S3 and not S4               said but not credited           depth, or judge
    S4 and not S1               the model had it unaided        graph not needed

GAIN ATTRIBUTION. Of the clusters an arm covers that baseline misses, what share
were actually retrieved (S1)? That is the content attribution `merge_v2_rand` was
meant to provide and never validly did -- computed here from files already on
disk.

WHAT THIS IS NOT. Matching a statement to a unit by cosine is CORRELATIONAL: the
model may have written that sentence without the fork. Read S1 as "the content
was available", not "the content was used". Thresholds are swept for the same
reason -- if the buckets move a lot across theta, the split is a threshold
artifact.

    python scripts/analysis/coverage_funnel.py \\
        --responses overton_responses_flat.jsonl \\
        --clusters overton_scores_flat_clusters.csv \\
        --conditions baseline,merge_v2,merge_v2_flat
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alignment.reward import split_units                       # noqa: E402

STAGES = ("retrieval_gap", "drafting_gap", "merge_loss", "delivery_gap",
          "covered", "unaided")


def load_clusters(path: str) -> tuple[dict, dict]:
    """(qid, cond) -> {cluster: covered0/1}, and qid -> n_clusters."""
    cov: dict = defaultdict(dict)
    n_cl: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = int(r["question_id"])
            cov[(qid, r["condition"])][int(r["cluster"])] = int(r["covered"])
            n_cl[qid] = int(r["n_clusters"])
    return cov, n_cl


def load_texts(path: str) -> dict:
    """(qid, cond) -> {"answer": [...], "fork": str, "drafts": [...]}.

    Rollouts are POOLED: the judge's covered set is the union over rollouts, so
    the text side must be the union too or the stages use different denominators.
    """
    out: dict = defaultdict(lambda: {"answer": [], "fork": "", "drafts": []})
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            e = out[(int(r["question_id"]), r["condition"])]
            if (r.get("response") or "").strip():
                e["answer"].append(r["response"])
            if not e["fork"] and (r.get("fork_context") or "").strip():
                e["fork"] = r["fork_context"]
            # New rows carry every draft with its own think; older ones only
            # draft_a/draft_b. Both are accepted so historical runs still work.
            if isinstance(r.get("draft_traces"), list):
                e["drafts"] += [d.get("text", "") for d in r["draft_traces"]]
            else:
                e["drafts"] += [r.get("draft_a") or "", r.get("draft_b") or ""]
    return out


def cluster_statements(split: str) -> dict:
    """qid -> {cluster: [participant statements]} -- the eval's own text."""
    from evaluation.overton.judge_overtonbench import load_index

    idx = load_index(split)
    out: dict = defaultdict(lambda: defaultdict(list))
    for (qid, _user), e in idx.items():
        t = (e.get("perspective") or "").strip()
        if t:
            out[int(qid)][int(e["cluster"])].append(t)
    return out


def _unit_vectors(texts: list[str], embed_fn, max_units: int = 40):
    """(units, matrix) for a list of text blobs, embedded in ONE call."""
    units: list[str] = []
    for t in texts:
        units += split_units(t or "", max_units)
    if not units:
        return [], np.zeros((0, 0))
    V = np.asarray(embed_fn(units), dtype=float)
    return units, V


def stage_hits(stmt_V: np.ndarray, units: list[str], V: np.ndarray,
               theta: float, depth: int) -> np.ndarray:
    """Per cluster: does any unit match one of its statements at >= theta?

    With ``depth`` > 0 a cluster also needs that many words of units assigned to
    it (argmax over clusters), the rule coverage_reward uses to separate naming a
    viewpoint from articulating it.
    """
    n_cl = stmt_V.shape[0]
    if not len(units) or not V.size or not n_cl:
        return np.zeros(n_cl, dtype=bool)
    S = V @ stmt_V.T                                  # (units, clusters)
    hit = S.max(axis=0) >= theta
    if depth <= 0:
        return hit
    words = np.zeros(n_cl)
    for u_i, (c_i, s) in enumerate(zip(S.argmax(axis=1), S.max(axis=1))):
        if s >= theta:
            words[c_i] += len(units[u_i].split())
    return hit & (words >= depth)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage attribution for missed viewpoints")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--clusters", required=True, help="judge --dump_clusters csv")
    ap.add_argument("--conditions", default="", help="default: all in the dump")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--split", default="full")
    ap.add_argument("--thresholds", default="0.35,0.45,0.55")
    ap.add_argument("--depth", type=int, default=0,
                    help="words an answer must spend on a cluster for S3 (0 = mention)")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--stub_embed", action="store_true", help="tests only")
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--out", default="docs/coverage_funnel.csv")
    args = ap.parse_args()

    cov, _n_cl = load_clusters(args.clusters)
    texts = load_texts(args.responses)
    stmts = cluster_statements(args.split)

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()] or \
        sorted({c for _q, c in cov})
    qids = sorted({q for q, _c in cov})
    if args.max_questions:
        qids = qids[: args.max_questions]

    if args.stub_embed:
        from alignment.group_reward import _topic_embed as embed_fn
    else:
        from alignment.reward import default_embed_fn
        embed_fn = default_embed_fn(args.embedder)

    thetas = [float(t) for t in args.thresholds.split(",") if t.strip()]
    counts = {(c, t): defaultdict(int) for c in conds for t in thetas}
    gain = {(c, t): [0, 0] for c in conds for t in thetas}   # [retrieved, total]

    for qid in qids:
        by_cl = stmts.get(qid)
        if not by_cl:
            continue
        cl_ids = sorted(by_cl)
        flat = [s for cl in cl_ids for s in by_cl[cl]]
        owner = [i for i, cl in enumerate(cl_ids) for _ in by_cl[cl]]
        SV = np.asarray(embed_fn(flat), dtype=float)

        def per_cluster(text_list, theta, depth=0):
            """Collapse statement-level hits to cluster level (any member hits)."""
            units, V = _unit_vectors(text_list, embed_fn)
            hits = stage_hits(SV, units, V, theta, depth)
            out = np.zeros(len(cl_ids), dtype=bool)
            for s_i, c_i in enumerate(owner):
                out[c_i] |= bool(hits[s_i])
            return out

        for cond in conds:
            e = texts.get((qid, cond))
            judged = cov.get((qid, cond))
            if e is None or not judged:
                continue
            base_judged = cov.get((qid, args.baseline), {})
            for theta in thetas:
                s1 = per_cluster([e["fork"]], theta) if e["fork"] else \
                    np.zeros(len(cl_ids), dtype=bool)
                s2 = per_cluster(e["drafts"], theta) if any(e["drafts"]) else s1 & False
                s3 = per_cluster(e["answer"], theta, args.depth)
                for i, cl in enumerate(cl_ids):
                    s4 = bool(judged.get(cl, 0))
                    c = counts[(cond, theta)]
                    c["n"] += 1
                    if s4:
                        c["covered"] += 1
                        if not s1[i]:
                            c["unaided"] += 1
                        if cond != args.baseline and not base_judged.get(cl, 0):
                            gain[(cond, theta)][1] += 1
                            gain[(cond, theta)][0] += int(bool(s1[i]))
                        continue
                    if not s1[i]:
                        c["retrieval_gap"] += 1
                    elif not s2[i]:
                        c["drafting_gap"] += 1
                    elif not s3[i]:
                        c["merge_loss"] += 1
                    else:
                        c["delivery_gap"] += 1

    rows = []
    print(f"=== stage attribution ({len(qids)} questions, depth={args.depth}) ===")
    hdr = f"  {'condition':<18}{'theta':>6}{'n':>7}" + "".join(f"{s:>15}" for s in STAGES)
    print(hdr)
    for cond in conds:
        for theta in thetas:
            c = counts[(cond, theta)]
            n = max(1, c["n"])
            row = {"condition": cond, "theta": theta, "n": c["n"]}
            row.update({s: c[s] for s in STAGES})
            row.update({f"{s}_rate": round(c[s] / n, 4) for s in STAGES})
            g_ret, g_tot = gain[(cond, theta)]
            row["gain_clusters"] = g_tot
            row["gain_retrieved_rate"] = round(g_ret / g_tot, 4) if g_tot else None
            rows.append(row)
            print(f"  {cond:<18}{theta:>6.2f}{c['n']:>7}" +
                  "".join(f"{c[s] / n:>14.1%}" for s in STAGES))

    print("\n=== of the clusters this arm covers that baseline misses, how many "
          "were retrieved? ===")
    for cond in conds:
        if cond == args.baseline:
            continue
        for theta in thetas:
            g_ret, g_tot = gain[(cond, theta)]
            if g_tot:
                print(f"  {cond:<18} theta={theta:.2f}  {g_ret}/{g_tot} "
                      f"({g_ret / g_tot:.1%}) were in the injected block")
    print("  A LOW share means the arm's extra clusters did not come from the "
          "graph -- the gain would be drafting variance, not retrieval.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    mid = thetas[len(thetas) // 2]
    worst = max(conds, key=lambda c: counts[(c, mid)]["n"])
    c = counts[(worst, mid)]
    misses = {s: c[s] for s in ("retrieval_gap", "drafting_gap", "merge_loss",
                                "delivery_gap")}
    if any(misses.values()):
        top = max(misses, key=misses.get)
        nxt = {"retrieval_gap": "graph coverage: more survey modules, or better "
                                "anchors -- the viewpoint was never retrieved",
               "drafting_gap": "delivery: the drafts ignored the injected block "
                               "(CAD, verbalized sampling, prompt work)",
               "merge_loss": "the merge: drafts held it and the merged answer "
                             "did not (guard, concat fallback)",
               "delivery_gap": "depth or the judge: the answer expressed it and "
                               "the judge did not credit it"}[top]
        print(f"\ndominant miss for {worst} at theta={mid:.2f}: {top} "
              f"({misses[top]}/{sum(misses.values())} of misses)\n  -> {nxt}")
    print("  NOTE: matching is correlational -- S1 means the content was "
          "available, not that the model used it.")
    return 0


def _selftest() -> None:
    """Each bucket, on hand-built rows with the topic-keyword embedder."""
    import tempfile

    from alignment.group_reward import _deep, _topic_embed

    d = tempfile.mkdtemp()
    rp, cp = os.path.join(d, "r.jsonl"), os.path.join(d, "c.csv")
    # taxes: only in the fork block (retrieval reached it, drafts did not)
    # guns:  in a draft, not in the answer      -> merge_loss
    # climate: in the answer, judge says no     -> delivery_gap
    # health: nowhere                           -> retrieval_gap
    with open(rp, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": 1, "condition": "merge_v2", "rollout": 0,
            "response": _deep("climate"), "fork_context": _deep("taxes"),
            "draft_traces": [{"label": "scout", "text": _deep("guns"), "think": ""}],
        }) + "\n")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "condition", "cluster", "covered", "n_clusters"])
        for cl in range(4):
            w.writerow([1, "merge_v2", cl, 0, 4])

    stmts = {1: {0: [_deep("taxes")], 1: [_deep("guns")], 2: [_deep("climate")],
                 3: [_deep("health")]}}
    cov, _ = load_clusters(cp)
    texts = load_texts(rp)
    e = texts[(1, "merge_v2")]
    cl_ids = sorted(stmts[1])
    SV = np.asarray(_topic_embed([stmts[1][c][0] for c in cl_ids]), dtype=float)

    def hits(blobs):
        units, V = _unit_vectors(blobs, _topic_embed)
        return stage_hits(SV, units, V, 0.55, 0)

    s1, s2, s3 = hits([e["fork"]]), hits(e["drafts"]), hits(e["answer"])
    assert list(s1) == [True, False, False, False], s1
    assert list(s2) == [False, True, False, False], s2
    assert list(s3) == [False, False, True, False], s3
    # drafts fall back to draft_a/draft_b on rows written before the change
    with open(rp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": 1, "condition": "merge_v2",
                            "response": "x", "draft_a": _deep("guns")}) + "\n")
    assert _deep("guns") in load_texts(rp)[(1, "merge_v2")]["drafts"]
    assert cov[(1, "merge_v2")][2] == 0
    print("coverage_funnel self-test OK (S1/S2/S3 buckets, legacy draft fields)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
