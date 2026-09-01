"""Can the graph even reach a new question set? Anchor resolution, before generating.

Moving to a new benchmark (WildSCOPE, 1.2K Reddit threads) does not need a new
graph: OvertonBench's questions are already everyday subjective ones -- "Should
family always stick together?", "Is piracy theft?", "Are soulmates real?" -- and
the ATP graph resolves anchors for 56 of 60 of them.

What it DOES need is for that to keep holding. If only 40% of the new questions
reach a relevant anchor, injection is inert on the rest and the benchmark cannot
test the method no matter how many questions it has. Every injected condition
degrades to `baseline` on an unresolved question, so a low rate does not show up
as an error -- it shows up as a diluted effect size, which is indistinguishable
from the method not working.

This measures that for the price of a graph load and one MiniLM pass. No LLM, no
generation, no judge. Run it BEFORE committing to a benchmark.

    # calibrate on the set whose answer we know, then measure the new one
    python scripts/analysis/anchor_coverage.py --embeddings embeddings_opinionqa.pt \\
        --reference overton --questions wildscope.jsonl --text_field question
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_texts(path: str, text_field: str) -> list[str]:
    """Questions from .jsonl / .csv / .txt. One question per record or line."""
    ext = os.path.splitext(path)[1].lower()
    out: list[str] = []
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                v = row.get(text_field)
                if v is None:
                    raise SystemExit(
                        f"field {text_field!r} not in {path}; available: "
                        f"{sorted(row)[:20]}")
                out.append(str(v).strip())
    elif ext == ".csv":
        import csv as _csv
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if rows and text_field not in rows[0]:
            raise SystemExit(f"column {text_field!r} not in {path}; available: "
                             f"{sorted(rows[0])[:20]}")
        out = [str(r[text_field]).strip() for r in rows]
    else:
        with open(path, encoding="utf-8") as f:
            out = [ln.strip() for ln in f]
    out = [t for t in out if t]
    # Threads repeat a prompt across comments; dedupe or the rate is weighted by
    # comment count rather than by question.
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if len(uniq) < len(out):
        print(f"  {path}: {len(out)} rows -> {len(uniq)} unique questions")
    return uniq


def _pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def measure(name: str, texts, graph, h_all, text_feat, manifold, cfg,
            embed_question, scout, tau: float) -> dict:
    n_forks, rels, resolved = [], [], 0
    for i, q in enumerate(texts):
        forks = scout(q, graph, h_all, text_feat, manifold, cfg=cfg,
                      q_emb=embed_question(q))
        n_forks.append(len(forks))
        if forks:
            resolved += 1
            rels.append(forks[0].relevance)
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(texts)} ...")
    rate = resolved / len(texts) if texts else float("nan")
    out = {"name": name, "n": len(texts), "resolved": resolved, "rate": rate,
           "rel_p25": _pct(rels, 25), "rel_p50": _pct(rels, 50),
           "rel_p75": _pct(rels, 75),
           "mean_forks": st.mean(n_forks) if n_forks else float("nan")}
    print(f"\n  {name}: {resolved}/{len(texts)} resolved ({rate:.1%})   "
          f"top-fork relevance p25/p50/p75 = "
          f"{out['rel_p25']:.3f}/{out['rel_p50']:.3f}/{out['rel_p75']:.3f}   "
          f"mean forks/question {out['mean_forks']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Anchor resolution on a question set")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--questions", default=None, help="jsonl/csv/txt of questions")
    ap.add_argument("--text_field", default="question")
    ap.add_argument("--reference", choices=["overton", "none"], default="overton",
                    help="also measure OvertonBench, so the new set has a "
                         "calibration point instead of an absolute number")
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="opinionqa")
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--out", default=None, help="csv of the summary rows")
    args = ap.parse_args()

    if not args.questions and args.reference == "none":
        ap.error("nothing to measure: pass --questions and/or --reference overton")

    import torch

    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.scout import (ScoutConfig, embed_question,
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

    rows = []
    if args.reference == "overton":
        from evaluation.overton.eval_overtonbench import load_questions
        ref = [q for _, q in load_questions("full")]
        if args.max_questions:
            ref = ref[: args.max_questions]
        print(f"\n=== reference: OvertonBench ({len(ref)} questions) ===")
        rows.append(measure("overton", ref, graph, h_all, text_feat, manifold,
                            cfg, embed_question, scout, args.tau))

    if args.questions:
        texts = load_texts(args.questions, args.text_field)
        if args.max_questions:
            texts = texts[: args.max_questions]
        print(f"\n=== new set: {os.path.basename(args.questions)} "
              f"({len(texts)} questions) ===")
        rows.append(measure(os.path.basename(args.questions), texts, graph,
                            h_all, text_feat, manifold, cfg, embed_question,
                            scout, args.tau))

    print("\n=== verdict ===")
    if len(rows) == 2:
        ref, new = rows
        d_rate = new["rate"] - ref["rate"]
        d_rel = new["rel_p50"] - ref["rel_p50"]
        print(f"  resolution {ref['rate']:.1%} -> {new['rate']:.1%} ({d_rate:+.1%})")
        print(f"  median top-fork relevance {ref['rel_p50']:.3f} -> "
              f"{new['rel_p50']:.3f} ({d_rel:+.3f})")
        if new["rate"] >= 0.80 and d_rel > -0.05:
            print("  USABLE: the graph reaches this set about as well as OvertonBench.")
        elif new["rate"] >= 0.60:
            print("  MARGINAL: injection is inert on a meaningful minority. Report "
                  "the resolved subset separately -- pooling them dilutes the "
                  "effect toward zero and looks like the method failing.")
        else:
            print("  NOT USABLE with this graph: most questions get no fork, so "
                  "the injected conditions collapse to baseline. A new benchmark "
                  "needs a graph that covers it.")
    else:
        r = rows[0]
        print(f"  {r['name']}: {r['rate']:.1%} resolved, median relevance "
              f"{r['rel_p50']:.3f} (no reference to compare against)")
    print("  NOTE: resolution is necessary, not sufficient -- a resolved anchor "
          "can still be the wrong survey question (g_relevance averaged ~0.50 on "
          "OvertonBench, which is not high).")

    if args.out and rows:
        import csv as _csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
