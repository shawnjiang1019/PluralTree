"""Merge probe (ceiling) + Map Reader (realized) results into one verdict table.

The probe says how much of each fact is *in* the embedding; the Map Reader says
how much the LM actually *reads*. Lining them up isolates the bottleneck per fact:

    probe ~= prior                     -> EMBEDDING  (info not encoded)
    probe ~= prior, both ~= 1.0        -> DEGENERATE (no target variance, e.g. rho collapse)
    probe >> prior, d_shuf ~= 0        -> INTERPRETATION (info there, LM not reading)
    probe >> prior, d_shuf > 0 < probe -> INTERP. HEADROOM (partly read)
    probe >> prior, MR ~= probe        -> AT CEILING (improve embedding to gain more)

Usage:
    python scripts/compare_probe_mapreader.py \
        --probe runs/probe_wn18rr.json --mapreader runs/mapreader_eval.json
"""

from __future__ import annotations

import argparse
import json


def verdict(p: dict, m: dict | None, gap_eps=0.05) -> str:
    prior, probe = p["prior"], p["best"]
    info_gap = probe - prior
    if info_gap < gap_eps:
        if prior > 0.95 and probe > 0.95:
            return "DEGENERATE (no target variance)"
        return "EMBEDDING (info not encoded)"
    # info is present -> look at what the Map Reader did with it
    if m is None:
        return "info present (no Map Reader result)"
    d_shuf = m.get("d_shuf", 0.0)
    mr = m.get("true", 0.0)
    if d_shuf < gap_eps:
        return "INTERPRETATION (LM not reading)"
    if mr >= probe - gap_eps:
        return "AT CEILING (improve embedding)"
    return "INTERP. HEADROOM (partly read)"


def main():
    ap = argparse.ArgumentParser(description="Compare probe ceiling vs Map Reader reading")
    ap.add_argument("--probe", required=True, help="JSON from probe_embeddings.py --json")
    ap.add_argument("--mapreader", default=None, help="JSON from eval_map_reader.py --json")
    args = ap.parse_args()

    probe = json.load(open(args.probe, encoding="utf-8"))
    mr = json.load(open(args.mapreader, encoding="utf-8")) if args.mapreader else {}

    hdr = (f"{'fact':<10}{'prior':>7}{'probe':>7}{'mr_true':>9}{'d_shuf':>8}"
           f"   verdict")
    print(hdr)
    print("-" * len(hdr))
    facts = sorted(set(probe) | set(mr))
    for f in facts:
        p = probe.get(f)
        m = mr.get(f)
        if p is None:                                   # fact only in MR (shouldn't happen)
            continue
        mr_true = f"{m['true']:>9.2f}" if m else f"{'-':>9}"
        d_shuf = f"{m['d_shuf']:>+8.2f}" if m else f"{'-':>8}"
        print(f"{f:<10}{p['prior']:>7.2f}{p['best']:>7.2f}{mr_true}{d_shuf}"
              f"   {verdict(p, m)}")
    print("\nprobe = best of linear/MLP (info ceiling); mr_true = Map Reader with own latent; "
          "d_shuf = drop when the latent is shuffled (the read signal).")


if __name__ == "__main__":
    main()
