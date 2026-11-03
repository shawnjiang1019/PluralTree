"""Compare retrieval_diag CSVs across encoders and retrievers. No LLM.

Answers checklist B before any generation is spent:
  gate      items passing tau per encoder (a gate that passes everything is
            not filtering; the cell's relevance then does all the work)
  personas  how often each cell gets two subgroups vs the plain-draft fallback
  pairs     how much the persona pairs actually disagree (JS / EMD / D on the
            real answers -- the same measure for every cell)
  overlap   how often two cells pick the SAME top survey item
and writes a side-by-side Markdown file for reading the questions by hand,
listing --focus questions (substring match, e.g. "russia") first.

    python scripts/analysis/compare_retrieval_diag.py \
        --csv retrieval_diag_minilm.csv retrieval_diag_bge.csv \
        --focus russia --out retrieval_diag_compare.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics as st


def _load(paths: list[str]) -> dict[tuple[str, str], dict[int, dict]]:
    """{(encoder, retriever): {question_id: row}}."""
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                enc = r["embed_model"].split("/")[-1]
                cells.setdefault((enc, r["retriever"]), {})[int(r["question_id"])] = r
    return cells


def _num(rows, col) -> list[float]:
    return [float(r[col]) for r in rows if r.get(col) not in ("", None)]


def _fmt(xs: list[float]) -> str:
    return f"{st.mean(xs):.3f}" if xs else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare retrieval diagnostics")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--focus", nargs="*", default=[],
                    help="substrings of questions to list first")
    ap.add_argument("--out", default="retrieval_diag_compare.md")
    args = ap.parse_args()

    cells = _load(args.csv)
    names = sorted(cells)

    print(f"{'cell':28} {'items>tau':>9} {'personas':>9} {'no fork':>8} "
          f"{'item rel':>9} {'pair JS':>8} {'pair EMD':>9} {'pair D':>7} "
          f"{'nominal':>8}")
    for c in names:
        rows = list(cells[c].values())
        made = [r for r in rows if r["personas"] == "True"]
        nom = sum(1 for r in made if r["scale"] == "nominal")
        print(f"{c[0] + '/' + c[1]:28} "
              f"{st.median(_num(rows, 'items_passing')):9.0f} "
              f"{len(made):4}/{len(rows):<4} "
              f"{sum(1 for r in rows if r['n_forks'] == '0'):8} "
              f"{_fmt(_num(made, 'top_item_rel')):>9} "
              f"{_fmt(_num(made, 'pair_js')):>8} {_fmt(_num(made, 'pair_emd')):>9} "
              f"{_fmt(_num(made, 'pair_d')):>7} {nom:8}")

    print("\nsame top survey item (questions where both cells retrieved):")
    for a, b in itertools.combinations(names, 2):
        shared = [q for q in cells[a] if q in cells[b]
                  and cells[a][q]["top_item"] and cells[b][q]["top_item"]]
        same = sum(1 for q in shared if cells[a][q]["top_item"] == cells[b][q]["top_item"])
        print(f"  {a[0]}/{a[1]:6} vs {b[0]}/{b[1]:6}: {same}/{len(shared)}")

    qids = sorted({q for c in names for q in cells[c]})
    text = {q: next(cells[c][q]["question"] for c in names if q in cells[c]) for q in qids}
    focus = [q for q in qids if any(s.lower() in text[q].lower() for s in args.focus)]
    order = focus + [q for q in qids if q not in focus]

    lines = ["# Retrieval diagnostic, side by side", "",
             f"Cells: {', '.join(e + '/' + r for e, r in names)}. "
             "Read each block for: is the item about the question, and do the "
             "two subgroups really take different positions?", ""]
    for q in order:
        lines += [f"## Q{q}: {text[q]}", ""]
        for c in names:
            r = cells[c].get(q)
            lines.append(f"**{c[0]} / {c[1]}**")
            if r is None or r["n_forks"] == "0":
                lines += ["- no forks (plain draft only)", ""]
                continue
            lines.append(f"- item (rel {r['top_item_rel']}): {r['top_item']}")
            lines.append(f"- anchor ({r['top_anchor_type']}, rel {r['top_relevance']}, "
                         f"score div {r['top_divergence']}): {r['top_anchor']}")
            if r["personas"] == "True":
                lines.append(f"- pair ({r['scale']}; JS {r['pair_js']}, "
                             f"EMD {r['pair_emd']}, D {r['pair_d']}):")
                lines.append(f"  - A: {r['persona_a']}")
                lines.append(f"  - B: {r['persona_b']}")
            else:
                lines.append("- no personas (top fork is not an axis) -> plain draft only")
            lines.append("")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nside-by-side ({len(focus)} focus question(s) first) -> {args.out}")


if __name__ == "__main__":
    main()
