"""What KIND of question did injection lose on? Print the losers, with their text.

injection_triage.py answers "which qids" and "does any feature separate them".
Both came back thin: no feature clears its confidence interval at 10 losses. That
is the point at which the useful move is to stop scoring features and READ the
questions, because a category you can name ("already-settled factual questions",
"questions with one dominant position") is a hypothesis you can then go build a
feature FOR -- rather than hoping one of the 21 you already have happens to fit.

Deliberately no graph and no embedder: this is a CSV join plus a jsonl lookup, so
it runs in seconds on a login node. Nothing here needs the 7-minute ATP load.

    # the 10 losing questions, with text
    python scripts/analysis/show_delta_questions.py \
        --scores overton_scores_v9.csv --condition merge_v2

    # and what actually changed in the answers
    python scripts/analysis/show_delta_questions.py \
        --scores overton_scores_v9.csv --condition merge_v2 --answers --fork
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.delta_regressor import (derive_responses_path, load_coverage,
                                              load_responses)


def _wrap(text: str, indent: str = "      ", width: int = 96) -> str:
    text = " ".join((text or "").split())
    if not text:
        return indent + "(empty)"
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def main():
    ap = argparse.ArgumentParser(description="Show the questions injection lost on")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--responses", default=None, help="default: derived from --scores")
    ap.add_argument("--condition", default="merge_v2")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--tol", type=float, default=0.027,
                    help="coverage noise floor; |delta| inside it is a tie")
    ap.add_argument("--side", choices=["loss", "win", "both"], default="loss")
    ap.add_argument("--n", type=int, default=0, help="0 = all on that side")
    ap.add_argument("--answers", action="store_true",
                    help="also print the baseline and injected answers")
    ap.add_argument("--fork", action="store_true",
                    help="also print the injected fork context")
    ap.add_argument("--chars", type=int, default=700, help="answer clip length")
    args = ap.parse_args()

    resp_path = args.responses or derive_responses_path(args.scores)
    cov = load_coverage(args.scores)
    resp = load_responses(resp_path)
    if not resp:
        print(f"WARNING: no responses at {resp_path} -- question text unavailable")

    items = []
    for qid, by_cond in cov.items():
        if args.baseline not in by_cond or args.condition not in by_cond:
            continue
        b, i = by_cond[args.baseline], by_cond[args.condition]
        rec = resp.get(int(qid), {})
        q = ((rec.get(args.condition) or {}).get("question")
             or (rec.get(args.baseline) or {}).get("question") or "")
        items.append({"qid": int(qid), "base": b, "inj": i, "delta": i - b,
                      "question": q, "rec": rec})
    if not items:
        ap.error(f"no question had both {args.baseline!r} and {args.condition!r} "
                 f"in {args.scores}")

    items.sort(key=lambda d: d["delta"])
    losses = [d for d in items if d["delta"] < -args.tol]
    wins = [d for d in items if d["delta"] > args.tol]
    ties = len(items) - len(losses) - len(wins)

    print(f"{args.scores}  |  {args.condition} vs {args.baseline}  |  "
          f"{len(items)} questions")
    print(f"  win {len(wins)}   loss {len(losses)}   tie {ties}   (tol={args.tol})")

    def show(group, label):
        sel = group if args.n <= 0 else (group[:args.n] if label == "LOST"
                                         else group[-args.n:])
        print(f"\n{'=' * 100}")
        print(f"=== {len(sel)} of {len(group)} questions where injection {label}")
        print(f"{'=' * 100}")
        for d in (sel if label == "LOST" else sel[::-1]):
            print(f"\n  [{d['qid']}]  base {d['base']:.3f} -> inj {d['inj']:.3f}  "
                  f"delta {d['delta']:+.3f}")
            print(_wrap(d["question"] or "(question text not in responses file)"))
            if args.fork:
                fc = (d["rec"].get(args.condition) or {}).get("fork_context") or ""
                print(f"      --- injected fork context ---")
                print(_wrap(_clip(fc, args.chars), indent="        "))
            if args.answers:
                for cond, tag in ((args.baseline, "BASELINE"), (args.condition, "INJECTED")):
                    txt = (d["rec"].get(cond) or {}).get("response") or ""
                    print(f"      --- {tag} answer ---")
                    print(_wrap(_clip(txt, args.chars), indent="        "))

    if args.side in ("loss", "both"):
        show(losses, "LOST")
    if args.side in ("win", "both"):
        show(wins, "WON")

    # A flat qid list is what you paste into the next command.
    if losses:
        print(f"\nloss qids: {','.join(str(d['qid']) for d in losses)}")
    if wins and args.side in ("win", "both"):
        print(f"win  qids: {','.join(str(d['qid']) for d in wins)}")


if __name__ == "__main__":
    main()
