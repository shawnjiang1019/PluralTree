"""WildSCOPE / PluralEval threads (ACL 2026), or any Reddit-style crowd corpus.

WHY IT MATTERS. Every conclusion here is capped by OvertonBench's 60 questions:
merge_v2 sits at p=0.014 only after three rollouts, 9 of 13 features flip sign
between arms, the probe lands at p=0.130, and resolving the +0.0154 content
effect would need ~365 questions. WildSCOPE is ~1,212 threads measuring the same
construct -- claim coverage against organic crowd responses -- which is the one
change that makes those questions answerable.

NO PUBLIC RELEASE AS OF WRITING. The ACL page carries no code or data link and
nothing surfaces on GitHub or HuggingFace. Reported composition: 1,212 threads
from 2019 Reddit archives across r/AmItheAsshole (moral reasoning),
r/AskEconomics (economic policy) and r/PoliticalDiscussion (deliberation).

So this loader is deliberately SHAPE-AGNOSTIC. Field names are guessed from a
candidate list and, when nothing matches, it raises with the keys actually
present rather than silently returning empty. Point it at the release when it
appears, or at a reconstruction from those three subreddits.

TWO LEVELS OF USE, and the first needs far less than the second:

  questions only   thread title/body -> `anchor_coverage.py --questions`. This is
                   the GATE: if the ATP graph does not resolve anchors for these
                   questions, injection is inert and the benchmark cannot test
                   the method regardless of its size. Needs no crowd data.
  full eval        also the per-thread atomic claims, as the coverage target.
                   If the authors released their decompositions, USE THEM --
                   reimplementing the claim split makes the numbers
                   incomparable to the paper's.

    python -m data.loaders.wildscope --selftest
    python -m data.loaders.wildscope --path wildscope.jsonl --out questions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Ordered by preference. A Reddit thread's question is usually the title, and the
# body adds context -- so both are taken and joined when present.
_TITLE = ("title", "question", "prompt", "query", "post_title")
_BODY = ("selftext", "body", "post", "context", "description", "question_body")
_ID = ("thread_id", "id", "post_id", "question_id", "qid", "link_id")
_CLAIMS = ("claims", "atomic_claims", "opinions", "positions", "responses",
           "comments", "crowd_responses")
_DOMAIN = ("subreddit", "domain", "source", "category")


def _first(row: dict, names) -> object:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def _as_claims(v) -> list[str]:
    """Claims may be strings, or dicts with the text under some key."""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    out = []
    for item in v:
        if isinstance(item, str):
            s = item
        elif isinstance(item, dict):
            s = _first(item, ("claim", "text", "body", "content", "opinion"))
        else:
            s = None
        if s and str(s).strip():
            out.append(str(s).strip())
    return out


def _rows(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # {"threads": [...]} or {"data": [...]}
            for k in ("threads", "data", "rows", "examples"):
                if isinstance(data.get(k), list):
                    return data[k]
            raise ValueError(f"{path}: top-level dict with no list under "
                             f"threads/data/rows/examples; keys={sorted(data)[:20]}")
        return data
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"{path}: expected .jsonl/.json/.csv, got {ext!r}")


def load_threads(path: str, *, require_claims: bool = False) -> list[dict]:
    """-> [{thread_id, question, claims, domain}]. Raises if fields are unfindable.

    ``require_claims=True`` for the full eval; leave False for the anchor-coverage
    gate, which only needs the questions.
    """
    rows = _rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")

    out, no_claims = [], 0
    for i, r in enumerate(rows):
        title = _first(r, _TITLE)
        body = _first(r, _BODY)
        if title is None and body is None:
            raise ValueError(
                f"{path} row {i}: no question field. Tried {_TITLE + _BODY}; "
                f"available keys: {sorted(r)[:25]}. Pass the real names by "
                f"editing _TITLE/_BODY, or pre-convert to {{question, claims}}.")
        q = " ".join(str(x).strip() for x in (title, body) if x)
        claims = _as_claims(_first(r, _CLAIMS))
        if not claims:
            no_claims += 1
        out.append({"thread_id": str(_first(r, _ID) or i),
                    "question": q, "claims": claims,
                    "domain": str(_first(r, _DOMAIN) or "")})

    if require_claims and no_claims:
        raise ValueError(
            f"{path}: {no_claims}/{len(out)} threads have no claims. Tried "
            f"{_CLAIMS}. The full eval needs the ground-truth decomposition; the "
            f"anchor-coverage gate does not (require_claims=False).")

    n_c = sum(len(t["claims"]) for t in out)
    doms = sorted({t["domain"] for t in out if t["domain"]})
    print(f"  wildscope: {len(out)} threads, {n_c} claims "
          f"({n_c / max(1, len(out)):.1f}/thread), {no_claims} without claims"
          + (f", domains {doms}" if doms else ""))
    return out


def questions_only(threads, out_path: str) -> str:
    """Write {question_id, question} jsonl for anchor_coverage.py --questions."""
    seen, n = set(), 0
    with open(out_path, "w", encoding="utf-8") as f:
        for t in threads:
            q = t["question"].strip()
            if not q or q in seen:          # threads repeat; dedupe or the
                continue                    # resolution rate is comment-weighted
            seen.add(q)
            f.write(json.dumps({"question_id": t["thread_id"], "question": q,
                                "domain": t["domain"]}) + "\n")
            n += 1
    print(f"  wrote {out_path}  ({n} unique questions)")
    return out_path


def _selftest() -> None:
    """Three plausible shapes, since the real one is unknown."""
    import tempfile

    d = tempfile.mkdtemp()
    shapes = {
        "a.jsonl": [  # reddit-ish: title + selftext, claims as strings
            {"id": "t1", "title": "AITA for skipping the wedding?",
             "selftext": "My cousin got married and I did not go.",
             "subreddit": "AmItheAsshole",
             "claims": ["You are not the asshole, it was your choice.",
                        "You should have gone to support family."]},
            {"id": "t2", "title": "Is rent control good policy?",
             "subreddit": "AskEconomics",
             "claims": ["It reduces supply over time."]},
        ],
        "b.json": {"threads": [  # nested, claims as dicts
            {"thread_id": "x9", "question": "Should voting be mandatory?",
             "domain": "PoliticalDiscussion",
             "atomic_claims": [{"claim": "It raises turnout."},
                               {"text": "It compels the uninformed."}]},
        ]},
        "c.csv": [  # flat csv, NO claims -- gate use only
            {"post_id": "c1", "question": "Is it rude to leave early?",
             "category": "AmItheAsshole"},
        ],
    }
    for name, payload in shapes.items():
        p = os.path.join(d, name)
        if name.endswith(".jsonl"):
            with open(p, "w", encoding="utf-8") as f:
                for r in payload:
                    f.write(json.dumps(r) + "\n")
        elif name.endswith(".json"):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        else:
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(payload[0]))
                w.writeheader(); w.writerows(payload)

    a = load_threads(os.path.join(d, "a.jsonl"))
    assert len(a) == 2 and a[0]["thread_id"] == "t1"
    assert a[0]["question"].startswith("AITA for skipping"), a[0]["question"]
    assert "My cousin" in a[0]["question"], "body must be appended to the title"
    assert len(a[0]["claims"]) == 2

    b = load_threads(os.path.join(d, "b.json"))
    assert b[0]["claims"] == ["It raises turnout.", "It compels the uninformed."], \
        b[0]["claims"]

    c = load_threads(os.path.join(d, "c.csv"))
    assert c[0]["claims"] == [] and c[0]["domain"] == "AmItheAsshole"
    try:
        load_threads(os.path.join(d, "c.csv"), require_claims=True)
    except ValueError as e:
        assert "no claims" in str(e)
    else:
        raise AssertionError("require_claims must reject a claimless file")

    # An unrecognisable schema must name the keys it saw, not return nothing.
    p = os.path.join(d, "bad.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"foo": 1, "bar": "baz"}) + "\n")
    try:
        load_threads(p)
    except ValueError as e:
        assert "available keys" in str(e) and "bar" in str(e), str(e)
    else:
        raise AssertionError("an unknown schema must raise with its keys")

    out = questions_only(a + a, os.path.join(d, "q.jsonl"))   # duplicated on purpose
    with open(out, encoding="utf-8") as f:
        got = [json.loads(l) for l in f]
    assert len(got) == 2, f"duplicates must be dropped, got {len(got)}"
    print("wildscope loader self-test OK (3 shapes, dedupe, loud on unknown schema)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WildSCOPE threads -> questions")
    ap.add_argument("--path", help="wildscope .jsonl/.json/.csv")
    ap.add_argument("--out", default="wildscope_questions.jsonl")
    ap.add_argument("--require_claims", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.path:
        questions_only(load_threads(a.path, require_claims=a.require_claims), a.out)
    else:
        ap.error("--path or --selftest")
