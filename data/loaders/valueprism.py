"""ValuePrism (Sorensen et al., Value Kaleidoscope) as a coverage benchmark.

WHY. Every open question here is capped by OvertonBench's 60 questions. ValuePrism
is ~31k human-written situations annotated with ~218k values, rights and duties --
roughly 500x the questions, measuring a coverage construct: does an answer express
the considerations people actually bring to this situation.

WHY NOT GlobalOpinionQA, which is already loaded. GOQA's targets are SURVEY
OPTIONS, the same object type the scout injects. Scoring option-coverage while
injecting options reads as evaluating on the format you inject, even with
disjoint data. ValuePrism's targets are VALUES ("autonomy", "duty of care") --
a different object, so covering them is a generalisation test.

RETRIEVAL STILL COMES FROM THE ATP GRAPH, which is what avoids the circularity
that killed "retrieve from ValuePrism, evaluate on ValuePrism". It is also the
risk: ATP is US political surveys and these are everyday moral scenarios, so
anchors may not resolve. Measure that FIRST with anchor_coverage.py -- it needs
only the situation texts, and an unreachable benchmark cannot test the method
however large it is.

SCHEMA IS GUESSED. The HF dataset card is license-gated, so field names come from
the paper's description, not from the file. Unmatched columns raise with the keys
actually present rather than returning empty. Fix the candidate lists below once
you have the real header.

    python -m data.loaders.valueprism --selftest
    python -m data.loaders.valueprism --path valueprism.csv --out vp_questions.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_SITUATION = ("situation", "action", "scenario", "context", "prompt", "query")
_TEXT = ("text", "value", "vrd_text", "statement", "title", "value_text")
_VRD = ("vrd", "type", "category", "kind", "vrd_type")
_VALENCE = ("valence", "stance", "polarity", "support")
_SID = ("situation_id", "sid", "id", "qid", "question_id")

# ValuePrism situations are DECLARATIVE ("Telling a white lie to spare feelings").
# Both the scout (which embeds a question) and the generator (which answers one)
# expect an interrogative, so the situation is wrapped. Keep this fixed across
# the gate check and the eval or the two measure different strings.
DEFAULT_TEMPLATE = ("Some people face this situation: {situation}\n"
                    "What considerations matter here, and how do people differ "
                    "on them?")


def _first(row: dict, names):
    for n in names:
        if n in row and str(row[n]).strip() not in ("", "None", "nan"):
            return row[n]
    return None


def _rows(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","))
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            for k in ("data", "rows", "situations", "examples"):
                if isinstance(d.get(k), list):
                    return d[k]
            raise ValueError(f"{path}: dict with no list under data/rows/"
                             f"situations/examples; keys={sorted(d)[:20]}")
        return d
    raise ValueError(f"{path}: expected .csv/.tsv/.jsonl/.json, got {ext!r}")


def load_situations(path: str, *, min_values: int = 2,
                    max_situations: int = 0) -> list[dict]:
    """Long-format rows -> one record per SITUATION with its value set.

    ValuePrism ships one row per (situation, value), so rows are grouped. A
    situation with fewer than ``min_values`` distinct values is dropped: coverage
    of a one-element target is a coin flip and would only add noise, the same
    reason persona_merge's <3-leaf anchors had to be excluded.
    """
    rows = _rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")

    probe = rows[0]
    if _first(probe, _SITUATION) is None:
        raise ValueError(
            f"{path}: no situation field. Tried {_SITUATION}; available keys: "
            f"{sorted(probe)[:25]}. Edit _SITUATION once you have the real header.")
    if _first(probe, _TEXT) is None:
        raise ValueError(
            f"{path}: no value-text field. Tried {_TEXT}; available keys: "
            f"{sorted(probe)[:25]}.")

    by_sit: dict = collections.OrderedDict()
    for i, r in enumerate(rows):
        sit = str(_first(r, _SITUATION) or "").strip()
        txt = _first(r, _TEXT)
        if not sit or txt is None:
            continue
        rec = by_sit.setdefault(sit, {
            "situation_id": str(_first(r, _SID) or len(by_sit)),
            "situation": sit, "values": [], "_seen": set()})
        t = str(txt).strip()
        if t in rec["_seen"]:                  # the same value can repeat per row
            continue
        rec["_seen"].add(t)
        rec["values"].append({
            "text": t,
            "vrd": str(_first(r, _VRD) or ""),
            "valence": str(_first(r, _VALENCE) or "")})

    out = [{k: v for k, v in rec.items() if k != "_seen"}
           for rec in by_sit.values() if len(rec["values"]) >= min_values]
    dropped = len(by_sit) - len(out)
    if max_situations:
        out = out[:max_situations]
    if not out:
        raise ValueError(f"{path}: no situation had >= {min_values} values")

    n_v = sum(len(s["values"]) for s in out)
    valences = collections.Counter(v["valence"] for s in out for v in s["values"])
    print(f"  valueprism: {len(out)} situations, {n_v} values "
          f"({n_v / len(out):.1f}/situation), dropped {dropped} with "
          f"< {min_values} values")
    if any(valences):
        print(f"  valence counts: {dict(valences.most_common(6))}")
    return out


def questions_only(situations, out_path: str,
                   template: str = DEFAULT_TEMPLATE) -> str:
    """{question_id, question} jsonl for anchor_coverage.py --questions."""
    seen, n = set(), 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in situations:
            q = template.format(situation=s["situation"]).strip()
            if not q or q in seen:
                continue
            seen.add(q)
            f.write(json.dumps({"question_id": s["situation_id"], "question": q,
                                "n_values": len(s["values"])}) + "\n")
            n += 1
    print(f"  wrote {out_path}  ({n} unique questions)")
    return out_path


def _selftest() -> None:
    """Long format, deduping, the min_values floor, and a loud unknown schema."""
    import tempfile

    d = tempfile.mkdtemp()
    p = os.path.join(d, "vp.csv")
    rows = [
        {"situation_id": "s1", "situation": "Telling a white lie to spare feelings",
         "vrd": "Value", "text": "Honesty", "valence": "opposes"},
        {"situation_id": "s1", "situation": "Telling a white lie to spare feelings",
         "vrd": "Value", "text": "Kindness", "valence": "supports"},
        {"situation_id": "s1", "situation": "Telling a white lie to spare feelings",
         "vrd": "Value", "text": "Honesty", "valence": "opposes"},   # duplicate
        {"situation_id": "s2", "situation": "Reporting a coworker's mistake",
         "vrd": "Duty", "text": "Duty of loyalty", "valence": "opposes"},
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    got = load_situations(p, min_values=2)
    assert len(got) == 1, f"s2 has 1 value and must be dropped, got {len(got)}"
    assert [v["text"] for v in got[0]["values"]] == ["Honesty", "Kindness"], \
        got[0]["values"]
    assert got[0]["values"][0]["valence"] == "opposes"

    both = load_situations(p, min_values=1)
    assert len(both) == 2, "min_values=1 must keep the single-value situation"

    out = questions_only(got, os.path.join(d, "q.jsonl"))
    with open(out, encoding="utf-8") as f:
        q = [json.loads(l) for l in f]
    assert len(q) == 1 and "white lie" in q[0]["question"]
    assert q[0]["question"].rstrip().endswith("?"), \
        "the template must yield an interrogative -- the scout embeds a question"
    assert q[0]["n_values"] == 2

    bad = os.path.join(d, "bad.csv")
    with open(bad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["foo", "bar"]); w.writeheader()
        w.writerow({"foo": 1, "bar": 2})
    try:
        load_situations(bad)
    except ValueError as e:
        assert "available keys" in str(e) and "bar" in str(e), str(e)
    else:
        raise AssertionError("an unknown schema must raise with its keys")
    print("valueprism loader self-test OK (grouping, dedupe, floor, loud schema)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ValuePrism -> situations / questions")
    ap.add_argument("--path", help="valueprism .csv/.tsv/.jsonl/.json")
    ap.add_argument("--out", default="valueprism_questions.jsonl")
    ap.add_argument("--min_values", type=int, default=2)
    ap.add_argument("--max_situations", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.path:
        questions_only(load_situations(a.path, min_values=a.min_values,
                                       max_situations=a.max_situations), a.out)
    else:
        ap.error("--path or --selftest")
