"""Perspectivist annotation datasets as a PluralTree graph (non-political sources).

WHY. Every result in this project comes from 60 US-political survey questions.
That is a real limitation and it is also unnecessary: the graph does not need
*surveys*, it needs

    items  x  a partition of people into groups  x  per-group label distributions

Perspectivist NLP datasets have exactly that shape. Each item is labelled by many
annotators whose demographics are recorded, so per-group label distributions fall
straight out -- and the domains are safety, offensiveness, and NLI rather than
politics.

  DICES-350   350 chatbot conversations rated for SAFETY by 104 raters across
              age, gender, race.
  D3          social-media comments rated for OFFENSIVENESS by ~4000 raters
              balanced on cultural region, gender, age.

Reported across this literature: race and age drive annotator disagreement most
consistently, so the axes the scout forks on transfer.

WHAT THIS IS NOT. A dataset with per-annotator labels but no annotator
attributes cannot be used -- there is no partition to fork on, and a graph built
from it would have one child per item. `rater_attr_cols` is required for that
reason rather than optional.

    from data.loaders.perspectivist import load_perspectivist
    g = load_perspectivist("dices350.csv", item_col="item_id",
                           text_col="context", label_col="Q_overall",
                           rater_attr_cols=["rater_age", "rater_gender",
                                            "rater_race"])

    python -m data.loaders.perspectivist --selftest    # no data needed
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.loaders.opinionqa import cluster_questions

# Values that are a non-answer rather than a position. Same rule as parse_atp_dir,
# which drops "Refused"/"Don't know" -- a refusal is not a subpopulation view.
_NON_ANSWERS = ("refused", "don't know", "dont know", "dk", "unsure",
                "no response", "n/a", "na", "none", "")


def _is_answer(v) -> bool:
    return str(v).strip().lower() not in _NON_ANSWERS


def parse_perspectivist(
    rows,
    *,
    item_col: str,
    text_col: str,
    label_col: str,
    rater_attr_cols: list[str],
    options: list[str] | None = None,
    min_group: int = 15,
    max_options: int = 12,
) -> list[dict]:
    """(item, rater, label) rows -> the {qkey, question, options, attribute,
    group, dist} records ``_build_graph`` consumes.

    One record per (item, attribute, group): the group's normalized distribution
    over the label options for that item.

    ``min_group`` defaults to 15, not ATP's 100. Perspectivist datasets have far
    fewer raters per item (DICES-350: ~104 raters total, so a demographic cell on
    one item can be single digits), and a 100 floor would empty the graph. The
    cost is noisier per-group distributions; that noise propagates into the
    Wasserstein fork scores, so report the achieved group sizes rather than
    assuming they are adequate.

    ``options`` fixes the label set and its ORDER. Leave it None to infer from
    the data, but pass it when the labels are ordinal (e.g. Likert) -- inference
    sorts lexically, which would order "1,10,2" and destroy the ordering the
    spectrum rendering depends on.
    """
    if not rater_attr_cols:
        raise ValueError("rater_attr_cols is required: without annotator "
                         "attributes there is no partition to fork on")

    by_item: dict = {}
    for row in rows:
        item = row.get(item_col)
        label = row.get(label_col)
        if item is None or label is None or not _is_answer(label):
            continue
        text = str(row.get(text_col, "")).strip()
        if not text:
            continue
        rec = by_item.setdefault(str(item), {"text": text, "rows": []})
        rec["rows"].append(row)

    if options is None:
        seen = {str(r[label_col]).strip()
                for v in by_item.values() for r in v["rows"]
                if _is_answer(r[label_col])}
        options = sorted(seen)
        if len(options) > max_options:
            raise ValueError(
                f"inferred {len(options)} label options from {label_col!r}; that "
                f"looks like free text, not a label set. Pass options= explicitly "
                f"or point label_col at the categorical column.")
        print(f"  perspectivist: inferred options {options} "
              f"(LEXICAL order -- pass options= if these are ordinal)")

    out: list[dict] = []
    for qkey, rec in by_item.items():
        for attr in rater_attr_cols:
            groups: dict = {}
            for r in rec["rows"]:
                g = r.get(attr)
                if g is None or not _is_answer(g):
                    continue
                groups.setdefault(str(g), []).append(str(r[label_col]).strip())
            for group, labels in groups.items():
                if len(labels) < min_group:
                    continue
                counts = [sum(1 for x in labels if x == o) for o in options]
                total = sum(counts)
                if total <= 0:
                    continue
                out.append({"qkey": qkey, "question": rec["text"],
                            "options": list(options), "attribute": attr,
                            "group": group,
                            "dist": [c / total for c in counts]})

    n_q = len({r["qkey"] for r in out})
    n_grp = len({(r["attribute"], r["group"]) for r in out})
    print(f"  perspectivist: {len(out)} records — {n_q} items, {n_grp} "
          f"(attribute, group) subpopulations, min_group={min_group}")
    if not out:
        raise ValueError("no records survived; min_group is probably too high "
                         "for this dataset's raters-per-item")
    return out


def parse_perspectivist_csv(path: str, **kw) -> list[dict]:
    """``parse_perspectivist`` over a CSV. Column names are dataset-specific --
    see docs/perspectivist_sources.md and VERIFY them against the real file."""
    import csv as _csv

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    for col in (kw["item_col"], kw["text_col"], kw["label_col"],
                *kw["rater_attr_cols"]):
        if col not in rows[0]:
            raise ValueError(f"column {col!r} not in {path}; available: "
                             f"{sorted(rows[0])[:20]}")
    return parse_perspectivist(rows, **kw)


def load_perspectivist(path: str, *, split_seed: int = 42, train_frac: float = 0.8,
                       val_frac: float = 0.1, leakage_safe: bool = True,
                       k_topics: int = 12, k_subtopics: int = 4,
                       model_name: str = "all-MiniLM-L6-v2", **kw):
    """Build the same graph object `load_opinionqa` returns, from annotations.

    Everything after the records is shared: MiniLM topic clustering, the
    topic->subtopic->item->subgroup hierarchy, and the leakage-safe triple split.
    Only the parser differs, which is the point -- the scout, the hyperbolic
    embedding and every downstream metric are domain-agnostic.
    """
    from data.loaders.opinionqa import _build_graph

    records = parse_perspectivist_csv(path, **kw)
    qkeys, qtexts, seen = [], [], set()
    for r in records:
        if r["qkey"] not in seen:
            seen.add(r["qkey"])
            qkeys.append(r["qkey"])
            qtexts.append(r["question"])
    assign, topic_text, sub_text = cluster_questions(
        qtexts, k_topics=k_topics, k_subtopics=k_subtopics, seed=split_seed,
        model_name=model_name)
    topic_of = {k: a for k, a in zip(qkeys, assign)}
    return _build_graph(records, topic_of, topic_text, sub_text,
                        split_seed=split_seed, train_frac=train_frac,
                        val_frac=val_frac, leakage_safe=leakage_safe)


def _selftest() -> None:
    """Synthetic annotations: the parser's contract, no data and no model needed.

    Plants a real demographic split -- group A mostly 'safe', group B mostly
    'unsafe' -- so the emitted distributions must differ. If they did not, the
    scout would have no fork to find and the whole pipeline would be inert on
    this source.
    """
    rows = []
    for item in range(4):
        for i in range(20):
            rows.append({"id": f"i{item}", "text": f"conversation number {item}",
                         "label": "safe" if i < 16 else "unsafe",
                         "age": "18-35", "gender": "M"})
        for i in range(20):
            rows.append({"id": f"i{item}", "text": f"conversation number {item}",
                         "label": "safe" if i < 4 else "unsafe",
                         "age": "36+", "gender": "F"})
        rows.append({"id": f"i{item}", "text": f"conversation number {item}",
                     "label": "Refused", "age": "36+", "gender": "F"})
        rows.append({"id": f"i{item}", "text": f"conversation number {item}",
                     "label": "safe", "age": "36+", "gender": None})

    recs = parse_perspectivist(rows, item_col="id", text_col="text",
                               label_col="label",
                               rater_attr_cols=["age", "gender"],
                               options=["safe", "unsafe"], min_group=15)

    assert {r["qkey"] for r in recs} == {"i0", "i1", "i2", "i3"}
    assert {r["attribute"] for r in recs} == {"age", "gender"}
    for r in recs:
        assert abs(sum(r["dist"]) - 1.0) < 1e-9, "distributions must normalize"
        assert len(r["dist"]) == len(r["options"])

    young = next(r for r in recs if r["attribute"] == "age" and r["group"] == "18-35")
    older = next(r for r in recs if r["attribute"] == "age" and r["group"] == "36+")
    assert abs(young["dist"][0] - 0.80) < 1e-9, young["dist"]
    # 'Refused' dropped, and the null-gender row still counts toward age.
    assert abs(older["dist"][0] - (4 + 1) / 21) < 1e-9, older["dist"]
    assert young["dist"][0] > older["dist"][0], \
        "the planted split must survive into the distributions"

    # A min_group that empties the graph must FAIL LOUDLY. Returning [] would
    # surface much later as an unrelated crash in the graph builder, and the
    # right floor is dataset-specific (ATP uses 100; 21 raters per cell here).
    try:
        parse_perspectivist(rows, item_col="id", text_col="text",
                            label_col="label", rater_attr_cols=["age"],
                            options=["safe", "unsafe"], min_group=100)
    except ValueError as e:
        assert "min_group" in str(e)
    else:
        raise AssertionError("an empty record set must raise, not return []")
    print("  distributions normalize, refusals dropped, planted split preserved")

    # Missing annotator attributes must be refused, not silently accepted.
    try:
        parse_perspectivist(rows, item_col="id", text_col="text",
                            label_col="label", rater_attr_cols=[])
    except ValueError:
        pass
    else:
        raise AssertionError("empty rater_attr_cols must raise")

    # Free-text mistaken for a label column must be caught, not turned into
    # thousands of one-hot 'options'.
    try:
        parse_perspectivist([{"id": "a", "text": "t", "label": f"free text {i}",
                              "age": "x"} for i in range(50)],
                            item_col="id", text_col="text", label_col="label",
                            rater_attr_cols=["age"], min_group=1)
    except ValueError as e:
        assert "free text" in str(e)
    else:
        raise AssertionError("a free-text label column must raise")
    print("perspectivist loader self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Perspectivist annotations -> graph")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.error("--selftest (or import load_perspectivist)")
