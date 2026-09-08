"""ISSP module files -> the canonical per-subgroup distribution record.

WHY ISSP. Every graph in this project so far is US political opinion (Pew ATP via
OpinionQA) or its cross-national sibling (GlobalOpinionQA, built from Pew Global
+ WVS). A second graph is what separates "merge_v2 helps" from "merge_v2 helps on
OpinionQA", and ISSP is the closest structural match with the furthest content:
religion, health, environment, work, leisure, family.

THE PROPERTY THAT MATTERS. ISSP ships ELEVEN NAMED TOPIC MODULES. OpinionQA has
no topic labels, so `cluster_questions` mints them by k-means over MiniLM
question embeddings -- which makes the hierarchy partly a product of an embedding
model, and testing a hyperbolic embedding on it is close to circular. ISSP's
topic level exists independently of any encoder, so `load_issp` passes native
labels straight to `_build_graph` and never calls the clusterer for level 1.
That is the point of using it, not an implementation detail.

    topic (module)  ->  subtopic (edition year)  ->  question
                    ->  demographic axis  ->  group

SCHEMA IS NOT GUESSED -- IT IS INSPECTED. GESIS documents harmonized background
variables (SEX, AGE, DEGREE, ...) but the documentation page is behind a 403 and
secondary descriptions have been wrong before (see data/loaders/valueprism.py,
whose column guesses came from a paper). So this module leads with `--inspect`:
point it at a real file, read the actual variables and value labels, and only
then fill in BACKGROUND / the missing-code filter. Nothing here assumes a name
without failing loudly when it is absent.

    python -m data.loaders.issp --selftest
    python -m data.loaders.issp --inspect ZA10000.dta            # DO THIS FIRST
    python -m data.loaders.issp --dir issp/ --out issp_records.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Harmonized ISSP background variables. Ordered by preference within each axis;
# the first name present in the file wins. VERIFY WITH --inspect before trusting
# this list -- ISSP renamed several of these between the 2012 and 2017 waves.
BACKGROUND: dict[str, tuple[str, ...]] = {
    "sex":        ("SEX", "sex", "V200", "C_ALPHAN_SEX"),
    "age":        ("AGE_GROUP", "AGEGRP", "AGE", "age"),
    "education":  ("DEGREE", "EDUCYRS", "degree", "ISCED"),
    "marital":    ("MARITAL", "marital"),
    "employment": ("WORK", "WRKST", "MAINSTAT", "EMPREL"),
    "religion":   ("RELIGGRP", "RELIG", "ATTEND"),
    "urbanrural": ("URBRURAL", "URBRUR", "PLACE"),
    "income":     ("INCOME", "HOMPOP_INC", "RINCOME"),
    "class":      ("TOPBOT", "SUBJCLASS"),
    "politics":   ("PARTY_LR", "PARTY", "VOTE_LE"),
}
COUNTRY = ("country", "COUNTRY", "C_ALPHAN", "V3", "cntry")
WEIGHT = ("WEIGHT", "weight", "WGT", "V5")

# ISSP codes non-responses inside the value range (8/9, 98/99, or negative) with
# labels rather than as SPSS system-missing. Filtering on the LABEL is safer than
# on the code, which differs per item width. Anything matching drops out of the
# distribution entirely -- it is not a viewpoint.
_MISSING_LABEL = re.compile(
    r"^\s*(no answer|don'?t know|dk|na\b|nap\b|not applicable|refused|"
    r"can'?t choose|cannot choose|no opinion|not available|missing|"
    r"other countries|nav\b)", re.IGNORECASE)

# Substantive ISSP items are Vnn / Qnn. Background, admin and technical
# variables are excluded by name so they never become graph questions.
_ITEM_NAME = re.compile(r"^(V\d{1,3}|Q\d{1,3}[a-z]?)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Reading: .dta preferred (pandas reads labels natively; no pyreadstat needed)
# ---------------------------------------------------------------------------
def read_table(path: str):
    """-> (DataFrame, {col: question text}, {col: {code: option text}}).

    .dta is the recommended download format precisely because pandas exposes
    both label tables without an extra dependency -- the cluster venv has no
    pyreadstat, and it is offline, so `pip install` is not a fallback there.
    """
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext == ".dta":
        with pd.io.stata.StataReader(path, convert_categoricals=False) as rd:
            df = rd.read()
            var_labels = dict(rd.variable_labels())
            val_labels = {k: dict(v) for k, v in rd.value_labels().items()}
        # Stata keeps value-label SETS, named separately from the columns that
        # use them; map set -> column so callers can key by column.
        with pd.io.stata.StataReader(path, convert_categoricals=False) as rd:
            fmt = getattr(rd, "lbllist", None) or []
            cols = list(df.columns)
        by_col = {}
        for col, setname in zip(cols, list(fmt) + [""] * len(cols)):
            if setname and setname in val_labels:
                by_col[col] = val_labels[setname]
            elif col in val_labels:
                by_col[col] = val_labels[col]
        return df, var_labels, by_col
    if ext == ".sav":
        try:
            import pyreadstat
        except ImportError as e:                              # noqa: BLE001
            raise SystemExit(
                f"{path}: .sav needs pyreadstat, which is not installed and "
                f"cannot be installed on an offline cluster. Re-download this "
                f"module from GESIS as Stata (.dta) instead.") from e
        df, meta = pyreadstat.read_sav(path)
        return df, dict(meta.column_names_to_labels), dict(meta.variable_value_labels)
    if ext == ".csv":
        return pd.read_csv(path), {}, {}
    raise ValueError(f"{path}: expected .dta/.sav/.csv, got {ext!r}")


def _first(cols, names):
    lower = {str(c).lower(): c for c in cols}
    for n in names:
        if n in cols:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def inspect_file(path: str, max_items: int = 40) -> None:
    """Print what is ACTUALLY in the file. Run this before editing BACKGROUND.

    The ValuePrism loader guessed its columns from a paper and the guesses were
    wrong; the cost was a failed cluster job and a wasted queue slot. One minute
    here removes that whole class of failure.
    """
    df, var_labels, val_labels = read_table(path)
    print(f"{path}: {len(df)} rows x {len(df.columns)} columns\n")

    print("=== background variables found (by axis) ===")
    for axis, names in BACKGROUND.items():
        got = _first(df.columns, names)
        mark = "OK " if got else "-- "
        print(f"  {mark}{axis:<12}{got or 'NOT FOUND':<16}"
              f"{('tried ' + ', '.join(names)) if not got else ''}")
    for label, names in (("country", COUNTRY), ("weight", WEIGHT)):
        got = _first(df.columns, names)
        print(f"  {'OK ' if got else '-- '}{label:<12}{got or 'NOT FOUND'}")

    print(f"\n=== candidate substantive items (first {max_items}) ===")
    n = 0
    for c in df.columns:
        if not _ITEM_NAME.match(str(c)):
            continue
        opts = val_labels.get(c, {})
        keep = [t for t in opts.values() if not _MISSING_LABEL.match(str(t))]
        print(f"  {str(c):<10}{str(var_labels.get(c, ''))[:66]}")
        if keep:
            print(f"  {'':<10}  options: {keep[:8]}")
        n += 1
        if n >= max_items:
            print(f"  ... ({sum(1 for c in df.columns if _ITEM_NAME.match(str(c)))} total)")
            break
    if n == 0:
        print("  NONE matched _ITEM_NAME. Real column names look like:")
        print(f"  {list(df.columns)[:30]}")
        print("  Fix _ITEM_NAME before parsing, or this file yields no questions.")


# ---------------------------------------------------------------------------
# Parsing: respondents -> per-(item, axis, group) distributions
# ---------------------------------------------------------------------------
def parse_issp_file(path: str, topic: str, year: str, *, min_group: int = 100,
                    axes: list[str] | None = None,
                    max_options: int = 12) -> list[dict]:
    """One module file -> canonical records.

    ``min_group`` matches parse_atp_dir's 100. Below ~15 the per-group
    distribution is mostly sampling noise and that noise goes straight into the
    Wasserstein fork scores, so a low floor does not buy more signal -- it buys
    more confident nonsense. Cells are pooled ACROSS COUNTRIES: crossing country
    with a demographic axis thins ISSP's ~1,200-per-country samples below the
    floor immediately.
    """
    import pandas as pd

    df, var_labels, val_labels = read_table(path)
    want = axes or list(BACKGROUND)
    found = {a: _first(df.columns, BACKGROUND[a]) for a in want if a in BACKGROUND}
    found = {a: c for a, c in found.items() if c is not None}
    if not found:
        raise ValueError(
            f"{path}: no background variables matched. Tried {sorted(BACKGROUND)}. "
            f"Columns present: {list(df.columns)[:30]}. Run --inspect and fix "
            f"BACKGROUND before parsing.")

    wcol = _first(df.columns, WEIGHT)
    w = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0) if wcol else None

    items = [c for c in df.columns if _ITEM_NAME.match(str(c))
             and c not in set(found.values())]
    if not items:
        raise ValueError(
            f"{path}: no substantive items matched _ITEM_NAME. "
            f"Columns: {list(df.columns)[:30]}")

    out: list[dict] = []
    n_thin = 0
    for item in items:
        labels = val_labels.get(item)
        if not labels:
            continue                       # no value labels -> no option vocabulary
        # Option order comes from the CODE order, which is the survey's own
        # ordering. Sorting the label strings would order "1","10","2" and
        # destroy the spectrum the fork rendering depends on.
        codes = sorted(c for c, t in labels.items()
                       if not _MISSING_LABEL.match(str(t)))
        if not 2 <= len(codes) <= max_options:
            continue
        options = [str(labels[c]).strip() for c in codes]
        qtext = str(var_labels.get(item, "")).strip() or str(item)
        col = pd.to_numeric(df[item], errors="coerce")

        for axis, gcol in found.items():
            glabels = val_labels.get(gcol, {})
            for gcode, gdf in df.groupby(df[gcol], dropna=True):
                gname = str(glabels.get(gcode, gcode)).strip()
                if _MISSING_LABEL.match(gname):
                    continue
                sel = col.loc[gdf.index]
                mask = sel.isin(codes)
                if int(mask.sum()) < min_group:
                    n_thin += 1
                    continue
                ww = (w.loc[gdf.index][mask] if w is not None
                      else pd.Series(1.0, index=sel[mask].index))
                tot = float(ww.sum())
                if tot <= 0:
                    continue
                dist = [float(ww[sel[mask] == c].sum()) / tot for c in codes]
                s = sum(dist)
                if not s or not math.isfinite(s):
                    continue
                out.append({
                    "qkey": f"{topic}|{year}|{item}",
                    "question": qtext,
                    "options": options,
                    "attribute": axis,
                    "group": gname,
                    "dist": [d / s for d in dist],
                    "topic": topic,
                    "year": str(year),
                })
    print(f"  {os.path.basename(path)}: {len(out)} records, "
          f"{len({r['qkey'] for r in out})} questions, axes {sorted(found)}"
          + (f", {n_thin} cells below min_group={min_group}" if n_thin else ""))
    return out


# "ZA10000_family_2022.dta" / "issp_religion_2018.dta" -> ("religion", "2018")
_FROM_NAME = re.compile(r"(?P<topic>[A-Za-z][A-Za-z_\- ]{2,40}?)[_\- ]*"
                        r"(?P<year>(19|20)\d{2})")


def topic_year_from_name(path: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^ZA\d+[_\- ]*", "", stem, flags=re.IGNORECASE)
    m = _FROM_NAME.search(stem)
    if not m:
        return stem.replace("_", " ").strip() or "unknown", "0000"
    return m.group("topic").replace("_", " ").strip().lower(), m.group("year")


def parse_issp_dir(data_dir: str, *, min_group: int = 100,
                   axes: list[str] | None = None) -> list[dict]:
    """Every module file in a directory -> records, topic/year from the filename.

    Rename downloads so the topic is readable ("religion_2018.dta"); the ZA
    number alone carries no topic and would collapse the native topic level this
    loader exists to provide.
    """
    files = sorted(f for f in os.listdir(data_dir)
                   if os.path.splitext(f)[1].lower() in (".dta", ".sav", ".csv"))
    if not files:
        raise ValueError(f"{data_dir}: no .dta/.sav/.csv files")
    recs: list[dict] = []
    for f in files:
        topic, year = topic_year_from_name(f)
        recs += parse_issp_file(os.path.join(data_dir, f), topic, year,
                                min_group=min_group, axes=axes)
    if not recs:
        raise ValueError(f"{data_dir}: parsed 0 records from {len(files)} files")
    topics = collections.Counter(r["topic"] for r in recs)
    print(f"  ISSP: {len(recs)} records, {len({r['qkey'] for r in recs})} questions, "
          f"{len(topics)} topics {dict(topics.most_common())}")
    return recs


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def native_topics(records, *, k_subtopics: int = 4, seed: int = 0,
                  model_name: str = "all-MiniLM-L6-v2"):
    """topic_of / topic_text / sub_text from ISSP's own module labels.

    Level 1 is never clustered -- that is the reason for using ISSP. Level 2 is
    the edition year where a topic has two or more editions (also native); a
    single-edition topic would otherwise get one child and flatten the tree, so
    those fall back to k-means WITHIN the module. The fallback is reported,
    because "how much of this hierarchy is native" is a property of the run that
    any claim about the geometry depends on.
    """
    by_topic: dict[str, list[str]] = collections.OrderedDict()
    years: dict[str, set] = collections.defaultdict(set)
    qtext: dict[str, str] = {}
    qyear: dict[str, str] = {}
    for r in records:
        k = r["qkey"]
        if k not in qtext:
            by_topic.setdefault(r["topic"], []).append(k)
            qtext[k] = r["question"]
            qyear[k] = r["year"]
        years[r["topic"]].add(r["year"])

    topic_of: dict[str, tuple[int, int]] = {}
    topic_text: dict[int, str] = {}
    sub_text: dict[tuple[int, int], str] = {}
    n_clustered = 0

    for t, (topic, qkeys) in enumerate(by_topic.items()):
        topic_text[t] = f"{topic} (ISSP module)"
        eds = sorted(years[topic])
        if len(eds) >= 2:                                  # native subtopic
            for s, ed in enumerate(eds):
                sub_text[(t, s)] = f"{topic}, {ed}"
            idx = {ed: s for s, ed in enumerate(eds)}
            for k in qkeys:
                topic_of[k] = (t, idx[qyear[k]])
        else:
            n_clustered += 1
            from data.loaders.opinionqa import cluster_questions
            assign, _, sub = cluster_questions(
                [qtext[k] for k in qkeys], k_topics=1,
                k_subtopics=min(k_subtopics, max(1, len(qkeys) // 4)),
                seed=seed + t, model_name=model_name)
            for k, (_, s) in zip(qkeys, assign):
                topic_of[k] = (t, s)
                sub_text[(t, s)] = sub.get((0, s), f"{topic} group {s}")

    native = len(by_topic) - n_clustered
    print(f"  ISSP hierarchy: {len(by_topic)} native topics; subtopics native for "
          f"{native}/{len(by_topic)} (multi-edition), clustered for {n_clustered}")
    return topic_of, topic_text, sub_text


def load_issp(split_seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1,
              leakage_safe: bool = True, data_dir: str | None = None,
              min_group: int = 100, axes: list[str] | None = None,
              k_subtopics: int = 4, model_name: str = "all-MiniLM-L6-v2"):
    """ISSP -> OpinionQAGraph, reusing opinionqa's builder unchanged.

    ``data_dir`` or the ISSP_DIR env var. Note the builder names its root node
    "US_Public"; that text is wrong for a cross-national graph but is only a node
    feature, so it is left alone rather than forking the builder.
    """
    from data.loaders.opinionqa import _build_graph

    data_dir = data_dir or os.environ.get("ISSP_DIR")
    if not data_dir:
        raise SystemExit("load_issp needs --data_dir or ISSP_DIR (the directory "
                         "of downloaded GESIS module files)")
    records = parse_issp_dir(data_dir, min_group=min_group, axes=axes)
    topic_of, topic_text, sub_text = native_topics(
        records, k_subtopics=k_subtopics, seed=split_seed, model_name=model_name)
    return _build_graph(records, topic_of, topic_text, sub_text,
                        split_seed=split_seed, train_frac=train_frac,
                        val_frac=val_frac, leakage_safe=leakage_safe)


def _selftest() -> None:
    """Option ordering, missing-code filtering, the min_group floor, native topics."""
    import tempfile

    import pandas as pd

    d = tempfile.mkdtemp()

    # Two editions of one topic (native subtopics) + one single-edition topic.
    def _write(name, n):
        rows = {
            "V1": ([1] * (n // 2)) + ([2] * (n // 4)) + ([9] * (n - n // 2 - n // 4)),
            "V2": ([1] * (n // 3)) + ([3] * (n - n // 3)),
            "SEX": ([1] * (n // 2)) + ([2] * (n - n // 2)),
            "AGE_GROUP": ([1] * (n // 2)) + ([2] * (n - n // 2)),
            "WEIGHT": [1.0] * n,
        }
        p = os.path.join(d, name)
        pd.DataFrame(rows).to_stata(
            p, write_index=False,
            variable_labels={"V1": "Is religion important?",
                             "V2": "Should the state fund churches?"},
            value_labels={"V1": {1: "Yes", 2: "No", 9: "Can't choose"},
                          "V2": {1: "Strongly agree", 3: "Strongly disagree"},
                          "SEX": {1: "Male", 2: "Female"},
                          "AGE_GROUP": {1: "18-44", 2: "45+"}})
        return p

    _write("religion_2018.dta", 400)
    _write("religion_2008.dta", 400)
    _write("environment_2020.dta", 400)

    recs = parse_issp_dir(d, min_group=50)
    assert recs, "parsed nothing"

    v1 = [r for r in recs if r["qkey"].endswith("V1")]
    assert v1, "V1 missing"
    assert v1[0]["options"] == ["Yes", "No"], \
        f"'Can't choose' must be dropped and code order kept: {v1[0]['options']}"
    assert abs(sum(v1[0]["dist"]) - 1.0) < 1e-9, v1[0]["dist"]
    assert {r["attribute"] for r in recs} == {"sex", "age"}, \
        sorted({r["attribute"] for r in recs})
    assert {r["topic"] for r in recs} == {"religion", "environment"}

    # min_group must EMPTY rather than silently keep tiny cells.
    try:
        parse_issp_dir(d, min_group=100000)
    except ValueError as e:
        assert "0 records" in str(e), str(e)
    else:
        raise AssertionError("an unreachable min_group must raise")

    topic_of, topic_text, sub_text = native_topics(recs)
    rel = {k for k in topic_of if k.startswith("religion|")}
    assert len({topic_of[k][1] for k in rel}) == 2, \
        "two religion editions must give two NATIVE subtopics, unclustered"
    assert len(topic_text) == 2 and all(sub_text.values())

    t, y = topic_year_from_name("ZA10000_family_2022.dta")
    assert (t, y) == ("family", "2022"), (t, y)

    print("issp loader self-test OK (code order, missing labels, floor, "
          "native topics, filename parse)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ISSP modules -> graph records")
    ap.add_argument("--inspect", metavar="FILE",
                    help="print the file's real columns and labels -- DO THIS FIRST")
    ap.add_argument("--dir", help="directory of downloaded module files")
    ap.add_argument("--out", default=None, help="write records as jsonl")
    ap.add_argument("--min_group", type=int, default=100)
    ap.add_argument("--axes", default=None, help="comma list; default all found")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.inspect:
        inspect_file(a.inspect)
    elif a.dir:
        rs = parse_issp_dir(a.dir, min_group=a.min_group,
                            axes=a.axes.split(",") if a.axes else None)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
            print(f"  wrote {a.out}")
    else:
        ap.error("--inspect, --dir or --selftest")
