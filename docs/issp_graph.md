# Building the ISSP graph

How the International Social Survey Programme becomes a PluralTree viewpoint
graph, and every transformation applied to make the raw GESIS release embeddable.
Written 2026-09-13, against ISSP 2016 Role of Government V (ZA6900).

---

## 1. Why a second graph at all

Every result in this project is retrieved from one corpus: Pew's American Trends
Panel via OpinionQA. Two distinct objections follow, and ISSP answers a different
one than it was originally chosen for.

**Original motivation (now largely spent).** OpinionQA has no topic labels, so
`cluster_questions` mints the topic and subtopic layers by k-means over MiniLM
question embeddings. Testing a *hyperbolic embedding* on a hierarchy that an
*embedding model* built is close to circular, and ISSP's eleven named modules
remove that objection. But the geometry ablation has since returned a
well-powered null (Δ = +0.005, 95% CI [−0.040, +0.057], p = 0.88 at NROLL=3),
so there is no geometry claim left to defend.

**Live motivation.** Does merge_v2's gain survive a different survey corpus
entirely? That is a generalisation question, and it is the one ISSP still
answers. Note the result is two-sided in an interesting way: reproducing ~+0.04
on ISSP supports "the method is robust" while simultaneously undercutting "this
specific graph's content is what matters" — the same tension `merge_v2_divrand`
tests from the selection side.

**What stays fixed.** The eval does not change. Same 60 OvertonBench questions,
same judge, same conditions. Only the retrieval graph is swapped. This is the
opposite of the VITAL experiment, which keeps the ATP graph and swaps the *task*.

---

## 2. Target structure

ISSP must produce the canonical record the rest of the pipeline consumes,
identical to `parse_atp_dir`'s output:

```python
{"qkey": str, "question": str, "options": [str],
 "attribute": str, "group": str, "dist": [float]}   # dist sums to 1
```

From those records, `data/loaders/opinionqa._build_graph` is reused unchanged,
producing:

```
root → topic (module) → subtopic (edition year) → question
                      → demographic axis → subgroup opinion leaf
```

The topic level comes from ISSP's own module names; the subtopic level from the
edition year when a topic has ≥2 editions, and from k-means within the module
when it has only one. `native_topics` reports the split explicitly, because
"how much of this hierarchy is native" is a property of each run.

---

## 3. Transformations applied to the raw release

Each of the following was required to make the GESIS file embeddable. All were
found by running `--inspect` against the real file rather than from
documentation — the GESIS variable-documentation page returns 403, and guessing
column names is what cost an earlier ValuePrism run.

### 3.1 Reading Stata value labels (`read_table`)

Download **`.dta`, not `.sav`**: pandas reads Stata variable and value labels
natively, whereas `.sav` needs `pyreadstat`, which is absent from the cluster
venv and cannot be installed on an offline node.

Stata stores value-label **sets** named independently of the columns that use
them — a real GESIS file shares one set (e.g. `AGREE5`) across dozens of items.
pandas exposes the per-variable set names as the private `_lbllist`. Reading the
public-looking `lbllist` yields nothing, every column falls through to a
name-match fallback, and the parser then skips *every* item for having no
options — a silent zero-record parse.

A synthetic selftest did not catch this: `DataFrame.to_stata` happens to name
each label set after its column, so the name-match branch worked there and only
failed on a real file. A categorical re-read fallback now recovers the mapping if
`_lbllist` is ever absent.

### 3.2 Item selection (`_ITEM_NAME`)

Substantive items match `^(V\d{1,3}|Q\d{1,3}[a-z]?)$`, case-insensitively — the
2016 module uses lowercase `v1`…`v63`. Background, administrative and technical
variables are excluded by name so they never become graph questions. 63 of 395
columns survive.

### 3.3 Option vocabulary and ordering

Options come from each item's value-label set, ordered by **numeric code**, not
by label text. This is load-bearing: a lexical sort would order `"1", "10", "2"`
and destroy the pole-to-pole ordering that `fork_context_full`'s spectrum render
and `pick_personas`' pole-first selection both depend on. Items with fewer than
2 or more than 12 options are dropped.

### 3.4 Non-response filtering (`_MISSING_LABEL`)

ISSP codes non-responses *inside* the value range (8/9, 98/99, or negative) with
labels, not as system-missing. Filtering on the **label** is safer than on the
code, which varies by item width. Two families are removed:

- standard non-response: "No answer", "Don't know", "Can't choose", "Refused",
  "Not applicable", "NAV"
- **coding artifacts**, added after inspecting `PARTY_LR`: "Invalid ballot",
  "Insufficient information to code into scheme", "Not classifiable". These are
  not viewpoints. Left in place, "Invalid ballot" becomes a subgroup whose
  opinion distribution the scout could select as a fork *pole*.

Legitimate categories survive — "Far right (fascist etc.)" and "Center, liberal"
are retained.

### 3.5 Binning raw numeric axes (`AXIS_BINS`) — the largest single change

ISSP ships `AGE` as **age in years**. With 48,720 pooled respondents, every
single year clears the `min_group=100` floor, so the unbinned axis produced
**69 single-year subgroups**:

```
age            69   ['18', '19', '20', '21', '22', '23', ...]
religion       11   ['Buddhist', 'Catholic', 'Hindu', ...]
sex             2   ['Female', 'Male']
```

That is not a viewpoint partition, and it would dominate the graph: roughly 69 of
~120 leaves per question would be one-year age slices, so the scout would score
forks almost entirely between adjacent birth years while sex, religion and party
competed for the remainder. Record count was inflated to 7,541 from an expected
~3,400.

Age is now banded to `18-29 / 30-44 / 45-59 / 60+` — the standard survey cut, and
the granularity of OpinionQA's own age axis.

**Binning happens before the groupby, not after.** Grouping by raw code and then
relabelling would leave 18 and 19 as separate cells that both happen to be named
"18-29".

### 3.6 Naming bare-numeral groups

`TOPBOT` (self-placement ladder) comes through as `"02"`…`"10"`. The group name
is what `describe_node` renders into the injected fork block, and "02" names
nothing to a reader, so bare numerals are prefixed with their axis → `"class 02"`.
The ladder is genuinely ordinal, so it is kept as a spectrum axis rather than
dropped.

### 3.7 Cell pooling and the group floor

Cells are pooled **across countries**. `min_group=100` matches `parse_atp_dir`.
Crossing country × demographic axis would thin cells below the floor
immediately — ISSP national samples run ~1,200 respondents, against ~48,000
pooled. Country is read and reported by `--inspect` but is *not* used as an axis.

Below ~15 respondents a per-group distribution is mostly sampling noise, and that
noise propagates directly into the Wasserstein fork scores — a low floor does not
buy more signal, it buys more confident nonsense. 1,216 cells fell below the
floor in the unbinned parse; binning should reduce this sharply, since each band
pools ~15 single years.

### 3.8 Downstream routing fixes

Two bugs outside the loader that only surfaced with a third dataset:

- `load_or_compute_text_feat` branched on dataset name and sent anything not
  `opinionqa`/`globalopinionqa` to CultureBank's feature builder. ISSP builds an
  `OpinionQAGraph`, so it now takes the same builder.
- The same function loaded a cached `text_feat` tensor **without a shape check**.
  The eval job's default is `feats_goqa.pt`; one forgotten variable would have
  silently misaligned every relevance score against the wrong nodes — no crash,
  wrong answers. A node-count guard now raises instead.

---

## 4. Pipeline

```bash
export ISSP_DIR=$HOME/projects/def-enaskt/shawnj/data/issp

# a. INSPECT FIRST — never parse on guessed column names
python -m data.loaders.issp --inspect $ISSP_DIR/role_of_government_2016.dta

# b. parse to canonical records
python -m data.loaders.issp --dir $ISSP_DIR --out issp_records.jsonl

# c. embed (EMB derives from DATASET)
DATASET=issp sbatch jobs/embed/job_embed_opinionqa.sh

# d. THE GATE — reference-only; the question set IS OvertonBench
SRC=none DATASET=issp EMB=embeddings_issp.pt OUT=docs/anchor_cov_issp.csv \
    sbatch jobs/eval/job_anchor_coverage.sh

# e. only if (d) passes — same questions, same judge, swapped graph
DATASET=issp EMB=embeddings_issp.pt FEATS=feats_issp.pt \
    CONDS=baseline,merge_v2 NROLL=3 \
    OUT=overton_responses_issp.jsonl SCORES=overton_scores_issp.csv \
    sbatch jobs/eval/job_overton_eval.sh
```

Filenames must be readable — `religion_2018.dta`, not `ZA6900_v2-0-0.dta`. The
loader parses topic and year from the filename, and the ZA number carries
neither. **This is the step that makes the topic level native rather than
invented.**

`OUT` must be passed explicitly to the gate: it defaults to
`docs/anchor_cov_${SRC}.csv`, and two `SRC=none` runs silently overwrite each
other (this already destroyed one VITAL gate result).

---

## 5. The gate, and how to read it

Step (d) is a hard gate, not a formality. ISSP 2016 is Role of Government;
OvertonBench is 60 contested social and political questions. If the ISSP graph
resolves anchors for only a small fraction of them, the scout returns no forks,
every arm silently collapses to baseline, and the run produces a confident null
that means nothing.

Read resolution and median top-fork relevance **against the OvertonBench
reference printed in the same run**, never in the abstract. For calibration, the
ATP graph resolves 60/60 at median relevance 0.490; VITAL's situations resolved
489/500 at 0.400; INFINITY-CHAT's creative queries resolved 71/100 at 0.326.

Resolution is necessary, not sufficient — a resolved anchor can still be the
wrong survey question.

---

## 6. Current state and limitations

**One module loaded.** With a single topic, `native_topics` reports "1 native
topic" and clusters the subtopics within it — so the native-hierarchy property
that justifies ISSP does not yet exist. Role of Government 2016 alone tests the
pipeline, not the argument.

**To make the hierarchy actually native**, add three more topics and, where
possible, two editions each:

| module | facet on GESIS | role |
|---|---|---|
| Role of Government 2016 | Government, political systems | loaded; carries the gate |
| Social Inequality 2019 | Equality, inequality, social exclusion | political |
| Environment 2020 | Environment and conservation | different domain |
| Religion 2018 | (scroll past "Language and linguistics") | different domain, overlaps on abortion |

Take the **integrated cross-national file**, not per-country releases — a single
national file is ~1,200 respondents and would thin every cell below the floor.

**`income` is unmatched** and expected to stay so: ISSP does not harmonise income
across countries, shipping country-specific variables instead. Nine axes (sex,
age, education, marital, employment, religion, urbanrural, class, politics) is
ample.

**Age bands are a judgement call.** `18-29 / 30-44 / 45-59 / 60+` is a standard
cut, not something ISSP prescribes. A different banding would produce a different
graph, and the choice should be stated in any writeup rather than presented as
given.
