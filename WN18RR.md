# Adding WN18RR — a deep hierarchy benchmark

This document explains the WN18RR integration: *why* we added it, *what* changed
(and, importantly, what didn't), and *how* to run it. It complements
`EXPERIMENTS.md` (roadmap C1: a real, deeper hierarchical KG) and
`LABEL_LEAKAGE.md` (why we needed a harder, non-trivial task).

---

## 1. Why WN18RR

CulturalBench's tree is **shallow (4 levels) and small (~1.3K nodes)**, and after
removing the label leak the bare Tree-GRU (`no_gki`) only matched the frozen-text
floor — structure wasn't helping. That is exactly the regime where hyperbolic
geometry has nothing to exploit. WN18RR fixes both problems:

- **Deep, genuine hierarchy** (WordNet hypernym chains) — where hyperbolic space
  should win, re-testing "does structure help?" under favorable conditions.
- **A real test set** (~3,134 test triples vs. 124), so a measured MRR gap is
  statistically meaningful instead of ~4-triple noise.
- **Leaderboard comparability** with HyperKGR-class baselines.
- **Multi-parent / polysemy structure** — the substrate the plurality directions
  (E2/E3) eventually need.

## 2. The key finding: almost no architecture changes were needed

The PluralTree encoder already handles what WN18RR requires:

- **DAG / multiple parents.** `topological_sort` (`pluraltree/utils/tree_utils.py`)
  is a generic Kahn-style sort that explicitly allows a node to have multiple
  parents. Bottom-up encoding reads each child's `h` once; a child shared by
  several parents just contributes to each. No change.
- **More relations.** `HyperbolicLinkPredictor(num_relations=...)` is
  parameterized; WN18RR's 11 relations work out of the box.
- **Larger graph / different text.** The encoder is size-agnostic; gloss text
  flows through the same `compute_text_embeddings`.

So the model (Tree-GRU cell, child attention, GKI, scoring) is **unchanged**. The
work was a data loader, a candidate/negative-sampling choice, and an evaluation
speedup.

## 3. What was added / changed

### `data/wordnet.py` (new) — `load_wn18rr(...)`
Returns the same `CulturalGraph` the trainer/encoder already consume.

- **Tree from hypernym edges.** For each `_hypernym` / `_instance_hypernym`
  triple `(head, r, tail)`, the **tail is the parent** (hypernym = more general)
  and the **head is the child** (hyponym). These become `children_indices`.
- **Virtual ROOT.** WordNet hypernymy is a *forest of DAGs*; the encoder wants one
  rooted structure. Every entity with no hypernym parent is attached under a
  single synthetic `__ROOT__`, giving one connected, single-rooted DAG that
  covers all entities.
- **Leakage-safe by default.** The tree uses **TRAIN hypernym edges only**, so a
  held-out (val/test) hypernym link is never baked into the structure that
  produces its own endpoints' embeddings — the same principle as the
  CulturalBench fix. Set `leakage_safe=False` to build from all splits.
- **Candidates = all entities (except ROOT).** WordNet has no clean type system,
  so ranking/negative-sampling corrupt against every entity (the standard WN18RR
  setting). All relations share one candidate list.
- **Entity text resolution order:** `entity2textlong.txt` / `entity2text.txt`
  (recommended) → nltk WordNet glosses (if ids are synset offsets and nltk is
  installed) → cleaned raw id (fallback; weak features, prints a warning).

### `evaluation/link_prediction.py` (rewritten ranking)
The old ranking used a **per-candidate Python loop**, fine for 45 countries but
~128M iterations on WN18RR (3,134 × 40,943). It is now **vectorized**:

- Score all candidates at once (`score_all_candidates`).
- Build the filtered set (other known-true objects of the query) and mask their
  scores to `-inf` with tensor indexing instead of a Python loop.
- `rank = (scores > correct_score).sum() + 1`.

Per-relation candidate tensors and an `id → position` map are cached. Behavior is
identical to the old loop (verified: vectorized MRR == reference MRR to 1e-6 on a
synthetic graph), just fast enough for ~40K candidates.

### `scripts/train.py` (dataset switch)
- `--dataset {culturalbench,wn18rr}` (default `culturalbench`).
- `--data_dir` for the WN18RR files (default `data/wn18rr`).
- Branches the loader; everything downstream (`compute_text_embeddings`,
  `build_full_tree_inputs`, trainer, predictor sized by
  `len(graph.relation_vocab)`) is unchanged.

## 4. Getting the data

Place the standard WN18RR release under `data/wn18rr/`:

```
data/wn18rr/train.txt    # head <TAB> relation <TAB> tail
data/wn18rr/valid.txt
data/wn18rr/test.txt
data/wn18rr/entity2text.txt        # optional, recommended (id <TAB> short text)
data/wn18rr/entity2textlong.txt    # optional (id <TAB> definition)
```

- The triple files ship with most KG toolkits (e.g. the ConvE / pykeen WN18RR
  release).
- The `entity2text*.txt` files (KG-BERT format) give meaningful node features. If
  absent, install nltk + `nltk.download('wordnet')` to resolve glosses, or the
  loader falls back to raw ids (not recommended).
- **Narval is offline:** download/prepare these on the login node, same as the
  sentence-transformer model.

## 5. Running

```bash
# train on WN18RR (hypernym tree + virtual root, leakage-safe)
python scripts/train.py --dataset wn18rr --device cuda \
    --d_hidden 128 --n_epochs 300 --embed_model all-mpnet-base-v2

# pure Tree-GRU baseline (no GKI) on WN18RR
python scripts/train.py --dataset wn18rr --no_gki --device cuda --d_hidden 128
```

## 6. Reading the results (important caveats)

- **Absolute MRR will look much lower than CulturalBench.** Ranking is against
  ~40K candidates, not 45 countries; random ≈ 1/40943. Compare variants against
  each other and against a frozen floor, **never across datasets**.
- **Frozen-NN floor:** `scripts/frozen_baseline.py` currently loads CulturalBench
  only. For a WN18RR floor it needs the same `--dataset` switch and the vectorized
  ranking (its inner loop is also per-candidate). This is the recommended next
  small follow-up so the A3 comparison exists on WN18RR too.
- **Text-feature quality matters.** Without `entity2text*.txt` the node features
  are weak; ensure glosses are available before trusting numbers.

## 7. What this unlocks on the roadmap

- **C1** (deeper hierarchy) — done, this loader.
- **A1/A3 re-run on a real hierarchy** — the honest test of whether the Tree-GRU
  structure beats a frozen floor when the hierarchy is deep.
- **C3 (DAG)** — already exercised: WordNet nodes genuinely have multiple
  hypernym parents, which the encoder now handles.
- **E2/E3 (plurality)** — polysemous synsets / multi-parent nodes are the natural
  data for distinct per-parent existences and measured variance.

---

**Files:** `data/wordnet.py` (loader), `evaluation/link_prediction.py`
(vectorized ranking), `scripts/train.py` (`--dataset` / `--data_dir`).
