# Plan — PrimeKG Encoding + Geometry-First Evaluation

End goal: construct good hyperbolic embeddings of a knowledge graph so that
**subtree similarity** and **path similarity** are recoverable from distances,
to hand an LLM diverse, structurally-related context for reasoning. Link
prediction is a training signal and sanity anchor, **not** the primary yardstick
(see `docs/EVALUATION.md`).

Two tracks. **Track A** (PrimeKG data) unblocks running anything; **Track B**
(evaluation) serves the real goal and can be built/tested on an existing WN18RR
checkpoint in parallel.

---

## Track A — Get PrimeKG into the existing pipeline

The encoder needs a rooted DAG (`children_indices` + `topo_order`). PrimeKG is a
general multi-relational biomedical KG (~129K nodes, ~8M directed edges, 10 node
types, 30 relations) but contains ontology hierarchies → same recipe as WN18RR:
extract an ontology backbone, return a `CulturalGraph`, change nothing in the
encoder/trainer.

### A1. `scripts/get_primekg.py` — downloader (run on Narval login node)
- Stdlib `urllib` download of `kg.csv` (+ `nodes.csv` if useful) from Harvard
  Dataverse into `data/primekg/`.
- Print size / row counts; idempotent (skip if present). Mirrors `get_wn18rr.py`.
- Compute nodes are offline → must be run on the login node.

### A2. `data/primekg.py` → `load_primekg(...) -> CulturalGraph`
- Parse `kg.csv` (`relation, x_index, x_type, x_name, y_index, y_type, y_name, …`).
- **Split relations into three roles** (parameterized):
  - *backbone* = ontology parent-child relations (`disease_disease`,
    `anatomy_anatomy`, `bioprocess_bioprocess`, `molfunc_molfunc`,
    `cellcomp_cellcomp`, `phenotype_phenotype`, `pathway_pathway`) → build
    `children_indices` + virtual `__ROOT__`, like `data/wordnet.py`.
  - *target* = held-out LP relation(s). **Default `drug↔disease`**
    (indication / contraindication / off-label); swappable via
    `--target_relations` to drug–protein / disease–protein.
  - *knowledge* = everything else → available to GKI.
- **Leakage-safe**: backbone tree from TRAIN edges only; target edges never
  appear in the tree.
- Node text feature = `"{x_name} ({x_type})"` for the sentence-transformer.
- **Real `type_constraints`**: candidates restricted to the object's node type
  (rank a disease only against diseases). Main new behavior vs WN18RR.
- Splits: random per-target-relation split with `split_seed` (PrimeKG ships none).
- Returns a `CulturalGraph` — zero encoder/trainer changes.

### A3. `scripts/train.py` integration
- Add `"primekg"` to `--dataset`; add `--target_relations`; branch to
  `load_primekg`. Reuse existing `--checkpoint`, `--data_dir`.

### A4. `jobs/job_primekg.sh`
- Clone `job_wn18rr.sh`: `gpu:1`, larger `--mem` (~48–64G), `--checkpoint`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--injection pre_agg`,
  `--d_hidden 128`, offline env vars.

### A5. Scale guard — `--encode_every N`
- Encode the full tree once per N epochs instead of every batch (129K nodes is
  too big to re-encode per step). Small change in `Trainer.train`.

### A6. Local CPU smoke test
- Tiny subset / few epochs to confirm load → encode → train → eval end-to-end
  before any cluster job.

---

## Track B — Evaluate the geometry (serves the real goal; from `docs/EVALUATION.md`)

### B1. `evaluation/structure_metrics.py`
Intrinsic, `torch.no_grad()`, takes `h_all` + `children_indices` + `topo_order`.
- Tier 1: distortion (sampled pairs), reconstruction MAP, **depth–radius
  Spearman ρ**.
- Tier 2: same-subtree retrieval AP, LCA-depth recovery.
- Tier 3: ancestor/descendant AUC.
- All pair-sampled with a seed (O(N²) guard for 129K nodes).

### B2. Wire depth–radius ρ into the trainer's eval log
- Cheapest signal; report every eval.

### B3. (optional, gated) structure-fidelity loss `λ_struct` in `train_step`
- Push ancestors closer than non-ancestors so training targets geometry, not
  just links. **Recommendation: eval-only first** (measure the gap before
  changing the objective).

### B4. Update `docs/EVALUATION.md` status checklist as items land.

---

## Execution order
1. **B1 + B2** — pure functions, testable today on a WN18RR `h_all`, no PrimeKG
   dependency.
2. **A2 → A3 → A5 → A6** — loader + integration + scale guard + local smoke.
3. **A1 + A4** — downloader + SLURM (run on Narval).
4. **B3** last, only if training should optimize geometry.

## Open decisions
- **LP target**: default **drug↔disease** unless changed (loader is
  parameterized regardless).
- **B3 structure loss**: include now vs eval-only first. Rec: eval-only first.

## Constraints
- Never run `git commit` / `push` — provide commands only.
- Narval compute nodes are offline: pre-download models/data on the login node;
  pyarrow from the `arrow` module.
