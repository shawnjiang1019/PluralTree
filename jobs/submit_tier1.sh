#!/bin/bash
# Run every Tier 1 experiment (docs/direction.txt). Login node, NOT an sbatch
# script:
#
#     bash jobs/submit_tier1.sh
#     DRY=1 bash jobs/submit_tier1.sh          # print what would be submitted
#     SKIP="4" bash jobs/submit_tier1.sh       # skip the 7B run
#
# NOT one sbatch job, deliberately. The arms need different allocations -- the
# 7B run wants 1 GPU, the 72B runs want 4 -- so a single script would idle three
# GPUs for hours, and one stage failing would take the rest with it. These are
# five independent submissions.
#
#   [1a] geometry c=0.5   4 GPU ~8h   hyperbolic: the reference arm
#   [1b] geometry c=0     4 GPU ~8h   Euclidean: does curvature buy coverage?
#   [2]  selection        4 GPU ~12h  max-W vs random sibling pairs, + flat
#   [3]  issp gate        1 GPU ~2h   does a second graph resolve OvertonBench?
#   [4]  scale 7B         1 GPU ~12h  does merge_v2's gain survive a small model?
#
# WHAT THIS ANSWERS. Nothing currently shows the hyperbolic embedding or the
# divergence scout helps COVERAGE -- link-prediction MRR 0.456 validates the
# embedding as a KG model and nothing more. [1a]/[1b] vary only the curvature;
# [2] varies only which child pairs survive; merge_v2_flat removes the tree
# outright. If merge_v2's gain survives all of them, the geometry and the
# selection are not what produced it, and that is worth knowing before writing
# it up.
#
# Knobs: MAXQ, MAXU, NROLL, SEED, TAU, ACCT, ISSP_DIR, SKIP, FORCE, DRY.
# Smoke everything first:  MAXQ=5 MAXU=10 NROLL=1 DRY=0 bash jobs/submit_tier1.sh

set -uo pipefail

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"
export ISSP_DIR="${ISSP_DIR:-$HOME/projects/def-enaskt/shawnj/data/issp}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$(git rev-parse --show-toplevel)" || exit 1
mkdir -p logs docs

ACCT="${ACCT:-def-enaskt}"
SEED="${SEED:-42}"
TAU="${TAU:-0.25}"
MAXQ="${MAXQ:-0}"
MAXU="${MAXU:-20}"
NROLL="${NROLL:-3}"
FEATS="${FEATS:-feats_opinionqa.pt}"
SKIP="${SKIP:-}"
DRY="${DRY:-0}"

skipped () { case " ${SKIP} " in *" $1 "*) return 0;; *) return 1;; esac; }

# A job already queued under this name is reused, not resubmitted -- rerunning
# after a partial failure must not double-book a 4-GPU allocation.
queued () {
    [ "${FORCE:-0}" = "1" ] && return 1
    squeue -h -u "${USER:-$(id -un)}" -n "$1" -t PD,R -o '%i' 2>/dev/null \
        | sort -n | head -1 | grep .
}

ALL_IDS=()
LAST_ID=""
submit () {   # submit <job-name> <sbatch args...>
    local name="$1"; shift
    local id
    if id=$(queued "${name}"); then
        echo "  ${id}  ${name}  (already queued -- reusing)"
        ALL_IDS+=("${id}"); LAST_ID="${id}"; return 0
    fi
    if [ "${DRY}" = "1" ]; then
        echo "  DRY   ${name}"; printf '        sbatch %s\n' "$*"; LAST_ID=""; return 0
    fi
    id=$(sbatch --parsable --job-name="${name}" --account="${ACCT}" \
                --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" \
                "$@") || { echo "  FAIL  ${name}"; return 1; }
    echo "  ${id}  ${name}"
    ALL_IDS+=("${id}"); LAST_ID="${id}"
}

# --- the shared text-feature cache -------------------------------------------
# [1a], [2] and [4] all call load_or_compute_text_feat on the SAME file. If it
# does not exist yet, three jobs would compute and write it concurrently and the
# loser's partial write is what the others read. So when it is missing, [1a]
# builds it and everything else waits on [1a]; when it is present, all four go
# in parallel. afterany, not afterok: [1a] failing in its EVAL stage still
# leaves a valid feats file, and blocking the other arms on that would be wrong.
DEP=""
if [ -f "${FEATS}" ]; then
    echo "${FEATS} present -- all arms submit in parallel"
else
    echo "${FEATS} MISSING -- [1a] builds it; [2]/[4] wait on it (avoids a"
    echo "  concurrent-write race on the text-feature cache)"
fi

echo ""
echo "=== [1] geometry ablation: does curvature buy coverage? ========="
# The curvature is baked into EMB/OUT/SCORES inside the job (c0p5 / c0), so the
# two arms cannot overwrite each other -- the likeliest way this experiment
# silently reports one number twice.
if ! skipped 1; then
    # NROLL is SHARED with [2], and --export=ALL ships whatever this shell holds
    # at submit time. Setting it to 1 here without restoring would silently give
    # the selection ablation 1 rollout instead of 3 -- the same clobber
    # submit_experiments.sh hit with SCORES. Save and restore.
    _nroll="${NROLL}"
    export TAU SEED MAXQ MAXU
    export CURV=0.5 CONDS=baseline,merge_v2 NROLL=1
    submit geom_c0p5 --export=ALL jobs/eval/job_geometry_ablation.sh
    [ -z "${DEP}" ] && [ ! -f "${FEATS}" ] && [ -n "${LAST_ID}" ] \
        && DEP="--dependency=afterany:${LAST_ID}"

    export CURV=0
    submit geom_c0 ${DEP} --export=ALL jobs/eval/job_geometry_ablation.sh
    unset CURV CONDS
    NROLL="${_nroll}"
else
    echo "  SKIP (SKIP contains 1)"
fi

echo ""
echo "=== [2] selection + hierarchy ablation =========================="
# CONDS holds commas and `sbatch --export=CONDS=...` SPLITS ON COMMAS, which
# would silently reduce it to "baseline" and the run would prove nothing.
# Export in this shell and let --export=ALL carry it.
if ! skipped 2; then
    export TAU SEED MAXQ MAXU NROLL
    export CONDS=baseline,merge_v2,merge_v2_divrand,merge_v2_flat
    export OUT=overton_responses_sel.jsonl SCORES=overton_scores_sel.csv
    submit selection_ablation ${DEP} --export=ALL jobs/eval/job_selection_ablation.sh
    unset CONDS OUT SCORES
else
    echo "  SKIP (SKIP contains 2)"
fi

echo ""
echo "=== [3] ISSP gate: does a second graph reach OvertonBench? ======"
# Gated on the data actually being there. ISSP is not on HuggingFace -- it is a
# GESIS download behind a free account -- so this stage is skipped rather than
# failed when the directory is empty.
if skipped 3; then
    echo "  SKIP (SKIP contains 3)"
elif ! ls "${ISSP_DIR}"/*.dta >/dev/null 2>&1; then
    echo "  SKIP: no .dta files in ${ISSP_DIR}"
    echo "  Download the modules from GESIS (Stata .dta, not .sav), then run"
    echo "    python -m data.loaders.issp --inspect ${ISSP_DIR}/<file>.dta"
    echo "  BEFORE parsing -- the column names are unverified and guessing them"
    echo "  is what cost the ValuePrism run."
else
    export DATASET=issp EMB=embeddings_issp.pt
    submit issp_embed --export=ALL jobs/embed/job_embed_opinionqa.sh
    ISSP_DEP=""
    [ -n "${LAST_ID}" ] && ISSP_DEP="--dependency=afterok:${LAST_ID}"
    export SRC=none
    submit issp_gate ${ISSP_DEP} --export=ALL jobs/eval/job_anchor_coverage.sh
    unset DATASET EMB SRC
    echo "  NOTE: the gate only MEASURES resolution. Do not submit the ISSP eval"
    echo "  until you have compared its rate against the ATP graph's -- a graph"
    echo "  that resolves few OvertonBench anchors makes every arm collapse to"
    echo "  baseline and produces a confident null that means nothing."
fi

echo ""
echo "=== [4] scale: does merge_v2 survive a 7B? ======================"
if ! skipped 4; then
    _nroll="${NROLL}"
    export TAU SEED MAXQ MAXU
    export CONDS=baseline,merge_v2 NROLL=1
    submit scale_7b ${DEP} --export=ALL jobs/eval/job_scale_7b.sh
    unset CONDS
    NROLL="${_nroll}"
else
    echo "  SKIP (SKIP contains 4)"
fi

echo ""
echo "================================================================"
if [ "${DRY}" = "1" ]; then
    echo "DRY run -- nothing submitted."
    exit 0
fi
echo "submitted: ${ALL_IDS[*]:-none}"
echo ""
echo "  squeue -u ${USER:-$(id -un)}"
echo ""
echo "READ THE MANIPULATION CHECK BEFORE THE SCORES for [2]. If merge_v2_divrand"
echo "does not show a LOWER mean_w than merge_v2 at the SAME n_forks, random"
echo "selection drew the same pairs, the ablation did nothing, and a tie means"
echo "nothing rather than something:"
echo ""
echo "  python -c \"import json,statistics as st,collections;"
echo "  by=collections.defaultdict(list);"
echo "  [by[r['condition']].append(r['pair_select']) for r in map(json.loads,"
echo "   open('overton_responses_sel.jsonl')) if r.get('pair_select')];"
echo "  [print(f'{c:18} n={st.mean(p[\\\"n_forks\\\"] for p in v):5.2f}"
echo "   w={st.mean(p[\\\"mean_w\\\"] for p in v if p[\\\"mean_w\\\"]):.4f}')"
echo "   for c,v in sorted(by.items())]\""
echo ""
echo "Then: merge_v2 c=0.5 vs c=0, merge_v2 vs merge_v2_flat,"
echo "      merge_v2 vs merge_v2_divrand, and baseline-vs-merge_v2 inside [4]."
echo "[4]'s absolute numbers are NOT comparable to v10/v11/v12 -- different"
echo "generator AND different judge."
