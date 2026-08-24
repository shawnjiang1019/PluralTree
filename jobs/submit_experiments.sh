#!/bin/bash
# Run the four oracle-gap / reward experiments. Login node, NOT an sbatch script:
#
#     bash jobs/submit_experiments.sh
#
# Step 0 (position statements) runs INLINE because everything else reads the
# artifact it writes. The four jobs are then mutually independent and go in
# parallel. Idempotent: a job already pending/running under the same name is
# left alone rather than duplicated (FORCE=1 overrides).
#
#   [0] build_position_statements    inline, ~1 min   declarative embed_text
#   [3] reward correlation gate      GPU, ~15 min     did the fix make the reward fire?
#   [5] merge_v2 OvertonBench        4 GPU, ~6 h      lossless merge -> union ceiling
#   [1] selector_search              GPU, ~30 min     can a rule reach oracle 0.6344?
#   [2] bestofk_selection            GPU, ~30 min     can the pool be its own rubric?
#
# Knobs: RESP/SCORES (default v8), OUT9/SCORES9 (merge run), ARTIFACT, SKIP,
#        FORCE, DRY. e.g.  SKIP="5" bash jobs/submit_experiments.sh
#                          DRY=1 bash jobs/submit_experiments.sh   # print only

set -uo pipefail

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"
export PYTHONUNBUFFERED=1
# Compute nodes have NO outbound network. Jobs [1]/[2] are --wrap one-liners with
# no preamble of their own, so they inherit THIS environment -- without these,
# sentence-transformers probes the hub for an adapter config and dies on
# "Network is unreachable" (selector_search_244676.err). The other jobs set these
# in their own scripts; these two cannot.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$(git rev-parse --show-toplevel)" || exit 1
mkdir -p logs artifacts docs

ACCT="${ACCT:-def-enaskt}"
RESP="${RESP:-overton_responses_v8.jsonl}"
SCORES="${SCORES:-overton_scores_v8.csv}"
OUT9="${OUT9:-overton_responses_v9.jsonl}"
SCORES9="${SCORES9:-overton_scores_v9.csv}"
ARTIFACT="${ARTIFACT:-artifacts/position_statements.jsonl}"
SKIP="${SKIP:-}"
DRY="${DRY:-0}"

skipped () { case " ${SKIP} " in *" $1 "*) return 0;; *) return 1;; esac; }

# A job whose name is already queued is reused, not resubmitted -- rerunning this
# script after a partial failure must not double-book the allocation.
queued () {
    [ "${FORCE:-0}" = "1" ] && return 1
    squeue -h -u "$USER" -n "$1" -t PD,R -o '%i' 2>/dev/null | sort -n | head -1 | grep .
}

ALL_IDS=()
submit () {   # submit <job-name> <sbatch args...>
    local name="$1"; shift
    local id
    if id=$(queued "${name}"); then
        echo "  ${id}  ${name}  (already queued -- reusing)"
        ALL_IDS+=("${id}"); return 0
    fi
    if [ "${DRY}" = "1" ]; then
        echo "  DRY   ${name}"; printf '        sbatch %s\n' "$*"; return 0
    fi
    id=$(sbatch --parsable --job-name="${name}" --account="${ACCT}" \
                --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" \
                "$@") || { echo "  FAIL  ${name}"; return 1; }
    echo "  ${id}  ${name}"
    ALL_IDS+=("${id}")
}

# --- [0] position statements: inline, everything downstream reads it ----------
echo "=== [0] position statements (inline) ============================"
if skipped 0; then
    echo "  SKIP (SKIP contains 0)"
elif [ -f "${ARTIFACT}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  ${ARTIFACT} exists -- reusing (FORCE=1 to rebuild)"
elif [ "${DRY}" = "1" ]; then
    echo "  DRY   python scripts/build_position_statements.py --dataset opinionqa --out ${ARTIFACT}"
else
    python scripts/build_position_statements.py --dataset opinionqa \
        --out "${ARTIFACT}" --force \
        || { echo "ARTIFACT BUILD FAILED -- 1/2/3 would silently fall back to"
             echo "the old '<question> <option>' target and stay broken. Stopping."
             exit 1; }
fi
export POSITION_STATEMENTS="${PWD}/${ARTIFACT}"
echo "  POSITION_STATEMENTS=${POSITION_STATEMENTS}"

EXPORTS="ALL,POSITION_STATEMENTS=${POSITION_STATEMENTS}"

echo ""
echo "=== submitting (all independent, run in parallel) ==============="

# --- [3] does the reward fire now? --------------------------------------------
skipped 3 || submit reward_corr --export="${EXPORTS}" \
    jobs/eval/job_reward_correlation.sh

# --- [5] merge_v2: the union ceiling ------------------------------------------
# CONDS holds commas, and sbatch --export SPLITS ON COMMAS -- passing it inline
# would silently reduce CONDS to "baseline" and the run would prove nothing.
# Export in this shell and let --export=ALL carry it.
if ! skipped 5; then
    # job_overton_eval.sh reads OUT/SCORES from the environment, and SCORES is
    # ALSO this script's name for the v8 scores file that jobs [1]/[2] need.
    # Unsetting it after the export killed them under `set -u`. Save and restore.
    _scores_v8="${SCORES}"
    export MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ TAU=0.25 MAXU=20 \
           DATASET=opinionqa EMB=embeddings_opinionqa.pt FEATS=feats_opinionqa.pt \
           CONDS=baseline,merge,merge_v2 OUT="${OUT9}" SCORES="${SCORES9}"
    submit overton_merge_v2 --time=06:00:00 --export=ALL jobs/eval/job_overton_eval.sh
    unset MODEL TAU MAXU DATASET EMB FEATS CONDS OUT
    SCORES="${_scores_v8}"
fi

# --- [1] can a rule reach oracle? ---------------------------------------------
skipped 1 || submit selector_search --gres=gpu:1 --time=00:30:00 --mem=32G \
    --cpus-per-task=8 --export="${EXPORTS}" \
    --wrap="python scripts/analysis/selector_search.py --responses ${RESP} --scores ${SCORES}"

# --- [2] can the pool be its own rubric? --------------------------------------
skipped 2 || submit bestofk --gres=gpu:1 --time=00:30:00 --mem=32G \
    --cpus-per-task=8 --export="${EXPORTS}" \
    --wrap="python scripts/analysis/bestofk_selection.py --responses ${RESP} --scores ${SCORES} --sim_thr 0.5,0.6,0.7 --min_support 1,2"

# --- [6] summary: one file to read in the morning -----------------------------
# afterANY, not afterok: a failed experiment is a result too, and the whole point
# is that nothing needs babysitting. Runs once every other job has stopped.
SUMMARY="${SUMMARY:-docs/experiment_summary.txt}"
if [ "${DRY}" != "1" ] && [ ${#ALL_IDS[@]} -gt 0 ]; then
    dep=$(printf ':%s' "${ALL_IDS[@]}")            # --dependency wants colons
    csv=$(IFS=,; echo "${ALL_IDS[*]}")             # sacct -j wants commas
    sbatch --parsable --job-name=exp_summary --account="${ACCT}" \
        --time=00:10:00 --mem=4G --cpus-per-task=1 \
        --output="logs/exp_summary_%j.out" --error="logs/exp_summary_%j.err" \
        --dependency="afterany${dep}" --kill-on-invalid-dep=no \
        --wrap="cd ${PWD} && {
          echo '=== experiment summary' \$(date) '==='
          echo; echo '--- [3] reward gate ---'
          grep -hE 'GATE:|pos_best|unit_best|tie_rate|conc\|separated|frac_with_ZERO' logs/reward_corr_*.out 2>/dev/null | tail -40
          echo; echo '--- [1] selector search ---'
          grep -hA25 'frac_gap' logs/selector_search_*.out 2>/dev/null | tail -40
          echo; echo '--- [2] best-of-K ---'
          tail -40 logs/bestofk_*.out 2>/dev/null
          echo; echo '--- [5] merge_v2 ---'
          grep -hE 'OvertonScore|condition|merge' logs/overton_merge_v2_*.out 2>/dev/null | tail -30
          echo \"  merge_fallback rows: \$(grep -c merge_fallback ${OUT9} 2>/dev/null || echo n/a)\"
          echo; echo '--- job states ---'
          sacct -X -j ${csv} --format=JobID,JobName%22,State,Elapsed 2>/dev/null
        } > ${SUMMARY} 2>&1" \
        && echo "  summary -> ${SUMMARY} (after all jobs stop)"
fi

echo ""
echo "Track:  squeue --me"
echo "Morning: cat ${SUMMARY}"
